from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from experiments.dendrite.phase_1_synthetic_experiments.discrete_selection import LearnableDiscreteDD

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
NX = NY = 8


def generate(n: int, seed: int, device: str = "cpu"):
    """Generate 20-parameter heterogeneous 2-D steady heat fields.

    Sixteen log-uniform conductivity blocks form a 4x4 material map; four
    boundary temperatures close the parameter vector. The 8x8 interior field
    solves div(k grad T)=0 with Dirichlet boundary values. The target is the
    center-cell temperature.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    k_blocks = torch.exp(torch.empty(n, 16).uniform_(math.log(.1), math.log(10.), generator=g))
    boundary = torch.empty(n, 4).uniform_(0., 100., generator=g)  # left,right,top,bottom
    k = k_blocks.reshape(n, 4, 4).repeat_interleave(2, 1).repeat_interleave(2, 2)
    k = k.to(device); boundary = boundary.to(device)
    total = NX * NY
    A = torch.zeros(n, total, total, device=device)
    rhs = torch.zeros(n, total, device=device)
    def idx(y, x): return y * NX + x
    def face(a, b): return 2. * a * b / (a + b).clamp_min(1e-8)
    for y in range(NY):
        for x in range(NX):
            p = idx(y, x); kp = k[:, y, x]; diag = torch.zeros(n, device=device)
            for dy, dx, side in ((0,-1,0),(0,1,1),(-1,0,2),(1,0,3)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < NY and 0 <= xx < NX:
                    w = face(kp, k[:, yy, xx]); diag += w; A[:, p, idx(yy, xx)] -= w
                else:
                    # Unit grid spacing; boundary face conductivity is kp.
                    diag += kp; rhs[:, p] += kp * boundary[:, side]
            A[:, p, p] = diag
    temperature = torch.linalg.solve(A, rhs.unsqueeze(-1)).squeeze(-1)
    target = temperature[:, idx(NY // 2, NX // 2)].unsqueeze(1).cpu()
    physical = torch.cat((k_blocks.cpu(), boundary.cpu()), 1)
    x = torch.cat((.5 + (torch.log(k_blocks.cpu()) - math.log(.1)) / math.log(100.), .5 + boundary.cpu() / 100.), 1)
    return {"physical": physical.float().cpu(), "x": x.float().cpu(), "y": target.float().cpu()}


class DDStack(nn.Module):
    def __init__(self, width, layers=4):
        super().__init__(); self.layers = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(layers))
    def forward(self, x):
        anchor = x; h = x
        for layer in self.layers: h = layer(h) * anchor
        return h


class BaseDD(nn.Module):
    name = "base_dd"
    def __init__(self): super().__init__(); self.dd=DDStack(20); self.output=nn.Linear(20,1)
    def forward(self,x): return self.output(self.dd(x))


class ReciprocalSharedDD(nn.Module):
    name = "reciprocal_shared_dd"
    def __init__(self): super().__init__(); self.dd=DDStack(20); self.output=nn.Linear(20,1)
    def forward(self,x): return self.output(self.dd(x)*self.dd(1./x.clamp_min(1e-6)))


class ReciprocalConcatDD(nn.Module):
    name = "reciprocal_concat_dd"
    def __init__(self): super().__init__(); self.dd=DDStack(40); self.output=nn.Linear(40,1)
    def forward(self,x): return self.output(self.dd(torch.cat((x,1./x.clamp_min(1e-6)),1)))


class DuplicateConcatDD(nn.Module):
    name = "duplicate_concat_dd"
    def __init__(self): super().__init__(); self.dd=DDStack(40); self.output=nn.Linear(40,1)
    def forward(self,x): return self.output(self.dd(torch.cat((x,x),1)))


class DeepBaseDD(nn.Module):
    name = "deep_base_dd_16_layers"
    def __init__(self): super().__init__(); self.dd=DDStack(20, 16); self.output=nn.Linear(20,1)
    def forward(self,x): return self.output(self.dd(x))


class LearnableDiscreteInputDD(LearnableDiscreteDD):
    name = "learnable_discrete_dd"

    def __init__(self):
        super().__init__(input_width=20, layers=4, output_dim=1, name=self.name)


MODELS=(BaseDD,ReciprocalSharedDD,ReciprocalConcatDD,DuplicateConcatDD,DeepBaseDD,LearnableDiscreteInputDD)
def counts(): return {c.name:sum(p.numel() for p in c().parameters()) for c in MODELS}


def metric(pred,target,mean,scale):
    actual=pred*scale+mean; truth=target*scale+mean
    return {"mse_normalized":float((pred-target).square().mean()),"temperature_mae":float((actual-truth).abs().mean()),"temperature_rmse":float((actual-truth).square().mean().sqrt()),"relative_error":float(((actual-truth).abs()/truth.abs().clamp_min(1.)).mean())}


def train_model(cls,train,valid,mean,scale,args):
    model=cls().to(args.device)
    if hasattr(model, "_seed"): model._seed = getattr(args, "seed", 0)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    loader=DataLoader(TensorDataset(train['x'],train['y']),batch_size=args.batch,shuffle=True,generator=torch.Generator().manual_seed(args.seed))
    val_loader=DataLoader(TensorDataset(valid['x'],valid['y']),batch_size=args.batch)
    folder=ARTIFACTS/cls.name; folder.mkdir(parents=True,exist_ok=True); best=float('inf')
    for epoch in range(1,args.epochs+1):
        if hasattr(model, "set_epoch"): model.set_epoch(epoch, args.epochs)
        t=time.perf_counter(); model.train(); total=0.; count=0; norms=[]
        for x,y in loader:
            x,y=x.to(args.device),y.to(args.device); opt.zero_grad(set_to_none=True); loss=(model(x)-y).square().mean(); loss.backward(); norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.))); opt.step(); total+=float(loss.detach())*len(x); count+=len(x)
        model.eval(); pp=[]; yy=[]
        with torch.inference_mode():
            for x,y in val_loader: pp.append(model(x.to(args.device)).cpu()); yy.append(y)
        row={"epoch":epoch,"seconds":time.perf_counter()-t,"train_loss":total/count,"gradient_norm":float(np.mean(norms)),**metric(torch.cat(pp),torch.cat(yy),mean,scale)}
        if hasattr(model, "selection_values"): row["selection_values"] = model.selection_values().tolist()
        with (folder/'metrics.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        if row['mse_normalized']<best:
            best=row['mse_normalized']; checkpoint={'model':model.state_dict(),'epoch':epoch,'metrics':row,'parameters':counts()[cls.name]}
            if hasattr(model, "selection_values"): checkpoint["selection_values"] = model.selection_values().tolist()
            torch.save(checkpoint,folder/'best.pt')
        if epoch==1 or epoch%25==0 or epoch==args.epochs: print(json.dumps({'model':cls.name,**row}),flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--train-size',type=int,default=30000); p.add_argument('--valid-size',type=int,default=6000); p.add_argument('--epochs',type=int,default=500); p.add_argument('--batch',type=int,default=4096); p.add_argument('--lr',type=float,default=2e-4); p.add_argument('--seed',type=int,default=20261017); p.add_argument('--device',choices=('cpu','cuda'),default='cuda'); p.add_argument('--only',choices=[c.name for c in MODELS]); args=p.parse_args(); args.device=torch.device(args.device)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); ARTIFACTS.mkdir(parents=True,exist_ok=True)
    train=generate(args.train_size,args.seed,str(args.device)); valid=generate(args.valid_size,args.seed+1,str(args.device)); mean=train['y'].mean(0); scale=train['y'].std(0).clamp_min(1e-6); train['y']=(train['y']-mean)/scale; valid['y']=(valid['y']-mean)/scale
    torch.save({'train':train,'valid':valid,'y_mean':mean,'y_scale':scale},ARTIFACTS/'dataset.pt'); (ARTIFACTS/'parameter_report.json').write_text(json.dumps(counts(),indent=2),encoding='utf-8'); (ARTIFACTS/'manifest.json').write_text(json.dumps({'seed':args.seed,'train_size':args.train_size,'valid_size':args.valid_size,'epochs':args.epochs,'models':[c.name for c in MODELS],'parameter_counts':counts(),'device':str(args.device)},indent=2),encoding='utf-8')
    for cls in MODELS:
        if args.only is None or args.only==cls.name: train_model(cls,train,valid,mean,scale,args)


if __name__=='__main__': main()
