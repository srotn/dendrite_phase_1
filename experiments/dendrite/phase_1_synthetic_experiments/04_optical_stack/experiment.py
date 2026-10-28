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


def lu(g, lo, hi, n):
    return torch.exp(torch.empty(n).uniform_(math.log(lo), math.log(hi), generator=g))


def generate(n: int, seed: int, device: str = "cuda"):
    """Generate six-layer absorbing thin-film reflectance targets.

    Inputs: six real refractive indices, six extinction coefficients, six
    thicknesses, wavelength and incidence angle (20 positive parameters).
    Output: s-polarized reflectance from the complex characteristic matrix.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    real_n = lu(g, 1.3, 3.0, n * 6).reshape(n, 6)
    kappa = lu(g, 1e-4, .08, n * 6).reshape(n, 6)
    thickness = lu(g, 20., 500., n * 6).reshape(n, 6)
    wavelength = lu(g, 400., 900., n)
    angle = torch.empty(n).uniform_(0., math.radians(70.), generator=g)
    real_n, kappa, thickness, wavelength, angle = [x.to(device) for x in (real_n, kappa, thickness, wavelength, angle)]
    complex_n = torch.complex(real_n, kappa)
    sin0 = torch.sin(angle)
    n0 = torch.ones(n, device=device, dtype=torch.complex64)
    cos0 = torch.cos(angle).to(torch.complex64)
    # Fixed glass substrate; the incident medium is air.
    ns = torch.full((n,), 1.52, device=device, dtype=torch.complex64)
    cos_layers = torch.sqrt(1. - (sin0.to(torch.complex64)[:, None] / complex_n).square())
    cos_sub = torch.sqrt(1. - (sin0.to(torch.complex64) / ns).square())
    eta_layers = complex_n * cos_layers
    eta0 = n0 * cos0; eta_s = ns * cos_sub
    M = torch.eye(2, device=device, dtype=torch.complex64).expand(n, 2, 2).clone()
    for layer in range(6):
        delta = 2. * math.pi * complex_n[:, layer] * thickness[:, layer] * cos_layers[:, layer] / wavelength.to(torch.complex64)
        c, s = torch.cos(delta), torch.sin(delta)
        eta = eta_layers[:, layer]
        L = torch.stack((torch.stack((c, 1j * s / eta), -1), torch.stack((1j * eta * s, c), -1)), -2)
        M = M @ L
    num = eta0 * M[:, 0, 0] + eta0 * eta_s * M[:, 0, 1] - M[:, 1, 0] - eta_s * M[:, 1, 1]
    den = eta0 * M[:, 0, 0] + eta0 * eta_s * M[:, 0, 1] + M[:, 1, 0] + eta_s * M[:, 1, 1]
    reflectance = (num / den).abs().square().clamp(1e-7, 1. - 1e-7)
    physical = torch.cat((real_n, kappa, thickness, wavelength[:, None], angle[:, None]), 1)
    # Log scaling preserves positive multiplicative structure; angle is scaled linearly.
    x = torch.cat((
        .5 + (torch.log(real_n.cpu()) - math.log(1.3)) / math.log(3.0 / 1.3),
        .5 + (torch.log(kappa.cpu()) - math.log(1e-4)) / math.log(.08 / 1e-4),
        .5 + (torch.log(thickness.cpu()) - math.log(20.)) / math.log(500. / 20.),
        .5 + (torch.log(wavelength.cpu()) - math.log(400.))[:, None] / math.log(900. / 400.),
        .5 + angle.cpu()[:, None] / math.radians(70.),
    ), 1)
    y = torch.logit(reflectance).cpu().unsqueeze(1)
    return {"physical": physical.float().cpu(), "x": x.float(), "y": y.float()}


class DDStack(nn.Module):
    def __init__(self, width, layers):
        super().__init__(); self.layers = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(layers))
    def forward(self, x):
        anchor = x; h = x
        for layer in self.layers: h = layer(h) * anchor
        return h


class BaseDD(nn.Module):
    name = "base_dd"
    def __init__(self): super().__init__(); self.dd=DDStack(20,4); self.output=nn.Linear(20,1)
    def forward(self,x): return self.output(self.dd(x))


class ReciprocalConcatDD(nn.Module):
    name = "reciprocal_concat_dd"
    def __init__(self): super().__init__(); self.dd=DDStack(40,4); self.output=nn.Linear(40,1)
    def forward(self,x): return self.output(self.dd(torch.cat((x,1./x.clamp_min(1e-6)),1)))


class DuplicateConcatDD(nn.Module):
    name = "duplicate_concat_dd"
    def __init__(self): super().__init__(); self.dd=DDStack(40,4); self.output=nn.Linear(40,1)
    def forward(self,x): return self.output(self.dd(torch.cat((x,x),1)))


class DeepBaseDD(nn.Module):
    name = "deep_base_dd_16_layers"
    def __init__(self): super().__init__(); self.dd=DDStack(20,16); self.output=nn.Linear(20,1)
    def forward(self,x): return self.output(self.dd(x))


class LearnableDiscreteInputDD(LearnableDiscreteDD):
    name = "learnable_discrete_dd"

    def __init__(self):
        super().__init__(input_width=20, layers=4, output_dim=1, name=self.name)


MODELS=(BaseDD,ReciprocalConcatDD,DuplicateConcatDD,DeepBaseDD,LearnableDiscreteInputDD)
def counts(): return {c.name:sum(p.numel() for p in c().parameters()) for c in MODELS}


def metric(pred,target,mean,scale):
    actual=pred*scale+mean; truth=target*scale+mean
    reflectance=torch.sigmoid(actual); truth_r=torch.sigmoid(truth)
    return {"mse_normalized":float((pred-target).square().mean()),"reflectance_mae":float((reflectance-truth_r).abs().mean()),"reflectance_rmse":float((reflectance-truth_r).square().mean().sqrt()),"relative_error":float(((reflectance-truth_r).abs()/truth_r.clamp_min(.01)).mean())}


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
