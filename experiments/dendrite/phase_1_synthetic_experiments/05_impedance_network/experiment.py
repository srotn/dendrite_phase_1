from __future__ import annotations
import argparse, json, math, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from experiments.dendrite.phase_1_synthetic_experiments.discrete_selection import LearnableDiscreteDD

ROOT=Path(__file__).resolve().parent; ARTIFACTS=ROOT/'artifacts'
# Eight-node fixed graph, 16 edges: ring, opposite-node chords, and short chords.
EDGES=((0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,0),(0,4),(1,5),(2,6),(3,7),(0,2),(2,4),(4,6),(6,0))
GROUND=4; NODES=8

def lu(g,lo,hi,n): return torch.exp(torch.empty(n).uniform_(math.log(lo),math.log(hi),generator=g))

def generate(n,seed,device='cuda',ood=False):
    g=torch.Generator(device='cpu').manual_seed(seed)
    if ood: er=lu(g,5.,20.,n*16); voltage=lu(g,2.,8.,n); rs=lu(g,5.,20.,n); load=lu(g,5.,20.,n); omega=lu(g,2.,8.,n)
    else: er=lu(g,.5,5.,n*16); voltage=lu(g,.5,2.,n); rs=lu(g,.5,5.,n); load=lu(g,.5,5.,n); omega=lu(g,.2,2.,n)
    er=er.reshape(n,16).to(device); voltage=voltage.to(device); rs=rs.to(device); load=load.to(device); omega=omega.to(device)
    unknown=[i for i in range(NODES) if i!=GROUND]; pos={node:i for i,node in enumerate(unknown)}; dim=len(unknown)
    A=torch.zeros(n,dim,dim,dtype=torch.complex64,device=device); b=torch.zeros(n,dim,dtype=torch.complex64,device=device)
    for edge,(u,v) in enumerate(EDGES):
        g_edge=1./er[:,edge];
        if u!=GROUND: A[:,pos[u],pos[u]]+=g_edge
        if v!=GROUND: A[:,pos[v],pos[v]]+=g_edge
        if u!=GROUND and v!=GROUND: A[:,pos[u],pos[v]]-=g_edge; A[:,pos[v],pos[u]]-=g_edge
    # Fixed node capacitance makes frequency part of the transfer relation.
    cap=0.12j*omega
    for node in unknown: A[:,pos[node],pos[node]]+=cap
    gs=1./rs; A[:,pos[0],pos[0]]+=gs; b[:,pos[0]]+=gs*voltage
    A[:,pos[7],pos[7]]+=1./load
    solution=torch.linalg.solve(A,b.unsqueeze(-1)).squeeze(-1)
    out=solution[:,pos[7]].abs().clamp_min(1e-8)
    physical=torch.cat((er.cpu(),voltage.cpu()[:,None],rs.cpu()[:,None],load.cpu()[:,None],omega.cpu()[:,None]),1)
    # Training normalization; OOD points intentionally leave this range.
    x=torch.cat((.5+(torch.log(er.cpu())-math.log(.5))/math.log(10.),.5+(torch.log(voltage.cpu())-math.log(.5))[:,None]/math.log(4.),.5+(torch.log(rs.cpu())-math.log(.5))[:,None]/math.log(10.),.5+(torch.log(load.cpu())-math.log(.5))[:,None]/math.log(10.),.5+(torch.log(omega.cpu())-math.log(.2))[:,None]/math.log(10.)),1)
    return {'physical':physical.float(),'x':x.float(),'y':torch.log(out).cpu().unsqueeze(1)}

class DDStack(nn.Module):
    def __init__(self,width,layers): super().__init__(); self.layers=nn.ModuleList(nn.Linear(width,width,bias=False) for _ in range(layers))
    def forward(self,x):
        anchor=x; h=x
        for layer in self.layers: h=layer(h)*anchor
        return h
class BaseDD(nn.Module):
    name='base_dd'
    def __init__(self): super().__init__(); self.dd=DDStack(20,4); self.output=nn.Linear(20,1)
    def forward(self,x): return self.output(self.dd(x))
class ReciprocalConcatDD(nn.Module):
    name='reciprocal_concat_dd'
    def __init__(self): super().__init__(); self.dd=DDStack(40,4); self.output=nn.Linear(40,1)
    def forward(self,x): return self.output(self.dd(torch.cat((x,1./x.clamp_min(1e-6)),1)))
class DuplicateConcatDD(nn.Module):
    name='duplicate_concat_dd'
    def __init__(self): super().__init__(); self.dd=DDStack(40,4); self.output=nn.Linear(40,1)
    def forward(self,x): return self.output(self.dd(torch.cat((x,x),1)))
class DeepBaseDD(nn.Module):
    name='deep_base_dd_16_layers'
    def __init__(self): super().__init__(); self.dd=DDStack(20,16); self.output=nn.Linear(20,1)
    def forward(self,x): return self.output(self.dd(x))
class LearnableDiscreteInputDD(LearnableDiscreteDD):
    name='learnable_discrete_dd'
    def __init__(self): super().__init__(input_width=20, layers=4, output_dim=1, name=self.name)
MODELS=(BaseDD,ReciprocalConcatDD,DuplicateConcatDD,DeepBaseDD,LearnableDiscreteInputDD)
def counts(): return {c.name:sum(p.numel() for p in c().parameters()) for c in MODELS}

def score(pred,target,mean,scale):
    pa=pred*scale+mean; ta=target*scale+mean; pv=torch.exp(pa); tv=torch.exp(ta); err=(pv-tv)
    return {'mse_normalized':float((pred-target).square().mean()),'output_mae':float(err.abs().mean()),'output_rmse':float(err.square().mean().sqrt()),'relative_error':float((err.abs()/tv.clamp_min(1e-6)).mean())}

def train_model(cls,train,iid,ood,mean,scale,args):
    model=cls().to(args.device)
    if hasattr(model, "_seed"): model._seed = getattr(args, "seed", 0)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    loader=DataLoader(TensorDataset(train['x'],train['y']),batch_size=args.batch,shuffle=True,generator=torch.Generator().manual_seed(args.seed)); iid_loader=DataLoader(TensorDataset(iid['x'],iid['y']),batch_size=args.batch); ood_loader=DataLoader(TensorDataset(ood['x'],ood['y']),batch_size=args.batch)
    folder=ARTIFACTS/cls.name; folder.mkdir(parents=True,exist_ok=True); best=float('inf')
    for epoch in range(1,args.epochs+1):
        if hasattr(model, 'set_epoch'): model.set_epoch(epoch, args.epochs)
        started=time.perf_counter(); model.train(); total=0.; count=0; norms=[]
        for x,y in loader:
            x,y=x.to(args.device),y.to(args.device); opt.zero_grad(set_to_none=True); loss=(model(x)-y).square().mean(); loss.backward(); norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.))); opt.step(); total+=float(loss.detach())*len(x); count+=len(x)
        model.eval(); pi,yi,po,yo=[],[],[],[]
        with torch.inference_mode():
            for x,y in iid_loader: pi.append(model(x.to(args.device)).cpu()); yi.append(y)
            for x,y in ood_loader: po.append(model(x.to(args.device)).cpu()); yo.append(y)
        mi=score(torch.cat(pi),torch.cat(yi),mean,scale); mo=score(torch.cat(po),torch.cat(yo),mean,scale)
        row={'epoch':epoch,'seconds':time.perf_counter()-started,'train_loss':total/count,'gradient_norm':float(np.mean(norms)),'iid':mi,'ood':mo}
        if hasattr(model, 'selection_values'): row['selection_values'] = model.selection_values().tolist()
        with (folder/'metrics.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        if mi['mse_normalized']<best:
            best=mi['mse_normalized']; checkpoint={'model':model.state_dict(),'epoch':epoch,'metrics':row,'parameters':counts()[cls.name]}
            if hasattr(model, 'selection_values'): checkpoint['selection_values'] = model.selection_values().tolist()
            torch.save(checkpoint,folder/'best.pt')
        if epoch==1 or epoch%25==0 or epoch==args.epochs: print(json.dumps({'model':cls.name,**row}),flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--train-size',type=int,default=30000); p.add_argument('--valid-size',type=int,default=6000); p.add_argument('--ood-size',type=int,default=6000); p.add_argument('--epochs',type=int,default=500); p.add_argument('--batch',type=int,default=4096); p.add_argument('--lr',type=float,default=2e-4); p.add_argument('--seed',type=int,default=20261017); p.add_argument('--device',choices=('cpu','cuda'),default='cuda'); p.add_argument('--only',choices=[c.name for c in MODELS]); args=p.parse_args(); args.device=torch.device(args.device)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); ARTIFACTS.mkdir(parents=True,exist_ok=True); train=generate(args.train_size,args.seed,str(args.device)); iid=generate(args.valid_size,args.seed+1,str(args.device)); ood=generate(args.ood_size,args.seed+2,str(args.device),ood=True); mean=train['y'].mean(0); scale=train['y'].std(0).clamp_min(1e-6)
    for d in (train,iid,ood): d['y']=(d['y']-mean)/scale
    torch.save({'train':train,'iid':iid,'ood':ood,'y_mean':mean,'y_scale':scale},ARTIFACTS/'dataset.pt'); (ARTIFACTS/'parameter_report.json').write_text(json.dumps(counts(),indent=2),encoding='utf-8'); (ARTIFACTS/'manifest.json').write_text(json.dumps({'seed':args.seed,'train_size':args.train_size,'iid_size':args.valid_size,'ood_size':args.ood_size,'epochs':args.epochs,'models':[c.name for c in MODELS],'parameter_counts':counts(),'device':str(args.device),'ood_definition':'all positive inputs sampled from higher ranges'},indent=2),encoding='utf-8')
    for cls in MODELS:
        if args.only is None or args.only==cls.name: train_model(cls,train,iid,ood,mean,scale,args)
if __name__=='__main__': main()
