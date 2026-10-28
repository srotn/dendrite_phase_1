from __future__ import annotations
import argparse, copy, hashlib, json, math, os, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent
DEPTHS=[1,2,3,4,8,16]
SEEDS=list(range(20261025,20261030))
METHODS=['base','duplicate','reciprocal']
TASKS={
 '01_forced': 'phase_1_synthetic_experiments/01_forced_oscillation/artifacts/dataset.pt',
 '02_thermal': 'phase_1_synthetic_experiments/03_thermal_field/artifacts/dataset.pt',
 '03_optical': 'phase_1_synthetic_experiments/04_optical_stack/artifacts/dataset.pt',
 '04_impedance': 'phase_1_synthetic_experiments/05_impedance_network/artifacts/dataset.pt',
 '05_concrete': 'phase_2_paper_regression/paper_regression_reproduction/09_concrete_compressive_strength/phase1_training_test/artifacts/dataset.pt',
 '06_co': 'phase_3_real_world_regression/01_gas_sensor_temperature_modulation/artifacts/dataset.pt'}

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()

def js(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');t.replace(p)

def cpu(v):
 if torch.is_tensor(v): return v.detach().cpu().clone()
 if isinstance(v,dict): return {k:cpu(a) for k,a in v.items()}
 if isinstance(v,list): return [cpu(a) for a in v]
 return v

def save(p,v):
 t=p.with_suffix('.tmp');torch.save(cpu(v),t);t.replace(p)

def prepare():
 audit=ROOT/'source_audit';audit.mkdir(exist_ok=True)
 inventory=[]; docs=[]
 for p in sorted(OLD.rglob('*')):
  if not p.is_file() or ROOT in p.parents or '__pycache__' in p.parts: continue
  row={'path':str(p.relative_to(OLD)),'bytes':p.stat().st_size}
  if p.suffix in ('.py','.md','.json','.jsonl'):
   text=p.read_text(encoding='utf-8-sig',errors='replace');row['sha256']=sha(p);row['lines']=len(text.splitlines())
   if p.suffix=='.md': docs.append('\n\n## '+row['path']+'\n\n'+text)
   if p.suffix=='.jsonl':
    rows=[json.loads(x) for x in text.splitlines() if x.strip()];row['records']=len(rows)
    row['epoch_resets']=sum(b.get('epoch',1)<=a.get('epoch',0) for a,b in zip(rows,rows[1:]))
  inventory.append(row)
 js(audit/'inventory.json',inventory)
 (audit/'historical_documents.md').write_text(''.join(docs),encoding='utf-8')
 for task,relative in TASKS.items():
  out=ROOT/task/'data';out.mkdir(parents=True,exist_ok=True)
  if (out/'dataset.pt').exists(): continue
  source=OLD/relative;d=torch.load(source,weights_only=False,map_location='cpu');result={};meta={}
  if task=='05_concrete':
   for s,old in [('train','train'),('valid','valid')]: result[s]={'x':d['x_'+old],'r':d['reciprocal_'+old],'y':d['y_'+old]}
   eligible=list(range(8));constant=True;scale=d['target_scale']/2
   meta={k:v for k,v in d.items() if k not in ('x_train','x_valid','y_train','y_valid','reciprocal_train','reciprocal_valid')}
  elif task=='06_co':
   for s,v in d['splits'].items(): result[s]={'x':v['x'],'r':v['physical'],'y':v['y']}
   eligible=list(range(4,18));constant=True;scale=d['transforms']['y_std'];meta=d['transforms']
  else:
   inv=lambda x:torch.where(x==0,0.,x.sign()/x.abs().clamp_min(1e-6))
   r=inv(d['train']['physical']);cap=torch.quantile(r.abs(),.995,dim=0).clamp_min(1e-6)
   r=r.sign()*torch.minimum(r.abs(),cap);mean=r.mean(0);std=r.std(0).clamp_min(1e-6)
   for s in ('train','valid','iid','ood'):
    if s not in d:continue
    v=d[s];r=inv(v['physical']);r=(r.sign()*torch.minimum(r.abs(),cap)-mean)/std
    result['valid' if s=='iid' else s]={'x':v['x'],'r':r,'y':v['y']}
   eligible=list(range(result['train']['x'].shape[1]));constant=False;scale=d['y_scale'];meta={'cap':cap,'mean':mean,'std':std}
  for v in result.values():
   assert all(torch.isfinite(t).all() for t in v.values())
  result.update(eligible=eligible,constant=constant,scale=scale,preprocessing=meta)
  save(out/'dataset.pt',result)
 js(out/'manifest.json',{'source':relative,'source_sha256':sha(source),'prepared_sha256':sha(out/'dataset.pt'),
    'sizes':{s:len(v['y']) for s,v in result.items() if isinstance(v,dict) and 'y' in v},'eligible':eligible,'constant':constant})
 js(ROOT/'protocol.json',{'tasks':list(TASKS),'depths':DEPTHS,'seeds':SEEDS,'methods':METHODS,'epochs':2000,'lr':2e-4,'batch':4096,'runs':540,'source_sha256':sha(Path(__file__))})

class DD(nn.Module):
 def __init__(self,width,depth,outputs,constant):
  super().__init__();self.constant=constant;w=width+int(constant)
  self.layers=nn.ModuleList(nn.Linear(w,w,bias=False) for _ in range(depth));self.output=nn.Linear(w,outputs,bias=not constant)
 def forward(self,x):
  a=torch.cat((torch.ones_like(x[:,:1]),x),1) if self.constant else x;h=a
  for l in self.layers:h=l(h)*a
  return self.output(h)

def build(d,depth,seed,device):
 models=[];width=d['train']['x'].shape[1];outputs=d['train']['y'].shape[1]
 for i in range(3):
  torch.manual_seed(seed);models.append(DD(width+(len(d['eligible']) if i else 0),depth,outputs,d['constant']).to(device))
 assert all(torch.equal(a,b) for a,b in zip(models[1].parameters(),models[2].parameters()))
 return models

def tensors(split,eligible,device):
 x,r,y=[split[k].to(device) for k in ('x','r','y')]
 return [x,torch.cat((x,x[:,eligible]),1),torch.cat((x,r),1)],y

def graph_step(models,opt,xs,y,batch):
 initial=[copy.deepcopy(m.state_dict()) for m in models]
 idx=torch.arange(batch,device='cuda')%len(y);weight=torch.ones(batch,1,device='cuda')
 accum=torch.zeros(3,device='cuda');norms=torch.zeros(3,device='cuda')
 def step():
  opt.zero_grad(set_to_none=True)
  losses=[((m(x[idx])-y[idx]).square()*weight).sum()/(weight.sum()*y.shape[1]) for m,x in zip(models,xs)]
  torch.stack(losses).sum().backward()
  for i,m in enumerate(models):norms[i].copy_(nn.utils.clip_grad_norm_(m.parameters(),5.))
  opt.step();accum.add_(torch.stack([l.detach() for l in losses])*weight.sum())
 stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(stream):
  for _ in range(3):step()
 torch.cuda.current_stream().wait_stream(stream)
 g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):step()
 for m,s in zip(models,initial):m.load_state_dict(s)
 for state in opt.state.values():
  for v in state.values():
   if torch.is_tensor(v):v.zero_()
 accum.zero_();norms.zero_()
 return g,idx,weight,accum,norms

@torch.no_grad()
def evaluate(models,xs,y):
 return torch.stack([sum((m(x[a:a+4096])-y[a:a+4096]).square().sum() for a in range(0,len(y),4096))/y.numel() for m,x in zip(models,xs)]).cpu().numpy()

def train(task,depth,seed,epochs=2000,pilot=False):
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
 d=torch.load(ROOT/task/'data/dataset.pt',weights_only=False)
 out=ROOT/task/('pilot' if pilot else 'runs')/f'seed_{seed}'/f'depth_{depth:02d}';out.mkdir(parents=True,exist_ok=True)
 if (out/'COMPLETED.json').exists():return
 if (out/'metrics.jsonl').exists():raise RuntimeError(f'Existing incomplete run requires explicit recovery: {out}')
 models=build(d,depth,seed,'cuda');xs,y=tensors(d['train'],d['eligible'],'cuda');vx,vy=tensors(d['valid'],d['eligible'],'cuda')
 opt=torch.optim.Adam([p for m in models for p in m.parameters()],lr=2e-4,capturable=True,foreach=True)
 batch=min(4096,len(y));g,idx,w,acc,norms=graph_step(models,opt,xs,y,batch)
 best=[float('inf')]*3;best_epoch=[0]*3;best_states=[None]*3
 js(out/'config.json',{'task':task,'depth':depth,'seed':seed,'epochs':epochs,'parameters':[sum(p.numel() for p in m.parameters()) for m in models],'data_sha256':sha(ROOT/task/'data/dataset.pt'),'source_sha256':sha(Path(__file__))})
 start=time.perf_counter();steps=math.ceil(len(y)/batch);rows=[]
 with (out/'metrics.jsonl').open('w',encoding='utf-8') as f:
  for epoch in range(1,epochs+1):
   order=torch.randperm(len(y),generator=torch.Generator().manual_seed(seed+epoch)).cuda()
   pad=steps*batch-len(y)
   if pad:order=torch.cat((order,order[:pad]))
   acc.zero_()
   for b in range(steps):
    idx.copy_(order[b*batch:(b+1)*batch]);w.fill_(1)
    if b==steps-1 and pad:w[-pad:]=0
    g.replay()
   val=evaluate(models,vx,vy);loss=acc.cpu().numpy()/len(y);grad=norms.cpu().numpy()
   if not np.isfinite(np.r_[val,loss,grad]).all():
    js(out/'FAILED.json',{'epoch':epoch,'reason':'nonfinite metric'});raise FloatingPointError(str(out))
   row={'epoch':epoch,'train_mse':loss.tolist(),'valid_mse':val.tolist(),'gradient_norm_last_batch':grad.tolist(),'elapsed':time.perf_counter()-start};rows.append(row);f.write(json.dumps(row)+'\n')
   for i in range(3):
    if val[i]<best[i]:
     best[i]=float(val[i]);best_epoch[i]=epoch
     best_states[i]={'model':cpu(models[i].state_dict()),'epoch':epoch,'valid_mse':best[i]}
   if epoch%500==0 or epoch==epochs:
    f.flush();save(out/'last.pt',{'models':[m.state_dict() for m in models],'optimizer':opt.state_dict(),'epoch':epoch,'best':best_states,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'next_order_seed':seed+epoch+1})
    print(task,depth,seed,epoch,[round(math.sqrt(v),6) for v in best],flush=True)
 save(out/'best.pt',best_states)
 for i,m in enumerate(models):m.load_state_dict(best_states[i]['model'])
 recomputed=evaluate(models,vx,vy);assert np.allclose(recomputed,best,rtol=2e-5,atol=1e-8)
 plot(out,rows)
 js(out/'COMPLETED.json',{'best_mse':best,'best_epoch':best_epoch,'epochs':epochs,'seconds':time.perf_counter()-start,'verified':True,'curve':'training_curves.png'})

def plot(out,rows):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(1,2,figsize=(10,4))
 for i,name in enumerate(METHODS):
  for ax,key in zip(axs,['train_mse','valid_mse']):ax.plot([r['epoch'] for r in rows],[math.sqrt(r[key][i]) for r in rows],label=name);ax.set_title(key);ax.set_yscale('log');ax.set_xlabel('Epoch')
 axs[0].legend();fig.tight_layout();fig.savefig(out/'training_curves.png',dpi=150);fig.savefig(out/'training_curves.pdf');plt.close(fig)

def runall():
 jobs=[(t,l,s) for t in TASKS for s in SEEDS for l in DEPTHS]
 for n,(t,l,s) in enumerate(jobs):
  js(ROOT/'status.json',{'status':'training','group':n+1,'groups':len(jobs),'task':t,'depth':l,'seed':s})
  train(t,l,s)
 js(ROOT/'training_lock.json',{'complete_runs':540,'best_hashes':{str(p.relative_to(ROOT)):sha(p) for p in ROOT.glob('*/runs/*/*/best.pt')}})
 js(ROOT/'status.json',{'status':'training_complete_pending_analysis'})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--run-all',action='store_true');p.add_argument('--pilot',action='store_true');a=p.parse_args()
 if a.prepare:prepare()
 if a.pilot:train('01_forced',4,20261025,20,True)
 if a.run_all:runall()
