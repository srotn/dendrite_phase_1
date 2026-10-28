"""Disposable numerical verification before formal runs; never touches old files."""
import copy
import torch
import study as s

def main():
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
 checks=[]
 for task in s.TASKS:
  d=torch.load(s.ROOT/task/'data/dataset.pt',weights_only=False)
  xs,y=s.tensors(d['train'],d['eligible'],'cuda')
  for depth in (1,4,16):
   models=s.build(d,depth,s.SEEDS[0],'cuda');refs=copy.deepcopy(models)
   opt=torch.optim.Adam([p for m in models for p in m.parameters()],lr=2e-4,capturable=True,foreach=True)
   opts=[torch.optim.Adam(m.parameters(),lr=2e-4,capturable=True,foreach=True) for m in refs]
   g,idx,w,acc,norms=s.graph_step(models,opt,xs,y,32)
   assert all(torch.equal(a,b) for m,r in zip(models,refs) for a,b in zip(m.parameters(),r.parameters()))
   assert all(float(st['step'])==0 for st in opt.state.values())
   for step in range(2):
    w.fill_(1)
    if step:w[-7:]=0
    g.replay()
    for ref,op,x in zip(refs,opts,xs):
     op.zero_grad(set_to_none=True)
     loss=((ref(x[idx])-y[idx]).square()*w).sum()/(w.sum()*y.shape[1]);loss.backward()
     torch.nn.utils.clip_grad_norm_(ref.parameters(),5.);op.step()
   error=max(float((a-b).abs().max()) for m,r in zip(models,refs) for a,b in zip(m.parameters(),r.parameters()))
   assert error<1e-6,(task,depth,error)
   for m,r,op in zip(models,refs,opts):
    for a,b in zip(m.parameters(),r.parameters()):
     for k in ('step','exp_avg','exp_avg_sq'):assert torch.allclose(opt.state[a][k],op.state[b][k],rtol=1e-5,atol=1e-7)
   checks.append({'task':task,'depth':depth,'max_parameter_error':error,'optimizer_match':True,'padded_batch_verified':True})
 s.js(s.ROOT/'numerical_verification.json',{'passed':True,'checks':checks,'torch':torch.__version__,'gpu':torch.cuda.get_device_name()})
 print('PASS',len(checks),'graph/eager comparisons')

if __name__=='__main__':main()
