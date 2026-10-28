from __future__ import annotations
import gc, math, time
from torch import nn
from common import *

class DD(nn.Module):
    def __init__(self,width,depth,outputs,constant,slot=None):
        super().__init__();self.constant=constant or slot is not None;self.slot=1. if slot is None else float(slot)
        w=width+int(self.constant)
        self.layers=nn.ModuleList(nn.Linear(w,w,bias=False) for _ in range(depth))
        self.output=nn.Linear(w,outputs,bias=(not constant) if slot is None else True)
    def forward(self,x):
        a=torch.cat((torch.ones_like(x[:,:1])*self.slot,x),1) if self.constant else x
        h=a
        for layer in self.layers:h=layer(h)*a
        return self.output(h)

def parameter_count(model):return sum(p.numel() for p in model.parameters())

def inputs(d,c,split):
    v=d[split] if isinstance(split,str) else split
    method=c['input'];x=v['x'];e=d['eligible']
    if method=='base':z=x
    elif method=='duplicate':z=torch.cat((x,x[:,e]),1)
    else:
        r=v['r']
        if method in ['matched','subset','complement']:
            tx=d['train']['x'][:,e];tr=d['train']['r']
            r=(r-tr.mean(0))/tr.std(0,correction=0).clamp_min(1e-6)*tx.std(0,correction=0)+tx.mean(0)
            if method in ['subset','complement']:
                mask=torch.tensor([i in FIELDS[d['task']] for i in e])
                if method=='complement':mask=~mask
                r=torch.where(mask[None,:],r,x[:,e])
        z=torch.cat((x,r),1)
    if c.get('standardize'):
        base=dict(c,standardize=False);t=inputs(d,base,'train')
        z=(z-t.mean(0))/t.std(0,correction=0).clamp_min(1e-6)
    return z

def build(d,c,seed):
    torch.manual_seed(seed)
    width=inputs(d,c,'train').shape[1];outputs=d['train']['y'].shape[1]
    if c['kind']=='dd':return DD(width,c['depth'],outputs,d['constant'],c.get('slot'))
    h=c['hidden']
    return nn.Sequential(nn.Linear(width,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,outputs))

def dd_configs(depth,methods=('base','duplicate','reciprocal'),lrs=(2e-4,)):
    return [dict(name=f'{method}_lr{lr:g}',kind='dd',depth=depth,input=method,lr=lr) for lr in lrs for method in methods]

def prepare_gpu(d,configs):
    cache={};xs=[];vx=[]
    for c in configs:
        key=(c['input'],c.get('standardize',False))
        if key not in cache:cache[key]=(inputs(d,c,'train').cuda(),inputs(d,c,'valid').cuda())
        a,b=cache[key];xs.append(a);vx.append(b)
    return xs,vx,d['train']['y'].cuda(),d['valid']['y'].cuda()

def graph_step(models,opt,xs,y,batch):
    initial=[copy.deepcopy(m.state_dict()) for m in models]
    idx=torch.arange(batch,device='cuda')%len(y);weight=torch.ones(batch,1,device='cuda')
    accum=torch.zeros(len(models),device='cuda');norms=torch.zeros_like(accum)
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

def restore_optimizer_inplace(opt,state):
    for group,savedgroup in zip(opt.param_groups,state['param_groups']):
        assert group['lr']==savedgroup['lr']
        for p,index in zip(group['params'],savedgroup['params']):
            for key,v in state['state'][index].items():
                if torch.is_tensor(v):opt.state[p][key].copy_(v.to(opt.state[p][key].device))
                else:assert opt.state[p][key]==v

@torch.no_grad()
def evaluate(models,xs,y):
    return torch.stack([sum((m(x[a:a+4096])-y[a:a+4096]).square().sum() for a in range(0,len(y),4096))/y.numel() for m,x in zip(models,xs)]).cpu().numpy()

def plot(out,rows,configs):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for i,c in enumerate(configs):
        for ax,key in zip(axes,['train_mse','valid_mse']):
            ax.plot([r['epoch'] for r in rows],[math.sqrt(r[key][i]) if r[key][i] is not None else np.nan for r in rows],label=c['name'],lw=.8)
            ax.set_title(key);ax.set_xlabel('Epoch');ax.set_yscale('log');ax.grid(alpha=.2)
    axes[0].legend(fontsize=6);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'training_curves.{ext}',dpi=150)
    plt.close(fig)

def train(out,d,configs,seed,epochs,budgets=None):
    configure();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETED.json').exists():return
    budgets=sorted(set((budgets or [])+[epochs]));n=len(configs)
    models=[build(d,c,seed).cuda() for c in configs]
    xs,vx,y,vy=prepare_gpu(d,configs)
    opt=torch.optim.Adam([dict(params=list(m.parameters()),lr=c['lr']) for m,c in zip(models,configs)],capturable=True,foreach=True)
    batch=min(4096,len(y));steps=math.ceil(len(y)/batch)
    resume=load(out/'last.pt') if (out/'last.pt').exists() else None
    if (out/'metrics.jsonl').exists() and resume is None:raise RuntimeError(f'Incomplete without checkpoint: {out}')
    if resume:
        for m,s in zip(models,resume['models']):m.load_state_dict(s)
    graph,idx,weight,acc,norms=graph_step(models,opt,xs,y,batch)
    best=[None]*n;failed=[None]*n;start_epoch=0;elapsed=0.;rows=[]
    if resume:
        restore_optimizer_inplace(opt,resume['optimizer']);best=resume['best'];failed=resume['failed'];start_epoch=resume['epoch'];elapsed=resume['elapsed']
        previous=(out/'metrics.jsonl').read_text(encoding='utf-8').splitlines();rows=[json.loads(s) for s in previous if s.strip()]
        tail=[r for r in rows if r['epoch']>start_epoch];rows=[r for r in rows if r['epoch']<=start_epoch]
        assert len(rows)==start_epoch and rows[-1]['epoch']==start_epoch
        if tail:
            js(out/f'recovery_tail_{int(time.time())}.json',tail)
            (out/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
        if start_epoch in budgets and not (out/f'budget_{start_epoch}.pt').exists():
            save(out/f'budget_{start_epoch}.pt',dict(best=best,models=resume['models'],failed=failed,epoch=start_epoch))
    config={'task':d['task'],'fold':d['fold'],'configs':configs,'seed':seed,'epochs':epochs,'budgets':budgets,
            'parameters':[parameter_count(m) for m in models],'data_sha256':sha(data_path(d['task'],d['fold'])),'engine_sha256':sha(__file__),
            'batch':batch,'steps_per_epoch':steps,'train_rows':len(y),'valid_rows':len(vy),'date':now()}
    if (out/'config.json').exists():
        old=json.loads((out/'config.json').read_text(encoding='utf-8'))
        for key in ['configs','seed','epochs','data_sha256','engine_sha256']:assert old[key]==config[key],key
    else:js(out/'config.json',config)
    version=ROOT/'source_versions'/(sha(__file__)+'.py')
    if not version.exists():version.parent.mkdir(exist_ok=True);version.write_bytes(Path(__file__).read_bytes())
    started=time.perf_counter()
    with (out/'metrics.jsonl').open('a',encoding='utf-8') as stream:
        for epoch in range(start_epoch+1,epochs+1):
            order=torch.randperm(len(y),generator=torch.Generator().manual_seed(seed+epoch)).cuda()
            pad=steps*batch-len(y)
            if pad:order=torch.cat((order,order[:pad]))
            acc.zero_()
            for b in range(steps):
                idx.copy_(order[b*batch:(b+1)*batch]);weight.fill_(1)
                if b==steps-1 and pad:weight[-pad:]=0
                graph.replay()
            val=evaluate(models,vx,vy);loss=acc.cpu().numpy()/len(y);grad=norms.cpu().numpy()
            for i in range(n):
                if not np.isfinite([val[i],loss[i],grad[i]]).all() and failed[i] is None:failed[i]=epoch
                if failed[i] is None and (best[i] is None or val[i]<best[i]['valid_mse']):
                    best[i]={'model':cpu(models[i].state_dict()),'epoch':epoch,'valid_mse':float(val[i])}
            row=clean(dict(epoch=epoch,train_mse=loss.tolist(),valid_mse=val.tolist(),gradient_norm=grad.tolist(),elapsed=elapsed+time.perf_counter()-started))
            rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
            if epoch%500==0 or epoch==epochs:
                stream.flush()
                state=dict(models=[m.state_dict() for m in models],optimizer=opt.state_dict(),epoch=epoch,best=best,failed=failed,elapsed=row['elapsed'],next_order_seed=seed+epoch+1)
                save(out/'last.pt',state)
                if epoch in budgets:save(out/f'budget_{epoch}.pt',state)
                print('TRAIN',str(out.relative_to(ROOT)),epoch,'best',[None if b is None else round(math.sqrt(b['valid_mse']),5) for b in best],'failed',failed,flush=True)
    last_val=evaluate(models,vx,vy)
    checks=[]
    for i,m in enumerate(models):
        if failed[i] is not None:checks.append({'candidate':i,'failed_epoch':failed[i]});continue
        assert np.isclose(last_val[i],rows[-1]['valid_mse'][i],rtol=2e-5,atol=1e-8)
        m.load_state_dict(best[i]['model']);replay=float(evaluate([m],[vx[i]],vy)[0])
        assert np.isclose(replay,best[i]['valid_mse'],rtol=2e-5,atol=1e-8)
        counters={int(v['step'].item()) for p,v in opt.state.items() if any(p is q for q in m.parameters())}
        assert counters=={epochs*steps},counters
        checks.append({'candidate':i,'best_replayed':replay,'final_replayed':float(last_val[i]),'optimizer_steps':epochs*steps})
    save(out/'best.pt',best);plot(out,rows,configs)
    js(out/'COMPLETED.json',{'date':now(),'epochs':epochs,'attempted_models':n,'successful_models':sum(f is None for f in failed),'failed_first_epoch':failed,
         'best_epoch':[None if b is None else b['epoch'] for b in best],'best_mse':[None if b is None else b['valid_mse'] for b in best],
         'verification':checks,'log_rows':len(rows),'group_seconds':rows[-1]['elapsed'],'curve':'training_curves.png','curve_pdf':'training_curves.pdf',
         'best_sha256':sha(out/'best.pt'),'last_sha256':sha(out/'last.pt')})
    del graph,models,opt,xs,vx,y,vy;gc.collect();torch.cuda.empty_cache()

@torch.no_grad()
def evaluate_test(out,d,configs,seed,tests,budget=None,save_predictions=True):
    out=Path(out);suffix='final' if budget is None else f'budget_{budget}'
    path=out/f'test_{suffix}.json'
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    state=load(out/'last.pt' if budget is None else out/f'budget_{budget}.pt');best=state['best'];failed=state['failed']
    results=[];predictions={}
    for i,c in enumerate(configs):
        if failed[i] is not None or best[i] is None:
            results.append({'candidate':i,'name':c['name'],'failed_epoch':failed[i]});continue
        m=build(d,c,seed).cuda();m.load_state_dict(best[i]['model']);m.eval()
        for name,v in tests.items():
            x=inputs(d,c,v).cuda();pred=torch.cat([m(x[a:a+4096]).cpu() for a in range(0,len(x),4096)])
            error=(pred.double()-v['y'].double());scale=d['target_scale'].double()
            rmse=float(error.square().mean().sqrt());raw=(error*scale).square().mean(0).sqrt()
            group_rows=[]
            for group in ([] if d['task'] in SYNTHETIC else torch.unique(v['groups'])):
                mask=v['groups']==group
                group_rows.append(dict(group=int(group),n=int(mask.sum()),mse=float(error[mask].square().mean()),raw_mse_per_output=(error[mask]*scale).square().mean(0).tolist()))
            # Individual synthetic observations are not independent model replicates; no enormous JSON row table.
            if d['task'] in SYNTHETIC:group_rows=[]
            record=dict(candidate=i,name=c['name'],split=name,normalized_rmse=rmse,raw_rmse_per_output=raw.tolist(),raw_mae_per_output=(error*scale).abs().mean(0).tolist(),
                 n=len(pred),valid_rmse=math.sqrt(best[i]['valid_mse']),best_epoch=best[i]['epoch'],scale_units=d['scale_units'],groups=group_rows)
            results.append(record)
            if save_predictions:
                predictions[f'pred_{i}_{name}']=pred.numpy()
                predictions[f'y_{name}']=v['y'].numpy();predictions[f'group_{name}']=v['groups'].numpy()
        del m
    if save_predictions:np.savez_compressed(out/f'predictions_{suffix}.npz',**predictions)
    js(path,results);return clean(results)
