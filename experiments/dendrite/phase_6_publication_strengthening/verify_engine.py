"""Compare graph/eager independent Adam and in-place resume on artificial data."""
from engine import *

def main():
    configure();lock_protocol();torch.manual_seed(372)
    x=torch.randn(11,4)*.2+.5;r=1/x.clamp_min(.1);y=torch.randn(11,2)
    d={'task':'01_forced','train':dict(x=x,r=r,y=y),'eligible':list(range(4)),'constant':False}
    configs=[dict(kind='dd',input='duplicate',depth=3,lr=2e-4),dict(kind='dd',input='reciprocal',depth=3,lr=2e-5,slot=1),
             dict(kind='dd',input='matched',depth=2,lr=2e-3,slot=0),dict(kind='mlp',input='base',hidden=7,standardize=True,lr=2e-4)]
    models=[build(d,c,12).cuda() for c in configs];reference=copy.deepcopy(models)
    opt=torch.optim.Adam([dict(params=list(m.parameters()),lr=c['lr']) for m,c in zip(models,configs)],capturable=True,foreach=True)
    refs=[torch.optim.Adam(m.parameters(),lr=c['lr'],capturable=True,foreach=True) for m,c in zip(reference,configs)]
    xs=[inputs(d,c,'train').cuda() for c in configs];gy=y.cuda()
    graph,idx,weight,acc,norm=graph_step(models,opt,xs,gy,8)
    schedules=[(torch.arange(8),8),(torch.tensor([8,9,10,0,1,2,3,4]),3)]
    maxdiff=0.
    for indices,real in schedules:
        idx.copy_(indices);weight.zero_();weight[:real]=1
        graph.replay()
        for model,optimizer,xx in zip(reference,refs,xs):
            optimizer.zero_grad(set_to_none=True)
            loss=((model(xx[idx])-gy[idx]).square()*weight).sum()/(real*2);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        for a,b in zip(models,reference):
            for pa,pb in zip(a.parameters(),b.parameters()):
                diff=float((pa-pb).abs().max());maxdiff=max(maxdiff,diff);assert torch.allclose(pa,pb,rtol=2e-5,atol=2e-7),(diff,)
    before=[cpu(m.state_dict()) for m in models];optimizer_before=cpu(opt.state_dict())
    graph.replay();expected=[cpu(m.state_dict()) for m in models]
    for m,s in zip(models,before):m.load_state_dict(s)
    restore_optimizer_inplace(opt,optimizer_before);graph.replay()
    resumediff=max(float((m.state_dict()[k].cpu()-s[k]).abs().max()) for m,s in zip(models,expected) for k in s)
    assert resumediff==0,resumediff
    js(ROOT/'engine_verification.json',{'date':now(),'passed':True,'engine_sha256':sha(ROOT/'engine.py'),'graph_vs_independent_eager_adam_max_abs':maxdiff,
        'inplace_resume_max_abs':resumediff,'models':['DD','DD constant one','DD constant zero matched moments','SiLU MLP'],
        'checks':['different learning rates','per-model norm clipping','padded incomplete minibatch','capture warmup optimizer reset','exact optimizer resume']})
    print('ENGINE VERIFIED',maxdiff,resumediff,flush=True)

if __name__=='__main__':main()
