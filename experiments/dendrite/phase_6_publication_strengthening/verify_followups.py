"""Verify planned controls on training inputs and classical prediction bookkeeping."""
from engine import *
from stage2 import mlp_configs,classic_predict
from stage4 import configs as mechanism_configs
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

def main():
    configure();records=[]
    for task,fold in datasets():
        d=load(data_path(task,fold))
        for depth in [4,16]:
            for c in mlp_configs(d,depth):
                model=build(d,c,SEEDS[0]);assert abs(c['parameter_difference_percent'])<=10
                assert model(inputs(d,c,'train')[:7]).shape==d['train']['y'][:7].shape
                records.append(dict(task=task,fold=fold,depth=depth,input=c['input'],parameter_difference_percent=c['parameter_difference_percent']))
        for depth in [1,4,16]:
            cs=mechanism_configs(depth);models=[build(d,c,SEEDS[0]) for c in cs]
            assert len({parameter_count(m) for m in models})==1
            assert all(all(torch.equal(v,models[0].state_dict()[k]) for k,v in m.state_dict().items()) for m in models[1:])
        base=inputs(d,dict(input='duplicate'),'train');matched=inputs(d,dict(input='matched'),'train')
        w=d['train']['x'].shape[1]
        assert torch.allclose(base[:,w:].mean(0),matched[:,w:].mean(0),atol=1e-5,rtol=1e-5)
        assert torch.allclose(base[:,w:].std(0,correction=0),matched[:,w:].std(0,correction=0),atol=1e-5,rtol=1e-5)
        subset=inputs(d,dict(input='subset'),'train');complement=inputs(d,dict(input='complement'),'train')
        mask=torch.tensor([i in FIELDS[task] for i in d['eligible']])
        assert torch.equal(subset[:,w:][:,~mask],base[:,w:][:,~mask])
        assert torch.equal(complement[:,w:][:,mask],base[:,w:][:,mask])
    with threadpool_limits(limits=1):
        x=np.random.default_rng(125).normal(size=(60,3));y=x[:,0]**2+x[:,1]
        m=HistGradientBoostingRegressor(max_iter=7,min_samples_leaf=3,early_stopping=False,random_state=1).fit(x,y)
        got=classic_predict({'model':[m],'best_iteration':3},'hist_gradient_boosting',x)
        assert np.array_equal(got[:,0],list(m.staged_predict(x))[2])
    js(ROOT/'followup_design_verification.json',dict(date=now(),passed=True,mlp_parameter_matches=records,
        max_mlp_parameter_mismatch_percent=max(abs(r['parameter_difference_percent']) for r in records),
        checks=['all 14 development-set shapes','MLP parameters within 10 percent of matched DD','all eight mechanism configurations equal nominal parameter counts and initial tensors',
                'matched training means and standard deviations','subset complement widths and duplicated coordinates','tree validation-selected iteration prediction on artificial data']))
    print('FOLLOW-UP DESIGN VERIFIED',flush=True)

if __name__=='__main__':main()
