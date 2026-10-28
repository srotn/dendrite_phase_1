from __future__ import annotations
import itertools
import joblib
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from engine import *
from selection import lock_selection

STAGE=ROOT/STAGES[1]

def mlp_configs(d,depth):
    target=parameter_count(build(d,dict(kind='dd',input='duplicate',depth=depth),SEEDS[0]))
    configs=[]
    for lr in LRS:
        for method in ['base','reciprocal']:
            width=inputs(d,dict(input=method),'train').shape[1];m=d['train']['y'].shape[1]
            h=min(range(1,1025),key=lambda h:abs(width*h+h+h*h+h+h*m+m-target))
            actual=width*h+h+h*h+h+h*m+m
            configs.append(dict(name=f'mlp_{method}_p{depth}_lr{lr:g}',kind='mlp',input=method,standardize=True,hidden=h,lr=lr,
                                target_dd_depth=depth,target_parameters=target,parameter_difference_percent=100*(actual/target-1)))
    return configs

def neural():
    jobs=[(task,fold,seed,depth) for task,fold in datasets() for seed in SEEDS for depth in [4,16]]
    for number,(task,fold,seed,depth) in enumerate(jobs,1):
        js(ROOT/'status.json',dict(pid=os.getpid(),stage=STAGES[1],status='mlp_training',group=number,total_groups=len(jobs),task=task,fold=fold,seed=seed,depth=depth,updated=now()))
        d=load(data_path(task,fold));out=STAGE/'runs'/task/fold/f'seed_{seed}'/f'target_depth_{depth:02d}'
        train(out,d,mlp_configs(d,depth),seed,2000)
    return jobs

def rmse(pred,y):
    pred=np.asarray(pred,dtype=np.float64); y=np.asarray(y,dtype=np.float64)
    # sklearn returns (N,) for a single-output target while torch exports (N,1).
    # Normalize both shapes before subtraction to avoid NxN broadcasting.
    if pred.ndim==1 and y.ndim==2 and y.shape[1]==1: pred=pred[:,None]
    if y.ndim==1 and pred.ndim==2 and pred.shape[1]==1: y=y[:,None]
    return float(np.sqrt(np.mean((pred-y)**2)))

def classic_plot(out,records,label):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    for key in ['train_rmse','valid_rmse']:ax.plot([r['step'] for r in records],[r[key] for r in records], 'o-' if len(records)<3 else '-',label=key)
    ax.set_xlabel(label);ax.set_ylabel('Normalized RMSE');ax.legend();ax.grid(alpha=.2);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'training_curves.{ext}',dpi=150)
    plt.close(fig)

def classic_training():
    with threadpool_limits(limits=1):
        for task,fold in datasets():
            d=load(data_path(task,fold));x=d['train']['x'].numpy().astype(np.float64);y=d['train']['y'].numpy().astype(np.float64)
            vx=d['valid']['x'].numpy().astype(np.float64);vy=d['valid']['y'].numpy().astype(np.float64)
            for method,values in [('polynomial_ridge',[1e-4,1.,1e4]),('hist_gradient_boosting',[15,31,63])]:
                for candidate,value in enumerate(values):
                    out=STAGE/'classic'/task/fold/method/f'candidate_{candidate}';out.mkdir(parents=True,exist_ok=True)
                    if (out/'COMPLETED.json').exists():continue
                    js(ROOT/'status.json',dict(pid=os.getpid(),stage=STAGES[1],status='classic_training',task=task,fold=fold,method=method,candidate=candidate,updated=now()))
                    start=time.perf_counter();records=[]
                    if method=='polynomial_ridge':
                        model=make_pipeline(StandardScaler(),PolynomialFeatures(degree=2,include_bias=False),StandardScaler(),Ridge(alpha=value,solver='cholesky'))
                        model.fit(x,y);best_iteration=1;best_val=rmse(model.predict(vx),vy)
                        records=[dict(step=1,train_rmse=rmse(model.predict(x),y),valid_rmse=best_val)]
                        checkpoint={'model':model,'best_iteration':1};features=model[1].n_output_features_
                    else:
                        models=[HistGradientBoostingRegressor(max_iter=500,max_leaf_nodes=value,l2_regularization=.1,learning_rate=.1,early_stopping=False,random_state=2026102601) for _ in range(y.shape[1])]
                        for i,m in enumerate(models):m.fit(x,y[:,i])
                        best_val=float('inf');best_iteration=0
                        trains=zip(*(m.staged_predict(x) for m in models));valids=zip(*(m.staged_predict(vx) for m in models))
                        for iteration,(tr,va) in enumerate(zip(trains,valids),1):
                            a=rmse(np.stack(tr,1),y);b=rmse(np.stack(va,1),vy);records.append(dict(step=iteration,train_rmse=a,valid_rmse=b))
                            if b<best_val:best_val=b;best_iteration=iteration
                        checkpoint={'model':models,'best_iteration':best_iteration};features=x.shape[1]
                    joblib.dump(checkpoint,out/'model.joblib')
                    (out/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
                    classic_plot(out,records,'Closed-form solve (not epochs)' if method=='polynomial_ridge' else 'Boosting iteration')
                    js(out/'COMPLETED.json',dict(task=task,fold=fold,method=method,candidate=candidate,value=value,valid_rmse=best_val,best_iteration=best_iteration,
                          seconds=time.perf_counter()-start,feature_count=features,data_sha256=sha(data_path(task,fold)),model_sha256=sha(out/'model.joblib'),
                          curves=['training_curves.png','training_curves.pdf'],deterministic=True,selection='validation only',date=now()))
                    print('CLASSIC',task,fold,method,value,'valid',best_val,flush=True)

def classic_predict(checkpoint,method,x):
    if method=='polynomial_ridge':return checkpoint['model'].predict(x)
    chosen=checkpoint['best_iteration'];values=[]
    for model in checkpoint['model']:
        values.append(next(itertools.islice(model.staged_predict(x),chosen-1,chosen)))
    return np.stack(values,1)

def evaluate_classic():
    with threadpool_limits(limits=1):
        for task,fold in datasets():
            d=load(data_path(task,fold));tests=load(test_path(task,fold))
            for method in ['polynomial_ridge','hist_gradient_boosting']:
                parent=STAGE/'classic'/task/fold/method
                candidates=[json.loads(p.read_text(encoding='utf-8')) for p in parent.glob('*/COMPLETED.json')]
                assert len(candidates)==3
                selected=min(candidates,key=lambda c:(c['valid_rmse'],c['candidate']))
                selection_file=parent/'VALIDATION_SELECTION.json'
                if selection_file.exists():assert json.loads(selection_file.read_text(encoding='utf-8'))['candidate']==selected['candidate']
                else:js(selection_file,dict(**selected,locked_before_test=now()))
                # All trials are retained; selection is already frozen before test prediction.
                for c in candidates:
                    out=parent/f"candidate_{c['candidate']}"
                    if (out/'test.json').exists():continue
                    checkpoint=joblib.load(out/'model.joblib');records=[];arrays={}
                    for name,v in tests.items():
                        pred=classic_predict(checkpoint,method,v['x'].numpy().astype(np.float64));target=v['y'].numpy();
                        if pred.ndim==1 and target.ndim==2 and target.shape[1]==1: pred=pred[:,None]
                        error=pred-target;scale=d['target_scale'].numpy()
                        groups=[]
                        if task not in SYNTHETIC:
                            for g in torch.unique(v['groups']):
                                mask=(v['groups']==g).numpy();groups.append(dict(group=int(g),n=int(mask.sum()),mse=float(np.mean(error[mask]**2))))
                        records.append(dict(split=name,normalized_rmse=float(np.sqrt(np.mean(error**2))),raw_rmse_per_output=np.sqrt(np.mean((error*scale)**2,axis=0)).tolist(),groups=groups,n=len(pred)))
                        arrays['pred_'+name]=pred;arrays['y_'+name]=v['y'].numpy();arrays['group_'+name]=v['groups'].numpy()
                    np.savez_compressed(out/'predictions.npz',**arrays);js(out/'test.json',records)

def main():
    configure();lock_protocol()
    assert json.loads((ROOT/STAGES[0]/'VERIFIED.json').read_text(encoding='utf-8'))['passed']
    verify=json.loads((ROOT/'engine_verification.json').read_text(encoding='utf-8'));assert verify['engine_sha256']==sha(ROOT/'engine.py')
    jobs=neural();classic_training()
    js(STAGE/'TRAINING_LOCK.json',dict(date=now(),neural_fits=504,classic_fits=84,checkpoints={str(p.relative_to(STAGE)):sha(p) for p in STAGE.rglob('best.pt')},
                                       classic_checkpoints={str(p.relative_to(STAGE)):sha(p) for p in STAGE.rglob('model.joblib')}))
    for task,fold,seed,depth in jobs:
        out=STAGE/'runs'/task/fold/f'seed_{seed}'/f'target_depth_{depth:02d}';d=load(data_path(task,fold))
        lock_selection(out,mlp_configs(d,depth))
    for task,fold,seed,depth in jobs:
        out=STAGE/'runs'/task/fold/f'seed_{seed}'/f'target_depth_{depth:02d}';d=load(data_path(task,fold))
        evaluate_test(out,d,mlp_configs(d,depth),seed,load(test_path(task,fold)))
    evaluate_classic();js(ROOT/'status.json',dict(pid=os.getpid(),stage=STAGES[1],status='trained_evaluated_pending_analysis',updated=now()))

if __name__=='__main__':main()
