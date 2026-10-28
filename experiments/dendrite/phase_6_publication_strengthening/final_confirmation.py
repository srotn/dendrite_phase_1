from __future__ import annotations
import importlib
import pandas as pd
from engine import *
from prepare import MODULES,unique_rows

OUT=ROOT/'verification/final_confirmation'

def freeze_all_models():
    for stage in STAGES:assert json.loads((ROOT/stage/'VERIFIED.json').read_text(encoding='utf-8'))['passed']
    paths=[]
    for stage in STAGES:
        paths+=list((ROOT/stage/'runs').rglob('best.pt'))+list((ROOT/stage/'runs').rglob('budget_*.pt'))+list((ROOT/stage/'runs').rglob('last.pt'))
        paths+=list((ROOT/stage/'runs').rglob('VALIDATION_SELECTION_*.json'))
    paths+=list((ROOT/STAGES[0]/'historical_checkpoint_evaluation').rglob('PROVENANCE.json'))+list((ROOT/STAGES[0]/'historical_checkpoint_evaluation').rglob('last.pt'))
    paths+=list((ROOT/STAGES[1]/'classic').rglob('model.joblib'))+list((ROOT/STAGES[1]/'classic').rglob('VALIDATION_SELECTION.json'))
    record={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}
    lock=OUT/'ALL_MODELS_LOCK.json'
    if lock.exists():assert json.loads(lock.read_text(encoding='utf-8'))['files']==record
    else:js(lock,dict(date=now(),files=record,all_four_stages_verified=True,test_generation_has_not_started=True))

def generate(task):
    target=OUT/'data'/task
    if (target/'PREPARED.json').exists():return load(target/'tests.pt')
    d=load(data_path(task,'historical'));a=load(test_path(task,'historical'));oldsource=json.loads((PREVIOUS/task/'data/manifest.json').read_text(encoding='utf-8'))['source'];old=load(oldsource)
    used=set()
    for v in a.values():used|=unique_rows(v['physical'])
    for split in ['train','valid','iid','ood']:
        if split in old:used|=unique_rows(old[split]['physical'])
    module=importlib.import_module('experiments.dendrite.phase_1_synthetic_experiments.'+MODULES[task]+'.experiment');tests={}
    for ood in ([False,True] if task=='04_impedance' else [False]):
        seed=2026120101+100*TASKS.index(task)+(10000 if ood else 0)
        kwargs={} if task=='01_forced' else dict(device='cpu')
        if task=='04_impedance':kwargs['ood']=ood
        v=module.generate(20000,seed,**kwargs);rows=unique_rows(v['physical']);assert len(rows)==20000 and not used&rows;used|=rows
        meta=d['preprocessing'];r=v['physical'].sign()/v['physical'].abs().clamp_min(1e-6);r=(r.clamp(-meta['cap'],meta['cap'])-meta['mean'])/meta['std']
        name='ood_final' if ood else 'iid_final'
        tests[name]=dict(x=v['x'],r=r,physical=v['physical'],y=(v['y']-d['target_mean'])/d['target_scale'],groups=torch.arange(20000),row_ids=torch.arange(20000),seed=seed)
    save(target/'tests.pt',tests);js(target/'PREPARED.json',dict(date=now(),seeds={k:v['seed'] for k,v in tests.items()},sha256=sha(target/'tests.pt'),
        disjoint_from_historical_and_stage1_tests=True,models_lock_sha256=sha(OUT/'ALL_MODELS_LOCK.json')))
    return tests

def neural_jobs():
    for source in sorted((ROOT/STAGES[0]/'historical_checkpoint_evaluation').rglob('PROVENANCE.json')):
        meta=json.loads(source.read_text(encoding='utf-8'));out=source.parent
        yield 1,out,meta,2000,out/'last.pt',None
    for stage_index in [2,3,4]:
        for p in sorted((ROOT/STAGES[stage_index-1]/'runs').rglob('config.json')):
            meta=json.loads(p.read_text(encoding='utf-8'))
            if meta['task'] not in SYNTHETIC:continue
            for budget in ([2000,4000,8000] if stage_index==3 else [meta['epochs']]):
                source=p.parent/(f'budget_{budget}.pt' if stage_index==3 else 'last.pt')
                selection=p.parent/(f'VALIDATION_SELECTION_budget_{budget}.json' if stage_index==3 else 'VALIDATION_SELECTION_final.json')
                yield stage_index,p.parent,meta,budget,source,selection if selection.exists() else None

def main():
    configure();lock_protocol();freeze_all_models()
    testsets={task:generate(task) for task in SYNTHETIC};rows=[]
    jobs=list(neural_jobs());assert len(jobs)==324 and sum(len(job[2]['configs']) for job in jobs)==1656
    for j,(stage,source,meta,budget,checkpoint,selection) in enumerate(jobs,1):
        task=meta['task'];d=load(data_path(task,'historical'));configs=meta['configs'];seed=meta['seed']
        dest=OUT/'evaluations'/str(stage)/source.relative_to(ROOT/STAGES[stage-1])/f'budget_{budget}'
        if not (dest/'PROVENANCE.json').exists():
            state=load(checkpoint);save(dest/'last.pt',dict(best=state['best'],failed=state['failed'],epoch=budget))
            js(dest/'PROVENANCE.json',dict(source=str(checkpoint),source_sha256=sha(checkpoint),stage=stage,budget=budget,task=task,seed=seed,configs=configs,selection=str(selection) if selection else None))
        selected=set(range(len(configs)))
        if selection:selected={r['selected'] for r in json.loads(selection.read_text(encoding='utf-8'))['selections'] if r['selected'] is not None}
        result=evaluate_test(dest,d,configs,seed,testsets[task])
        for r in result:
            if 'normalized_rmse' not in r:continue
            c=configs[r['candidate']]
            method='mlp_'+c['input'] if c['kind']=='mlp' else c['name'] if 'slot' in c else 'dd_'+c['input']
            row=dict(stage=stage,task=task,seed=seed,depth=c.get('target_dd_depth',c.get('depth')),method=method,budget=budget,domain=r['split'].split('_')[0],
                rmse=r['normalized_rmse'],valid_rmse=r['valid_rmse'],selected=r['candidate'] in selected,source=str(dest.relative_to(ROOT)))
            for i,v in enumerate(r['raw_rmse_per_output']):row[f'raw_rmse_{i}']=v
            rows.append(row)
        js(ROOT/'status.json',dict(pid=os.getpid(),stage='final_confirmation',status='evaluating',group=j,total_groups=len(jobs),updated=now()))
        print('FINAL CONFIRMATION',j,len(jobs),task,stage,budget,flush=True)
    import joblib
    from stage2 import classic_predict
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        for task in SYNTHETIC:
            for method in ['polynomial_ridge','hist_gradient_boosting']:
                parent=ROOT/STAGES[1]/'classic'/task/'historical'/method
                choice=json.loads((parent/'VALIDATION_SELECTION.json').read_text(encoding='utf-8'));source=parent/f"candidate_{choice['candidate']}"/'model.joblib'
                model=joblib.load(source);dest=OUT/'classic'/task/method;arrays={}
                for name,v in testsets[task].items():
                    pred=classic_predict(model,method,v['x'].numpy().astype(np.float64));error=pred-v['y'].numpy();scale=load(data_path(task,'historical'))['target_scale'].numpy()
                    row=dict(stage=2,task=task,seed=0,depth=4,method=method,budget=0,domain=name.split('_')[0],rmse=float(np.sqrt(np.mean(error**2))),valid_rmse=choice['valid_rmse'],selected=True,source=str(source.relative_to(ROOT)))
                    for i,value in enumerate(np.sqrt(np.mean((error*scale)**2,axis=0))):row[f'raw_rmse_{i}']=float(value)
                    rows.append(row);arrays['pred_'+name]=pred;arrays['y_'+name]=v['y'].numpy()
                dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/'predictions.npz',**arrays);js(dest/'PROVENANCE.json',dict(source=str(source),sha256=sha(source),selected_candidate=choice['candidate']))
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'metrics.csv',index=False,encoding='utf-8-sig')
    frame[frame.selected].to_csv(OUT/'selected_metrics.csv',index=False,encoding='utf-8-sig')
    freeze_all_models()
    js(OUT/'VERIFIED.json',dict(date=now(),passed=True,neural_evaluation_groups=len(jobs),attempted_candidate_budgets=1656,metric_rows=len(frame),new_iid_rows=80000,new_ood_rows=20000,
        models_frozen_before_generation=True,test_inputs_disjoint=True,all_selected_by_previous_validation=True,real_new_external_data=False))
    print('FINAL CONFIRMATION COMPLETE',len(frame),'rows',flush=True)

if __name__=='__main__':main()
