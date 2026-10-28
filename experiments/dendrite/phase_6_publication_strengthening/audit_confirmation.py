"""Independently replay final-test predictions from the original frozen weights."""
from __future__ import annotations
import importlib
import pandas as pd
from engine import *
from prepare import MODULES, unique_rows
from final_confirmation import OUT, neural_jobs, freeze_all_models

@torch.no_grad()
def main():
    configure();freeze_all_models()
    lock=json.loads((OUT/'ALL_MODELS_LOCK.json').read_text(encoding='utf-8'))
    final_tests={};data={};datasets_checked=[]
    for task in SYNTHETIC:
        d=load(data_path(task,'historical'));data[task]=d
        file=OUT/'data'/task/'tests.pt';info=json.loads((file.parent/'PREPARED.json').read_text(encoding='utf-8'))
        assert info['date']>=lock['date'] and info['models_lock_sha256']==sha(OUT/'ALL_MODELS_LOCK.json')
        assert sha(file)==info['sha256']
        tests=load(file);final_tests[task]=tests
        earlier=load(test_path(task,'historical'))
        source=json.loads((PREVIOUS/task/'data/manifest.json').read_text(encoding='utf-8'))['source']
        historical=load(source);used=set()
        for split in ['train','valid','iid','ood']:
            if split in historical:used|=unique_rows(historical[split]['physical'])
        for v in earlier.values():used|=unique_rows(v['physical'])
        module=importlib.import_module('experiments.dendrite.phase_1_synthetic_experiments.'+MODULES[task]+'.experiment')
        for name,v in tests.items():
            kwargs={} if task=='01_forced' else dict(device='cpu')
            if task=='04_impedance':kwargs['ood']=name.startswith('ood')
            expected_seed=2026120101+100*TASKS.index(task)+(10000 if name.startswith('ood') else 0)
            assert v['seed']==expected_seed
            generated=module.generate(20000,expected_seed,**kwargs)
            assert torch.equal(v['physical'],generated['physical']) and torch.equal(v['x'],generated['x'])
            assert torch.equal(v['y'],(generated['y']-d['target_mean'])/d['target_scale'])
            meta=d['preprocessing'];r=generated['physical'].sign()/generated['physical'].abs().clamp_min(1e-6)
            expected_r=(r.clamp(-meta['cap'],meta['cap'])-meta['mean'])/meta['std']
            assert torch.equal(v['r'],expected_r)
            rows=unique_rows(v['physical']);assert len(rows)==20000 and not rows&used;used|=rows
            datasets_checked.append(dict(task=task,split=name,seed=expected_seed,n=20000,regenerated_exactly=True,disjoint=True))
    frame=pd.read_csv(OUT/'metrics.csv');jobs=list(neural_jobs());assert len(jobs)==324
    checked=[];failures=[];matched_rows=0
    for stage,source,meta,budget,checkpoint,selection in jobs:
        task=meta['task'];d=data[task];tests=final_tests[task]
        dest=OUT/'evaluations'/str(stage)/source.relative_to(ROOT/STAGES[stage-1])/f'budget_{budget}'
        provenance=json.loads((dest/'PROVENANCE.json').read_text(encoding='utf-8'))
        assert provenance['source_sha256']==sha(checkpoint)
        assert lock['files'][str(checkpoint.relative_to(ROOT))]==sha(checkpoint)
        state=load(checkpoint);records=json.loads((dest/'test_final.json').read_text(encoding='utf-8'))
        selected=set(range(len(meta['configs'])))
        if selection:
            s=json.loads(selection.read_text(encoding='utf-8'));assert s['date']<=lock['date']
            assert s['checkpoint_sha256']==sha(checkpoint)
            selected=set()
            for group in s['selections']:
                valid=[i for i in group['candidates'] if state['failed'][i] is None and state['best'][i] is not None]
                winner=min(valid,key=lambda i:(state['best'][i]['valid_mse'],i)) if valid else None
                assert winner==group['selected']
                if winner is not None:selected.add(winner)
        with np.load(dest/'predictions_final.npz') as arrays:
            for i,config in enumerate(meta['configs']):
                if state['failed'][i] is not None or state['best'][i] is None:
                    failed_records=[r for r in records if r['candidate']==i]
                    assert len(failed_records)==1 and 'normalized_rmse' not in failed_records[0]
                    failures.append(dict(source=str(checkpoint.relative_to(ROOT)),candidate=i,budget=budget));continue
                m=build(d,config,meta['seed']).cuda();m.load_state_dict(state['best'][i]['model']);m.eval()
                for name,v in tests.items():
                    x=inputs(d,config,v).cuda()
                    pred=torch.cat([m(x[a:a+4096]).cpu() for a in range(0,len(x),4096)])
                    saved=arrays[f'pred_{i}_{name}'];assert np.allclose(pred.numpy(),saved,rtol=2e-5,atol=1e-7)
                    assert np.array_equal(arrays['y_'+name],v['y'].numpy())
                    error=pred.double()-v['y'].double();rmse=float(error.square().mean().sqrt())
                    raw=(error*d['target_scale'].double()).square().mean(0).sqrt().tolist()
                    rec=next(r for r in records if r['candidate']==i and r.get('split')==name)
                    assert np.isclose(rmse,rec['normalized_rmse'],rtol=2e-5,atol=1e-8)
                    assert np.allclose(raw,rec['raw_rmse_per_output'],rtol=2e-5,atol=1e-8)
                    method='mlp_'+config['input'] if config['kind']=='mlp' else config['name'] if 'slot' in config else 'dd_'+config['input']
                    # Multiple LR candidates can share a method, but validation RMSE and candidate order disambiguate via JSON.
                    matches=frame[(frame.source==str(dest.relative_to(ROOT)))&(frame.method==method)&(frame.domain==name.split('_')[0])]
                    matches=matches[np.isclose(matches.valid_rmse,rec['valid_rmse'],rtol=1e-12,atol=1e-12)&np.isclose(matches.rmse,rec['normalized_rmse'],rtol=1e-12,atol=1e-12)]
                    assert len(matches)>=1 and ((matches.selected==(i in selected)).any())
                    matched_rows+=1
                del m
        checked.append(dict(source=str(checkpoint.relative_to(ROOT)),budget=budget,candidates=len(meta['configs'])))
        print('FINAL AUDIT',len(checked),324,task,flush=True)
    import joblib
    from stage2 import classic_predict
    from threadpoolctl import threadpool_limits
    classical=0
    with threadpool_limits(limits=1):
        for task in SYNTHETIC:
            for method in ['polynomial_ridge','hist_gradient_boosting']:
                dest=OUT/'classic'/task/method;provenance=json.loads((dest/'PROVENANCE.json').read_text(encoding='utf-8'))
                source=Path(provenance['source']);assert sha(source)==provenance['sha256']==lock['files'][str(source.relative_to(ROOT))]
                chosen=json.loads((source.parent.parent/'VALIDATION_SELECTION.json').read_text(encoding='utf-8'))
                assert chosen['locked_before_test']<=lock['date'] and chosen['candidate']==provenance['selected_candidate']
                model=joblib.load(source)
                with np.load(dest/'predictions.npz') as arrays:
                    for name,v in final_tests[task].items():
                        pred=classic_predict(model,method,v['x'].numpy().astype(np.float64))
                        assert np.array_equal(pred,arrays['pred_'+name]) and np.array_equal(v['y'].numpy(),arrays['y_'+name])
                        rmse=float(np.sqrt(np.mean((pred-v['y'].numpy())**2)))
                        rec=frame[(frame.task==task)&(frame.method==method)&(frame.domain==name.split('_')[0])]
                        assert len(rec)==1 and np.isclose(rec.iloc[0].rmse,rmse,rtol=1e-12,atol=1e-12)
                        matched_rows+=1
                classical+=1
    assert matched_rows==len(frame),(matched_rows,len(frame))
    freeze_all_models()
    js(OUT/'INDEPENDENT_AUDIT.json',dict(date=now(),passed=True,regenerated_datasets=datasets_checked,
       checked_neural_groups=checked,failed_candidate_budgets=failures,classic_models_replayed=classical,
       csv_metric_rows_recomputed=matched_rows,source_checkpoint_replay=True,validation_selection_recomputed=True,
       models_and_selections_predate_test_generation=True,models_unchanged_after_confirmation=True))
    print('INDEPENDENT FINAL AUDIT PASSED',flush=True)

if __name__=='__main__':main()
