from engine import *

def audit_neural(stage,expected_groups,expected_models,epochs,budgets=None):
    configure();stage=Path(stage);completed=sorted((stage/'runs').rglob('COMPLETED.json'))
    assert len(completed)==expected_groups,(stage,len(completed),expected_groups)
    stage_number=STAGES.index(stage.name)+1
    planned={}
    for task,fold in datasets():
        if stage_number==1 and task in SYNTHETIC:continue
        depths={1:DEPTHS,2:[4,16],3:[1,4,8,16],4:[1,4,16]}[stage_number]
        for seed in SEEDS:
            for depth in depths:
                name=f'target_depth_{depth:02d}' if stage_number==2 else f'depth_{depth:02d}'
                planned[stage/'runs'/task/fold/f'seed_{seed}'/name]=(task,fold,seed,depth)
    assert {p.parent for p in completed}==set(planned), 'Completed identities must exactly match the frozen factorial design'
    total=0;success=0;failures=[];replayed=[]
    cache={}
    for path in completed:
        out=path.parent;c=json.loads((out/'config.json').read_text(encoding='utf-8'));done=json.loads(path.read_text(encoding='utf-8'))
        from render_curves import render
        render(out)
        curve_record=json.loads((out/'CURVES.json').read_text(encoding='utf-8'))
        assert curve_record['source_log_sha256']==sha(out/'metrics.jsonl') and curve_record['quantity']=='sqrt(logged MSE) = normalized RMSE'
        key=(c['task'],c['fold'])
        if key not in cache:cache={key:load(data_path(*key))}
        d=cache[key]
        task,fold,seed,depth=planned[out]
        assert (c['task'],c['fold'],c['seed'])==(task,fold,seed)
        if stage_number==2:
            from stage2 import mlp_configs
            expected_configs=mlp_configs(d,depth)
        elif stage_number==4:
            from stage4 import configs
            expected_configs=configs(depth)
        else:expected_configs=dd_configs(depth) if stage_number==1 else dd_configs(depth,('duplicate','reciprocal'),LRS)
        assert c['configs']==expected_configs and c['epochs']==epochs
        assert c['budgets']==([2000,4000,8000] if stage_number==3 else [epochs])
        assert c['train_rows']==len(d['train']['y']) and c['valid_rows']==len(d['valid']['y'])
        assert c['batch']==min(4096,c['train_rows']) and c['steps_per_epoch']==math.ceil(c['train_rows']/c['batch'])
        source_version=ROOT/'source_versions'/(c['engine_sha256']+'.py')
        assert source_version.is_file() and sha(source_version)==c['engine_sha256']
        assert sha(data_path(*key))==c['data_sha256']
        assert sha(out/'last.pt')==done['last_sha256'] and sha(out/'best.pt')==done['best_sha256']
        rows=[json.loads(x) for x in (out/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(rows)==epochs and [r['epoch'] for r in rows]==list(range(1,epochs+1))
        for f in ['training_curves.png','training_curves.pdf']:assert (out/f).stat().st_size>1000
        last=load(out/'last.pt');best=load(out/'best.pt');assert last['epoch']==epochs
        count=len(c['configs']);assert count==done['attempted_models']
        total+=count;success+=done['successful_models']
        vy=d['valid']['y'].cuda()
        for i,conf in enumerate(c['configs']):
            if done['failed_first_epoch'][i] is not None:
                failures.append(dict(run=str(out.relative_to(stage)),candidate=i,epoch=done['failed_first_epoch'][i]));continue
            model=build(d,conf,c['seed']).cuda();xx=inputs(d,conf,'valid').cuda()
            assert parameter_count(model)==c['parameters'][i]
            model.load_state_dict(last['models'][i]);final=float(evaluate([model],[xx],vy)[0])
            model.load_state_dict(best[i]['model']);selected=float(evaluate([model],[xx],vy)[0])
            assert np.isclose(final,rows[-1]['valid_mse'][i],rtol=2e-5,atol=1e-8)
            assert np.isclose(selected,best[i]['valid_mse'],rtol=2e-5,atol=1e-8)
            steps={int(last['optimizer']['state'][j]['step']) for j in last['optimizer']['param_groups'][i]['params']}
            assert steps=={epochs*c['steps_per_epoch']}
            replayed.append(dict(run=str(out.relative_to(stage)),candidate=i,best=selected,final=final,steps=list(steps)[0]))
        for budget in (budgets or [None]):
            suffix='final' if budget is None else f'budget_{budget}'
            r=json.loads((out/f'test_{suffix}.json').read_text(encoding='utf-8'))
            with np.load(out/f'predictions_{suffix}.npz') as predictions:
                for record in r:
                    if 'normalized_rmse' not in record:continue
                    pred=predictions[f"pred_{record['candidate']}_{record['split']}"];y=predictions['y_'+record['split']]
                    value=float(np.sqrt(np.mean((pred.astype(np.float64)-y)**2)))
                    assert np.isclose(value,record['normalized_rmse'],rtol=1e-7,atol=1e-9)
            if budget is not None:
                snap=load(out/f'budget_{budget}.pt');assert snap['epoch']==budget
                assert all(int(v['step'])==budget*c['steps_per_epoch'] for v in snap['optimizer']['state'].values())
                # Independently replay all successful budget-best checkpoints from disk.
                for i,conf in enumerate(c['configs']):
                    if snap['failed'][i] is not None or snap['best'][i] is None:continue
                    m=build(d,conf,c['seed']).cuda();m.load_state_dict(snap['best'][i]['model'])
                    val=float(evaluate([m],[inputs(d,conf,'valid').cuda()],vy)[0])
                    assert np.isclose(val,snap['best'][i]['valid_mse'],rtol=2e-5,atol=1e-8)
                    del m
        print('AUDIT',str(out.relative_to(stage)),flush=True)
    assert total==expected_models,(total,expected_models)
    result=dict(date=now(),passed=True,groups=len(completed),attempts=total,successful=success,failures=failures,replayed=replayed,
                checks=['disk best and final weights replayed','budget checkpoints replayed where applicable','all per-epoch rows','optimizer steps','data and checkpoint hashes','all saved test predictions recomputed','PNG/PDF curves'])
    js(stage/'analysis/disk_replay_audit.json',result);return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',type=int);a=p.parse_args()
    setup={1:(180,540,2000,None),2:(84,504,2000,None),3:(168,1008,8000,[2000,4000,8000]),4:(126,1008,4000,None)}
    audit_neural(ROOT/STAGES[a.stage-1],*setup[a.stage])
