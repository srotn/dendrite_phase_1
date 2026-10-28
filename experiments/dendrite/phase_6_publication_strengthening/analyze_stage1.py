from __future__ import annotations
import csv
import pandas as pd
from scipy import stats
from common import *
from audit_runs import audit_neural

STAGE=ROOT/STAGES[0]

def gather():
    rows=[];groups=[]
    for p in STAGE.rglob('test_final.json'):
        if 'historical_checkpoint_evaluation' in p.parts:
            meta=json.loads((p.parent/'PROVENANCE.json').read_text(encoding='utf-8'));fold='historical'
        else:
            meta=json.loads((p.parent/'config.json').read_text(encoding='utf-8'));fold=meta['fold']
        for r in json.loads(p.read_text(encoding='utf-8')):
            if 'normalized_rmse' not in r:continue
            config=meta['configs'][r['candidate']]
            row={k:v for k,v in r.items() if k not in ['groups','raw_rmse_per_output','raw_mae_per_output']}
            row.update(task=meta['task'],fold=fold,seed=meta['seed'],depth=config['depth'],method=config['input'],source=str(p.relative_to(STAGE)))
            for i,v in enumerate(r['raw_rmse_per_output']):row[f'raw_rmse_output_{i}']=v
            rows.append(row)
            for group in r['groups']:groups.append(dict(task=meta['task'],fold=fold,seed=meta['seed'],depth=config['depth'],method=config['input'],**group))
    return pd.DataFrame(rows),groups

def verify():
    records=[];total=0;success=0;failed=[];source_cache={};test_groups={t:[] for t in TASKS[4:]}
    for task,fold in datasets():
        if task in SYNTHETIC:
            source=STAGE/'historical_checkpoint_evaluation'/task
            for seed in OLD_SEEDS:
                for depth in DEPTHS:
                    out=source/f'seed_{seed}'/f'depth_{depth:02d}'
                    provenance=json.loads((out/'PROVENANCE.json').read_text(encoding='utf-8'))
                    assert sha(provenance['source_best'])==provenance['source_sha256']
                    r=json.loads((out/'test_final.json').read_text(encoding='utf-8'))
                    assert len(r)==(18 if task=='04_impedance' else 9)
            continue
        d=load(data_path(task,fold));tests=load(test_path(task,fold));test=tests['test']
        if task not in source_cache:
            from prepare import real_source
            source_cache[task]=real_source(task)
        sx,sy,sg,srow,_=source_cache[task]
        gs=[set(d[s]['groups'].tolist()) for s in ['train','valid']]+[set(test['groups'].tolist())]
        assert all(not gs[a]&gs[b] for a,b in [(0,1),(0,2),(1,2)])
        assert set.union(*gs)==set(sg.tolist())
        test_groups[task]+=list(gs[2])
        physical=d['train']['physical'];e=d['eligible'];meta=d['preprocessing']
        r=physical[:,e].sign()/physical[:,e].abs().clamp_min(1e-6)
        cap=torch.quantile(r.abs(),.995,dim=0).clamp_min(1e-6)
        assert torch.equal(cap,meta['cap'])
        capped=r.clamp(-cap,cap)
        ty=sy[d['train']['source_indices']]
        if task=='05_concrete':
            assert torch.equal(physical.min(0).values,meta['x_mean'])
            calc=2*(physical-meta['x_mean'])/meta['x_scale']-1
            assert torch.equal(capped.min(0).values,meta['r_mean'])
            assert torch.equal((capped.max(0).values-capped.min(0).values).clamp_min(1e-6),meta['r_scale'])
            target_scale=(ty.max(0).values-ty.min(0).values).clamp_min(1e-6)/2
            target_mean=ty.min(0).values+target_scale
        else:
            assert torch.equal(physical.mean(0),meta['x_mean'])
            calc=((physical-meta['x_mean'])/meta['x_scale']).clamp(-3,3)
            assert torch.equal(capped.mean(0),meta['r_mean'])
            assert torch.equal(capped.std(0,correction=0).clamp_min(1e-6),meta['r_scale'])
            target_scale=ty.std(0,correction=0).clamp_min(1e-6);target_mean=ty.mean(0)
        assert torch.allclose(calc,d['train']['x'])
        assert torch.equal(target_scale,d['target_scale']) and torch.equal(target_mean,d['target_mean'])
        for v in [d['train'],d['valid'],test]:
            index=v['source_indices']
            assert torch.equal(v['physical'],sx[index]) and torch.equal(v['groups'],sg[index]) and torch.equal(v['row_ids'],srow[index])
            assert torch.allclose(v['y'],(sy[index]-target_mean)/target_scale)
            rr=v['physical'][:,e].sign()/v['physical'][:,e].abs().clamp_min(1e-6);rr=rr.clamp(-cap,cap)
            rr=(rr-meta['r_mean'])/meta['r_scale']
            rr=2*rr-1 if task=='05_concrete' else rr.clamp(-3,3)
            assert torch.allclose(rr,v['r'])
        for seed in SEEDS:
            for depth in DEPTHS:
                out=STAGE/'runs'/task/fold/f'seed_{seed}'/f'depth_{depth:02d}'
                c=json.loads((out/'COMPLETED.json').read_text(encoding='utf-8'))
                config=json.loads((out/'config.json').read_text(encoding='utf-8'))
                assert config['data_sha256']==sha(data_path(task,fold))
                logs=[json.loads(x) for x in (out/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
                assert len(logs)==2000 and [r['epoch'] for r in logs]==list(range(1,2001))
                assert sha(out/'best.pt')==c['best_sha256'] and sha(out/'last.pt')==c['last_sha256']
                assert (out/'training_curves.png').stat().st_size>1000 and (out/'training_curves.pdf').stat().st_size>1000
                assert len(c['verification'])==3
                results=json.loads((out/'test_final.json').read_text(encoding='utf-8'));assert len(results)==3
                for i,f in enumerate(c['failed_first_epoch']):
                    if f is not None:failed.append(dict(run=str(out.relative_to(STAGE)),candidate=i,epoch=f))
                    else:assert c['verification'][i]['optimizer_steps']==config['steps_per_epoch']*2000
                total+=c['attempted_models'];success+=c['successful_models'];records.append(str(out.relative_to(STAGE)))
    assert len(records)==180 and total==540
    for task,groups in test_groups.items():
        assert len(groups)==len(set(groups)) and set(groups)==set(source_cache[task][2].tolist())
    return dict(date=now(),passed=True,new_attempts=total,new_successful=success,failures=failed,historical_models=360,training_groups=len(records),
                checked=['all planned groups','all per-epoch logs','all saved replay records','optimizer step counts','checkpoint hashes','curves','test counts','group separation','train-only preprocessing recomputation'])

def main():
    configure();audit=verify();disk_audit=audit_neural(STAGE,180,540,2000);audit['disk_checkpoint_replay_passed']=disk_audit['passed'];out=STAGE/'analysis';out.mkdir(exist_ok=True)
    frame,groups=gather();frame.to_csv(out/'per_run_test_metrics.csv',index=False,encoding='utf-8-sig');js(out/'per_group_test_metrics.json',groups)
    # Pool the three equally sized independent test batches by mean square error.
    frame['domain']=frame['split'].str.split('_').str[0];frame['mse']=frame.normalized_rmse**2
    pooled=frame.groupby(['task','fold','seed','depth','method','domain']).mse.mean().pow(.5).rename('rmse').reset_index()
    pooled.to_csv(out/'pooled_test_metrics.csv',index=False,encoding='utf-8-sig')
    pivot=pooled.pivot(index=['task','fold','seed','depth','domain'],columns='method',values='rmse').reset_index()
    for control in ['base','duplicate']:pivot['gain_vs_'+control]=100*(1-pivot.reciprocal/pivot[control])
    pivot.to_csv(out/'paired_gains.csv',index=False,encoding='utf-8-sig')
    summary=[];slopes=[]
    for task in TASKS:
        part=pivot[(pivot.task==task)&(pivot.domain.isin(['iid','test']))]
        unit='optimization_seed' if task in SYNTHETIC else 'test_fold_mean_over_3_seeds'
        for control in ['base','duplicate']:
            key='gain_vs_'+control
            for depth in DEPTHS:
                z=part[part.depth==depth]
                values=z[key].to_numpy() if task in SYNTHETIC else z.groupby('fold')[key].mean().to_numpy()
                if task in SYNTHETIC:
                    radius=stats.t.ppf(.975,len(values)-1)*stats.sem(values)
                    low,high=np.mean(values)-radius,np.mean(values)+radius;interval='95% t over fixed-data optimization seeds'
                else:low,high=min(values),max(values);interval='range of fold means; NOT a confidence interval'
                summary.append(dict(task=task,control=control,depth=depth,mean=float(np.mean(values)),low=low,high=high,unit=unit,interval=interval,values=values.tolist()))
            units=part.groupby('seed' if task in SYNTHETIC else 'fold')
            vals=[]
            for _,z in units:
                z=z.groupby('depth')[key].mean().reindex(DEPTHS)
                vals.append(float(np.polyfit(np.log2(DEPTHS),z.to_numpy(),1)[0]))
            slopes.append(dict(task=task,control=control,mean=float(np.mean(vals)),unit=unit,values=vals,
                p_unadjusted=float(stats.ttest_1samp(vals,0).pvalue) if task in SYNTHETIC else None))
    # Only four synthetic tasks x two controls; real overlapping folds are descriptive.
    tested=[r for r in slopes if r['p_unadjusted'] is not None];order=sorted(tested,key=lambda r:r['p_unadjusted']);last=0
    for i,r in enumerate(order):last=max(last,min(1.,(len(order)-i)*r['p_unadjusted']));r['p_holm_8']=last
    js(out/'gain_summary.json',summary);js(out/'slopes.json',slopes)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for task,ax in zip(TASKS,axes.flat):
        for control,color in [('base','#6b7280'),('duplicate','#087f8c')]:
            rs=[r for r in summary if r['task']==task and r['control']==control]
            means=np.array([r['mean'] for r in rs]);lo=np.array([r['low'] for r in rs]);hi=np.array([r['high'] for r in rs])
            ax.plot(DEPTHS,means,'o-',color=color,label=control);ax.fill_between(DEPTHS,lo,hi,color=color,alpha=.12)
        ax.axhline(0,c='gray',lw=.6);ax.set_xscale('log',base=2);ax.set_xticks(DEPTHS,DEPTHS);ax.set_title(task);ax.set_xlabel('DD depth');ax.set_ylabel('RMSE reduction (%)');ax.grid(alpha=.15)
    axes[0,0].legend();fig.suptitle('New synthetic tests / grouped real-data tests\nBands: synthetic seed 95% t interval; real fold-mean range',fontsize=11);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'depth_gain_new_tests.{ext}',dpi=180)
    plt.close(fig)
    lines=['# 第一项：新测试与真实数据分组评估','',f"完成 {audit['new_attempts']} 次新拟合（成功 {audit['new_successful']}，数值失败 {len(audit['failures'])}），复评 360 个历史合成模型。",'',
      '四合成任务各三份 20000 样本 IID 测试，阻抗另三份 OOD。混凝土按 428 个配方组、CO 按 13 份采集文件冻结五折；每折三个优化种子。新分组结果不是新增外部采集或未来时间测试。','',
      '## 同参数复制对照的测试 RMSE 降幅','', '|任务|1|2|3|4|8|16|','|---|---:|---:|---:|---:|---:|---:|']
    for task in TASKS:
        rs=[r for r in summary if r['task']==task and r['control']=='duplicate'];lines.append('|'+task+'|'+'|'.join(f"{r['mean']:.2f}%" for r in rs)+'|')
    lines+=['','真实数据先在折内平均三种子增益，再平均五折；合成先按 MSE 合并三份测试，再逐优化种子计算配对增益。不同任务的原始单位不混合。',
       '', '## 不确定性与选择边界','', '合成区间仅覆盖固定训练数据下的五个优化种子。真实数据图带是折均值范围，不是校准置信区间；五折训练集重叠，不进行天真的五独立实验 t 检验。全部检查点只按验证选择。后续方案在本项结果前已冻结。',
       '', '逐运行指标、组级误差、全部配对值、趋势摘要与图表位于 `analysis/`；失败清单见 `VERIFIED.json`。不将大量测试行当作模型重复数。']
    (STAGE/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    js(STAGE/'VERIFIED.json',audit);print('STAGE 1 VERIFIED',audit['new_successful'],'successful',flush=True)

if __name__=='__main__':main()
