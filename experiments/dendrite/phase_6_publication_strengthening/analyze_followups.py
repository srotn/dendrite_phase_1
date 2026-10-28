from __future__ import annotations
import argparse
import pandas as pd
from common import *
from audit_runs import audit_neural

def collect(stage,budgets=(None,),selected_only=True):
    rows=[]
    for path in sorted((stage/'runs').rglob('config.json')):
        c=json.loads(path.read_text(encoding='utf-8'));out=path.parent
        for budget in budgets:
            suffix='final' if budget is None else f'budget_{budget}'
            selection=out/f'VALIDATION_SELECTION_{suffix}.json'
            indices=set(range(len(c['configs'])))
            if selection.exists():
                choice=json.loads(selection.read_text(encoding='utf-8'))
                indices={r['selected'] for r in choice['selections'] if r['selected'] is not None}
                state=load(out/'last.pt' if budget is None else out/f'budget_{budget}.pt')
                assert sha(out/'last.pt' if budget is None else out/f'budget_{budget}.pt')==choice['checkpoint_sha256']
                for r in choice['selections']:
                    eligible=[i for i in r['candidates'] if state['failed'][i] is None and state['best'][i] is not None]
                    expected=min(eligible,key=lambda i:(state['best'][i]['valid_mse'],i)) if eligible else None
                    assert r['selected']==expected
            for r in json.loads((out/f'test_{suffix}.json').read_text(encoding='utf-8')):
                if 'normalized_rmse' not in r:continue
                i=r['candidate'];conf=c['configs'][i]
                if selected_only and i not in indices:continue
                depth=conf.get('target_dd_depth',conf.get('depth'))
                method='mlp_'+conf['input'] if conf['kind']=='mlp' else conf['name'] if 'slot' in conf else 'dd_'+conf['input']
                row=dict(task=c['task'],fold=c['fold'],seed=c['seed'],depth=depth,method=method,input=conf['input'],lr=conf['lr'],
                    budget=c['epochs'] if budget is None else budget,split=r['split'],domain=r['split'].split('_')[0],rmse=r['normalized_rmse'],
                    valid_rmse=r['valid_rmse'],best_epoch=r['best_epoch'],selected=i in indices,parameters=c['parameters'][i],
                    candidate=i,source=str(out.relative_to(ROOT)),n=r['n'])
                for j,v in enumerate(r['raw_rmse_per_output']):row[f'raw_rmse_{j}']=v
                rows.append(row)
    return pd.DataFrame(rows)

def pool(frame):
    keys=['task','fold','seed','depth','method','budget','domain']
    values=['rmse']+[x for x in frame.columns if x.startswith('raw_rmse_')]
    a=frame.copy()
    for key in values:a[key]=a[key]**2
    p=a.groupby(keys,dropna=False)[values].mean().pow(.5).reset_index()
    return p

def summary(frame):
    # Explicit equal-fold weighting; neural seeds are averaged within each fold first.
    a=frame.groupby(['task','fold','depth','method','budget','domain'],dropna=False).rmse.mean().reset_index()
    return a.groupby(['task','depth','method','budget','domain'],dropna=False).rmse.agg(['mean','min','max']).reset_index()

def fixed_lr_budget_table(all_trials):
    """Keep LR constant when comparing windows; never pool distinct candidates."""
    keys=['task','fold','seed','depth','lr','budget','domain','method']
    a=all_trials.assign(mse=all_trials.rmse**2)
    pooled=a.groupby(keys,dropna=False).mse.mean().pow(.5).rename('rmse').reset_index()
    paired=pooled.pivot(index=keys[:-1],columns='method',values='rmse').reset_index()
    for method in ['dd_duplicate','dd_reciprocal']:
        if method not in paired:paired[method]=np.nan
    paired['paired_available']=paired[['dd_duplicate','dd_reciprocal']].notna().all(axis=1)
    paired['gain_percent']=100*(1-paired.dd_reciprocal/paired.dd_duplicate)
    summary_keys=['task','depth','lr','budget','domain']
    counts=paired.groupby(summary_keys).paired_available.agg(['sum','count']).rename(columns={'sum':'available_pairs','count':'observed_units'})
    means=paired.groupby(summary_keys+['fold']).gain_percent.mean().groupby(summary_keys).mean().rename('gain_percent')
    return paired,means.to_frame().join(counts).reset_index()

def make_plot(frame,path,title):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for task,ax in zip(TASKS,axes.flat):
        part=frame[(frame.task==task)&frame.domain.isin(['test','iid'])]
        for method,p in part.groupby('method'):
            p=p.groupby('depth')['mean'].mean().sort_index();ax.plot(p.index,p.values,'o-',label=method,lw=1,ms=3)
        ax.set_title(task);ax.set_xscale('log',base=2);ax.set_xlabel('Depth / matched DD parameter target');ax.set_ylabel('Normalized test RMSE');ax.grid(alpha=.2)
        if len(part):ax.legend(fontsize=6)
    fig.suptitle(title);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(path.with_suffix('.'+ext),dpi=170)
    plt.close(fig)

def stage2():
    stage=ROOT/STAGES[1];out=stage/'analysis';out.mkdir(exist_ok=True)
    audit=audit_neural(stage,84,504,2000)
    all_trials=collect(stage,selected_only=False);all_trials.to_csv(out/'all_neural_trials.csv',index=False,encoding='utf-8-sig')
    selected=collect(stage);selected.to_csv(out/'selected_neural_test.csv',index=False,encoding='utf-8-sig')
    pooled=pool(selected);classical=[];classic_count=0
    import joblib
    from stage2 import classic_predict,rmse
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        for task,fold in datasets():
            d=load(data_path(task,fold))
            for method in ['polynomial_ridge','hist_gradient_boosting']:
                parent=stage/'classic'/task/fold/method;select=json.loads((parent/'VALIDATION_SELECTION.json').read_text(encoding='utf-8'))
                records=[json.loads(p.read_text(encoding='utf-8')) for p in parent.glob('*/COMPLETED.json')]
                assert len(records)==3 and select['candidate']==min(records,key=lambda r:(r['valid_rmse'],r['candidate']))['candidate']
                for c in records:
                    folder=parent/f"candidate_{c['candidate']}";assert sha(folder/'model.joblib')==c['model_sha256']
                    model=joblib.load(folder/'model.joblib');v=classic_predict(model,method,d['valid']['x'].numpy().astype(np.float64))
                    check_rmse=rmse(v,d['valid']['y'].numpy())
                    if not np.isclose(check_rmse,c['valid_rmse'],rtol=1e-7,atol=1e-9):
                        raise AssertionError(f'classic validation mismatch: {task} {fold} {method} candidate={c["candidate"]} recomputed={check_rmse} recorded={c["valid_rmse"]} pred_shape={np.asarray(v).shape} target_shape={d["valid"]["y"].shape}')
                    assert (folder/'training_curves.png').stat().st_size>1000 and (folder/'training_curves.pdf').stat().st_size>1000
                    logs=[json.loads(x) for x in (folder/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
                    assert len(logs)==(1 if method=='polynomial_ridge' else 500)
                    test=json.loads((folder/'test.json').read_text(encoding='utf-8'))
                    with np.load(folder/'predictions.npz') as arrays:
                        for r in test:assert np.isclose(rmse(arrays['pred_'+r['split']],arrays['y_'+r['split']]),r['normalized_rmse'],rtol=1e-7,atol=1e-9)
                    classic_count+=1
                    if c['candidate']!=select['candidate']:continue
                    for r in test:
                        row=dict(task=task,fold=fold,seed=0,depth=4,method=method,budget=0,training_steps=1 if method=='polynomial_ridge' else 500,domain=r['split'].split('_')[0],rmse=r['normalized_rmse'],split=r['split'],
                                 valid_rmse=c['valid_rmse'],candidate=c['candidate'],hyperparameter=c['value'])
                        for j,v in enumerate(r['raw_rmse_per_output']):row[f'raw_rmse_{j}']=v
                        classical.append(row)
    assert classic_count==84
    classics=pd.DataFrame(classical);classics.to_csv(out/'selected_classic_test.csv',index=False,encoding='utf-8-sig')
    combined=pd.concat([pooled,pool(classics)],ignore_index=True);combined.to_csv(out/'selected_pooled_test.csv',index=False,encoding='utf-8-sig')
    sums=summary(combined);sums.to_csv(out/'test_summary.csv',index=False,encoding='utf-8-sig');make_plot(sums,out/'external_baselines','External baselines: validation-selected; classic depth marker is only a plotting anchor')
    lines=['# 第二项：外部基线','',f"完成 {audit['attempts']} 次 MLP 拟合（成功 {audit['successful']}）及 {classic_count} 次经典基线拟合。",'',
      'MLP 两个隐层、SiLU，原值/固定倒数两表示，参数量分别接近复制 DD 4/16 层；每条件三学习率、三种子、2000 轮。多项式基线为显式二阶交互 + Ridge，不冒称 Pi-Nets 复现；树模型为 500 轮 HistGradientBoosting。', '',
      '所有超参数和树迭代数按验证选择。经典模型确定性运行不扩成三种子；经典模型的图中深度 4 仅用于放置，不表示模型有 DD 层。','',
      '|任务|MLP 原值 d4|MLP 倒数 d4|MLP 原值 d16|MLP 倒数 d16|多项式 Ridge|梯度提升树|','|---|---:|---:|---:|---:|---:|---:|']
    for task in TASKS:
        cells=[]
        for method,depth in [('mlp_base',4),('mlp_reciprocal',4),('mlp_base',16),('mlp_reciprocal',16),('polynomial_ridge',4),('hist_gradient_boosting',4)]:
            r=sums[(sums.task==task)&(sums.method==method)&(sums.depth==depth)&sums.domain.isin(['iid','test'])]
            cells.append('缺失/失败' if len(r)==0 else f"{r.iloc[0]['mean']:.5f}")
        lines.append('|'+task+'|'+'|'.join(cells)+'|')
    lines+=['','表为归一化测试 RMSE，先合并合成测试批，再在真实任务折内平均种子、最后等权平均折。真实物理单位逐运行指标另存 CSV。','',
      '此阶段不依据 MLP 的三学习率搜索结果与旧单学习率 DD 作公平优劣结论；同三学习率、同 2000 轮的正式比较将在第三项生成。MLP 与 DD 是参数近似匹配，不是 FLOPs 或训练耗时完全相同。']
    (stage/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    audit['classic_attempts']=classic_count;audit['validation_selection_checked']=True;js(stage/'VERIFIED.json',audit)

def stage3():
    stage=ROOT/STAGES[2];out=stage/'analysis';out.mkdir(exist_ok=True)
    audit=audit_neural(stage,168,1008,8000,[2000,4000,8000])
    all_trials=collect(stage,[2000,4000,8000],False);all_trials.to_csv(out/'all_trials.csv',index=False,encoding='utf-8-sig')
    fixed_pairs,fixed_summary=fixed_lr_budget_table(all_trials)
    fixed_pairs.to_csv(out/'fixed_lr_budget_pairs.csv',index=False,encoding='utf-8-sig')
    fixed_summary.to_csv(out/'fixed_lr_budget_summary.csv',index=False,encoding='utf-8-sig')
    selected=collect(stage,[2000,4000,8000]);selected.to_csv(out/'validation_selected_test.csv',index=False,encoding='utf-8-sig')
    pooled=pool(selected);pooled.to_csv(out/'selected_pooled_test.csv',index=False,encoding='utf-8-sig')
    sums=summary(pooled);sums.to_csv(out/'budget_summary.csv',index=False,encoding='utf-8-sig')
    piv=pooled.pivot(index=['task','fold','seed','depth','budget','domain'],columns='method',values='rmse').reset_index()
    piv['gain_percent']=100*(1-piv.dd_reciprocal/piv.dd_duplicate);piv.to_csv(out/'paired_budget_gains.csv',index=False,encoding='utf-8-sig')
    baseline=pd.read_csv(ROOT/STAGES[1]/'analysis/selected_pooled_test.csv')
    fair=pd.concat([baseline,pooled[(pooled.budget==2000)&pooled.depth.isin([4,16])]],ignore_index=True)
    fair.to_csv(out/'fair_2000_epoch_comparison.csv',index=False,encoding='utf-8-sig');fair_sums=summary(fair);fair_sums.to_csv(out/'fair_comparison_summary.csv',index=False,encoding='utf-8-sig')
    for budget in [2000,4000,8000]:make_plot(sums[sums.budget==budget],out/f'absolute_error_budget_{budget}',f'DD fair learning-rate search: {budget} epochs')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for task,ax in zip(TASKS,axes.flat):
        a=piv[(piv.task==task)&piv.domain.isin(['test','iid'])]
        for budget,z in a.groupby('budget'):
            z=z.groupby(['fold','depth']).gain_percent.mean().groupby('depth').mean()
            ax.plot(z.index,z.values,'o-',label=str(budget))
        ax.set_title(task);ax.set_xscale('log',base=2);ax.set_xticks([1,4,8,16],[1,4,8,16]);ax.axhline(0,c='gray',lw=.6);ax.set_ylabel('Reciprocal gain vs duplicate (%)');ax.grid(alpha=.2)
    axes[0,0].legend(title='Epoch budget');fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'budget_gains.{ext}',dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for task,ax in zip(TASKS,axes.flat):
        part=fixed_summary[(fixed_summary.task==task)&(fixed_summary.depth==16)&fixed_summary.domain.isin(['iid','test'])]
        for lr,z in part.groupby('lr'):
            z=z.sort_values('budget');ax.plot(z.budget,z.gain_percent,'o-',label=f'lr={lr:g}')
        ax.set_title(task);ax.set_xlabel('Continuous epoch budget');ax.set_ylabel('Gain vs duplicate (%)');ax.axhline(0,color='gray',lw=.6);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('Depth 16: fixed-learning-rate trajectories (available pairs; failures separately counted)',fontsize=10);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'fixed_lr_budget_depth16.{ext}',dpi=170)
    plt.close(fig)
    # Report how often the selected best remains near the end of each window.
    checkpoint_rows=selected.drop_duplicates(['source','candidate','budget'])
    tail=pd.DataFrame([dict(budget=int(b),selected_models=len(g),best_in_last_5pct=int((g.best_epoch>=.95*b).sum()))
                       for b,g in checkpoint_rows.groupby('budget')])
    tail.to_csv(out/'optimization_tail.csv',index=False,encoding='utf-8-sig')
    lines=['# 第三项：延长训练与公平学习率搜索','',f"完成 {audit['attempts']} 次连续 8000 轮 DD 拟合（成功 {audit['successful']}）。保存 2000/4000/8000 轮预算检查点，所有预算状态的 Adam 步数与最佳权重均独立核验。",'',
      '每个任务/折、深度、表示和种子均给予 2e-5、2e-4、2e-3 三个候选；按各预算窗口内验证 RMSE 选候选，不看测试。不同预算来自连续轨迹，不是独立实验。','',
      '|任务|16 层 2000 轮增益|16 层 4000 轮增益|16 层 8000 轮增益|','|---|---:|---:|---:|']
    for task in TASKS:
        vals=[]
        for budget in [2000,4000,8000]:
            a=piv[(piv.task==task)&(piv.depth==16)&(piv.budget==budget)&piv.domain.isin(['iid','test'])]
            v=a.groupby('fold').gain_percent.mean().mean();vals.append(f'{v:.2f}%')
        lines.append('|'+task+'|'+'|'.join(vals)+'|')
    lines+=['','完整同预算 DD/MLP/经典基线比较见 `analysis/fair_2000_epoch_comparison.csv`；其选择空间均事先固定。经典模型的迭代预算不同，因此不能称所有算法同 FLOPs。','',
      '同一学习率的预算曲线另见 `fixed_lr_budget_depth16.pdf`，配对明细与可用对数见 `fixed_lr_budget_pairs.csv` 和 `fixed_lr_budget_summary.csv`。它们不重新挑选学习率，可与验证选择后的预算曲线区分阅读。任何失败造成的缺失都不计为零增益。', '',
      '延长训练不能自动证明已收敛；`optimization_tail.csv` 报告选中检查点仍靠近预算尾部的频率。必须同时看绝对测试误差、验证曲线和 OOD。候选数值失败计入审计并排除该预算窗口的选择，不删除失败记录。']
    (stage/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');js(stage/'VERIFIED.json',audit)

def stage4():
    stage=ROOT/STAGES[3];out=stage/'analysis';out.mkdir(exist_ok=True)
    audit=audit_neural(stage,126,1008,4000)
    frame=collect(stage);frame.to_csv(out/'per_run_test.csv',index=False,encoding='utf-8-sig');pooled=pool(frame);pooled.to_csv(out/'pooled_test.csv',index=False,encoding='utf-8-sig')
    piv=pooled.pivot(index=['task','fold','seed','depth','budget','domain'],columns='method',values='rmse').reset_index()
    for slot in [0,1]:
        for method in ['reciprocal','matched']:
            piv[f'gain_c{slot}_{method}']=100*(1-piv[f'c{slot}_{method}']/piv[f'c{slot}_duplicate'])
    for method in ['subset','complement']:piv[f'gain_c1_{method}']=100*(1-piv[f'c1_{method}']/piv.c1_duplicate)
    piv['constant_interaction_native']=piv.gain_c1_reciprocal-piv.gain_c0_reciprocal
    piv['constant_interaction_matched']=piv.gain_c1_matched-piv.gain_c0_matched
    piv['moment_matching_change_c0']=piv.gain_c0_matched-piv.gain_c0_reciprocal
    piv['moment_matching_change_c1']=piv.gain_c1_matched-piv.gain_c1_reciprocal
    piv.to_csv(out/'paired_mechanism_effects.csv',index=False,encoding='utf-8-sig')
    from stage4 import configs
    from engine import inputs,build,parameter_count
    diagnostics=[]
    for task,fold in datasets():
        d=load(data_path(task,fold));cs=configs(4)
        a=inputs(d,dict(input='duplicate'),'train')[:,d['train']['x'].shape[1]:]
        b=inputs(d,dict(input='matched'),'train')[:,d['train']['x'].shape[1]:]
        delta_mean=float((a.mean(0)-b.mean(0)).abs().max());delta_std=float((a.std(0,correction=0)-b.std(0,correction=0)).abs().max())
        assert delta_mean<1e-4 and delta_std<1e-4
        models=[build(d,c,SEEDS[0]) for c in cs]
        assert len({parameter_count(m) for m in models})==1
        assert all(all(torch.equal(v,models[0].state_dict()[k]) for k,v in m.state_dict().items()) for m in models[1:])
        diagnostics.append(dict(task=task,fold=fold,mean_match_max_abs=delta_mean,std_match_max_abs=delta_std,parameters=parameter_count(models[0]),init_identical=True,subset=FIELDS[task]))
    js(out/'mechanism_control_verification.json',diagnostics)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for label,columns in [('constant_and_moments',['gain_c0_reciprocal','gain_c1_reciprocal','gain_c0_matched','gain_c1_matched']),('field_subsets',['gain_c1_matched','gain_c1_subset','gain_c1_complement'])]:
        fig,axes=plt.subplots(2,3,figsize=(14,7))
        for task,ax in zip(TASKS,axes.flat):
            a=piv[(piv.task==task)&piv.domain.isin(['iid','test'])]
            for key in columns:
                z=a.groupby(['fold','depth'])[key].mean().groupby('depth').mean();ax.plot(z.index,z.values,'o-',label=key,lw=1)
            ax.set_title(task);ax.set_xscale('log',base=2);ax.set_xticks([1,4,16],[1,4,16]);ax.axhline(0,color='gray',lw=.6);ax.grid(alpha=.2);ax.set_ylabel('RMSE reduction vs matched duplicate (%)')
        axes[0,0].legend(fontsize=6);fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(out/f'{label}.{ext}',dpi=170)
        plt.close(fig)
    lines=['# 第四项：常数坐标、二阶矩与字段机制对照','',f"完成 {audit['attempts']} 次拟合（成功 {audit['successful']}），每次 4000 轮。八条件保持名义参数、初始化和输出偏置完全一致。",'',
      '常数槽 0/1 控制是否可通过该槽形成低阶项；零槽会使部分路径失效，控制的是名义参数而非有效自由度。原生倒数/矩匹配倒数只改变训练均值和标准差，不消除尾部、符号结构或全部优化差异。字段子集及补集在试验前固定，未使用测试挑选字段。','',
      '|任务|16 层原生倒数 c0|16 层矩匹配 c0|16 层原生倒数 c1|16 层矩匹配 c1|','|---|---:|---:|---:|---:|']
    for task in TASKS:
        a=piv[(piv.task==task)&(piv.depth==16)&piv.domain.isin(['iid','test'])]
        keys=['gain_c0_reciprocal','gain_c0_matched','gain_c1_reciprocal','gain_c1_matched'];values=a.groupby('fold')[keys].mean().mean()
        lines.append('|'+task+'|'+'|'.join(f'{values[k]:.2f}%' for k in keys)+'|')
    lines+=['','字段子集/补集完整曲线、逐种子配对变化、各折矩匹配残差与初始化核验均在 `analysis/`。','',
      '这些对照检验常数槽、前两阶边际矩与字段选择是否改变增益，能够缩小机制解释范围；不能单凭它们证明负幂表达是唯一原因。固定学习率机制组仍可能有优化差异，需结合第三项一起解释。']
    (stage/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');audit['mechanism_controls_passed']=True;js(stage/'VERIFIED.json',audit)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',type=int,choices=[2,3,4]);a=p.parse_args();configure()
    {2:stage2,3:stage3,4:stage4}[a.stage]()
