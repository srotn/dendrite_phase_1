from __future__ import annotations
import pandas as pd
from common import *

NAMES={'01_forced':'强迫振荡','02_thermal':'热场','03_optical':'光学','04_impedance':'阻抗','05_concrete':'混凝土','06_co':'CO'}

def scalar(frame,task,depth,method,budget=None):
    f=frame[(frame.task==task)&(frame.depth==depth)&(frame.method==method)&frame.domain.isin(['iid','test'])]
    if budget is not None:f=f[f.budget==budget]
    return float(f.groupby('fold').rmse.mean().mean()) if 'fold' in f else float(f.rmse.mean())

def main():
    audits=[json.loads((ROOT/s/'VERIFIED.json').read_text(encoding='utf-8')) for s in STAGES]
    assert all(a['passed'] for a in audits)
    confirmation=json.loads((ROOT/'verification/final_confirmation/VERIFIED.json').read_text(encoding='utf-8'));assert confirmation['passed']
    from audit_confirmation import main as audit_confirmation
    audit_confirmation()
    confirmation_audit=json.loads((ROOT/'verification/final_confirmation/INDEPENDENT_AUDIT.json').read_text(encoding='utf-8'));assert confirmation_audit['passed']
    historical=json.loads((ROOT/'historical_source_hashes.json').read_text(encoding='utf-8'))
    changed=[name for name,h in historical.items() if not (DENDRITE/name).exists() or sha(DENDRITE/name)!=h]
    assert not changed,changed[:10]
    counts=[audits[0]['new_attempts']]+[a['attempts'] for a in audits[1:]]
    assert counts==[540,504,1008,1008] and audits[1]['classic_attempts']==84
    successes=[audits[0]['new_successful']]+[a['successful'] for a in audits[1:]]
    out=ROOT/'analysis';out.mkdir(exist_ok=True)
    gains=pd.DataFrame(json.loads((ROOT/STAGES[0]/'analysis/gain_summary.json').read_text(encoding='utf-8')))
    original=pd.read_csv(PREVIOUS/'analysis/gain_summary.csv')
    compare=[]
    for task in TASKS:
        old=original[(original.task==task)&(original.depth==16)&(original.budget==2000)&(original.control=='duplicate')].iloc[0]['mean']
        new=gains[(gains.task==task)&(gains.depth==16)&(gains.control=='duplicate')].iloc[0]['mean']
        compare.append(dict(task=task,historical_validation_gain=float(old),new_test_gain=float(new),difference_pp=float(new-old)))
    pd.DataFrame(compare).to_csv(out/'historical_vs_new_evaluation.csv',index=False,encoding='utf-8-sig')
    fair=pd.read_csv(ROOT/STAGES[2]/'analysis/fair_2000_epoch_comparison.csv')
    fair_rows=[]
    for task in TASKS:
        for method in ['dd_duplicate','dd_reciprocal','mlp_base','mlp_reciprocal','polynomial_ridge','hist_gradient_boosting']:
            for depth in ([4] if method in ['polynomial_ridge','hist_gradient_boosting'] else [4,16]):
                value=scalar(fair,task,depth,method);fair_rows.append(dict(task=task,method=method,depth=depth,normalized_test_rmse=value))
    pd.DataFrame(fair_rows).to_csv(out/'fair_baseline_table.csv',index=False,encoding='utf-8-sig')
    budget=pd.read_csv(ROOT/STAGES[2]/'analysis/paired_budget_gains.csv')
    mechanisms=pd.read_csv(ROOT/STAGES[3]/'analysis/paired_mechanism_effects.csv')
    final=pd.read_csv(ROOT/'verification/final_confirmation/selected_metrics.csv')
    # Final confirmation is exclusively synthetic and is not reused for selection.
    final_dd=final[(final.stage==3)&(final.budget==8000)]
    pivot=final_dd.pivot(index=['task','seed','depth','budget','domain'],columns='method',values='rmse').reset_index()
    pivot['gain_percent']=100*(1-pivot.dd_reciprocal/pivot.dd_duplicate)
    pivot.to_csv(out/'final_confirmation_gains.csv',index=False,encoding='utf-8-sig')
    ood=final_dd[(final_dd.task=='04_impedance')&(final_dd.domain=='ood')]
    ood_absolute=ood.groupby(['depth','method']).raw_rmse_0.agg(['mean','std','min','max']).reset_index()
    ood_absolute.to_csv(out/'final_impedance_ood_absolute_log_voltage.csv',index=False,encoding='utf-8-sig')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    for method,g in ood_absolute.groupby('method'):
        g=g.sort_values('depth');ax.plot(g.depth,g['mean'],'o-',label=method);ax.fill_between(g.depth,g['min'],g['max'],alpha=.12)
    ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xticks([1,4,8,16],[1,4,8,16]);ax.set_xlabel('DD multiplicative depth')
    ax.set_ylabel('Absolute RMSE in log output voltage');ax.set_title('Final independent impedance OOD test\n8000 epochs; LR chosen on validation; band = seed range',fontsize=10)
    ax.legend();ax.grid(alpha=.2);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'final_impedance_ood_absolute.{ext}',dpi=170)
    plt.close(fig)
    lines=['# 论文证据补强：四项完成报告','',f'完成时间：{now()}。范围仅为 dendrite。实验方案在新测试生成前冻结；最终合成确认集在全部四项模型锁定后生成。','',
      '## 完成量与文件入口','', '|顺序|补强项|新增拟合|成功神经模型|报告|','|---|---|---:|---:|---|']
    titles=['新测试与分组评估','外部基线','连续延长训练与公平调参','常数/矩匹配/字段机制']
    for i,(s,title,n,ok) in enumerate(zip(STAGES,titles,counts,successes),1):
        lines.append(f'|{i}|{title}|{n}'+(' + 84 经典模型' if i==2 else '')+f'|{ok}|[{s}]({s}/RESULTS.md)|')
    lines+=['',f'合计新增 **{sum(counts)} 次神经拟合 + 84 次经典拟合**；另复评既有 360 个合成模型。神经模型失败数为 {sum(counts)-sum(successes)}，失败逐项保留在阶段审计中，不冒充成功模型。',
       '', '第一项新合成测试共 30 万行（24 万 IID + 6 万阻抗 OOD）；最后再生成 10 万行确认数据（8 万 IID + 2 万 OOD）。真实任务没有新增外部采集，完成的是五折分组评估。样本行数不能当作独立模型重复数。',
       '', '## 1. 历史验证结论能否在新评估中复现','', '|任务|历史 16 层验证增益|新 16 层测试增益|差值/百分点|','|---|---:|---:|---:|']
    for r in compare:lines.append(f"|{NAMES[r['task']]}|{r['historical_validation_gain']:.2f}%|{r['new_test_gain']:.2f}%|{r['difference_pp']:+.2f}|")
    lines+=['','增益均相对同参数复制。合成模型不重训，在全新样本上复评；真实模型则使用新分组训练/验证/测试折。真实任务还改变了训练样本量和初始化重复，不能把上述差异唯一归因于分组泄漏或某单一因素。',
       '', '混凝土按相同配方（不含龄期）形成 428 组，每折 618/206/206 行；CO 按整采集文件划分，测试批次互斥，所有 13 批次均恰好作为测试一次。各折的预处理参数仅由该折训练部分估计。',
       '', '## 2. 与外部方法的同搜索预算比较','',
       '下表固定比较与 DD 4 层参数规模相近的 MLP；全部神经模型均 2000 轮、三个学习率、三个优化种子。各模型仅用验证选择学习率/检查点。经典方法各三个固定候选，训练算法与计算量不同，不声称相同 FLOPs。完整 4/16 层表见 `analysis/fair_baseline_table.csv`。','',
       '|任务|DD 复制|DD 倒数|MLP 原值|MLP 倒数|二阶多项式|梯度提升树|','|---|---:|---:|---:|---:|---:|---:|']
    for task in TASKS:
        vals=[scalar(fair,task,4,m) for m in ['dd_duplicate','dd_reciprocal','mlp_base','mlp_reciprocal','polynomial_ridge','hist_gradient_boosting']]
        lines.append('|'+NAMES[task]+'|'+'|'.join(f'{x:.5f}' for x in vals)+'|')
    lines+=['','表为归一化测试 RMSE。各真实折内先平均种子，再等权平均折；原始单位及各折分数保留于阶段 CSV。这里不依据测试排名选出“最佳深度”重新报告。外部算法超过 DD 的结果也完整保留，本文定位仍是表示—深度的实证研究，而非自动转为最优算法声明。',
       '', '## 3. 延长训练和学习率搜索是否改变结论','', '|任务|16 层 2000 轮|16 层 4000 轮|16 层 8000 轮|','|---|---:|---:|---:|']
    for task in TASKS:
        vals=[]
        for b in [2000,4000,8000]:
            z=budget[(budget.task==task)&(budget.depth==16)&(budget.budget==b)&budget.domain.isin(['iid','test'])]
            vals.append(float(z.groupby('fold').gain_percent.mean().mean()))
        lines.append('|'+NAMES[task]+'|'+'|'.join(f'{x:.2f}%' for x in vals)+'|')
    lines+=['','连续训练保存 Adam 状态，不在 2000/4000 轮重置优化器。每个预算窗口独立按验证选择三个候选中的最佳；所以上表同时体现预算与验证选择的结果。固定学习率逐轨迹数据同样保留，可用于分离这两部分。',
       '', '接近预算尾部的最优轮次比例见第三项 `optimization_tail.csv`。延长到 8000 轮仍不是数学意义的收敛证明；报告包含实际失败和优化残余，不把有限预算性能直接等同表达能力上限。',
       '', '## 4. 机制对照揭示哪些依赖','', '|任务|c0 原生倒数|c0 矩匹配|c1 原生倒数|c1 矩匹配|c1 字段子集|c1 字段补集|','|---|---:|---:|---:|---:|---:|---:|']
    keys=['gain_c0_reciprocal','gain_c0_matched','gain_c1_reciprocal','gain_c1_matched','gain_c1_subset','gain_c1_complement']
    for task in TASKS:
        z=mechanisms[(mechanisms.task==task)&(mechanisms.depth==16)&mechanisms.domain.isin(['iid','test'])]
        values=z.groupby('fold')[keys].mean().mean()
        lines.append('|'+NAMES[task]+'|'+'|'.join(f'{values[k]:.2f}%' for k in keys)+'|')
    lines+=['','表为 16 层、4000 轮相对对应复制条件的增益；完整 1/4/16 层结果在第四项。c0/c1 仅改变同一预留常数槽的值，模型名义参数和初始张量匹配。矩匹配仅对齐训练集前两阶边际矩；字段子集不基于测试筛选。',
       '', '这些结果可支持“增益对常数坐标、前两阶尺度或某类字段敏感/不敏感”的条件性判断。它们不能单独证明负幂表达是唯一原因，不能消除边际尾形、有效自由度与优化之间的全部耦合。',
       '', '## 最终未用于选择的合成确认','', '|任务|1 层|4 层|8 层|16 层|','|---|---:|---:|---:|---:|']
    for task in SYNTHETIC:
        vals=[]
        for depth in [1,4,8,16]:
            z=pivot[(pivot.task==task)&(pivot.depth==depth)&(pivot.domain=='iid')];vals.append(float(z.gain_percent.mean()))
        lines.append('|'+NAMES[task]+'|'+'|'.join(f'{x:.2f}%' for x in vals)+'|')
    lines+=['','上表为第三项 8000 轮验证选择模型的独立最终确认增益。全部模型在生成此数据前锁定；所有阶段的确认指标都已保存，不只保留上表。',
       '', '阻抗最终 OOD 的绝对误差如下，单位是 log 输出电压，不是伏特；±为三个优化种子的样本标准差。', '',
       '|层数|复制绝对 RMSE|倒数绝对 RMSE|','|---|---:|---:|']
    for depth in [1,4,8,16]:
        cells=[]
        for method in ['dd_duplicate','dd_reciprocal']:
            r=ood_absolute[(ood_absolute.depth==depth)&(ood_absolute.method==method)]
            cells.append('缺失/失败' if not len(r) else f"{r.iloc[0]['mean']:.4f} ± {r.iloc[0]['std']:.4f}")
        lines.append(f'|{depth}|'+ '|'.join(cells)+'|')
    lines+=['','对应对数纵轴图为 `analysis/final_impedance_ood_absolute.pdf`；须同时结合绝对值与相对降幅，不把基线恶化造成的高相对改善解释为可靠外推。',
       '', '## 论文应如何更新','',
       '1. 主协议增加真实数据分组五折，将历史固定划分结果作为前期探索；新增任务规模与训练占比差异如实列出。',
       '2. 主结果保留同参数 DD 表示对照，并新增外部方法表；公平神经比较使用同三候选、同 2000 轮数据。',
       '3. 将“训练预算可能影响结果”升级为 2000/4000/8000 连续路径的直接证据，同时保留最优轮次靠近尾部的限制。',
       '4. 方法与讨论加入统一常数槽、矩匹配和预设字段干预，不把部分机制控制写成完整因果识别。',
       '5. 真实折之间训练集重叠，均值/范围和组级误差优先；不能把五折×三种子当 15 次独立数据实验。新增证据不自动保证期刊录用或提高分区。',
       '', '## 复现与核验','',
       '冻结协议与哈希：`PROTOCOL_LOCK.json`。训练引擎图执行/恢复核验：`engine_verification.json`。MLP 参数匹配及机制控制核验：`followup_design_verification.json`。逐阶段独立磁盘检查点重放：各 `analysis/disk_replay_audit.json`。最终模型锁与确认数据：`verification/final_confirmation/`。每个神经运行的 PNG/PDF 曲线与逐轮日志同级。',
       '', f'历史源码/文档/JSON 共 {len(historical)} 个哈希检查全部保持一致；未覆盖既有实验文件。']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (ROOT/'MANUSCRIPT_ADDENDUM.md').write_text('# 原论文大纲的证据增补\n\n本文件补充 phase_5 的写作材料，不覆盖历史大纲。\n\n'+ '\n'.join(lines[lines.index('## 论文应如何更新'):])+'\n',encoding='utf-8')
    artifacts={}
    volatile={'COMPLETION_AUDIT.json','status.json','pipeline_state.json','completion_progress.json','PIPELINE_FAILED.json'}
    for p in ROOT.rglob('*'):
        if p.is_file() and p.name not in volatile and p.suffix in ['.md','.csv','.json','.png','.pdf'] and ('analysis' in p.parts or p.parent==ROOT):artifacts[str(p.relative_to(ROOT))]=sha(p)
    js(ROOT/'COMPLETION_AUDIT.json',dict(date=now(),passed=True,requirements={
        '1_new_synthetic_tests_and_real_grouped_folds':audits[0]['passed'],
        '2_parameter_matched_mlp_tree_polynomial_baselines':audits[1]['passed'],
        '3_long_budget_equal_learning_rate_search':audits[2]['passed'],
        '4_constant_moments_field_mechanisms':audits[3]['passed'],
        'final_unselected_synthetic_confirmation':confirmation['passed'],
        'final_confirmation_independent_regeneration_and_prediction_replay':confirmation_audit['passed'],
        'historical_files_unchanged':not changed},neural_attempts=sum(counts),neural_successes=sum(successes),classic_fits=84,
        verified_historical_files=len(historical),failure_count=sum(counts)-sum(successes),artifacts=artifacts))
    js(ROOT/'status.json',dict(pid=os.getpid(),status='complete_verified',updated=now(),report='RESULTS.md',audit='COMPLETION_AUDIT.json'))
    js(ROOT/'completion_progress.json',dict(status='complete_verified',collected_at=now(),neural_attempts=sum(counts),neural_successes=sum(successes),classic_fits=84,
        requirements=[dict(stage=s,required_neural_fits=n,completed_neural_fits=n,stage_verified=True) for s,n in zip(STAGES,counts)],
        final_independent_confirmation_verified=True,historical_integrity_verified=True,final_report='RESULTS.md'))
    for relative,digest in artifacts.items():assert sha(ROOT/relative)==digest,relative
    print('ALL FOUR REQUIREMENTS VERIFIED; REPORT WRITTEN',flush=True)

if __name__=='__main__':main()
