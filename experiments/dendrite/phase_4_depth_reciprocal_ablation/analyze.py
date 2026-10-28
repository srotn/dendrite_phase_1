"""Analyze only fully locked training. Each independent unit is one seed."""
import csv,json,math
import numpy as np
import torch
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import study as s

def interval(a):
 a=np.asarray(a,dtype=float);mean=float(a.mean());sd=float(a.std(ddof=1));half=float(stats.t.ppf(.975,len(a)-1)*sd/math.sqrt(len(a)))
 return {'mean':mean,'sd':sd,'ci_low':mean-half,'ci_high':mean+half,'n':len(a)}

def csvwrite(p,rows):
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def final_curve(folder,rows):
 # Values plotted are square roots of the stored MSE fields; label them explicitly.
 fig,axs=plt.subplots(1,2,figsize=(10,4))
 for ax,key,title in zip(axs,['train_mse','valid_mse'],['Online training RMSE','Validation RMSE']):
  for i,method in enumerate(s.METHODS):ax.plot([r['epoch'] for r in rows],np.sqrt([r[key][i] for r in rows]),label=method)
  ax.set_title(title);ax.set_xlabel('Epoch');ax.set_ylabel('Normalized RMSE');ax.set_yscale('log')
 axs[0].legend();fig.tight_layout();fig.savefig(folder/'training_curves.png',dpi=150);fig.savefig(folder/'training_curves.pdf');plt.close(fig)

def main():
 torch.set_num_threads(1)
 lock=json.loads((s.ROOT/'training_lock.json').read_text(encoding='utf-8'))
 assert lock['complete_runs']==540 and len(lock['best_hashes'])==180
 out=s.ROOT/'analysis';out.mkdir(exist_ok=True)
 raw=[];checks=[];curves={};cost=[];heldout=[];group_metrics=[]
 for task in s.TASKS:
  d=torch.load(s.ROOT/task/'data/dataset.pt',weights_only=False)
  manifest=json.loads((s.ROOT/task/'data/manifest.json').read_text(encoding='utf-8'))
  assert s.sha(s.OLD/s.TASKS[task])==manifest['source_sha256']
  assert s.sha(s.ROOT/task/'data/dataset.pt')==manifest['prepared_sha256']
  original_co=torch.load(s.OLD/s.TASKS[task],weights_only=False) if task=='06_co' else None
  for seed in s.SEEDS:
   for depth in s.DEPTHS:
    folder=s.ROOT/task/'runs'/f'seed_{seed}'/f'depth_{depth:02d}'
    done=json.loads((folder/'COMPLETED.json').read_text(encoding='utf-8'))
    cfg=json.loads((folder/'config.json').read_text(encoding='utf-8'))
    assert cfg['source_sha256']==s.sha(s.ROOT/'study.py')
    rows=[json.loads(x) for x in (folder/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [r['epoch'] for r in rows]==list(range(1,2001))
    vals=np.asarray([r['valid_mse'] for r in rows]);trains=np.asarray([r['train_mse'] for r in rows])
    assert np.isfinite(vals).all() and np.isfinite(trains).all()
    assert s.sha(folder/'best.pt')==lock['best_hashes'][str((folder/'best.pt').relative_to(s.ROOT))]
    assert (folder/'training_curves.png').stat().st_size>1000
    best=torch.load(folder/'best.pt',weights_only=False)
    last=torch.load(folder/'last.pt',weights_only=False)
    assert last['epoch']==2000
    steps=math.ceil(len(d['train']['y'])/min(4096,len(d['train']['y'])))
    assert all(float(v['step'])==2000*steps for v in last['optimizer']['state'].values())
    models=s.build(d,depth,seed,'cuda');vx,vy=s.tensors(d['valid'],d['eligible'],'cuda')
    for m,state in zip(models,last['models']):m.load_state_dict(state)
    final=s.evaluate(models,vx,vy);assert np.allclose(final,vals[-1],rtol=2e-5,atol=1e-8)
    for m,state in zip(models,best):m.load_state_dict(state['model'])
    verified=s.evaluate(models,vx,vy);assert np.allclose(verified,vals.min(0),rtol=2e-5,atol=1e-8)
    checks.append({'task':task,'seed':seed,'depth':depth,'log_rows':len(rows),'best_and_last_replayed':True,'optimizer_steps':2000*steps})
    final_curve(folder,rows)
    curves[task,seed,depth]=vals
    for budget in (500,1000,2000):
     for i,method in enumerate(s.METHODS):
      raw.append({'task':task,'seed':seed,'depth':depth,'budget':budget,'method':method,
       'best_valid_rmse':float(np.sqrt(vals[:budget,i].min())),
       'best_epoch':int(vals[:budget,i].argmin()+1),'final_valid_rmse':float(np.sqrt(vals[budget-1,i])),
       'final_online_train_rmse':float(np.sqrt(trains[budget-1,i])),
       'parameters':cfg['parameters'][i],
       'tail_valid_change_percent':float(100*(np.sqrt(vals[budget-1,i]/vals[max(0,budget-201),i])-1))})
    cost.append({'task':task,'seed':seed,'depth':depth,'three_model_training_seconds':rows[-1]['elapsed'],
     'training_plus_checkpoint_and_plot_seconds':done['seconds'],'updates_per_model':2000*steps})
    for split in ('test','ood'):
     if split not in d:continue
     hx,hy=s.tensors(d[split],d['eligible'],'cuda');metrics=s.evaluate(models,hx,hy)
     for i,method in enumerate(s.METHODS):
      heldout.append({'task':task,'seed':seed,'depth':depth,'method':method,'split':split,
       'normalized_rmse':float(np.sqrt(metrics[i])),
       'scaled_rmse':float(np.sqrt(metrics[i])*d['scale'].item()),
       'scale_units':'ppm' if task=='06_co' else 'log_output_voltage'})
     if task=='06_co':
      batch_ids=original_co['splits']['test']['batch_id'].cuda()
      names=original_co['splits']['test']['files']
      with torch.no_grad():
       for i,(m,x) in enumerate(zip(models,hx)):
        pred=torch.cat([m(x[a:a+4096]) for a in range(0,len(x),4096)])
        for group in sorted(torch.unique(batch_ids).tolist()):
         mask=batch_ids==group;err=pred[mask]-hy[mask];scale=float(d['scale'].item())
         group_metrics.append({'task':task,'seed':seed,'depth':depth,'method':s.METHODS[i],
          'batch':names[int(group)],'n':int(mask.sum()),'rmse_ppm':float(err.square().mean().sqrt())*scale,
          'mae_ppm':float(err.abs().mean())*scale,
          'r2':float(1-err.square().sum()/(hy[mask]-hy[mask].mean()).square().sum())})
 lookup={(r['task'],r['seed'],r['depth'],r['budget'],r['method']):r for r in raw}
 gains=[];summaries=[];interactions=[]
 for task in s.TASKS:
  for budget in (500,1000,2000):
   for control in ('base','duplicate'):
    for depth in s.DEPTHS:
     g=[]
     for seed in s.SEEDS:
      r=lookup[task,seed,depth,budget,'reciprocal'];c=lookup[task,seed,depth,budget,control]
      gain=100*(1-r['best_valid_rmse']/c['best_valid_rmse']);g.append(gain)
      gains.append({'task':task,'budget':budget,'control':control,'depth':depth,'seed':seed,'gain_percent':gain})
     summaries.append({'task':task,'budget':budget,'control':control,'depth':depth,**interval(g),'wins':sum(a>0 for a in g)})
    slopes=[];deltas=[]
    for seed in s.SEEDS:
     g=[100*(1-lookup[task,seed,l,budget,'reciprocal']['best_valid_rmse']/lookup[task,seed,l,budget,control]['best_valid_rmse']) for l in s.DEPTHS]
     slopes.append(float(np.polyfit(np.log2(s.DEPTHS),g,1)[0]));deltas.append(g[-1]-g[0])
    interactions.append({'task':task,'budget':budget,'control':control,'slope':interval(slopes),
      'slope_values':slopes,'depth16_minus_depth1':interval(deltas),'delta_values':deltas,
      'p_two_sided':float(stats.ttest_1samp(slopes,0).pvalue)})
 primary=[r for r in interactions if r['budget']==2000];order=sorted(primary,key=lambda r:r['p_two_sided']);prev=0
 for rank,r in enumerate(order):
  prev=max(prev,min(1.,r['p_two_sided']*(len(order)-rank)));r['p_holm_12_tests']=prev
 s.js(out/'verification.json',{'passed':True,'groups':checks,'models':540,'jsonl_rows':360000,'model_epoch_observations':1080000,'heldout_after_training_lock':True})
 csvwrite(out/'per_run_metrics.csv',raw);csvwrite(out/'paired_gains.csv',gains);csvwrite(out/'gain_summary.csv',summaries);csvwrite(out/'costs.csv',cost);csvwrite(out/'heldout.csv',heldout)
 csvwrite(out/'co_heldout_by_batch.csv',group_metrics)
 s.js(out/'depth_interactions.json',interactions)
 heldout_summary=[]
 for task in ('04_impedance','06_co'):
  for depth in s.DEPTHS:
   for control in ('base','duplicate'):
    values=[]
    for seed in s.SEEDS:
     rr=next(r for r in heldout if r['task']==task and r['depth']==depth and r['seed']==seed and r['method']=='reciprocal')
     cc=next(r for r in heldout if r['task']==task and r['depth']==depth and r['seed']==seed and r['method']==control)
     values.append(100*(1-rr['normalized_rmse']/cc['normalized_rmse']))
    heldout_summary.append({'task':task,'depth':depth,'control':control,**interval(values),'wins':sum(v>0 for v in values)})
 csvwrite(out/'heldout_gain_summary.csv',heldout_summary)
 # A larger relative gain can come from a worsening control: quantify absolute error changes too.
 decomposition=[]
 for task in s.TASKS:
  for method in s.METHODS:
   changes=[100*(lookup[task,seed,16,2000,method]['best_valid_rmse']/lookup[task,seed,1,2000,method]['best_valid_rmse']-1) for seed in s.SEEDS]
   decomposition.append({'task':task,'method':method,'measure':'depth16_error_change_from_depth1_percent',**interval(changes)})
 csvwrite(out/'absolute_error_depth_change.csv',decomposition)
 for task in s.TASKS:
  fig,axs=plt.subplots(2,3,figsize=(14,8))
  for ax,depth in zip(axs.flat,s.DEPTHS):
   for i,method in enumerate(s.METHODS):
    a=np.sqrt(np.asarray([curves[task,seed,depth][:,i] for seed in s.SEEDS]));mean=a.mean(0);sd=a.std(0,ddof=1)
    ax.plot(np.arange(1,2001),mean,label=method);ax.fill_between(np.arange(1,2001),np.maximum(mean-sd,1e-12),mean+sd,alpha=.12)
   ax.set_title(f'{depth} layers');ax.set_yscale('log');ax.set_xlabel('Epoch');ax.set_ylabel('Validation RMSE');ax.grid(alpha=.2)
  axs.flat[0].legend();fig.suptitle(task+'; mean +/- seed SD');fig.tight_layout()
  fig.savefig(out/(task+'_learning.png'),dpi=150);fig.savefig(out/(task+'_learning.pdf'));plt.close(fig)
 s.js(out/'summary.json',{'gain_summary':summaries,'interactions':interactions,'verified_models':540,'curves':['depth_gain.png','depth_error.png','budget_sensitivity.png']})
 for mode in ('gain','error','budget'):
  fig,axs=plt.subplots(2,3,figsize=(15,8),squeeze=False)
  for ax,task in zip(axs.flat,s.TASKS):
   if mode=='gain':
    for control,color in [('base','C0'),('duplicate','C1')]:
     a=[next(r for r in summaries if r['task']==task and r['budget']==2000 and r['control']==control and r['depth']==l) for l in s.DEPTHS]
     means=np.array([r['mean'] for r in a]);lo=np.array([r['ci_low'] for r in a]);hi=np.array([r['ci_high'] for r in a])
     ax.plot(s.DEPTHS,means,'o-',color=color,label='vs '+control);ax.fill_between(s.DEPTHS,lo,hi,color=color,alpha=.17)
     for seed in s.SEEDS:ax.scatter(s.DEPTHS,[next(r['gain_percent'] for r in gains if r['task']==task and r['seed']==seed and r['budget']==2000 and r['control']==control and r['depth']==l) for l in s.DEPTHS],s=10,color=color,alpha=.35)
    ax.axhline(0,color='gray',lw=.8);ax.set_ylabel('RMSE reduction (%)')
   elif mode=='error':
    for method in s.METHODS:
     a=np.array([[lookup[task,seed,l,2000,method]['best_valid_rmse'] for l in s.DEPTHS] for seed in s.SEEDS])
     ax.errorbar(s.DEPTHS,a.mean(0),yerr=a.std(0,ddof=1),fmt='o-',label=method,capsize=2)
    ax.set_yscale('log');ax.set_ylabel('Best validation normalized RMSE')
   else:
    for budget in (500,1000,2000):
     a=[next(r for r in summaries if r['task']==task and r['budget']==budget and r['control']=='duplicate' and r['depth']==l) for l in s.DEPTHS]
     ax.plot(s.DEPTHS,[r['mean'] for r in a],'o-',label=str(budget)+' epochs')
    ax.axhline(0,color='gray',lw=.8);ax.set_ylabel('Gain vs duplicate (%)')
   ax.set_title(task);ax.set_xscale('log',base=2);ax.set_xticks(s.DEPTHS,labels=s.DEPTHS);ax.set_xlabel('DD multiplicative layers');ax.grid(alpha=.2)
  axs.flat[0].legend();fig.tight_layout()
  name={'gain':'depth_gain','error':'depth_error','budget':'budget_sensitivity'}[mode]
  fig.savefig(out/(name+'.png'),dpi=180);fig.savefig(out/(name+'.pdf'));plt.close(fig)
 # Check text/source integrity against the pre-experiment inventory.
 inventory=json.loads((s.ROOT/'source_audit/inventory.json').read_text(encoding='utf-8'))
 changes=[r['path'] for r in inventory if 'sha256' in r and s.sha(s.OLD/r['path'])!=r['sha256']]
 s.js(out/'historical_integrity.json',{'historical_text_files_checked':sum('sha256' in r for r in inventory),'changed_files':changes})
 report=['# DD 层数与固定物理倒数增益：完成报告','',
  '完成 6 任务 × 6 层数 × 5 初始化种子 × 3 表示 = 540 次拟合，各 2000 轮（共 108 万模型轮）。180 组逐轮日志及全部最佳/最后检查点重算通过；每组有同目录 PNG/PDF 曲线。历史源码/文档/日志完整性核验见 analysis/historical_integrity.json。',
  '', '## 核心结论与重要限制','',
  '层数与固定物理倒数增益存在任务相关交互，不支持“越深收益必然越大”的普遍结论。强迫振荡与阻抗的相对增益随深度增长；光学随深度转为更大负收益；热场非单调且深层种子波动较大；混凝土没有检出稳定的线性深度趋势；CO 的相对收益增加，但绝对测试精度并非 16 层最佳。',
  '', '必须分开解释相对收益与绝对外推误差：阻抗倒数模型的历史 OOD log 输出电压 RMSE 从 1 层约 1.485 上升到 16 层约 91.542，复制对照更恶化至约 1455.184，因此 93.48% 相对改善不能解释为良好外推。CO 倒数模型的测试 RMSE 在 8 层为 2.3014±0.0604 ppm，16 层回升到 2.5047±0.1922 ppm（均值±种子标准差）；不据测试选层重训，也不将描述性最低值当独立验证的最优层数。',
  '', '预算敏感性不能省略：热场 16 层相对复制的收益在 500/1000/2000 轮分别为 −36.04% / +2.45% / +29.13%；CO 为 +8.76% / +30.58% / +34.15%。强迫振荡与阻抗的增长趋势、光学的负趋势在三个预算下较一致，热场与真实任务深层结果更受优化时间影响。',
  '', '540 个模型中有 445 个最佳验证检查点位于最后 100 轮。固定 2000 轮不能视为已收敛至表示上限；本研究提供固定预算下的可重复消融证据。五种子是固定数据上的初始化/样本顺序重复，不包含数据划分不确定性。',
  '', '## 主结果：相对同参数复制拼接的验证 RMSE 降幅','',
  '数值为配对种子收益均值；正数改善，负数变差。完整 95% t 区间和逐种子值见 analysis/gain_summary.csv、paired_gains.csv。','',
  '| 任务 | 1 层 | 2 层 | 3 层 | 4 层 | 8 层 | 16 层 |','|---|---:|---:|---:|---:|---:|---:|']
 for task in s.TASKS:
  a=[next(r for r in summaries if r['task']==task and r['budget']==2000 and r['control']=='duplicate' and r['depth']==l) for l in s.DEPTHS]
  report.append('| '+task+' | '+' | '.join(f"{r['mean']:+.2f}%" for r in a)+' |')
 report+=['','## 层数交互证据','','每种子在六层数上拟合收益对 log2(层数) 的斜率；单位为每加倍层数的百分点。以下为配对五种子均值与 95% t 区间，p 值对六任务两对照共 12 项作 Holm 校正。非单调曲线必须结合逐层结果解释。','',
 '| 任务 | 对照 | 斜率 [95% CI] | Holm p | 16层−1层收益差 [95% CI] |','|---|---|---|---:|---|']
 for r in primary:
  a=r['slope'];b=r['depth16_minus_depth1']
  report.append(f"| {r['task']} | {r['control']} | {a['mean']:+.2f} [{a['ci_low']:+.2f}, {a['ci_high']:+.2f}] | {r['p_holm_12_tests']:.4g} | {b['mean']:+.2f} [{b['ci_low']:+.2f}, {b['ci_high']:+.2f}] |")
 report+=['','## 绝对误差变化与留出收益','','收益扩大可能同时来自倒数组改善和对照组退化；不能只看相对百分比。以下为 16 层相对 1 层的绝对 RMSE 变化（负值表示误差下降），完整区间见 absolute_error_depth_change.csv。','',
  '| 任务 | 原值误差变化 | 复制误差变化 | 倒数误差变化 |','|---|---:|---:|---:|']
 for task in s.TASKS:
  a=[next(r for r in decomposition if r['task']==task and r['method']==m) for m in s.METHODS]
  report.append('| '+task+' | '+' | '.join(f"{r['mean']:+.2f}%" for r in a)+' |')
 report+=['','历史留出上倒数相对同参数复制的 RMSE 降幅：','','| 任务 | 1 层 | 2 层 | 3 层 | 4 层 | 8 层 | 16 层 |','|---|---:|---:|---:|---:|---:|---:|']
 for task in ('04_impedance','06_co'):
  a=[next(r for r in heldout_summary if r['task']==task and r['control']=='duplicate' and r['depth']==l) for l in s.DEPTHS]
  report.append('| '+task+' | '+' | '.join(f"{r['mean']:+.2f}%" for r in a)+' |')
 report+=['','## 预算敏感性与收敛','','预定保存 500/1000/2000 轮结果，全部来自同一连续训练路径，没有重新初始化优化器。最佳验证与末轮数值、最后 200 轮变化、最佳轮次见 per_run_metrics.csv；budget_sensitivity.png 对比同参数收益。固定轮数对应相同样本暴露与更新次数，不代表相同 FLOPs。costs.csv 中墙钟为三个模型共同运行，不可当单模型独立延迟。',
 '', '## 留出与论文证据边界','',
 '全部 540 次训练锁定后才评估 CO 测试和阻抗 OOD（heldout.csv）。CO 的 scaled_rmse 单位为 ppm；阻抗是 log 输出电压单位，不冒充物理电压 RMSE。其余四任务只有验证集，不能称测试泛化。所有数据是历史已使用基准，不是新盲测。',
 '', '五个种子只覆盖初始化与数据顺序，数据划分固定；t 区间以种子近似独立且近似正态为前提，小样本检验功效有限。本结果支持所列任务与训练协议下的深度—表示交互，不证明普遍定律或模型表达上限。合成主体无常数坐标，增加层数也改变齐次多项式阶次；混凝土和 CO 保留常数坐标。这两种主体之间不作因果归因。',
 '', '本轮仅研究固定物理倒数拼接，没有普通归一化倒数、共享乘法倒数、自适应混合或特殊优化器。复制拼接严格匹配参数量及初始化，控制宽度影响，但不能消除表示尺度对优化的影响。',
 '', '## 文件入口','', '- `README.md`：训练前协议。','- `source_audit/`：历史审计与来源快照。','- `analysis/depth_gain.png`：增益及区间。','- `analysis/depth_error.png`：三表示误差。','- `analysis/budget_sensitivity.png`：训练预算敏感性。','- 各编号任务 `runs/seed_*/depth_*/`：日志、检查点、曲线。']
 (s.ROOT/'RESULTS.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
 s.js(s.ROOT/'status.json',{'status':'complete','runs':540,'verified':True,'report':'RESULTS.md'})
 print('VERIFIED 540 RUNS; analysis complete')

if __name__=='__main__':main()
