"""Render logged MSE as correctly labelled RMSE figures; training files are read-only."""
from common import *
import argparse, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def render(out):
    out=Path(out);meta=json.loads((out/'config.json').read_text(encoding='utf-8'))
    marker=out/'CURVES.json'
    if marker.exists():
        record=json.loads(marker.read_text(encoding='utf-8'))
        if record['source_log_sha256']==sha(out/'metrics.jsonl') and record['renderer_sha256']==sha(__file__) and all((out/f).is_file() for f in record['files']):return
    rows=[json.loads(line) for line in (out/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for i,c in enumerate(meta['configs']):
        label=c['name'].replace('_lr',' | lr=')
        for ax,key,title in zip(axes,['train_mse','valid_mse'],['Training (online minibatch aggregate)','Validation']):
            values=[np.nan if r[key][i] is None else math.sqrt(r[key][i]) for r in rows]
            ax.plot([r['epoch'] for r in rows],values,label=label,lw=.9)
            ax.set_title(title,fontsize=10);ax.set_xlabel('Epoch');ax.set_ylabel('Normalized RMSE');ax.set_yscale('log');ax.grid(alpha=.2)
    axes[0].legend(fontsize=6)
    depth=meta['configs'][0].get('depth',meta['configs'][0].get('target_dd_depth'))
    fig.suptitle(f"{meta['task']} | {meta['fold']} | seed {meta['seed']} | depth / DD parameter target {depth}",fontsize=10)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'training_curves.{ext}',dpi=170)
    plt.close(fig)
    js(out/'CURVES.json',{'quantity':'sqrt(logged MSE) = normalized RMSE','training_metric':'online minibatch aggregate','files':['training_curves.png','training_curves.pdf'],
        'source_log_sha256':sha(out/'metrics.jsonl'),'renderer_sha256':sha(__file__)})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',type=int,nargs='?');p.add_argument('--run');a=p.parse_args()
    if a.run:render(a.run)
    else:
        assert a.stage in [1,2,3,4]
        folders=[f.parent for f in (ROOT/STAGES[a.stage-1]/'runs').rglob('COMPLETED.json')]
        for folder in folders:render(folder)
        print('RENDERED',len(folders),'RMSE curve pairs',flush=True)
