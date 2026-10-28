from common import *
import math

def lock_selection(out,configs,budgets=(None,)):
    out=Path(out)
    for budget in budgets:
        suffix='final' if budget is None else f'budget_{budget}'
        file=out/f'VALIDATION_SELECTION_{suffix}.json'
        state=load(out/'last.pt' if budget is None else out/f'budget_{budget}.pt')
        groups={}
        for i,c in enumerate(configs):
            key=f"{c['kind']}:{c['input']}:{c.get('target_dd_depth',c.get('depth'))}"
            groups.setdefault(key,[]).append(i)
        selected=[]
        for key,indices in groups.items():
            valid=[i for i in indices if state['failed'][i] is None and state['best'][i] is not None]
            winner=min(valid,key=lambda i:(state['best'][i]['valid_mse'],i)) if valid else None
            selected.append(dict(group=key,candidates=indices,selected=winner,valid_rmse=None if winner is None else math.sqrt(state['best'][winner]['valid_mse'])))
        if file.exists():assert json.loads(file.read_text(encoding='utf-8'))['selections']==selected
        else:js(file,dict(date=now(),criterion='minimum validation RMSE, tie by candidate index; failed candidates excluded and retained in accounting',budget=budget,
                         selections=selected,checkpoint_sha256=sha(out/'last.pt' if budget is None else out/f'budget_{budget}.pt')))
