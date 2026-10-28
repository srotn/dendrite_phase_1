from engine import *
from selection import lock_selection

STAGE=ROOT/STAGES[2]

def main():
    configure();lock_protocol()
    assert json.loads((ROOT/STAGES[1]/'VERIFIED.json').read_text(encoding='utf-8'))['passed']
    jobs=[(task,fold,seed,depth) for task,fold in datasets() for seed in SEEDS for depth in [1,4,8,16]]
    for n,(task,fold,seed,depth) in enumerate(jobs,1):
        js(ROOT/'status.json',dict(pid=os.getpid(),stage=STAGES[2],status='training',group=n,total_groups=len(jobs),task=task,fold=fold,seed=seed,depth=depth,updated=now()))
        out=STAGE/'runs'/task/fold/f'seed_{seed}'/f'depth_{depth:02d}'
        train(out,load(data_path(task,fold)),dd_configs(depth,('duplicate','reciprocal'),LRS),seed,8000,[2000,4000,8000])
    js(STAGE/'TRAINING_LOCK.json',dict(date=now(),fits=1008,checkpoints={str(p.relative_to(STAGE)):sha(p) for p in STAGE.rglob('best.pt')}))
    for task,fold,seed,depth in jobs:
        out=STAGE/'runs'/task/fold/f'seed_{seed}'/f'depth_{depth:02d}'
        lock_selection(out,dd_configs(depth,('duplicate','reciprocal'),LRS),[2000,4000,8000])
    for task,fold,seed,depth in jobs:
        out=STAGE/'runs'/task/fold/f'seed_{seed}'/f'depth_{depth:02d}';d=load(data_path(task,fold));tests=load(test_path(task,fold))
        for budget in [2000,4000,8000]:evaluate_test(out,d,dd_configs(depth,('duplicate','reciprocal'),LRS),seed,tests,budget)
    js(ROOT/'status.json',dict(pid=os.getpid(),stage=STAGES[2],status='trained_evaluated_pending_analysis',updated=now()))

if __name__=='__main__':main()
