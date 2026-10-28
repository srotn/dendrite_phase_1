from engine import *

STAGE=ROOT/STAGES[3]

def configs(depth):
    return [dict(name=f'c{slot}_{method}',kind='dd',depth=depth,input=method,slot=slot,lr=2e-4)
            for slot,method in [(0,'duplicate'),(0,'reciprocal'),(0,'matched'),(1,'duplicate'),(1,'reciprocal'),(1,'matched'),(1,'subset'),(1,'complement')]]

def main():
    configure();lock_protocol()
    assert json.loads((ROOT/STAGES[2]/'VERIFIED.json').read_text(encoding='utf-8'))['passed']
    jobs=[(task,fold,seed,depth) for task,fold in datasets() for seed in SEEDS for depth in [1,4,16]]
    for n,(task,fold,seed,depth) in enumerate(jobs,1):
        js(ROOT/'status.json',dict(pid=os.getpid(),stage=STAGES[3],status='training',group=n,total_groups=len(jobs),task=task,fold=fold,seed=seed,depth=depth,updated=now()))
        out=STAGE/'runs'/task/fold/f'seed_{seed}'/f'depth_{depth:02d}';d=load(data_path(task,fold));cs=configs(depth)
        models=[build(d,c,seed) for c in cs]
        assert len({parameter_count(m) for m in models})==1
        assert all(all(torch.equal(v,models[0].state_dict()[k]) for k,v in m.state_dict().items()) for m in models[1:])
        train(out,d,cs,seed,4000)
    js(STAGE/'TRAINING_LOCK.json',dict(date=now(),fits=1008,checkpoints={str(p.relative_to(STAGE)):sha(p) for p in STAGE.rglob('best.pt')}))
    for task,fold,seed,depth in jobs:
        out=STAGE/'runs'/task/fold/f'seed_{seed}'/f'depth_{depth:02d}'
        evaluate_test(out,load(data_path(task,fold)),configs(depth),seed,load(test_path(task,fold)))
    js(ROOT/'status.json',dict(pid=os.getpid(),stage=STAGES[3],status='trained_evaluated_pending_analysis',updated=now()))

if __name__=='__main__':main()
