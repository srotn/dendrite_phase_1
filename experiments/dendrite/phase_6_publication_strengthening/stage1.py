from engine import *
import csv

STAGE=ROOT/STAGES[0]

def old_evaluation():
    for task in SYNTHETIC:
        d=load(data_path(task,'historical'));tests=load(test_path(task,'historical'))
        for seed in OLD_SEEDS:
            for depth in DEPTHS:
                source=PREVIOUS/task/'runs'/f'seed_{seed}'/f'depth_{depth:02d}'
                out=STAGE/'historical_checkpoint_evaluation'/task/f'seed_{seed}'/f'depth_{depth:02d}'
                if (out/'test_final.json').exists():continue
                configs=dd_configs(depth);best=load(source/'best.pt')
                save(out/'last.pt',{'best':best,'failed':[None]*3,'epoch':2000})
                js(out/'PROVENANCE.json',{'kind':'evaluation of existing checkpoint; not new training','source_best':str(source/'best.pt'),
                    'source_sha256':sha(source/'best.pt'),'configs':configs,'seed':seed,'depth':depth,'task':task,'date':now()})
                evaluate_test(out,d,configs,seed,tests)
        print('OLD MODELS TESTED',task,90,flush=True)

def new_training():
    planned=[(t,f,s,l) for t,f in datasets() if t not in SYNTHETIC for s in SEEDS for l in DEPTHS]
    for number,(task,fold,seed,depth) in enumerate(planned,1):
        js(ROOT/'status.json',dict(pid=os.getpid(),stage=STAGES[0],status='training',group=number,total_groups=len(planned),task=task,fold=fold,seed=seed,depth=depth,updated=now()))
        d=load(data_path(task,fold));out=STAGE/'runs'/task/fold/f'seed_{seed}'/f'depth_{depth:02d}'
        train(out,d,dd_configs(depth),seed,2000)
    js(STAGE/'TRAINING_LOCK.json',{'date':now(),'new_models':540,'checkpoints':{str(p.relative_to(STAGE)):sha(p) for p in (STAGE/'runs').rglob('best.pt')}})
    # Evaluate only after every new real-data training job is complete.
    for task,fold,seed,depth in planned:
        out=STAGE/'runs'/task/fold/f'seed_{seed}'/f'depth_{depth:02d}'
        evaluate_test(out,load(data_path(task,fold)),dd_configs(depth),seed,load(test_path(task,fold)))

def main():
    configure();lock_protocol()
    check=json.loads((ROOT/'engine_verification.json').read_text(encoding='utf-8'))
    assert check['passed'] and check['engine_sha256']==sha(ROOT/'engine.py')
    old_evaluation();new_training()
    js(ROOT/'status.json',dict(pid=os.getpid(),stage=STAGES[0],status='trained_evaluated_pending_analysis',updated=now()))
    print('STAGE 1 FITTING AND EVALUATION DONE; ANALYSIS REQUIRED',flush=True)

if __name__=='__main__':main()
