from __future__ import annotations
import copy, hashlib, json, os, platform, sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
DENDRITE=ROOT.parent
PREVIOUS=DENDRITE/'phase_4_depth_reciprocal_ablation'
REPO=DENDRITE.parents[1]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
TASKS=['01_forced','02_thermal','03_optical','04_impedance','05_concrete','06_co']
SYNTHETIC=TASKS[:4]
DEPTHS=[1,2,3,4,8,16]
SEEDS=[20261101,20261102,20261103]
OLD_SEEDS=list(range(20261025,20261030))
LRS=[2e-5,2e-4,2e-3]
STAGES=['01_independent_evaluation','02_external_baselines','03_optimization_fairness','04_mechanism_controls']
FIELDS={'01_forced':[1,2],'02_thermal':list(range(16)),'03_optical':[18],
        '04_impedance':list(range(16))+[17,18],'05_concrete':list(range(7)),'06_co':list(range(4,11))}

def now(): return datetime.now(timezone(timedelta(hours=8))).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def clean(v):
    if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
    if isinstance(v,np.integer):return int(v)
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if torch.is_tensor(v):return clean(v.detach().cpu().tolist())
    if isinstance(v,Path):return str(v)
    return v
def js(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(path)
def cpu(v):
    if torch.is_tensor(v):return v.detach().cpu().clone()
    if isinstance(v,dict):return {k:cpu(x) for k,x in v.items()}
    if isinstance(v,list):return [cpu(x) for x in v]
    return v
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');torch.save(cpu(value),tmp);tmp.replace(path)
def load(path):return torch.load(path,weights_only=False,map_location='cpu')
def configure():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
def datasets():
    return [(t,'historical') for t in SYNTHETIC]+[(t,f'fold_{i}') for t in TASKS[4:] for i in range(5)]
def data_path(task,fold):return ROOT/STAGES[0]/'data'/task/fold/'development.pt'
def test_path(task,fold):return ROOT/STAGES[0]/'data'/task/fold/'tests.pt'
def lock_protocol():
    if (ROOT/'PROTOCOL_LOCK.json').exists():
        lock=json.loads((ROOT/'PROTOCOL_LOCK.json').read_text(encoding='utf-8'))
        assert sha(ROOT/'protocol.json')==lock['protocol_sha256']
        assert sha(ROOT/'README.md')==lock['readme_sha256']
        return
    for s in STAGES:(ROOT/s).mkdir(exist_ok=True)
    protocol=dict(date=now(),tasks=TASKS,depths=DEPTHS,optimization_seeds=SEEDS,learning_rates=LRS,
        stage_order=STAGES,split_seed=2026102601,synthetic_test_seeds=[2026111101,2026111102,2026111103],
        synthetic_final_seed=2026120101,test_rows=20000,mechanism_fields=FIELDS,
        stage1=dict(epochs=2000,real_fits=540,old_synthetic_models=360,lr=2e-4),
        stage2=dict(epochs=2000,mlp_fits=504,classic_fits=84,mlp_activation='SiLU',mlp_hidden_layers=2,
                    parameter_targets=[4,16],polynomial_degree=2,ridge_alpha=[1e-4,1.,1e4],tree_leaf_nodes=[15,31,63],tree_max_iter=500,tree_l2=.1),
        stage3=dict(epochs=8000,depths=[1,4,8,16],fits=1008,budgets=[2000,4000,8000]),
        stage4=dict(epochs=4000,depths=[1,4,16],fits=1008,lr=2e-4,
             variants=['c0_duplicate','c0_reciprocal','c0_matched','c1_duplicate','c1_reciprocal','c1_matched','c1_subset','c1_complement']),
        validation_selection='per seed, method, depth and budget: select learning rate by minimum validation RMSE; never test',
        runtime=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,platform=platform.platform()))
    js(ROOT/'protocol.json',protocol)
    previous={str(p.relative_to(DENDRITE)):sha(p) for p in DENDRITE.rglob('*') if p.is_file() and ROOT not in p.parents and p.suffix in ['.py','.md','.json'] and '__pycache__' not in p.parts}
    js(ROOT/'historical_source_hashes.json',previous)
    js(ROOT/'PROTOCOL_LOCK.json',dict(date=now(),protocol_sha256=sha(ROOT/'protocol.json'),readme_sha256=sha(ROOT/'README.md'),results_observed=False))
    print('PROTOCOL FROZEN',now(),flush=True)

if __name__=='__main__':configure();lock_protocol()
