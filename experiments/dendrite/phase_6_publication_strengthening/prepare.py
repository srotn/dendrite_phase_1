from __future__ import annotations
import importlib
import numpy as np
import pandas as pd
from common import *

MODULES={'01_forced':'01_forced_oscillation','02_thermal':'03_thermal_field','03_optical':'04_optical_stack','04_impedance':'05_impedance_network'}

def unique_rows(x):
    a=np.ascontiguousarray(x.numpy())
    return set(a.view(np.dtype((np.void,a.dtype.itemsize*a.shape[1]))).ravel().tolist())

def synthetic(task):
    out=data_path(task,'historical').parent
    if (out/'PREPARED.json').exists():return
    d=load(PREVIOUS/task/'data/dataset.pt')
    src=DENDRITE/json.loads((PREVIOUS/task/'data/manifest.json').read_text(encoding='utf-8'))['source']
    original=load(src);meta=d['preprocessing']
    d['task']=task;d['fold']='historical';d['scale_units']='mixed_log_amplitude_phase' if task=='01_forced' else 'temperature_C' if task=='02_thermal' else 'logit_reflectance' if task=='03_optical' else 'log_output_voltage'
    d['target_mean']=original['y_mean'];d['target_scale']=original['y_scale']
    for split,source_split in [('train','train'),('valid','iid' if 'iid' in original else 'valid')]:
        d[split]['physical']=original[source_split]['physical']
        d[split]['groups']=torch.arange(len(d[split]['y']))
        d[split]['row_ids']=torch.arange(len(d[split]['y']))
    module=importlib.import_module('experiments.dendrite.phase_1_synthetic_experiments.'+MODULES[task]+'.experiment')
    tests={};used=set()
    for split in ('train','valid','iid','ood'):
        if split in original:used |= unique_rows(original[split]['physical'])
    count=0
    for rep,seed in enumerate([2026111101,2026111102,2026111103]):
        for ood in ([False,True] if task=='04_impedance' else [False]):
            name=('ood' if ood else 'iid')+f'_{rep}'
            actual_seed=seed+100*TASKS.index(task)+(10000 if ood else 0)
            kwargs={} if task=='01_forced' else {'device':'cpu'}
            if task=='04_impedance':kwargs['ood']=ood
            v=module.generate(20000,actual_seed,**kwargs)
            new=unique_rows(v['physical']);assert len(new)==len(v['x']) and not used.intersection(new)
            used |= new;count+=len(new)
            r=v['physical'].sign()/v['physical'].abs().clamp_min(1e-6)
            r=(r.sign()*torch.minimum(r.abs(),meta['cap'])-meta['mean'])/meta['std']
            tests[name]={'x':v['x'],'r':r,'physical':v['physical'],'y':(v['y']-d['target_mean'])/d['target_scale'],
                         'groups':torch.arange(len(v['y'])),'row_ids':torch.arange(len(v['y'])),'seed':actual_seed}
    save(data_path(task,'historical'),d);save(test_path(task,'historical'),tests)
    js(out/'PREPARED.json',{'date':now(),'task':task,'source':src,'source_sha256':sha(src),
       'development_sha256':sha(data_path(task,'historical')),'tests_sha256':sha(test_path(task,'historical')),
       'new_test_rows':count,'new_input_rows_disjoint_from_history_and_each_other':True,
       'generator_source_sha256':sha(module.__file__),'test_seed_map':{k:v['seed'] for k,v in tests.items()}})
    print('PREPARED',task,count,flush=True)

def real_source(task):
    if task=='05_concrete':
        path=DENDRITE/'phase_2_paper_regression/paper_regression_datasets/data/09_concrete_compressive_strength/Concrete_Data.xls'
        a=pd.read_excel(path,engine='xlrd').to_numpy(dtype=np.float32)
        assert a.shape==(1030,9) and np.isfinite(a).all()
        _,groups=np.unique(a[:,:7],axis=0,return_inverse=True)
        return torch.from_numpy(a[:,:8].copy()),torch.from_numpy(a[:,8:].copy()),torch.from_numpy(groups),torch.arange(len(a)),{'source':str(path),'source_sha256':sha(path),'group_definition':'exact first seven mixture ingredients; age excluded','unique_groups':len(np.unique(groups))}
    path=DENDRITE/'phase_3_real_world_regression/01_gas_sensor_temperature_modulation/artifacts/dataset.pt'
    old=load(path);xs=[];ys=[];groups=[];rowids=[];files=[]
    for oldsplit in ['train','valid','test']:
        s=old['splits'][oldsplit]
        for local,file in enumerate(s['files']):
            mask=s['batch_id']==local
            xs.append(s['physical_x'][mask]);ys.append(s['raw_y'][mask]);groups.append(torch.full((int(mask.sum()),),len(files)))
            rowids.append(s['source_rows'][local]);files.append(file)
    assert len(files)==len(set(files))==13
    return torch.cat(xs),torch.cat(ys),torch.cat(groups),torch.cat(rowids),{'source':str(path),'source_sha256':sha(path),'group_definition':'whole acquisition file; 10-second historical sampling','files':files,'unique_groups':13,'new_external_acquisition':False}

def allocate_groups(groups):
    rng=np.random.default_rng(2026102601)
    ids,counts=np.unique(groups.numpy(),return_counts=True)
    order=rng.permutation(len(ids));order=sorted(order,key=lambda i:-counts[i])
    bins=[[] for _ in range(5)];sizes=[0]*5
    for i in order:
        f=int(np.argmin(sizes));bins[f].append(int(ids[i]));sizes[f]+=int(counts[i])
    return bins

def real(task):
    base=ROOT/STAGES[0]/'data'/task
    if (base/'PREPARED.json').exists():return
    x,y,g,row,source=real_source(task);folds=allocate_groups(g)
    fold_records=[]
    for fold in range(5):
        assignment={name:torch.isin(g,torch.tensor(ids)) for name,ids in [('test',folds[fold]),('valid',folds[(fold+1)%5]),('train',[i for k,v in enumerate(folds) if k not in [fold,(fold+1)%5] for i in v])]}
        assert torch.all(sum(v.int() for v in assignment.values())==1)
        tr=x[assignment['train']];ty=y[assignment['train']]
        eligible=list(range(8)) if task=='05_concrete' else list(range(4,18))
        inv=lambda v:v[:,eligible].sign()/v[:,eligible].abs().clamp_min(1e-6)
        rr=inv(tr);cap=torch.quantile(rr.abs(),.995,dim=0).clamp_min(1e-6);rr=rr.clamp(-cap,cap)
        if task=='05_concrete':
            xm=tr.min(0).values;xs=(tr.max(0).values-xm).clamp_min(1e-6)
            rm=rr.min(0).values;rs=(rr.max(0).values-rm).clamp_min(1e-6)
            ym=ty.min(0).values;ys=(ty.max(0).values-ym).clamp_min(1e-6)
            transform=lambda v,m,s:2*(v-m)/s-1
            target_mean=ym+ys/2;target_scale=ys/2
        else:
            xm=tr.mean(0);xs=tr.std(0,correction=0);xs=torch.where(xs<1e-6,torch.ones_like(xs),xs)
            rm=rr.mean(0);rs=rr.std(0,correction=0).clamp_min(1e-6)
            ym=ty.mean(0);ys=ty.std(0,correction=0).clamp_min(1e-6)
            transform=lambda v,m,s:(v-m)/s
            target_mean=ym;target_scale=ys
        d={'task':task,'fold':f'fold_{fold}','eligible':eligible,'constant':True,'target_mean':target_mean,'target_scale':target_scale,'scale':target_scale,
           'scale_units':'MPa' if task=='05_concrete' else 'ppm','preprocessing':dict(x_mean=xm,x_scale=xs,r_mean=rm,r_scale=rs,cap=cap,kind='minmax' if task=='05_concrete' else 'zscore'),
           'source':source,'group_folds':folds}
        test={}
        for split,mask in assignment.items():
            xx=transform(x[mask],xm,xs);r=transform(inv(x[mask]).clamp(-cap,cap),rm,rs)
            if task=='06_co':xx=xx.clamp(-3,3);r=r.clamp(-3,3)
            v=dict(x=xx,r=r,y=(y[mask]-target_mean)/target_scale,physical=x[mask],groups=g[mask],row_ids=row[mask],source_indices=torch.where(mask)[0])
            assert all(torch.isfinite(v[k]).all() for k in ['x','r','y'])
            if split=='test':test['test']=v
            else:d[split]=v
        out=data_path(task,f'fold_{fold}').parent;save(out/'development.pt',d);save(out/'tests.pt',test)
        record={'fold':fold,'sizes':{s:int(m.sum()) for s,m in assignment.items()},'groups':{s:sorted(torch.unique(g[m]).tolist()) for s,m in assignment.items()},
                'development_sha256':sha(out/'development.pt'),'tests_sha256':sha(out/'tests.pt'),'train_only_transform':True,'groups_disjoint':True}
        fold_records.append(record);js(out/'PREPARED.json',record)
    js(base/'PREPARED.json',{'source':source,'folds':fold_records,'each_group_tested_exactly_once':True,'date':now()})
    print('PREPARED',task,source['unique_groups'],'groups',[r['sizes'] for r in fold_records],flush=True)

def main():
    configure();lock_protocol()
    for task in TASKS:
        if task in SYNTHETIC:synthetic(task)
        else:real(task)

if __name__=='__main__':main()
