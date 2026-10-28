"""Durable sequential controller. Never overlaps the four experimental stages."""
from __future__ import annotations
import argparse,subprocess,time
import psutil
from common import *

def invoke(script,*args):
    logs=ROOT/'logs';logs.mkdir(exist_ok=True)
    log=logs/(Path(script).stem+('_'+'_'.join(args) if args else '')+'.log')
    with log.open('a',encoding='utf-8') as stream:
        stream.write('\nSTART '+now()+'\n');stream.flush()
        process=subprocess.Popen([sys.executable,'-X','utf8',str(ROOT/script),*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                                 creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        print('STARTED',script,args,'PID',process.pid,'LOG',log,flush=True)
        while process.poll() is None:
            js(ROOT/'pipeline_state.json',dict(controller_pid=os.getpid(),child_pid=process.pid,script=script,args=args,log=str(log),status='running',updated=now()))
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:pass
        if process.returncode:
            js(ROOT/'PIPELINE_FAILED.json',dict(date=now(),script=script,args=args,exit_code=process.returncode,log=str(log)))
            raise RuntimeError(f'{script} failed ({process.returncode}); see {log}')
        print('FINISHED',script,args,flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait-stage1-pid',type=int);args=parser.parse_args()
    lock_protocol()
    if args.wait_stage1_pid:
        try:
            process=psutil.Process(args.wait_stage1_pid)
            assert any(str(ROOT/'stage1.py').lower()==a.lower() for a in process.cmdline()),process.cmdline()
            created=process.create_time()
            while process.is_running() and process.create_time()==created:
                js(ROOT/'pipeline_state.json',dict(controller_pid=os.getpid(),child_pid=process.pid,status='waiting_for_existing_stage1',updated=now()))
                try:process.wait(timeout=30)
                except psutil.TimeoutExpired:continue
                break
        except psutil.NoSuchProcess:pass
    if not (ROOT/STAGES[0]/'VERIFIED.json').exists():
        invoke('stage1.py');invoke('render_curves.py','1');invoke('analyze_stage1.py')
    for stage in [2,3,4]:
        marker=ROOT/STAGES[stage-1]/'VERIFIED.json'
        if not marker.exists():invoke(f'stage{stage}.py');invoke('render_curves.py',str(stage));invoke('analyze_followups.py',str(stage))
        assert json.loads(marker.read_text(encoding='utf-8'))['passed']
    if not (ROOT/'verification/final_confirmation/VERIFIED.json').exists():invoke('final_confirmation.py')
    invoke('write_report.py')
    js(ROOT/'pipeline_state.json',dict(controller_pid=os.getpid(),status='complete_verified',updated=now()))

if __name__=='__main__':main()
