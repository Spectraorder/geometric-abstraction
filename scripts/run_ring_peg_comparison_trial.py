"""Execute a restricted single policy with sensor geometry and no truth RPC."""
import argparse
import ast
import json
import os
from pathlib import Path
import selectors
import shutil
import subprocess
import tempfile
import time
import hashlib

from ring_peg_scene import RingPegEnv
from ring_peg_runtime import plain
from ring_peg_comparison_runtime import ComparisonSensors
from ring_peg_development import installed
from run_vision_trial import validate_source, policy_worker_command


def validate_policy(source):
    validate_source(source)
    tree=ast.parse(source)
    if len(source)>50000:raise ValueError('Policy exceeds 50000 characters')
    for node in tree.body:
        if not isinstance(node,(ast.FunctionDef,ast.Import,ast.ImportFrom,ast.Expr)):
            raise ValueError('Only imports, functions and docstrings at module scope')
        if isinstance(node,ast.Expr) and not (isinstance(node.value,ast.Constant) and isinstance(node.value.value,str)):
            raise ValueError('Only docstrings at module scope')
    for node in ast.walk(tree):
        if isinstance(node,(ast.ClassDef,ast.Global,ast.Nonlocal)):raise ValueError('Forbidden policy structure')
        if isinstance(node,ast.Attribute) and isinstance(node.ctx,ast.Store):raise ValueError('Attribute mutation forbidden')
        if isinstance(node,ast.Attribute) and node.attr in ('request','env','model','data'):raise ValueError('Private API access forbidden')
    if not any(isinstance(n,ast.FunctionDef) and n.name=='run' for n in tree.body):raise ValueError('Missing run(api)')


def run(output,candidate,seed,arm):
    output.mkdir(parents=True,exist_ok=False)
    source=candidate.read_text();result={'seed':seed,'success':False,'candidate_sha256':hashlib.sha256(source.encode()).hexdigest(),'status':'running','arm':arm,'started_at':time.time()}
    try:validate_policy(source)
    except (ValueError,SyntaxError) as error:
        result.update(status='invalid_policy',error=str(error),rollout_executed=False)
        (output/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result
    env=RingPegEnv(output/'env',scene_seed=seed);env.reset();api=ComparisonSensors(env,output,arm)
    calls=[];started=time.monotonic()
    try:
        api.camera('side')
        with tempfile.TemporaryDirectory(prefix='ring_policy_') as folder:
            scratch=Path(folder);(scratch/'candidate.py').write_text(source)
            shutil.copyfile(Path(__file__).with_name('ring_peg_comparison_worker.py'),scratch/'worker.py')
            command=policy_worker_command(scratch,scratch/'candidate.py')
            (output/'sandbox.json').write_text(json.dumps({'command':command,'network':'disabled','policy_access':'scratch and runtime dependencies only'})+'\n')
            clean={'PATH':'/usr/bin:/bin','HOME':folder,'TMPDIR':folder,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
            with (output/'worker.stderr').open('w') as stderr:
                proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,text=True,cwd=folder,env=clean,bufsize=1)
                selector=selectors.DefaultSelector();selector.register(proc.stdout,selectors.EVENT_READ)
                try:
                    while True:
                        if time.monotonic()-started>120:raise TimeoutError('Policy wall-time limit')
                        if not selector.select(timeout=.5):continue
                        line=proc.stdout.readline()
                        if not line:raise RuntimeError('Worker exited without completion')
                        request=json.loads(line);method=request['method'];params=request['params']
                        if method=='done':break
                        if method=='failed':raise RuntimeError(params['error'])
                        if len(calls)>=600:raise RuntimeError('API call budget exceeded')
                        row={'method':method,'params':params,'time':float(env.data.time)}
                        try:
                            value=plain(api.dispatch(method,params));answer={'result':value}
                            if method=='observe':row['result']=value
                        except (ValueError,RuntimeError) as error:
                            answer={'error':type(error).__name__+': '+str(error)};row.update(answer)
                        calls.append(row);proc.stdin.write(json.dumps(answer,allow_nan=False)+'\n');proc.stdin.flush()
                finally:
                    proc.terminate();proc.wait(timeout=10);selector.close()
        result['status']='completed'
    except (RuntimeError,ValueError,TimeoutError) as error:result.update(status='policy_error',error=type(error).__name__+': '+str(error))
    finally:
        evaluation=env.evaluate_on_peg();result.update(success=bool(evaluation['success']),rollout_executed=True,
            wall_seconds=time.monotonic()-started,finished_at=time.time(),camera_calls=api.frames,private_evaluation=plain(evaluation))
        api.camera('side');api.camera('wrist')
        (output/'api_calls.json').write_text(json.dumps(plain(calls),indent=2)+'\n')
        (output/'sensor_records.json').write_text(json.dumps(plain(api.records),indent=2)+'\n')
        (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        feedback={k:result[k] for k in ('seed','success','status','error') if k in result}
        feedback['observations']=plain(api.records[-4:])
        if arm=='basic' and any(set(row)-{'note','time'} for row in api.records):
            raise RuntimeError('Base feedback contains supplied geometry')
        (output/'actor_feedback.json').write_text(json.dumps(feedback,indent=2)+'\n');env.close()
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--arm',choices=('basic','symbolic'),required=True)
    args=parser.parse_args()
    with installed():
        result=run(args.output,args.candidate,args.seed,args.arm)
    print(json.dumps({k:result[k] for k in ('seed','arm','success','status','error') if k in result}),flush=True)
