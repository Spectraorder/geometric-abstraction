"""Run a real sensor-only trial with an initial model gate and time-based occlusion."""
import argparse,hashlib,json,os,runpy,sys,types
from pathlib import Path
import numpy as np
from aspire_so100.instance_geometry import inventory_points,inventory_table,inventory_color_mask
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['hybrid','astra_only','jev_only'],required=True)
    p.add_argument('--fault',choices=['none','transient_occlusion','persistent_occlusion'],default='transient_occlusion')
    p.add_argument('--seed',type=int,default=4001);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--jev-key-file',type=Path);p.add_argument('--confidence',type=float,default=.8)
    a=p.parse_args()
    if a.output.exists():p.error('Output must be new')
    if a.jev_key_file:
        key=a.jev_key_file.read_text().strip()
        if not key or any(c.isspace() for c in key):p.error('Expected a single API token in credential file')
        os.environ['TYPESAFE_API_KEY']=key
    if a.mode!='astra_only' and not os.environ.get('TYPESAFE_API_KEY'):p.error('TypeSafe credentials required')
    import tempfile
    frozen_sources={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__),ROOT/'scripts/run_vision_trial.py',ROOT/'scripts/build_dual_system_policy.py',ROOT/'src/aspire_so100/dual_system.py',ROOT/'src/aspire_so100/astra_decision.py',ROOT/'src/aspire_so100/decision_session.py']}
    events=[];origin=[None]
    def camera_hook(env,frame,view):
        if view!='side':return
        if origin[0] is None:origin[0]=float(env.data.time)
        age=float(env.data.time)-origin[0]
        if a.fault=='none' or (a.fault=='transient_occlusion' and age>=.35):return
        points,valid=inventory_points(frame);plane=inventory_table(frame,points,valid)
        heights=points[:,:,2]-points[:,:,:2]@plane[:2]-plane[2]
        mask=inventory_color_mask(frame['rgb'],'blue')&valid&(heights>.004)&(heights<.15)
        ys,xs=np.nonzero(mask)
        if not len(xs):raise RuntimeError('No visible blue ROI for fault injection')
        y0,y1=max(0,int(ys.min())-4),min(len(mask),int(ys.max())+5)
        x0,x1=max(0,int(xs.min())-4),min(mask.shape[1],int(xs.max())+5)
        frame['rgb'][y0:y1,x0:x1]=55;frame['depth_m'][y0:y1,x0:x1]=np.nan;frame['valid_depth'][y0:y1,x0:x1]=False
        events.append({'event':'sensor_rectangle','time':float(env.data.time),'rectangle':[x0,y0,x1,y1]})
    runner=ROOT/'scripts/run_vision_trial.py';source=runner.read_text()
    anchor="                                if a.sensor_blackout:"
    if source.count(anchor)!=1:raise RuntimeError('Runner changed; review sensor hook')
    source=source.replace(anchor,"                                camera_hook(env,response,view)\n"+anchor)
    module=types.ModuleType('dual_trial_runner');module.__file__=str(runner);module.camera_hook=camera_hook
    sys.modules[module.__name__]=module;exec(compile(source,str(runner),'exec'),module.__dict__)
    with tempfile.TemporaryDirectory(prefix='dual-policy-') as temp:
        candidate=Path(temp)/'candidate.py';candidate.write_text(runpy.run_path(str(ROOT/'scripts/build_dual_system_policy.py'))['source']())
        sys.argv=[str(runner),'--candidate',str(candidate),'--output',str(a.output),'--seed',str(a.seed),'--layout','random_xy','--timeout','900','--decision-mode',a.mode,'--decision-confidence',str(a.confidence)]
        try:module.main()
        finally:
            if a.output.exists():
                (a.output/'candidate.py').write_text(candidate.read_text())
                (a.output/'fault_events.private.json').write_text(json.dumps({'fault':a.fault,'time_anchor':'first side observation','duration_s':.35 if a.fault=='transient_occlusion' else None,'events':events,'actor_receives_fault_metadata':False},indent=2)+'\n')
                (a.output/'harness.json').write_text(json.dumps({'protocol':'phase-conditional-plan/2','frozen_sources':frozen_sources,'mode':a.mode,'runner_sha256':hashlib.sha256(runner.read_bytes()).hexdigest(),'adapter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'model_wait_simulation':'paused','wall_timeout_s':900,'source_base_commit':'3a2a48a'},indent=2)+'\n')
if __name__=='__main__':main()
