"""PRIVATE offline accuracy audit; output must never be supplied to a policy actor.

Estimates are produced from archived RGB-D before simulator states are restored.
Simulator truth is used only to score the resulting estimates, never to fit them.
"""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import mujoco
from geometric_feedback import make_feedback
from aspire_so100.playback import recorded_env_for_result


def audit(trials):
    rows=[]
    for trial in trials:
        # The complete estimate is computed before any private file is opened.
        packet=make_feedback(trial)
        result=json.loads((trial/'result.json').read_text())
        with np.load(trial/'rollout.npz') as record:
            states=record['states'];state_type=mujoco.mjtState(int(record['state_type']))
        with TemporaryDirectory(prefix='geometry-offline-audit-') as tmp:
            env=recorded_env_for_result(result,Path(tmp))
            try:
                if not state_type & mujoco.mjtState.mjSTATE_TIME:
                    raise ValueError('Recorded state must contain time')
                for frame in packet['observations']:
                    index=int(np.argmin(abs(states[:,0]-frame['time'])))
                    mismatch=float(abs(states[index,0]-frame['time']))
                    if mismatch>env.model.opt.timestep*1.1:
                        raise ValueError('No aligned recorded state for sensor frame')
                    mujoco.mj_setState(env.model,env.data,states[index],state_type)
                    mujoco.mj_forward(env.model,env.data)
                    for color,estimate in frame['packet']['objects'].items():
                        name=color+'_cube';gid=env.model.geom(name+'/'+name+'_geom').id
                        actual=env.data.geom_xpos[gid].copy()
                        residual=np.array(estimate['center_mm'])-actual*1000
                        rows.append({'seed':result['seed'],'time':frame['time'],'color':color,
                            'camera_index':frame['camera_index'],'sensor_sha256':frame['sensor_sha256'],
                            'observed_translation_rank':estimate['observed_translation_rank'],
                            'center_error_mm':float(np.linalg.norm(residual)),
                            'xy_error_mm':float(np.linalg.norm(residual[:2])),
                            'z_error_mm':float(abs(residual[2])),
                            'residual_xyz_mm':residual.tolist(),'state_time_mismatch_s':mismatch})
            finally:env.close()
    groups={}
    for label,chosen in [('all',rows),('fully_observed',[r for r in rows if r['observed_translation_rank']==3]),
                         ('partial',[r for r in rows if r['observed_translation_rank']<3])]:
        errors=[r['center_error_mm'] for r in chosen]
        groups[label]={'count':len(chosen),'median_center_error_mm':float(np.median(errors)) if errors else None,
            'p95_center_error_mm':float(np.percentile(errors,95)) if errors else None,
            'max_center_error_mm':max(errors) if errors else None}
    return {'scope':'PRIVATE OFFLINE diagnostic; never policy input','groups':groups,'rows':rows,
            'limits':['Initial/final side observations only; not all motion or occlusion frames.',
                'This evaluates instantaneous center estimates, not vertex accuracy or memory benefit.',
                'Reported center positions were rounded to 0.01 mm by the actor packet.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();value=audit(args.trials)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value['groups'],indent=2))
