"""Offline contact-solver sensitivity audit; never a policy success benchmark.

Restores identical saved final states and changes only friction post-processing.
Default scene physics and all previously scored trial results remain unchanged.
"""
import argparse
import json
from pathlib import Path
import tempfile

import mujoco
import numpy as np
from aspire_so100.playback import recorded_env_for_result
from score_cube_alignment import measure


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=60)
    args=parser.parse_args();rows=[]
    for trial in args.trials:
        result=json.loads((trial/'result.json').read_text());order=result['scene']['order']
        with np.load(trial/'rollout.npz') as archive:
            state=archive['states'][-1].copy();stype=int(archive['state_type'])
        for iterations in (0,10,50):
            with tempfile.TemporaryDirectory() as tmp:
                env=recorded_env_for_result(result,Path(tmp))
                try:
                    default=int(env.model.opt.noslip_iterations)
                    env.model.opt.noslip_iterations=iterations
                    mujoco.mj_setState(env.model,env.data,state,stype);mujoco.mj_forward(env.model,env.data)
                    before=measure(env,order)
                    centers={n:env.data.xpos[env.model.body(f'{n}/{n}').id].copy() for n in order}
                    check=env.check_structure(order,duration=args.seconds)
                    after=measure(env,order)
                    row=dict(trial=str(trial),seed=result['seed'],mujoco_version=mujoco.__version__,
                             original_policy_sha256=result['candidate_sha256'],default_noslip_iterations=default,
                             noslip_iterations=iterations,impratio=float(env.model.opt.impratio),
                             cone=int(env.model.opt.cone),solver=int(env.model.opt.solver),
                             before=before,after=after,check=check,
                             displacement_mm={n:float(np.linalg.norm(env.data.xpos[env.model.body(f'{n}/{n}').id]-centers[n])*1000) for n in order})
                    rows.append(row)
                    print(trial,iterations,check['stable'],row['displacement_mm'],flush=True)
                finally:env.close()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(scope='Offline sensitivity only. No policy execution, no runtime privileged data, no relabeling of baseline or candidate success. Default physics unchanged.',documentation='https://mujoco.readthedocs.io/en/latest/modeling.html#preventing-slip',cases=rows),indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')

if __name__=='__main__':main()
