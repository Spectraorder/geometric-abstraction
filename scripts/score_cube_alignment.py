"""Offline-only edge alignment and optional durability scoring of saved trials."""
import argparse
import json
from pathlib import Path
import tempfile

import mujoco
import numpy as np
from aspire_so100.playback import recorded_env_for_result
from aspire_so100.cube_alignment import cube_edge_alignment


def measure(env, order):
    models = []
    for name in order:
        geom = env.model.geom(f'{name}/{name}_geom')
        rotation = env.data.geom_xmat[geom.id].reshape(3,3)
        center = env.data.geom_xpos[geom.id].copy()
        corners = np.array([[x,y,1] for x in (-1,1) for y in (-1,1)])*geom.size
        models.append(dict(center=center,top_vertices=center+corners@rotation.T))
    rows = []
    for i in range(1,len(order)):
        metrics = cube_edge_alignment(models[i],models[i-1])
        rows.append(dict(lower=order[i-1],upper=order[i],**metrics,
                         yaw_error_deg=float(np.rad2deg(abs(metrics['yaw_error_rad'])))))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--durability',type=float,default=0)
    args=parser.parse_args()
    rows=[]
    for root in args.trials:
        for path in sorted(root.rglob('result.json')):
            result=json.loads(path.read_text())
            if result.get('task')!='inventory_tower':continue
            order=result['scene']['order']
            if not all(n.endswith('_cube') for n in order):continue
            with np.load(path.parent/'rollout.npz') as data:
                state=data['states'][-1].copy();state_type=int(data['state_type'])
            with tempfile.TemporaryDirectory() as tmp:
                env=recorded_env_for_result(result,Path(tmp))
                try:
                    mujoco.mj_setState(env.model,env.data,state,state_type);mujoco.mj_forward(env.model,env.data)
                    pairs=measure(env,order)
                    row=dict(trial=str(path.parent),seed=result['seed'],success=result['success'],
                             candidate_sha256=result['candidate_sha256'],physics=result.get('physics', {'noslip_iterations':0}),policy_error=result['policy_error'],
                             pairs=pairs, max_corner_m=max(p['corner_max_m'] for p in pairs),
                             max_xy_m=max(p['xy_error_m'] for p in pairs),max_yaw_deg=max(p['yaw_error_deg'] for p in pairs))
                    if args.durability:
                        row['durability']=env.check_structure(order,duration=args.durability)
                        row['durability_pairs']=measure(env,order)
                    rows.append(row)
                    print(path.parent, result['success'], 'max corner mm',round(row['max_corner_m']*1000,3),'yaw deg',round(row['max_yaw_deg'],3),flush=True)
                finally:env.close()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'boundary':'Privileged state used ONLY for offline scoring, never returned to policy. Final saved frame after original stability evaluation.','cases':rows},indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')

if __name__=='__main__':main()
