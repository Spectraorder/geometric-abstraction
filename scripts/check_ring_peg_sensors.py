"""Development-only sensor-driven physical check; never supplied as a policy starter."""
import argparse
import json
from pathlib import Path
import numpy as np
from ring_peg_scene import RingPegEnv
from ring_peg_runtime import RingSensors,plain


def run(output,seed):
    output.mkdir(parents=True,exist_ok=False)
    env=RingPegEnv(output/'env',scene_seed=seed);env.reset()
    api=RingSensors(env,output)
    result={'seed':seed,'success':False,'scope':'sensor-driven development only; no model'}
    try:
        ring,peg=api.observe('ring'),api.observe('peg')
        table=api.table
        plans=[]
        for wall in ring['walls']:
            normal=np.array(wall['outward_normal']);point=np.array(wall['outer_mid_m'])+normal*.003
            try:
                plan=api.robot.plan_grasp(point+np.array([0,0,.06]),-normal,.15)
                plans.append((abs(plan['roll']-api.robot.current()['roll']),point,plan))
            except (ValueError,RuntimeError):pass
        if not plans: raise ValueError('No reachable visible wall')
        _,point,plan=min(plans,key=lambda row:row[0]);roll=plan['roll']
        api.dispatch('gripper',{'angle':.15})
        api.robot.move_pad(point+np.array([0,0,.06]),roll,1.8)
        for drop in np.arange(.06,-.03,-.004):
            desired=point+np.array([0,0,drop]);p=api.robot.plan_pad(desired,roll,.15)
            vertices=p['jaw_vertices_m'];clearance=vertices[:,2]-vertices[:,:2]@table[:2]-table[2]
            if np.min(clearance)<.003: break
            api.robot.move_pad(desired,roll,.35)
        api.dispatch('gripper',{'angle':-.174,'seconds':1.})
        robot=api.robot.current();lift=robot['fixed_pad_m'].copy();lift[2]+=.12
        api.robot.move_pad(lift,roll,1.7)
        api.observe('ring')
        for _ in range(5):
            ring=api.observe('ring');robot=api.robot.current()
            error=np.array(peg['top_center_m'])[:2]-np.array(ring['hole']['center_m'])[:2]
            lowest=min(float(robot['jaw_vertices_m'][:,2].min()),ring['minimum_z_m'])
            dz=peg['top_center_m'][2]+.015-lowest
            desired=robot['fixed_pad_m']+np.r_[error,dz]
            if np.linalg.norm(error)<.002 and abs(dz)<.002:break
            api.robot.move_pad(desired,roll,1.4)
        api.dispatch('gripper',{'angle':.15})
        robot=api.robot.current();api.robot.move_pad(robot['fixed_pad_m']+np.array([0,0,.05]),roll,1.2)
        result['evaluation']=env.evaluate_on_peg();result['success']=result['evaluation']['success']
    except Exception as error:
        result['error']=type(error).__name__+': '+str(error)
        import traceback
        result['traceback']=traceback.format_exc()
    finally:
        api.camera('side');api.camera('wrist')
        (output/'sensor_records.json').write_text(json.dumps(plain(api.records),indent=2)+'\n')
        (output/'result.json').write_text(json.dumps(plain(result),indent=2)+'\n');env.close()
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    print(json.dumps(run(a.output,a.seed)),flush=True)
