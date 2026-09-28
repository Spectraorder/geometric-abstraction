"""Run one continuous online Ring development or registered episode."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
from ring_peg_development import installed
from ring_peg_runtime import plain
from online_bridge_scene import EpisodeSuccess
from online_ring_scene import OnlineRingEnv
from online_ring_loop import OnlineRingSensors, RingLoop
from online_ring_skills import RingSkills, choose_training_prior, digest

ROOT=Path(__file__).resolve().parents[1]
DEVELOPMENT_SEEDS=(1,2)


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n')


def scripted_check(api,skills):
    """Sensor-driven physical smoke test; no model-reasoning claim."""
    result=skills.grasp_lift()
    write(api.output/'scripted_grasp.json',result)
    if result['status']!='lift_observed':return
    # The peg is declared fixed. Retain its measured pre-transport position,
    # with its original timestamp, when the ring later occludes its cap.
    peg=api.observe('peg')
    for _ in range(10):
        ring=api.observe('ring');robot=api.robot.current()
        center=np.array(ring['hole']['center_m']);top=np.array(peg['top_center_m'])
        clearance=min(ring['minimum_z_m'],np.asarray(robot['jaw_vertices_m'])[:,2].min())
        delta=np.r_[top[:2]-center[:2],top[2]+.008-clearance]
        if np.linalg.norm(delta[:2])<.003 and .003 <= clearance-top[2] <= .016:
            opening=min(1.75,skills.last_grasp_opening+.5)
            skills.execute(dict(action='release',jaw=opening))
            return
        if delta[2]>.004:delta[:2]=0.
        target=np.asarray(robot['fixed_pad_m'])+delta
        outcome=skills.execute(dict(action='move_pad',point_m=target.tolist(),roll=float(robot['roll']),seconds=1.5))
        if outcome['status']!='motion_completed':return


def run(output,seed,arm,prior,*,scripted=False,budget=200.,kind='development'):
    if arm not in ('basic','symbolic'):raise ValueError('Unknown arm')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    from online_ring_evidence import snapshot
    snapshot(output)
    prior_copy=output/'skill_prior.py';prior_copy.write_bytes(Path(prior).read_bytes())
    prior=prior_copy
    result=dict(seed=seed,arm=arm,budget_s=budget,kind=kind,scripted=scripted,
                status='initializing',success=False,skill_prior_sha256=digest(prior))
    env=api=skills=loop=None
    try:
        with installed():
            env=OnlineRingEnv(output/'env',scene_seed=seed);env.reset(seed)
            result['private_initial_state_sha256']=hashlib.sha256(env.data.qpos.tobytes()+env.data.qvel.tobytes()).hexdigest()
            env.start_episode(budget,realtime=not scripted)
            api=OnlineRingSensors(env,output);skills=RingSkills(api,prior)
            write(output/'goal.json',api.task())
            if scripted:
                scripted_check(api,skills)
                env.wait(3.)
                result['status']='scripted_incomplete'
            else:
                loop=RingLoop(env,api,skills,arm);loop.run()
    except EpisodeSuccess:
        result.update(status='success',success=True,time_to_success_s=env.success_time)
    except TimeoutError:
        result['status']='timeout'
    except Exception as error:
        result.update(status='infrastructure_error',error=type(error).__name__+': '+str(error))
        (output/'error.txt').write_text(traceback.format_exc())
    finally:
        receipts=[] if loop is None else loop.actor.receipts
        incomplete=list(output.glob('context_*/decision_*/attempt_failure.json'))
        result.update(model_requests_completed=len(receipts),model_requests_attempted=0 if loop is None else loop.actor.turn,
                      incomplete_requests=len(incomplete),usage_complete=not incomplete,
                      usage={k:sum(r['usage'].get(k,0) for r in receipts) for k in
                             ('input_tokens','cached_input_tokens','output_tokens','total_tokens')},
                      model_seconds=sum(r['wall_seconds'] for r in receipts),
                      incomplete_model_seconds=sum(json.loads(p.read_text())['wall_seconds'] for p in incomplete),
                      capped_completion_seconds=env.success_time if result['success'] else budget,
                      operations=0 if loop is None else loop.operations,
                      feedback_seconds=0. if loop is None else loop.feedback_seconds,
                      action_seconds=0. if skills is None else skills.action_seconds)
        if api is not None:
            result.update(camera_acquisitions=api.frames,camera_seconds=api.camera_seconds,
                          depth_queries=api.depth_query_count,depth_query_seconds=api.depth_query_seconds)
            write(output/'private_sensor_records.json',api.records)
        if env is not None:
            try:
                if env.episode_started is not None:
                    result.update(timing=env.timing(),private_success_evidence=env.success_evidence)
                result['private_final_status']=env.ring_status()
                write(output/'private_monitor.json',env.monitor_rows)
                env.archive_terminal_physics(output)
                if api is not None:api.camera('side')
            except Exception as error:result['archive_error']=str(error)
            finally:env.close()
        write(output/'result.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,choices=DEVELOPMENT_SEEDS,required=True)
    parser.add_argument('--arm',choices=('basic','symbolic'),required=True)
    parser.add_argument('--scripted',action='store_true')
    args=parser.parse_args()
    prior,provenance=choose_training_prior()
    result=run(args.output,args.seed,args.arm,prior,scripted=args.scripted)
    write(args.output/'skill_provenance.json',provenance)
    print(json.dumps(result,indent=2))
    if result['status']=='infrastructure_error' or result.get('archive_error'):raise SystemExit(1)
