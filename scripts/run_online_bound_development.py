"""Run an explicit-target development episode, with no reset after failure."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import traceback

from online_bound_actor import BoundActor
from online_bound_decisions import BoundDecisionLoop, ExplicitSpatialSkills
from online_bound_target_skills import drive_bound
from online_metric_depth import MetricDepthSensorMixin
from online_bridge_scene import EpisodeSuccess
from online_contact_scene import ContactCompleteEnv
from online_bridge_skills import SensorAPI, append, plain

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_BUDGET_S=200.
DEVELOPMENT_SEEDS=tuple(range(2028100101,2028100104))


class TargetSensorAPI(MetricDepthSensorMixin,SensorAPI):
    pass


def write(path,value):
    path.write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n')


def snapshot_sources(output, *, compact_text=False):
    names=('run_online_bound_development.py','online_bound_actor.py','online_bound_decisions.py',
           'online_bound_target_skills.py','online_metric_depth.py','online_bridge_actor.py',
           'online_contact_scene.py','online_bridge_scene.py','online_bridge_skills.py',
           'online_event_skills.py','online_event_memory.py','online_reasoning_state.py',
           'build_online_bridge_geometry.py','verified_token_usage.py')
    if compact_text:
        names += ('online_compact_geometry.py', 'online_compact_actor.py')
    sources={name:ROOT/'scripts'/name for name in names}
    sources.update({'pick_place_rgbd_shared.py':ROOT/'policies/pick_place_rgbd_shared.py',
                    'mixed_scene_session.py':ROOT/'policies/mixed_scene_session.py',
                    'symbolic_scene.py':ROOT/'src/aspire_so100/symbolic_scene.py',
                    'rgbd.py':ROOT/'src/aspire_so100/rgbd.py'})
    (output/'sources').mkdir()
    for name,path in sources.items():
        (output/'sources'/name).write_bytes(path.read_bytes())
    write(output/'sources.json',{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in sources.items()})
    write(output/'source_locations.json',{name:str(path.relative_to(ROOT)) for name,path in sources.items()})


def scripted_check(api,skills,result):
    """Physical API check using a visual driver, not model reasoning evidence."""
    b,session=skills.bridge,skills.session
    fits={key:b.observe(session,api,skills.nodes[key]['color'],attempts=2) for key in skills.order}
    if any(value is None for value in fits.values()):
        raise ValueError('Scripted check needs current source fits')
    region=b.table_region(api)
    if region is None:raise ValueError('No observed table for scripted check')
    origin,_,_=b.select_workspace(api,session,skills.nodes,skills.order,fits,region,skills.scale)
    if origin is None:raise ValueError('No observed worksite for scripted check')
    append(api.output/'scripted_driver.jsonl',{'origin':origin,'source':'shared_visual_driver_not_actor'})
    for key in skills.order:
        node=skills.nodes[key]
        target,_,available=b.current_target(api,session,node,skills.nodes,origin,skills.scale)
        if not available:
            result['status']='scripted_target_unavailable'
            return
        operation=dict(action='execute',skill='stack' if len(node['supports'])==1 else 'bridge',
                       node=key,surface_point_world_m=target.tolist())
        result['dispatched_operations']+=1
        outcome=drive_bound(skills,operation)
        result['completed_operations']+=1
        append(api.output/'scripted_driver.jsonl',{'operation':operation,'outcome':outcome})
        if outcome['status']!='visually_consistent':
            result['status']='scripted_incomplete'
            return
    api.wait(6.)
    result['status']='scripted_incomplete'


def run(output,seed,arm,budget=DEFAULT_BUDGET_S,*,scripted=False,compact_text=False):
    if seed not in DEVELOPMENT_SEEDS or arm not in ('basic','symbolic'):
        raise ValueError('Only the three existing development layouts and known arms are allowed')
    return _run_episode(output,seed,arm,budget,scripted=scripted,compact_text=compact_text)


def _run_episode(output,seed,arm,budget=DEFAULT_BUDGET_S,*,scripted=False,kind='explicit_target_development',compact_text=False):
    """Shared episode body; reserved callers must first verify their registration."""
    if arm not in ('basic','symbolic'):
        raise ValueError('Unknown interface')
    if not math.isfinite(budget) or not 0<budget<=DEFAULT_BUDGET_S:
        raise ValueError('Episode budget must be positive and at most 200 seconds')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    snapshot_sources(output,compact_text=compact_text)
    result=dict(kind=kind,seed=seed,arm=arm,budget_s=budget,scripted=scripted,
                geometry_serialization='shared_reference_metadata_v1' if compact_text else 'reference_records',
                status='initializing',success=False,completed_operations=0,dispatched_operations=0,
                scenario='ordinary_unassembled',intervention=None)
    write(output/'settings.json',result)
    env=api=skills=loop=None
    try:
        env=ContactCompleteEnv(output,scene_seed=seed)
        env.reset(seed)
        result['private_initial_state_sha256']=hashlib.sha256(env.data.qpos.tobytes()+env.data.qvel.tobytes()).hexdigest()
        env.start_episode(budget,realtime=not scripted)
        api=TargetSensorAPI(env,output)
        skills=ExplicitSpatialSkills(api)
        write(output/'goal.json',api.task())
        write(output/'skill_provenance.json',skills.provenance)
        if scripted:
            scripted_check(api,skills,result)
        else:
            actor_factory=BoundActor
            if compact_text:
                from online_compact_actor import CompactBoundActor
                actor_factory=CompactBoundActor
            loop=BoundDecisionLoop(env,api,skills,arm,actor_factory=actor_factory)
            loop.run()
    except EpisodeSuccess:
        result.update(status='success',success=True,time_to_success_s=env.success_time)
    except TimeoutError:
        result['status']='timeout'
    except Exception as error:
        result.update(status='infrastructure_error',error=type(error).__name__+': '+str(error))
        (output/'error.txt').write_text(traceback.format_exc())
    finally:
        actor=None if loop is None else loop.actor
        receipts=[] if actor is None else actor.receipts
        result.update(model_requests_completed=len(receipts),model_requests_attempted=0 if actor is None else actor.turn,
            usage={key:sum(row['usage'].get(key,0) for row in receipts)
                   for key in ('input_tokens','cached_input_tokens','output_tokens','total_tokens')},
            model_seconds=sum(row['wall_seconds'] for row in receipts),
            incomplete_requests=len(list(output.glob('context_*/decision_*/attempt_failure.json'))))
        if loop is not None:
            result.update(completed_operations=loop.completed_operations,dispatched_operations=loop.dispatched_operations,
                          feedback_seconds=loop.feedback_seconds)
        if api is not None:
            result.update(camera_acquisitions=api.camera_count,camera_seconds=api.camera_seconds,
                          action_seconds=api.action_seconds,depth_queries=api.depth_query_count,
                          depth_query_seconds=api.depth_query_seconds)
        result['capped_completion_seconds']=env.success_time if result['success'] else budget
        if env is not None:
            try:
                if env.episode_started is not None:
                    result.update(timing=env.timing(),private_success_evidence=env.success_evidence)
                result.update(private_final_structure=env.structure_status(list(env.order)),
                              private_contact_evaluator='geometry_and_body_ancestry_v1')
                env.archive_terminal_physics(output)
            except Exception as error:
                result['archive_error']=type(error).__name__+': '+str(error)
            finally:
                env.close()
        write(output/'result.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,choices=DEVELOPMENT_SEEDS,required=True)
    parser.add_argument('--arm',choices=('basic','symbolic'),required=True)
    parser.add_argument('--budget',type=float,default=DEFAULT_BUDGET_S)
    parser.add_argument('--scripted',action='store_true')
    parser.add_argument('--compact-text',action='store_true',
                        help='Opt in to lossless reference metadata factoring for development')
    args=parser.parse_args()
    result=run(args.output,args.seed,args.arm,args.budget,scripted=args.scripted,compact_text=args.compact_text)
    print(json.dumps(result,indent=2))
    if result['status']=='infrastructure_error' or result.get('archive_error'):
        raise SystemExit(1)
