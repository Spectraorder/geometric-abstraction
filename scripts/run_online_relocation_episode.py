"""Develop continuous tower relocation without changing frozen online runners."""
import argparse
import json
from pathlib import Path
from online_bridge_skills import append
from online_bound_target_skills import drive_bound
from online_relocation_scene import RelocationEnv
from online_relocation_runtime import RelocationSkills, RelocationLoop, clone
from online_ring_evidence import snapshot
import run_online_bound_development as shared

DEVELOPMENT_SEEDS = (1,2)


def snapshot_sources(output, **kwargs):
    snapshot(output)


def scripted_check(api,skills,result):
    """Use sensor references for a fixed development sequence, never model evidence."""
    api.image('scripted_initial', 'side')
    roles = api.task()['roles']
    session = skills.session
    # Temporary location is a development-driver intent; its Z comes from observed table depth.
    staging = [-.005,-.355,session.table_z([-.005,-.355])]
    for key,support in ((roles['upper'],None),(roles['middle'],roles['destination']),
                        (roles['upper'],roles['middle'])):
        if support is None:
            target = staging
        else:
            session.observe(support)
            target = session.reference(support+'.top_face')['center_m']
        operation = dict(action='execute',skill='stack',node=key,surface_point_world_m=list(target))
        result['dispatched_operations'] += 1
        def boundary(event):
            api.image(f"scripted_{result['dispatched_operations']}_{len(list(api.output.glob('scripted_*_*.png'))):03d}",'side')
            append(api.output/'private_development_motion.jsonl',{'event':event,
                'positions':{r:api.env._object_state(n)['position'] for r,n in api.env.roles.items()}})
            return False
        outcome = drive_bound(skills,operation,boundary)
        api.image(f"scripted_step_{result['dispatched_operations']}",'side')
        result['completed_operations'] += 1
        append(api.output/'scripted_driver.jsonl',{'operation':operation,'outcome':outcome})
        if outcome['status'] != 'visually_consistent':
            result['status'] = 'scripted_incomplete'
            return
    api.wait(6.)
    result['status'] = 'scripted_incomplete'


_episode = clone(shared._run_episode,ContactCompleteEnv=RelocationEnv,
    ExplicitSpatialSkills=RelocationSkills,BoundDecisionLoop=RelocationLoop,
    snapshot_sources=snapshot_sources,scripted_check=scripted_check)


def run(output,seed,arm,*,scripted=False,kind='relocation_development'):
    result = _episode(output,seed,arm,200.,scripted=scripted,kind=kind,compact_text=True)
    result['scenario'] = 'initial_three_cube_column_and_separate_destination'
    result['incomplete_model_seconds'] = sum(json.loads(p.read_text())['wall_seconds'] for p in Path(output).glob('context_*/decision_*/attempt_failure.json'))
    shared.write(Path(output)/'result.json',result)
    settings=json.loads((Path(output)/'settings.json').read_text())
    settings['scenario']=result['scenario']
    shared.write(Path(output)/'settings.json',settings)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,choices=DEVELOPMENT_SEEDS,required=True)
    parser.add_argument('--arm',choices=('basic','symbolic'),default='symbolic')
    parser.add_argument('--scripted',action='store_true')
    args=parser.parse_args()
    result=run(args.output,args.seed,args.arm,scripted=args.scripted)
    print(json.dumps(result,indent=2))
    if result['status']=='infrastructure_error' or result.get('archive_error'):raise SystemExit(1)
