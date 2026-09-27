"""Small visual/robot-only held-point probe, separate from policy learning.

Six development trials: two layouts, cube/beam/cylinder targets, five prescribed
robot offsets. These do not attempt the bridge task and are not task-SR trials.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from replay_held_geometry import robot_pose, predict_center
from run_vision_trial import validate_source

ROOT = Path(__file__).resolve().parents[1]
POLICY = '''
def run(api):
    nodes = api.task()['nodes']
    node = next(n for n in nodes if n['key'] == PROBE_KEY)
    session = Session(api, nodes)
    picked = session.pick(node['color'])
    if picked['status'] != 'lift_visually_confirmed':
        api.note('Probe stopped: lift not visually confirmed')
        return
    fit = session.get(node['color'])
    robot = api.proprioception()
    origin = np.asarray(robot['tool_position'], dtype=float).copy()
    offsets = [[.015,0,0],[-.015,0,0],[0,.015,0],[0,-.015,0],[0,0,.02]]
    for index, delta in enumerate(offsets):
        target = origin + np.asarray(delta, dtype=float)
        plan = api.plan_tool(target)
        if float(plan['position_error']) > .005:
            api.note('Probe stopped: prescribed robot offset is unreachable')
            return
        before = fit
        before_robot = robot
        api.move_tool(target, seconds=1.2, orientation=plan['tool_rotation'])
        robot = api.proprioception()
        predicted = predict_center(before['center'], before_robot, robot)
        record = {'kind':'held_prediction_probe','object':node['color'],'shape':node['shape'],
                  'step':index,'from_time':float(before_robot['time']),'time':float(robot['time']),
                  'condition':'Rigid attachment without slip after confirmed lift; not contact certified.',
                  'predicted_center_m':predicted.tolist(),
                  'source_position_fully_observed':bool(before['position_fully_observed']),
                  'from_tool_position_m':list(before_robot['tool_position']),
                  'to_tool_position_m':list(robot['tool_position']),
                  'visual_observation_available':False}
        try:
            fit = session.observe(node['color'])
        except (ValueError, RuntimeError) as error:
            record['reason'] = str(error)
            api.record_geometry(record)
            return
        residual = np.asarray(fit['center']) - predicted
        record.update(visual_observation_available=True,
                      later_visual_center_m=np.asarray(fit['center']).tolist(),
                      visual_residual_m=residual.tolist(),
                      visual_residual_norm_m=float(np.linalg.norm(residual)),
                      later_position_fully_observed=bool(fit['position_fully_observed']))
        api.record_geometry(record)
'''


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def allow_initial_unknown(source):
    old = """        for color in self.colors:
            self.pnp.observe(color)
"""
    new = """        for color in self.colors:
            try:
                self.pnp.observe(color)
            except (ValueError, RuntimeError) as error:
                self.ledger.unavailable(color, self.time(), str(error))
"""
    if source.count(old) != 1:
        raise ValueError('Expected exactly one initial-observation loop')
    return source.replace(old, new, 1)


def run(output, frozen_library, tolerate_initial_unknown=False):
    output.mkdir(parents=True, exist_ok=False)
    library = output/'geometry_library.py'
    source = frozen_library.read_text()
    if tolerate_initial_unknown:
        source = allow_initial_unknown(source)
    validate_source(source);library.write_text(source)
    seeds = [960001,960002];keys = ['left','beam','cap']
    common = 'import numpy as np\nSession = geometry.session\n'+inspect.getsource(robot_pose)+'\n'+inspect.getsource(predict_center)+'\n'
    for key in keys:
        code = common+'\nPROBE_KEY = '+repr(key)+'\n'+POLICY
        validate_source(code);(output/(key+'_probe.py')).write_text(code)
    manifest = {'scope':__doc__, 'seeds':seeds, 'public_target_keys':keys,
                'tolerate_initial_unknown':tolerate_initial_unknown,
                'original_library_sha256':digest(frozen_library),
                'library_sha256':digest(library), 'driver_sha256':digest(Path(__file__)),
                'prediction_source_sha256':digest(Path(__file__).with_name('replay_held_geometry.py')),
                'policy_sha256':{key:digest(output/(key+'_probe.py')) for key in keys},
                'separate_from_learning':True,'object_truth_read':False,
                'planned_motion_checks':len(seeds)*len(keys)*5}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    shutil.copy2(Path(__file__),output/'driver_snapshot.py')
    env = dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MUJOCO_GL='egl')
    results = [];rows = []
    for key in keys:
        for seed in seeds:
            trial = output/key/f'seed_{seed}'
            command = [sys.executable,str(ROOT/'scripts/run_vision_trial.py'),'--candidate',str(output/(key+'_probe.py')),
                       '--output',str(trial),'--seed',str(seed),'--task','inventory_structure','--structure-recipe','thin_beam_bridge',
                       '--object-count','4','--layout','sampled','--wrist-depth','--record-geometry',
                       '--geometry-library',str(library),'--timeout','120']
            with (output/f'{key}_{seed}.log').open('w') as log:
                subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            records_path = trial/'geometry_records.json'
            records = json.loads(records_path.read_text()) if records_path.exists() else []
            records = [r for r in records if r.get('kind')=='held_prediction_probe']
            calls = json.loads((trial/'api_calls.json').read_text())
            # Verify the asserted no-gripper-command interval from policy-visible logs.
            for row in records:
                assert not any(c['method']=='gripper' and row['from_time']+1e-6 < c['time'] <= row['time']+1e-6 for c in calls)
                rows.append(dict(row,target_key=key,seed=seed))
            # Read only execution error metadata, never evaluator geometry or success.
            execution_error = json.loads((trial/'result.json').read_text())['policy_error']
            results.append({'target_key':key,'seed':seed,'policy_error':execution_error,'recorded_motion_checks':len(records),
                            'visual_observations':sum(r['visual_observation_available'] for r in records),
                            'notes':[c['params']['text'] for c in calls if c['method']=='note']})
            print('Completed '+key+' '+str(seed)+': '+str(len(records))+' checks',flush=True)
    full = [r for r in rows if r['visual_observation_available'] and r['source_position_fully_observed'] and r['later_position_fully_observed']]
    errors = [r['visual_residual_norm_m'] for r in full]
    report = {'scope':manifest['scope'],'tolerate_initial_unknown':tolerate_initial_unknown,
              'planned_motion_checks':manifest['planned_motion_checks'],
              'recorded_motion_checks':len(rows),'fully_observed_pairs':len(full),
              'fully_observed_shapes':sorted({r['shape'] for r in full}),
              'median_visual_residual_m':float(np.median(errors)) if errors else None,
              'max_visual_residual_m':max(errors,default=None),'trials':results,'records':rows,
              'limitations':['Agreement with visual estimates, not simulator truth or certified attachment.',
                             'Two development layouts per shape; no generalization or task-SR claim.',
                             'Prescribed offsets and no injected disturbances; no live policy change.'],
              'artifact_sha256':{str(p.relative_to(output)):digest(p) for p in output.glob('*/seed_*/*.json') if p.name in ('api_calls.json','geometry_records.json')}}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'docs/evaluations'/('held_geometry_development_unknown.json' if tolerate_initial_unknown else 'held_geometry_development.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('records','artifact_sha256')},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--allow-initial-unknown',action='store_true')
    args=parser.parse_args();run(args.output.resolve(),args.library.resolve(),args.allow_initial_unknown)
