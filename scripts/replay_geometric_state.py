"""Replay recorded sensor frames into text memory; never load evaluator object state."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from aspire_so100.ablation_geometry import observe_cube
from aspire_so100.instance_geometry import inventory_table
from aspire_so100.geometric_state import GeometricState
from aspire_so100.rigid_state import geometry_json
from aspire_so100.symbolic_geometry import symbolic_packet


def replay(trial, output, colors):
    output.mkdir(parents=True, exist_ok=False)
    # Read camera identity/timestamps only. No trace.json, result.json or rollout
    # object qpos enters this observer. It never replays privileged object states.
    calls = json.loads((trial/'api_calls.json').read_text())
    cameras = [c for c in calls if c['method']=='camera']
    memory, models, plane = GeometricState(), {}, None
    records, durations = [], []
    inputs = []
    for index, call in enumerate(cameras):
        path = trial/f'camera_{index:03d}.npz'
        frame = dict(np.load(path))
        view, stamp = call['params'].get('view', 'side'), float(frame['time'])
        if 'depth_m' not in frame:
            continue
        if plane is None:
            if view != 'side':
                continue
            plane = inventory_table(frame)
        start = time.process_time()
        instantaneous, unavailable = {}, {}
        for color in colors:
            try:
                if color not in models and (view != 'side' or stamp != float(cameras[0]['time'])):
                    raise ValueError('No initial tabletop dimension measurement')
                observation = observe_cube(frame, color, model=models.get(color), table_plane=plane)
                models[color] = observation['model']
                instantaneous[color] = observation
                memory.observe(color, observation, stamp, view)
            except ValueError as error:
                unavailable[color] = str(error)
                if color in memory.objects:
                    memory.missing(color, stamp, str(error))
        if not memory.objects:
            continue
        packet = memory.packet(stamp)
        text = memory.text(stamp)
        durations.append(time.process_time()-start)
        records.append({'camera_index': index, 'view': view, 'time': stamp,
                        'instantaneous': geometry_json(instantaneous),
                        'unavailable': unavailable, 'memory': packet, 'text': text})
        inputs.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (output/'states.json').write_text(json.dumps(records, indent=2, allow_nan=False)+'\n')
    if not records:
        raise RuntimeError('No usable visual state records')
    (output/'final_state.txt').write_text(records[-1]['text']+'\n')
    (output/'final_symbolic_state.json').write_text(json.dumps(symbolic_packet(memory), indent=2, allow_nan=False)+'\n')
    counts = {'instantaneous_fits': 0, 'instantaneous_full_translation': 0,
              'remembered_object_records': 0, 'conditional_prediction_records': 0,
              'fresh_full_translation_records': 0}
    for record in records:
        counts['instantaneous_fits'] += len(record['instantaneous'])
        counts['instantaneous_full_translation'] += sum(o['position_fully_observed'] for o in record['instantaneous'].values())
        for obj in record['memory']['objects'].values():
            counts['remembered_object_records'] += obj['status']=='remembered'
            counts['conditional_prediction_records'] += obj['status']=='conditional_prediction'
            counts['fresh_full_translation_records'] += obj['observed_translation_rank']==3
    summary = {'kind': 'sensor_only_temporal_replay_not_a_policy_success_experiment',
               'trial': str(trial), 'frame_count': len(records), 'colors': colors, 'counts': counts,
               'observed_sim_seconds': records[-1]['time']-records[0]['time'],
               'text_characters_median': float(np.median([len(r['text']) for r in records])),
               'fit_and_packet_cpu_ms_median': float(np.median(durations)*1000),
               'timing_scope': 'offline CPU fits plus state and text; excludes acquisition, IO and agent reasoning',
               'prediction_mode': 'disabled; camera-only replay has no assumed grasp attachment',
               'limitations': ['No private truth read; no position accuracy or SR claim.',
                               'Memory assumes unobserved coordinates persist; moving occluded objects can invalidate them.',
                               'Heuristic covariance is not calibrated uncertainty.'],
               'input_sha256': inputs,
               'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
                   Path(__file__), Path('src/aspire_so100/geometric_state.py'),
                   Path('src/aspire_so100/symbolic_geometry.py'), Path('src/aspire_so100/ablation_geometry.py'),
                   Path('src/aspire_so100/cube_geometry.py'), Path('src/aspire_so100/rigid_state.py')]}}
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('input_sha256','source_sha256')},indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--colors', nargs='+', default=['blue','red','green'])
    args = parser.parse_args()
    replay(args.trial, args.output, args.colors)
