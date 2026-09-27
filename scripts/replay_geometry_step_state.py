"""Project recorded training action checkpoints into active-object records.

This is an offline interface check, not a new rollout or success-rate comparison.
Only observer state and public task labels are consumed; no held-out trials.
"""
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'policies'))
from geometry_step_state import GeometryStepState


def size(packet):
    return len(json.dumps(packet, separators=(',', ':'), allow_nan=False))


def update(recorder, label, scene):
    row = scene['objects'].get(recorder.object_id)
    fit = None if row is None else dict(
        row, coordinate_frame=scene['coordinate_frame'], units=scene['units'],
        observation_time=row.get('last_observed'))
    desired = scene.get('desired_states', {}).get(recorder.object_id, {})
    if desired.get('center') is not None:
        recorder.set_goal(desired['center'])
    else:
        recorder.clear_goal()
    return recorder.update(label, fit, scene['time'])


def main():
    source = ROOT / 'outputs/shared_observer_learning_pilot_01/symbolic/round_001/trials'
    records, sources = [], []
    for seed in range(910001, 910006):
        trial = source / f'seed_{seed}'
        path = trial / 'geometry_records.json'
        result_path = trial / 'result.json'
        # Read only public task identities from the result envelope.
        nodes = json.loads(result_path.read_text())['goal']['nodes']
        identities = {node['key']: node['color'] for node in nodes}
        sources.append({'seed': seed, 'path': str(path.relative_to(ROOT)),
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'public_node_identities': identities})
        previous_scene, previous_label, recorder = None, None, None
        for entry in json.loads(path.read_text()):
            if entry.get('kind') != 'planner_state':
                continue
            label, scene = entry['label'], entry['state']
            parts = label.split('_')
            if len(parts) >= 2 and parts[0] in identities and parts[1] in ('pick', 'place', 'settled', 'ready'):
                identity = identities[parts[0]]
                if recorder is None or recorder.object_id != identity:
                    recorder = GeometryStepState(identity)
                    if previous_scene is not None:
                        update(recorder, previous_label, previous_scene)
                packet = update(recorder, label, scene)
                records.append({'seed': seed, 'source_label': label,
                                'global_chars': size(scene), 'focused_chars': size(packet),
                                'record': packet})
            previous_scene, previous_label = scene, label
    if not records:
        raise RuntimeError('No training action checkpoints found')
    report = {'kind': 'training_checkpoint_projection', 'live_policy_integration': False,
              'semantics': 'Estimated geometry and changes; no certified contact or stability.',
              'sources': sources, 'checkpoints': len(records),
              'median_global_chars': statistics.median(r['global_chars'] for r in records),
              'median_focused_chars': statistics.median(r['focused_chars'] for r in records),
              'aggregate_char_reduction': 1-sum(r['focused_chars'] for r in records)/sum(r['global_chars'] for r in records),
              'records': records}
    output = ROOT / 'docs/evaluations/geometry_step_state_replay.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('sources', 'records')}, indent=2))
    print(json.dumps(records[0]['record'], indent=2))


if __name__ == '__main__':
    main()
