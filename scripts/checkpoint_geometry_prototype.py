"""Compare explicit sensor-state checkpoints without fusing or predicting poses.

This offline prototype is not part of the frozen policy library. Deltas describe
estimated centers across caller-selected milestones, not certified physical motion.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def compare_checkpoints(before, after):
    for packet in (before, after):
        if packet.get('coordinate_frame') != 'world' or packet.get('units') != 'meters':
            raise ValueError('Require current-state packets in world meters')
    start, finish = float(before['time']), float(after['time'])
    if not np.isfinite([start, finish]).all() or finish < start:
        raise ValueError('Require finite nondecreasing checkpoint times')
    rows = {}
    for name in sorted(set(before['objects']) | set(after['objects'])):
        first, second = before['objects'].get(name), after['objects'].get(name)
        if first is None or second is None or first['status'] != 'available' or second['status'] != 'available':
            rows[name] = {'status': 'unknown', 'reason': 'Both checkpoints need an available visual estimate'}
            continue
        a, b = np.asarray(first['center'], float), np.asarray(second['center'], float)
        if a.shape != (3,) or b.shape != (3,) or not np.isfinite([a, b]).all():
            raise ValueError('Require finite three-dimensional estimated centers')
        delta = b-a
        rows[name] = {'status': 'estimated_change', 'from_center_m': a.tolist(), 'to_center_m': b.tolist(),
                      'estimated_translation_m': delta.tolist(), 'estimated_distance_m': float(np.linalg.norm(delta)),
                      'both_estimates_fully_observed': bool(first.get('position_fully_observed', False)
                                                            and second.get('position_fully_observed', False))}
    return {'kind': 'checkpoint_estimate_difference', 'coordinate_frame': 'world', 'units': 'meters',
            'from_time': start, 'to_time': finish, 'objects': rows,
            'semantics': 'difference_between_estimates_not_certified_physical_motion',
            'poses_fused_or_predicted': False}


def audit(paths, large_change_m=.010, small_internal_change_m=.001):
    events = []; source_hashes = {}; comparisons = 0; available_pairs = 0
    for path in paths:
        source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        records = [row for row in json.loads(path.read_text()) if row.get('kind') == 'planner_state']
        for first, second in zip(records, records[1:]):
            before, after = first['state'], second['state']
            if float(after['time']) <= float(before['time'])+1e-6:
                continue
            change = compare_checkpoints(before, after); comparisons += 1
            for name, value in change['objects'].items():
                if value['status'] != 'estimated_change':
                    continue
                available_pairs += 1
                if value['estimated_distance_m'] < large_change_m:
                    continue
                internal = after['objects'][name].get('change', {})
                vector = internal.get('estimated_translation_m')
                internal_norm = None if vector is None else float(np.linalg.norm(vector))
                events.append({'source': str(path), 'from_label': first['label'], 'to_label': second['label'],
                    'object': name, **value, 'checkpoint_from_time': change['from_time'], 'checkpoint_to_time': change['to_time'],
                    'internal_from_time': internal.get('from_time'), 'internal_to_time': internal.get('to_time'),
                    'internal_estimated_distance_m': internal_norm,
                    'small_internal_change': internal_norm is not None and internal_norm < small_internal_change_m})
    return {'scope': 'Offline sensor-derived training checkpoints; no physical truth, success labels or reserved data read.',
        'thresholds_m': {'large_checkpoint_change': large_change_m, 'small_internal_change': small_internal_change_m},
        'source_sha256': source_hashes, 'checkpoint_pairs': comparisons, 'available_object_pairs': available_pairs,
        'large_estimate_changes': len(events), 'large_changes_with_small_internal_delta': sum(row['small_internal_change'] for row in events),
        'fully_observed_large_changes': sum(row['both_estimates_fully_observed'] for row in events),
        'fully_observed_large_changes_with_small_internal_delta': sum(row['both_estimates_fully_observed'] and row['small_internal_change'] for row in events),
        'events': events,
        'interpretation': 'Internal successive-observation deltas and caller-checkpoint deltas answer different questions. Zero internal change does not imply zero change across an action.',
        'not_demonstrated': ['Certified physical displacement', 'Improved perception accuracy', 'Improved policy success or learning speed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); result = audit(args.records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('checkpoint_pairs','available_object_pairs','large_estimate_changes',
        'large_changes_with_small_internal_delta','fully_observed_large_changes','fully_observed_large_changes_with_small_internal_delta')}, indent=2))
