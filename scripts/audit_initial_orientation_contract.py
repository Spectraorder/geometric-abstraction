"""Paired sensor replay of the optional initial-orientation descriptor."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SensorReplay:
    def __init__(self, frame):
        self.frame = frame

    def camera(self):
        return self.frame


def audit(root):
    protocol = read(root/'protocol.json')
    rows = []
    for replica in protocol['replicas']:
        directory = root/replica['directory']
        manifest = read(directory/'manifest.json')
        source = directory/'numeric_library.py'
        namespace = {}
        exec(compile(source.read_text(), str(source), 'exec'), namespace)
        for seed in replica['train_seeds']:
            trial = directory/'numeric/revision_000/trials'/f'seed_{seed}'
            # Read only the public goal from result.json, never scene/metrics/poses.
            goal = read(trial/'result.json')['goal']
            beam = next(n for n in goal['nodes'] if n['shape'] == 'thin_prism')
            sensor = trial/'camera_000.npz'
            with np.load(sensor) as saved:
                frame = {k: saved[k].copy() for k in saved.files}
            plane = namespace['RGBDPickPlace'](SensorReplay(frame)).plane
            cases = {}
            for name, descriptor in [('public_orientation', dict(beam)),
                                     ('shape_color_only', {k: beam[k] for k in ('shape','color')})]:
                observer = namespace['MixedPnPObservation']([descriptor], correspondence=True)
                try:
                    measured = observer.observe_views([('side', frame)], beam['color'], None, plane)
                    cases[name] = {'available': True, 'center': np.asarray(measured['center']).tolist(),
                                   'height': float(measured['height']),
                                   'position_fully_observed': bool(measured['position_fully_observed'])}
                except (ValueError, RuntimeError) as error:
                    cases[name] = {'available': False, 'error': str(error)}
            rows.append({'replica': replica['directory'], 'seed': seed, 'beam_color': beam['color'],
                         'public_initial_orientation': beam['orientation'], 'cases': cases,
                         'camera_sha256': digest(sensor), 'library_sha256': digest(source)})
    counts = {name: sum(row['cases'][name]['available'] for row in rows)
              for name in ('public_orientation', 'shape_color_only')}
    return {'scope': 'Same saved R0 training side RGB-D and calibration, same sensor-fitted table, frozen numeric observer. Only descriptor orientation differs. No robot actions or reserved scenes.',
            'limits': 'Unmasked initial side frames; this isolates orientation dispatch, not a full session or side-plus-wrist replay. No simulator object truth or physical success claim.',
            'protocol_sha256': digest(root/'protocol.json'), 'auditor_sha256': digest(Path(__file__)),
            'available_counts': counts, 'pairs': len(rows), 'rows': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.root)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'pairs': report['pairs'], 'available_counts': report['available_counts']}, indent=2))
