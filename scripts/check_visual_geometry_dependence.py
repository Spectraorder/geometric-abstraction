"""Sensor-only metamorphic checks on recorded RGB-D and a frozen geometry library.

Translation and scale are synthetic calibration/depth transformations, not new
physical rollouts. They test sensor dependence, not absolute pose accuracy or SR.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FrameAPI:
    def __init__(self, frame):
        self.frame = frame

    def camera(self, view='side'):
        return self.frame


def check(library, fixture):
    namespace = {}
    exec(compile(library.read_text(), str(library), 'exec'), namespace)
    with np.load(fixture) as data:
        original = {key[len('initial_'):]: data[key].copy()
                    for key in data.files if key.startswith('initial_')}
    # Public labels for this recorded fixture; no metric object geometry.
    objects = [{'color': 'blue', 'shape': 'cube'}, {'color': 'yellow', 'shape': 'cube'},
               {'color': 'green', 'shape': 'thin_prism', 'orientation': 'flat_x'},
               {'color': 'red', 'shape': 'cylinder'}]
    offset = np.array([.031, -.017, .012])
    scale = 1.15
    transformed = {name: {k: v.copy() for k,v in original.items()}
                   for name in ('original', 'translated', 'scaled', 'blackout')}
    transformed['translated']['camera_to_world'][:3, 3] += offset
    transformed['scaled']['camera_to_world'][:3, 3] *= scale
    transformed['scaled']['depth_m'] *= scale
    transformed['blackout']['valid_depth'][:] = False
    fits = {}
    for name in ('original', 'translated', 'scaled'):
        frame = transformed[name]
        observer = namespace['MixedPnPObservation'](objects)
        plane = namespace['RGBDPickPlace'](FrameAPI(frame)).plane
        fits[name] = {obj['color']: observer.observe(frame, obj['color'], None, plane)
                      for obj in objects}
    rows = []
    for obj in objects:
        color = obj['color']
        first, moved, resized = [fits[name][color] for name in ('original', 'translated', 'scaled')]
        dims = lambda fit: np.r_[fit['model']['dims'], fit['model']['height']]
        row = {'color': color, 'shape': obj['shape'],
               'translation_center_residual_m': float(np.linalg.norm(moved['center']-first['center']-offset)),
               'translation_dimension_residual_m': float(np.linalg.norm(dims(moved)-dims(first))),
               'scale_center_residual_m': float(np.linalg.norm(resized['center']-scale*first['center'])),
               'scale_dimension_residual_m': float(np.linalg.norm(dims(resized)-scale*dims(first))),
               'original_visual_dimensions_m': dims(first).tolist(),
               'scaled_visual_dimensions_m': dims(resized).tolist()}
        try:
            namespace['MixedPnPObservation'](objects).observe(transformed['blackout'], color, None, np.zeros(3))
        except (ValueError, RuntimeError) as error:
            row['blackout_rejected'] = True
            row['blackout_reason'] = str(error)
        else:
            row['blackout_rejected'] = False
        row['passed'] = (row['blackout_rejected'] and
                         max(row[k] for k in row if k.endswith('_residual_m')) < .001)
        rows.append(row)
    return {'scope': __doc__, 'passed': all(r['passed'] for r in rows),
            'translation_m': offset.tolist(), 'scale': scale, 'residual_tolerance_m': .001,
            'records': rows,
            'source_sha256': {str(p): digest(p) for p in (Path(__file__), library, fixture)},
            'limitations': ['One recorded initial scene, four uniquely colored objects.',
                            'Shape labels, initial on-table orientation assumptions and estimator shape/range priors remain.',
                            'This checks dependence on transformed visual data, not absence of every possible information leak.',
                            'No simulator object state or task outcomes are read. No actor feedback is changed.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite diagnostic evidence')
    result = check(args.library, args.fixture)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','scope')}, indent=2))
    if not result['passed']:
        raise SystemExit(1)
