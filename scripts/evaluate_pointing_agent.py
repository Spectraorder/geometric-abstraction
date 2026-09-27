"""Validate saved RGB-only subagent proposals against local camera RGB-D.

No model calls or robot actions. Component support is not localization accuracy.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from aspire_so100.beam_geometry import estimate_flat_beam_geometry
from aspire_so100.pointing_geometry import locate_pointed_component


def indexed(records):
    result = {}
    for record in records:
        key = record['task_id']
        if not isinstance(key, str) or key in result:
            raise ValueError('Task IDs must be unique strings')
        result[key] = record
    return result


def validate_prediction(frame, prediction, color, flat_beam=False):
    status, point = prediction.get('status'), prediction.get('point_px')
    if status == 'abstain' and point is None:
        return {'status': 'abstained'}, None
    if (status != 'point' or not isinstance(point, list) or len(point) != 2
            or any(type(v) not in (int, float) for v in point)):
        return {'status': 'rejected', 'reason': 'Invalid prediction schema'}, None
    try:
        located = locate_pointed_component(frame, point, color)
    except ValueError as error:
        return {'status': 'rejected', 'reason': str(error)}, None
    result = {k: v for k, v in located.items() if k not in ('mask', 'table_plane')}
    result.update(status='color_depth_supported', geometry_check='not_run')
    if flat_beam:
        selected = dict(frame, rgb=np.where(located['mask'][..., None], frame['rgb'], 0),
                        valid_depth=frame['valid_depth'] & located['mask'])
        try:
            fitted = estimate_flat_beam_geometry(selected, color, located['table_plane'])
            result.update(geometry_check='flat_beam_fit_accepted',
                          geometry={k: fitted[k] for k in (
                              'center', 'axis_dimensions', 'vertices', 'geometry_reliable',
                              'inferred_orientation', 'observed_face_count')})
        except ValueError as error:
            result.update(geometry_check='flat_beam_fit_rejected', geometry_reason=str(error))
    return result, located['mask']


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--frames', type=Path, required=True, help='Host-only task-to-RGB-D mapping')
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    tasks, frames, predictions = [indexed(json.loads(path.read_text())) for path in
                                  (args.tasks, args.frames, args.predictions)]
    if not tasks or tasks.keys() != frames.keys() or tasks.keys() != predictions.keys():
        raise ValueError('Expected exactly one frame and prediction for every task')
    args.output.mkdir(parents=True, exist_ok=True)
    panels, rows = [], []
    for key, task in tasks.items():
        entry = frames[key]
        source = Path(entry['frame_path'])
        with np.load(source, allow_pickle=False) as archive:
            frame = {k: archive[k] for k in ('rgb', 'depth_m', 'valid_depth',
                                             'intrinsics', 'camera_to_world')}
        with Image.open(task['image_path']) as rgb:
            pixels = np.array(rgb.convert('RGB'))
        if (not np.array_equal(pixels, frame['rgb'])
                or pixels.shape[:2] != (task['image_height'], task['image_width'])):
            raise ValueError('RGB input and camera frame differ for task ' + key)
        checked, mask = validate_prediction(frame, predictions[key], entry['color'],
                                            entry.get('flat_beam', False))
        rows.append({'task_id': key, 'target': task['target'],
                     'image_size': [task['image_width'], task['image_height']],
                     'source_frame': str(source),
                     'source_frame_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                     'input_png_sha256': hashlib.sha256(Path(task['image_path']).read_bytes()).hexdigest(),
                     'prediction': predictions[key], 'validation': checked})
        # Show the original image and proposed pixel, including rejected proposals.
        im = Image.fromarray(pixels).resize((640, 480), Image.Resampling.NEAREST)
        panel = Image.new('RGB', (640, 548), 'white')
        panel.paste(im, (0, 68))
        draw = ImageDraw.Draw(panel)
        draw.text((8, 6), key + ': ' + task['target'], fill='black')
        draw.text((8, 24), checked['status'] + ' | ' + checked.get('geometry_check', ''), fill='black')
        draw.text((8, 42), 'RGB-only proposal; color/depth support does not verify identity', fill='black')
        point = predictions[key].get('point_px')
        if (isinstance(point, list) and len(point) == 2
                and all(type(v) in (int, float) and np.isfinite(v) for v in point)):
            x, y = point[0] * 2, point[1] * 2 + 68
            if 0 <= x < 640 and 68 <= y < 548:
                draw.ellipse((x-7, y-7, x+7, y+7), outline='magenta', width=2)
                draw.line((x-10, y, x+10, y), fill='magenta')
                draw.line((x, y-10, x, y+10), fill='magenta')
        panels.append(panel)
    gallery = Image.new('RGB', (1280, 548 * ((len(panels)+1)//2)), '#dddddd')
    for i, panel in enumerate(panels):
        gallery.paste(panel, ((i % 2)*640, (i//2)*548))
    gallery.save(args.output / 'gallery.png')
    summary = {name: sum(row['validation']['status'] == name for row in rows)
               for name in ('color_depth_supported', 'abstained', 'rejected')}
    summary['flat_beam_fit_accepted'] = sum(row['validation'].get('geometry_check') == 'flat_beam_fit_accepted' for row in rows)
    summary['flat_beam_fit_rejected'] = sum(row['validation'].get('geometry_check') == 'flat_beam_fit_rejected' for row in rows)
    report = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Selected development images; no robot actions or full-policy evaluation.',
              'runtime_inputs': 'Subagent: RGB and public target text. Host: RGB-D and calibration.',
              'identity_accuracy_measured': False, 'predictions_modified_by_validator': False,
              'summary': summary, 'cases': rows}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, default=jsonable) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
