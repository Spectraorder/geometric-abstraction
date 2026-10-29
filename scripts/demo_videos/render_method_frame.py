"""Re-render the method figure's panel (a) side-camera frame at high resolution.

Rebuilds the three-cube scene for a seed at reset and captures the calibrated side
RGB-D camera (the same camera and intrinsics model as the policy sensor) at a larger
size. The result is a calibrated RGB-D archive for render_agentic_geometry_method.py.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]

import numpy as np  # noqa: E402
import run_three_cube_trial as three  # noqa: E402
import run_vision_trial as trial  # noqa: E402
from aspire_so100 import inventory_structure  # noqa: E402
from aspire_so100.playback import RecordedInventoryStructureEnv  # noqa: E402
from aspire_so100.rgbd import OverheadRGBD  # noqa: E402
from aspire_so100.tower_tasks import RECIPES  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=2027100302)
    p.add_argument('--color-map', type=json.loads, required=True, help='Four-color bijection, as in the study')
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=960)
    p.add_argument('--crop', type=int, nargs=4, metavar=('X0', 'Y0', 'W', 'H'),
                   help='Crop pixels; intrinsics are shifted so the archive stays calibrated')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    with patch.object(trial, 'STRUCTURE_RECIPES', {three.RECIPE: RECIPES[three.RECIPE]}), \
            patch.object(trial, 'structure_recipe', three.cube_recipe), \
            patch.object(inventory_structure, 'structure_recipe', three.cube_recipe), \
            tempfile.TemporaryDirectory() as temp:
        env = RecordedInventoryStructureEnv(Path(temp), scene_seed=a.seed, object_count=3,
                                            recipe=three.RECIPE, color_map=a.color_map)
        try:
            three.apply_physics(env.model, {'noslip_iterations': 0})
            env.reset(a.seed, layout='sampled')
            env.model.vis.global_.offwidth = max(env.model.vis.global_.offwidth, a.width)
            env.model.vis.global_.offheight = max(env.model.vis.global_.offheight, a.height)
            env.model.vis.quality.offsamples = 8
            sensor = OverheadRGBD(env.model, width=a.width, height=a.height)
            frame = sensor.capture(env.data)
            arrays = {k: np.asarray(v) for k, v in frame.items()}
            arrays.setdefault('time', np.asarray(float(env.data.time)))
            if a.crop:
                x0, y0, w, h = a.crop
                for key in ('rgb', 'depth_m', 'valid_depth'):
                    arrays[key] = arrays[key][y0:y0 + h, x0:x0 + w].copy()
                arrays['intrinsics'] = arrays['intrinsics'].copy()
                arrays['intrinsics'][0, 2] -= x0
                arrays['intrinsics'][1, 2] -= y0
            np.savez_compressed(a.output, seed=a.seed, **arrays)
            sensor.close()
            print(json.dumps({'output': str(a.output), 'keys': sorted(frame), 'shape': list(frame['rgb'].shape)}))
        finally:
            env.close()


if __name__ == '__main__':
    main()
