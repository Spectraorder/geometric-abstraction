"""Rendered metric depth must correspond to the supplied pixel-center rays."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class RGBDCalibrationTests(unittest.TestCase):
    def test_pixel_aligned_matches_an_analytic_plane_without_global_mutation(self):
        # Execute in a fresh process so EGL selection precedes any MuJoCo import.
        code = '''
import mujoco
import numpy as np
from aspire_so100.rgbd import OverheadRGBD
xml = """<mujoco>
  <visual><global offwidth="320" offheight="240"/><quality offsamples="4"/></visual>
  <worldbody>
    <camera name="table/overhead_camera" pos=".2 -.3 .8"
      xyaxes="1 0 0 0 .8660254 .5" fovy="50"/>
    <geom type="plane" size="5 5 .1" rgba=".6 .4 .2 1"/>
  </worldbody>
</mujoco>"""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
errors = {}
for aligned in (False, True):
    sensor = OverheadRGBD(model, pixel_aligned=aligned)
    assert model.vis.quality.offsamples == 4, 'sensor construction mutated model defaults'
    try:
        frame = sensor.capture(data)
        v, u = np.indices(frame['depth_m'].shape)
        rays = np.stack((u, v, np.ones_like(u)), axis=-1) @ np.linalg.inv(frame['intrinsics']).T
        rays = rays @ frame['camera_to_world'][:3, :3].T
        expected = -frame['camera_to_world'][2, 3] / rays[..., 2]
        errors[aligned] = float(np.max(abs(frame['depth_m'] - expected)))
    finally:
        sensor.close()
assert errors[True] < .00001, errors
assert errors[False] > .0001, errors
assert errors[True] < errors[False] / 50, errors
print(errors)
'''
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1')
        if sys.platform == 'linux' and not os.environ.get('DISPLAY'):
            env['MUJOCO_GL'] = 'egl'
        result = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[1],
                                env=env, capture_output=True, text=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
