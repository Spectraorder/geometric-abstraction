"""The wrist RPC exposes RGB/calibration, never a synthetic depth channel."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class WristSensorTests(unittest.TestCase):
    def test_camera_contract_and_blackout_through_isolated_worker(self):
        source = '''import numpy as np
def run(api):
    wrist = api.camera('wrist')
    assert set(wrist) == {'rgb', 'intrinsics', 'camera_to_world', 'time'}
    assert wrist['rgb'].shape == (240, 320, 3)
    assert np.isfinite(wrist['intrinsics']).all()
    assert np.isfinite(wrist['camera_to_world']).all()
    side = api.camera('side')
    assert 'depth_m' in side and 'valid_depth' in side
    assert np.max(wrist['rgb']) == 0 and np.max(side['rgb']) == 0
    assert not np.any(side['valid_depth'])
    robot = api.proprioception()
    rotation = robot['tool_rotation']
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
    plan = api.plan_tool(robot['tool_position'], orientation=rotation)
    assert set(plan) == {'joint_angles', 'tool_position', 'tool_rotation', 'position_error'}
    assert plan['position_error'] < .001
    assert api.report('identify', base_center=[0,0,0], target_center=[0,0,0],
                      target_dimensions=[0,0,0]) is None
    assert api.report('plan', role='grasp', position=robot['tool_position'].tolist()) is None
    assert api.proprioception()['time'] == robot['time']
    api.move_tool(robot['tool_position'], seconds=.1, orientation=rotation)
    assert np.linalg.norm(api.proprioception()['tool_position']-robot['tool_position']) < .004
    try:
        api.move_tool(robot['tool_position'], orientation=-rotation)
    except RuntimeError:
        api.note('reflection rejected')
    else:
        raise RuntimeError('reflection accepted as a robot rotation')
    try:
        api.camera('hidden_state')
    except RuntimeError:
        api.note('invalid view rejected')
    else:
        raise RuntimeError('invalid view accepted')
'''
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            candidate = folder / 'policy.py'
            candidate.write_text(source)
            output = folder / 'trial'
            worker_env = dict(os.environ, OPENBLAS_NUM_THREADS='1')
            if sys.platform == 'linux' and not os.environ.get('DISPLAY'):
                worker_env.setdefault('MUJOCO_GL', 'egl')
            result = subprocess.run([
                sys.executable, str(ROOT / 'scripts/run_vision_trial.py'),
                '--candidate', str(candidate), '--output', str(output),
                '--task', 'inventory_repair', '--layout', 'sampled', '--seed', '1501',
                '--sensor-blackout'], capture_output=True, text=True, timeout=90,
                env=worker_env)
            self.assertEqual(result.returncode, 0, result.stderr)
            metrics = json.loads((output / 'result.json').read_text())
            self.assertIsNone(metrics['policy_error'])
            self.assertFalse(metrics['success'])
            self.assertFalse(any(metrics['repair_progress']['verified'].values()))
            with np.load(output / 'camera_000.npz') as wrist:
                self.assertNotIn('depth_m', wrist.files)
            calls = json.loads((output / 'api_calls.json').read_text())
            preview=next(c for c in calls if c['method']=='plan_tool')
            self.assertLess(preview['observation']['position_error'],.001)
            self.assertTrue(preview['completed'])
            feedback=json.loads((output/'actor_feedback.json').read_text())
            self.assertIn('diagnostics',feedback)
            self.assertTrue(feedback['post_trial_progress']['checks'][0]['reasons'])
            self.assertTrue(any(c['params'].get('text') == 'invalid view rejected' for c in calls))
            self.assertTrue(any(c['params'].get('text') == 'reflection rejected' for c in calls))


if __name__ == '__main__':
    unittest.main()
