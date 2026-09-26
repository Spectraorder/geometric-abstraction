"""Checks for the Linux worker and saved-state Viser replay path."""

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from aspire_so100.playback import load_vision_rollout, load_trajectory_review, RolloutPlayback


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("vision_runner", ROOT / "scripts/run_vision_trial.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class VisionViewerTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "linux", "Linux bwrap worker")
    def test_linux_worker_is_network_isolated(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            with patch.object(RUNNER.shutil, "which", return_value="/usr/bin/bwrap"):
                command = RUNNER.policy_worker_command(directory, directory / "candidate.py")
            self.assertEqual(command[0], "bwrap")
            self.assertIn("--unshare-net", command)
            self.assertIn("--ro-bind", command)
            self.assertIn("--bind", command)
            self.assertEqual(command[-2:], [str(directory / "worker.py"), str(directory / "candidate.py")])

    def test_saved_rollout_is_validated_and_downsampled(self):
        model = mujoco.MjModel.from_xml_string("<mujoco><worldbody><body><freejoint/><geom size='.1'/></body></worldbody></mujoco>")
        state_type = mujoco.mjtState.mjSTATE_INTEGRATION
        width = mujoco.mj_stateSize(model, state_type)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "rollout.npz"
            states = np.zeros((101, width))
            np.savez_compressed(path, states=states, timestep=model.opt.timestep,
                                state_type=int(state_type))
            sampled, actual_type, seconds, playback_dt = load_vision_rollout(Path(tmp), model, fps=50)
            self.assertEqual(actual_type, state_type)
            self.assertEqual(sampled.shape[0], 11)
            self.assertAlmostEqual(seconds, .2)
            self.assertAlmostEqual(playback_dt, .02)
            self.assertAlmostEqual((len(sampled) - 1) * playback_dt, seconds)
            self.assertAlmostEqual(model.opt.timestep, .002)
            with self.assertRaisesRegex(ValueError, "FPS"):
                load_vision_rollout(Path(tmp), model, fps=0)
            np.savez_compressed(path, states=np.zeros((2, width + 1)),
                                timestep=model.opt.timestep, state_type=int(state_type))
            with self.assertRaisesRegex(ValueError, "state size"):
                load_vision_rollout(Path(tmp), model)

    def test_trajectory_review_aligns_calls_with_saved_tool_motion(self):
        model = mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody><body name='robot/link'><joint name='x' type='slide' axis='1 0 0'/>"
            "<geom size='.01'/><site name='tool' pos='0 0 .1'/></body></worldbody></mujoco>")
        data = mujoco.MjData(model)
        state_type = mujoco.mjtState.mjSTATE_INTEGRATION
        states=[]
        for time,x in ((1.,0.),(2.,.1),(3.,.2)):
            data.time=time;data.qpos[0]=x;mujoco.mj_forward(model,data)
            state=np.empty(mujoco.mj_stateSize(model,state_type))
            mujoco.mj_getState(model,data,state,state_type);states.append(state)
        with TemporaryDirectory() as tmp:
            trial=Path(tmp)
            (trial/'api_calls.json').write_text(json.dumps([
                {'time':1.2,'method':'plan_tool','params':{'position':[.1,0,.1]}},
                {'time':1.5,'method':'move_tool','params':{'position':[.2,0,.1],'seconds':1.}},
                {'time':2.5,'method':'gripper','params':{'angle':-.1,'seconds':.5}},
            ]))
            (trial/'result.json').write_text(json.dumps({'policy_sim_seconds':2.0}))
            review=load_trajectory_review(trial,model,data,np.asarray(states),state_type,
                                          model.site('tool').id)
            np.testing.assert_allclose(review['times'],[0.,1.,2.])
            np.testing.assert_allclose(review['positions'][:,0],[0.,.1,.2])
            np.testing.assert_array_equal(review['robot_body_ids'],[1])
            self.assertEqual(review['robot_positions'].shape,(3,1,3))
            self.assertEqual(review['robot_quaternions'].shape,(3,1,4))
            np.testing.assert_allclose(review['robot_positions'][:,0,0],[0.,.1,.2])
            self.assertAlmostEqual(review['plans'][0]['time'],.2)
            self.assertAlmostEqual(review['motions'][0]['time'],.5)
            self.assertEqual(review['events'][-1]['method'],'gripper')
            self.assertEqual(review['policy_end'],2.)
            env=type('Saved',(),{'frames':np.asarray(states),'state_type':state_type})()
            playback=RolloutPlayback(env,review['times'])
            playback.seek(model,data,1.6)
            self.assertEqual(playback.index,2)
            self.assertAlmostEqual(data.qpos[0],.2)
