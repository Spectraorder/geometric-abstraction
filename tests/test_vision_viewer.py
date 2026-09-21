"""Checks for the Linux worker and saved-state Viser replay path."""

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from aspire_so100.playback import load_vision_rollout


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
