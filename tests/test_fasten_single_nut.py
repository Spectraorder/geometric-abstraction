"""Single-nut curriculum isolation, scoring, and replay compatibility."""
import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from aspire_so100.fasten_screws import FastenScrewsEnv, learning_layout
from aspire_so100.playback import recorded_env_for_result

ROOT = Path(__file__).resolve().parents[1]
trial = runpy.run_path(str(ROOT/'scripts/learn_fasten_screws.py'))['trial']


class SingleNutTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.env = FastenScrewsEnv(self.root/'env', pair_count=1)
        self.env.reset(5101)

    def tearDown(self):
        self.env.close()
        self.tmp.cleanup()

    def test_inventory_goal_and_seeded_layout(self):
        e = self.env
        self.assertEqual(set(e.objects), {'red_nut', 'red_screw'})
        self.assertEqual(set(e.nuts), {'red'})
        self.assertEqual(e.scene_metadata['object_count'], 2)
        self.assertEqual(e.scene_metadata['pair_count'], 1)
        self.assertEqual(e.public_goal()['pairs'], [{'nut':'red_nut','bolt':'red_screw'}])
        self.assertNotIn('position', str(e.public_goal()))
        self.assertEqual(learning_layout(5101, 1), learning_layout(5101, 1))
        self.assertNotEqual(learning_layout(5101, 1), learning_layout(5102, 1))
        self.assertEqual(len(learning_layout(5101)), 6)
        self.assertEqual(mujoco.mj_name2id(e.model, mujoco.mjtObj.mjOBJ_BODY, 'blue_nut/blue_nut'), -1)
        self.assertFalse(e.evaluate().success)

    def test_one_pair_requires_sustained_release_home_and_no_failure(self):
        e = self.env
        e.data.qpos[e.arm_qpos_addresses] = e.home
        e.data.qpos[e.jaw_address] = .8
        status = dict(valid=True, touching_gripper=False)
        with patch.object(e, 'pair_status', return_value=status):
            e.held_seconds['red'] = 1.99
            self.assertFalse(e.evaluate().success)
            e.held_seconds['red'] = 2.
            self.assertTrue(e.evaluate().success)
            self.assertEqual(e.evaluate().score, 20)
            e.data.qpos[e.arm_qpos_addresses[0]] += .2
            self.assertFalse(e.evaluate().success)
            e.data.qpos[e.arm_qpos_addresses] = e.home
            e.data.qpos[e.jaw_address] = .2
            self.assertFalse(e.evaluate().success)
            e.data.qpos[e.jaw_address] = .8
            status['touching_gripper'] = True
            self.assertFalse(e.evaluate().success)
            status['touching_gripper'] = False
            e.failure = 'policy failed'
            self.assertFalse(e.evaluate().success)

    def test_replay_reconstructs_single_pair_and_historical_default(self):
        for count in (1, 3):
            original = FastenScrewsEnv(self.root/f'original_{count}', pair_count=count)
            try:
                result = dict(task='fasten_screws', seed=5101, scene=original.scene_metadata)
                replay = recorded_env_for_result(result, self.root/f'replay_{count}')
                try:
                    self.assertEqual(replay.objects, original.objects)
                    self.assertEqual(replay.model.nq, original.model.nq)
                    self.assertEqual(replay.scene_metadata, original.scene_metadata)
                    np.testing.assert_array_equal(replay.model.qpos0, original.model.qpos0)
                finally:
                    replay.close()
                if count == 3:
                    self.assertNotIn('pair_count', original.scene_metadata)
            finally:
                original.close()

    def test_cached_three_pair_trial_cannot_be_reused(self):
        import hashlib
        candidate = self.root/'candidate.py'
        candidate.write_text('def run(api): pass\n')
        folder = self.root/'cached'
        folder.mkdir()
        scene = dict(self.env.scene_metadata)
        scene.pop('pair_count')
        (folder/'result.json').write_text(json.dumps(dict(
            candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
            seed=5101, task='fasten_screws', scene=scene)))
        with self.assertRaisesRegex(RuntimeError, 'does not match'):
            trial(candidate, folder, 5101, 300, pair_count=1)


if __name__ == '__main__':
    unittest.main()
