"""Controlled intervention isolation and paired fixture determinism."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import sys
import unittest
import json
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import online_recovery_fixture as fixture


class FixtureTests(unittest.TestCase):
    def make(self, path):
        names = ['a_cube', 'b_cube', 'c_thin_prism', 'd_cube', 'e_cube']
        env = SimpleNamespace(order=names, reset_count=1,
            data=SimpleNamespace(qpos=np.arange(35, dtype=float), qvel=np.ones(30), time=0.))
        env.model = SimpleNamespace(joint=lambda name: SimpleNamespace(
            qposadr=[names.index(name.split('/')[0])*7], dofadr=[names.index(name.split('/')[0])*6]))
        colors = ['a', 'b', 'c', 'd', 'e']
        skills = SimpleNamespace(order=colors, nodes={key:{'supports': [] if i<2 else colors[:2] if i==2 else [colors[i-1]]}
                                                     for i,key in enumerate(colors)})
        images = []
        def image(name, view):
            images.append(env.data.qpos.copy())
            return path/(name+'.png'), {'time': env.data.time}
        api = SimpleNamespace(output=path, wait=lambda seconds:None, image=image)
        return env, api, skills, images

    def test_intervention_is_single_body_translation_without_reset_or_actor_truth(self):
        with TemporaryDirectory() as folder:
            path=Path(folder);env,api,skills,images=self.make(path)
            old=env.data.qpos.copy()
            with patch.object(fixture, 'drive', return_value={'status':'visually_consistent'}), \
                 patch.object(fixture.mujoco, 'mj_forward'):
                prepared=fixture.RecoveryFixture('foundation')(env,api,skills)
            changed=np.flatnonzero(env.data.qpos != old).tolist()
            self.assertEqual(changed,[8])
            self.assertAlmostEqual(env.data.qpos[8]-old[8],.025)
            self.assertTrue(all(np.array_equal(frame,old) for frame in images))
            self.assertEqual(env.reset_count,1)
            self.assertNotIn('displacement',json.dumps(prepared['public']))
            self.assertNotIn('moved_private',prepared['public'])
            self.assertEqual(len(prepared['public']['completed_prefix']),2)
            self.assertEqual(prepared['scripted_plan'][0]['node'],'b')
            self.assertEqual(json.loads((path/'fixture.json').read_text())['status'],'ready')

    def test_setup_failure_does_not_apply_intervention(self):
        with TemporaryDirectory() as folder:
            env,api,skills,_=self.make(Path(folder));old=env.data.qpos.copy()
            with patch.object(fixture,'drive',return_value={'status':'object_unavailable'}):
                with self.assertRaisesRegex(RuntimeError,'prefix failed'):
                    fixture.RecoveryFixture('foundation')(env,api,skills)
            self.assertTrue(np.array_equal(old,env.data.qpos))
            self.assertEqual(env.reset_count,1)

    def test_two_arms_have_identical_fixture_states(self):
        with TemporaryDirectory() as folder:
            states=[]
            for arm in ('basic','symbolic'):
                path=Path(folder)/arm;path.mkdir()
                env,api,skills,_=self.make(path)
                with patch.object(fixture,'drive',return_value={'status':'visually_consistent'}), \
                     patch.object(fixture.mujoco,'mj_forward'):
                    fixture.RecoveryFixture('foundation')(env,api,skills)
                states.append(fixture.state_digest(env))
            self.assertEqual(states[0],states[1])

if __name__ == '__main__':unittest.main()
