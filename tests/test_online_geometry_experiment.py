"""Ensure the experimental treatment cannot silently replace shared execution."""
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_online_geometry_experiment import validate_planner


class FrozenPlannerTests(unittest.TestCase):
    def test_original_adapter_is_allowed(self):
        validate_planner((ROOT/'policies/stack_from_rgbd_pnp_adapter.py').read_text())

    def test_rejects_actuation_replacement_and_tuning(self):
        candidates=[
            'def run(api):\n api.move_tool([0,0,0])',
            'def run(api):\n p=RGBDPickPlace(api,position_tolerance=.02)',
            'def run(api):\n p=RGBDPickPlace(api)\n p.move=lambda x: None',
            'def RGBDPickPlace(api):\n pass\ndef run(api):\n pass',
            'def run(api):\n RGBDPickPlace=lambda x: None',
            'def run(api):\n p=RGBDPickPlace(api)\n p.api.gripper(1)',
            'class Fake:\n pass\ndef run(api):\n pass',
        ]
        for source in candidates:
            with self.subTest(source=source),self.assertRaises(ValueError):validate_planner(source)

    def test_allows_sensor_based_high_level_planning(self):
        validate_planner('''import numpy as np
def run(api):
    pnp=RGBDPickPlace(api)
    color=api.task()['levels'][0]['color']
    observed=pnp.observe(color)
    position=observed['center'].copy()
    position[2]=pnp.table_z(position[:2])
    preview=api.plan_tool(position)
    if preview['position_error']<.003:
        api.note(str(pnp.pick_and_place(color,position)))
''')

    def test_numerical_closure_counter_does_not_modify_executor(self):
        validate_planner("def run(api):\n    attempts=0\n    def count():\n        nonlocal attempts\n        attempts+=1\n    count()\n")

    def test_explicit_larger_budget_and_geometry_namespace(self):
        validate_planner("def run(api):\n    api.note('"+'x'*20000+"')\n")
        with self.assertRaises(ValueError):
            validate_planner("def run(api):\n    geometry=None\n")
        with self.assertRaises(ValueError):
            validate_planner("def run(api):\n    scene=geometry.scene(api.camera(),['blue'])\n    scene.belief.predict('blue',None,0.,'if held')\n")
        validate_planner("def run(api):\n    scene=geometry.scene(api.camera(),['blue'])\n    api.record_geometry(scene.packet())\n")


class FeedbackSelectionTests(unittest.TestCase):
    def test_shows_a_failure_and_success_with_equal_frame_selection(self):
        import json
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from run_online_geometry_experiment import context
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'executor.py').write_text('# frozen executor')
            (root/'starter.py').write_text('def run(api): pass')
            for seed,success in [(1,True),(2,False),(3,False)]:
                d=root/'bootstrap'/('seed_'+str(seed));d.mkdir(parents=True)
                (d/'api_calls.json').write_text('[]')
                (d/'actor_feedback.json').write_text(json.dumps({'success':success,'policy_error':None,'notes':[]}))
            selections=[]
            def sensor(directory):
                selections.append(directory.name)
                return {'observations':[]},[],[]
            manifest={'train_seeds':[1,2,3],'max_prompt_characters':100000}
            for arm in ['raw','symbolic']:
                selections.clear()
                with patch('run_online_geometry_experiment.sensor_context',side_effect=sensor):
                    prompt,images=context(root,arm,1,manifest)
                self.assertEqual(selections,['seed_2','seed_1'])
                self.assertIn('final success=False',prompt)
                self.assertIn('final success=True',prompt)


class CampaignPhaseTests(unittest.TestCase):
    def test_training_freezes_both_before_explicit_reserved_phase(self):
        import json
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from run_online_geometry_experiment import prepare,campaign,digest,write
        calls=[]
        def execute(root,candidate,directory,seeds,manifest,error=None,library=None):
            calls.append(str(directory.relative_to(root)))
            results=[]
            for seed in seeds:
                out=directory/f'seed_{seed}';out.mkdir(parents=True,exist_ok=True)
                value={'seed':seed,'success':True,'policy_error':None,'notes':[],
                       'rollout_executed':True,'failure_stage':None}
                write(out/'actor_feedback.json',value);results.append(value)
            return results
        def propose(directory,prompt,images,manifest):
            directory.mkdir(parents=True)
            (directory/'candidate.py').write_text('def run(api):\n    pass\n')
            write(directory/'generation.json',{'candidate_sha256':digest(directory/'candidate.py')})
            write(directory/'response.json',{'code':(directory/'candidate.py').read_text(),'hypothesis':'test'})
        with TemporaryDirectory() as tmp:
            root=Path(tmp)/'campaign';manifest=prepare(root,1,[1],[2])
            with patch('run_online_geometry_experiment.execute',side_effect=execute),patch('run_online_geometry_experiment.propose',side_effect=propose):
                campaign(root,manifest,'train')
                self.assertFalse(any('heldout' in call for call in calls))
                self.assertTrue(all((root/arm/'frozen.py').exists() for arm in ['raw','symbolic']))
                self.assertEqual(json.loads((root/'progress.json').read_text())['status'],'trained')
                calls.clear()
                campaign(root,manifest,'evaluate')
                self.assertEqual(calls,['raw/heldout','symbolic/heldout'])
                self.assertEqual(json.loads((root/'summary.json').read_text())['status'],'completed')
