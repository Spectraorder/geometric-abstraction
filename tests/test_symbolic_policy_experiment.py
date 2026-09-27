"""Ensure the experimental treatment cannot silently replace shared execution."""
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_symbolic_policy_experiment import validate_planner


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
