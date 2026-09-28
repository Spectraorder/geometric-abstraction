"""Sensor evidence, privileged interface exclusion and training budget checks."""
import json
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import ring_peg_geometry as geometry
from ring_peg_runtime import RingSensors
from run_ring_peg_trial import validate_policy
import run_ring_peg_training as training


class RingPolicyTests(unittest.TestCase):
    def test_observation_only_starter_and_privileged_access_rejection(self):
        validate_policy(training.STARTER)
        for code in ("def run(api):\n    api.env.reset()", "def run(api):\n    api.request('private_ring_state')",
                     "import os\ndef run(api):\n    pass", "def run(api):\n    api.foo = 1"):
            with self.assertRaises(ValueError):validate_policy(code)

    def test_dispatch_rejects_private_methods(self):
        api=object.__new__(RingSensors)
        for method in ('ring_status','private_ring_state','reset','evaluate_on_peg','peg_axis_xy'):
            with self.assertRaisesRegex(ValueError,'Unknown public method'):api.dispatch(method,{})

    def test_sensor_geometry_has_no_scene_or_oracle_dependency(self):
        source=Path(geometry.__file__).read_text()
        for forbidden in ('ring_peg_scene','ring_peg_oracle','private_ring_state','peg_axis_xy','RingPegGeometry'):
            self.assertNotIn(forbidden,source)

    def test_actual_sensor_initialization_and_missing_depth_rejection(self):
        root=training.ROOT/'outputs/ring_peg_dev/sensor_physical_05/seed_1'
        if not root.exists():self.skipTest('Optional development frames not installed')
        frame=dict(np.load(next(root.glob('camera_000*.npz'))))
        table=geometry.inventory_table(frame)
        record=geometry.observe_ring(frame,'yellow',table)
        self.assertGreater(record['hole']['inscribed_radius_m'],.01)
        self.assertGreater(len(record['outer_boundary_m']),2)
        invalid=dict(frame,valid_depth=np.zeros_like(frame['valid_depth']))
        with self.assertRaises(ValueError):geometry.observe_ring(invalid,'yellow',table)

    def test_missing_current_geometry_cannot_be_replaced_by_model(self):
        frame={'rgb':np.zeros((32,32,3),dtype=np.uint8),'depth_m':np.ones((32,32)),
               'valid_depth':np.ones((32,32),bool),'intrinsics':np.eye(3),'camera_to_world':np.eye(4),'time':1.}
        with self.assertRaises(ValueError):geometry.register_ring(frame,'red',np.array([0,0,0]),{})

    def test_training_has_five_new_layouts_and_no_oracle_prompt(self):
        self.assertEqual(len(set(training.SEEDS)),5)
        self.assertFalse(set(training.SEEDS)&{1,2,3})
        self.assertNotIn('ring_peg_oracle.py',training.CONTRACT)
        self.assertNotIn('class PadActuator',training.CONTRACT)


if __name__=='__main__':unittest.main()
