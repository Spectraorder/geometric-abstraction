"""Development physics variants preserve defaults, paired control and strict scoring."""
import ast
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from build_polygon_prism_session import library
from validate_polygon_prism_physics import (
    DESTINATION, PhysicsEnv, configure_collision_pipeline, mujoco, physics_settings,
)


class PhysicsDiagnosticTests(unittest.TestCase):
    def test_collision_selection_preserves_other_options_and_flags(self):
        model = mujoco.MjModel.from_xml_string('<mujoco/>')
        model.opt.disableflags = int(mujoco.mjtDisableBit.mjDSBL_GRAVITY)
        before = physics_settings(model)
        configure_collision_pipeline(model, 'scene')
        self.assertEqual(physics_settings(model), before)
        configure_collision_pipeline(model, 'libccd')
        after = physics_settings(model)
        self.assertEqual(after.pop('collision_pipeline'), 'libccd')
        self.assertEqual(after.pop('disableflags'), before['disableflags'] | int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD))
        self.assertEqual(after, {key: value for key, value in before.items()
                                 if key not in ('collision_pipeline', 'disableflags')})
        configure_collision_pipeline(model, 'native')
        self.assertEqual(physics_settings(model), before)
        with self.assertRaises(ValueError):
            configure_collision_pipeline(model, 'unknown')
        self.assertEqual(physics_settings(model), before)

    def test_grasp_override_changes_only_the_existing_default_in_both_arms(self):
        actuators = []
        for arm in ('basic', 'symbolic'):
            original = ast.parse(library(arm))
            changed = ast.parse(library(arm, grasp_fraction=.4))
            actuator = next(node for node in changed.body if isinstance(node, ast.ClassDef) and node.name == 'RGBDPickPlace')
            constructor = next(node for node in actuator.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
            parameters = constructor.args.args[-len(constructor.args.defaults):]
            index = next(i for i, parameter in enumerate(parameters) if parameter.arg == 'grasp_fraction')
            self.assertEqual(ast.literal_eval(constructor.args.defaults[index]), .4)
            actuators.append(ast.dump(actuator))
            constructor.args.defaults[index] = ast.Constant(.7)
            self.assertEqual(ast.dump(original), ast.dump(changed))
        self.assertEqual(*actuators)
        for value in (np.nan, np.inf, -.1, 1.1, '0.4'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                library('symbolic', grasp_fraction=value)

    def test_stationary_position_does_not_hide_velocity_failure(self):
        for angular_speed, linear_speed, expected in ((.02, .001, True), (.11, 0., False), (0., .016, False)):
            with self.subTest(angular=angular_speed, linear=linear_speed):
                env = object.__new__(PhysicsEnv)
                state = {'position': np.array(DESTINATION)+[0., 0., .02], 'bottom': DESTINATION[2],
                         'linear_velocity': np.array([linear_speed, 0., 0.]),
                         'angular_velocity': np.array([0., angular_speed, 0.]),
                         'upright': 1., 'table_contact': True, 'robot_contact': False}
                env.private_samples = [dict(state, bottom=DESTINATION[2]+.04, robot_contact=True)]
                env._step_for = lambda seconds: env.private_samples.extend(dict(state) for _ in range(100))
                env.private_state = lambda: state
                result = env.evaluate_placement()
                self.assertEqual(result['checks']['stable_for_2s'], expected)
                self.assertEqual(result['physical_success'], expected)
                self.assertEqual(result['stability']['maximum_drift_m'], 0.)


if __name__ == '__main__':
    unittest.main()
