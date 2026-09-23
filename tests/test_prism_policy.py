"""Sensor boundary and parity of the isolated policy's embedded geometry."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

from aspire_so100.cuboid_geometry import color_mask, estimate_cuboid_geometry
from test_cube_geometry import cube_frame
from test_vision_boundary import runner


SOURCE = Path(__file__).resolve().parents[1] / 'policies/stack_prism_upright_visual.py'
spec = importlib.util.spec_from_file_location('prism_policy', SOURCE)
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


class PrismPolicyTests(unittest.TestCase):
    def test_published_source_passes_isolated_worker_validator(self):
        runner.validate_source(SOURCE.read_text())

    def test_embedded_geometry_matches_reusable_module(self):
        dimensions = np.array([.015, .03, .06])
        frame, _ = cube_frame(np.eye(3), half_size=dimensions/2)
        frame['rgb'][frame['rgb'][:, :, 1] == 30] = [160, 30, 210]
        expected = estimate_cuboid_geometry(frame, 'purple', dimensions)
        actual = policy.estimate_cuboid_geometry(frame, 'purple', dimensions)
        for key in ('centroid', 'vertices', 'top_vertices', 'constrained_axes'):
            np.testing.assert_allclose(actual[key], expected[key])
        colors = np.array([[[160, 31, 27], [154, 65, 8], [25, 85, 180], [25, 160, 190]]])
        for name in ('red', 'green', 'blue', 'orange', 'purple', 'cyan'):
            np.testing.assert_array_equal(policy.color_mask(colors, name), color_mask(colors, name))

    def test_missing_pixels_stop_before_any_robot_action(self):
        frame, _ = cube_frame(np.eye(3))
        frame['rgb'][:] = 0
        frame['depth_m'][:] = np.nan
        frame['valid_depth'][:] = False

        class BlackoutAPI:
            # Any proprioception, planning, or motion call fails this test.
            def __init__(self):
                self.notes = []

            def camera(self):
                return frame

            def note(self, text):
                self.notes.append(text)

        api = BlackoutAPI()
        policy.run(api)
        self.assertEqual(api.notes, ['Insufficient tabletop depth.'])
