"""Clearing targets must avoid both observed blocks and the intended structure."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from online_bridge_recovery import clearing_candidates


class ClearingTests(unittest.TestCase):
    def test_existing_free_location_is_preferred_but_worksite_is_not_a_clearance(self):
        bridge = SimpleNamespace(inside_table=lambda region, xy, half, margin:
            bool(np.all(xy-half-margin >= -.3) and np.all(xy+half+margin <= .3)))
        region = {'bounds': [[-.3, -.3], [.3, .3]]}
        nodes = {'a': {'grid': [0., 0.]}, 'b': {'grid': [1.6, 0.]}}
        fits = {'a': {'center': np.array([-.2, -.2, .015]), 'footprint': np.array([.03, .03])},
                'b': {'center': np.array([.15, .1, .015]), 'footprint': np.array([.03, .03])}}
        points = clearing_candidates(bridge, region, nodes, fits, np.zeros(2), .03, 'a')
        np.testing.assert_allclose(points[0], fits['a']['center'][:2])
        fits['a']['center'][:2] = 0.
        points = clearing_candidates(bridge, region, nodes, fits, np.zeros(2), .03, 'a')
        self.assertTrue(points)
        for point in points:
            self.assertFalse(np.all(np.abs(point) < .0405))
            self.assertFalse(np.all(np.abs(point-np.array([.048, 0.])) < .0405))
            self.assertFalse(np.all(np.abs(point-np.array([.15, .1])) < .0405))


if __name__ == '__main__':
    unittest.main()
