"""Check region transport and intersection without simulator state."""
import sys
from pathlib import Path
import numpy as np
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from replay_center_regions import transport, compare_regions


def pose(position, rotation):
    return {'tool_position':position, 'tool_rotation':rotation}


def test_transport_preserves_every_corner_in_rotated_basis():
    rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    bounds = np.array([[-.02, .03], [.01, .04], [-.07, -.05]])
    before, after = np.array([.4, -.1, .2]), np.array([-.2, .3, .5])
    basis, moved = transport(np.eye(3), bounds, pose(before, np.eye(3)), pose(after, rotation))
    for bits in np.ndindex(2, 2, 2):
        point = bounds[np.arange(3), bits]
        projected = basis @ (after + rotation @ (point-before))
        assert np.allclose(projected, moved[np.arange(3), bits], atol=1e-12)


def test_region_overlap_and_separation():
    bounds = np.array([[-1., 1.]]*3)
    assert compare_regions(np.eye(3), bounds, np.eye(3), bounds)['overlap_at_solver_tolerance']
    other = bounds.copy(); other[0] += 3
    result = compare_regions(np.eye(3), bounds, np.eye(3), other)
    assert not result['overlap_at_solver_tolerance']
    assert abs(result['minimum_uniform_halfspace_expansion_m'] - .5) < 1e-12


def test_comparison_invariant_to_world_rotation_and_translation():
    q = np.array([[-.001, .002], [-.002, .001], [-.003, .003]])
    r = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    other = q.copy(); other[0] += .006
    expected = compare_regions(np.eye(3), q, r, other)
    first = pose([0, 0, 0], np.eye(3)); last = pose([.1, -.2, .3], r)
    a, b = transport(np.eye(3), q, first, last)
    c, d = transport(r, other, first, last)
    actual = compare_regions(a, b, c, d)
    assert abs(actual['minimum_uniform_halfspace_expansion_m'] - expected['minimum_uniform_halfspace_expansion_m']) < 1e-12


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(test) for test in (
        test_transport_preserves_every_corner_in_rotated_basis,
        test_region_overlap_and_separation,
        test_comparison_invariant_to_world_rotation_and_translation))
