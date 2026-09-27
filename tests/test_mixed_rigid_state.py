"""Measured-dimension support goals and shape-specific geometry error semantics."""
import unittest
import numpy as np
from aspire_so100.structure_tasks import structure_recipe
from aspire_so100.mixed_rigid_state import graph_targets, shape_target, mixed_state_error, mixed_support_relation


class MixedRigidStateTests(unittest.TestCase):
    def setUp(self):
        self.nodes = structure_recipe('thin_beam_bridge').public_goal()['nodes']
        self.models = {'blue': {'dims': [.03, .03], 'height': .03},
                       'yellow': {'dims': [.03, .03], 'height': .03},
                       'green': {'dims': [.06, .03], 'height': .015},
                       'red': {'dims': [.03, .03], 'height': .06, 'radius': .015}}

    def test_bridge_targets_and_support_graph_use_measured_scale(self):
        targets = graph_targets(self.nodes, self.models, [0., 0., .4], [.02, -.2], .03)
        np.testing.assert_allclose(targets['blue']['center'], [-.01, -.2, .415])
        np.testing.assert_allclose(targets['yellow']['center'], [.05, -.2, .415])
        np.testing.assert_allclose(targets['green']['center'], [.02, -.2, .4375])
        np.testing.assert_allclose(targets['red']['center'], [.02, -.2, .475])
        self.assertEqual(targets['green']['support'], ['blue', 'yellow'])
        self.assertEqual(targets['red']['support'], ['green'])
        self.assertEqual(len(targets['green']['vertices']), 8)
        self.assertEqual(targets['red']['vertices_kind'], 'sampled_model_rims_not_physical_vertices')
        self.models['blue']['height'] = .04
        changed = graph_targets(self.nodes, self.models, [0., 0., .4], [.02, -.2], .04)
        self.assertAlmostEqual(changed['yellow']['center'][0]-changed['blue']['center'][0], .08)
        self.assertAlmostEqual(changed['green']['center'][2], .4475)

    def test_same_color_is_explicitly_out_of_scope(self):
        self.nodes[1]['color'] = self.nodes[0]['color']
        with self.assertRaises(ValueError):
            graph_targets(self.nodes, self.models, [0., 0., .4], [0., 0.], .03)

    def test_cylinder_yaw_is_not_an_alignment_error(self):
        target = shape_target('cylinder', [0., 0., .4], [.03, .03, .06])
        observed = dict(target, geometry_reliable=True, inferred_orientation=False, model={'radius': .015})
        angle = .71
        observed['rotation'] = np.array([[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
        error = mixed_state_error(observed, target)
        self.assertAlmostEqual(error['rotation_error_rad'], 0.)
        self.assertNotIn('vertex_rms_m', error)
        observed['normal'] = np.array([0., np.sin(.2), np.cos(.2)])
        self.assertAlmostEqual(mixed_state_error(observed, target)['rotation_error_rad'], .2)

    def test_beam_axis_swap_is_not_a_cube_symmetry(self):
        target = shape_target('thin_prism', [0., 0., .4], [.06, .03, .015])
        observed = dict(target, geometry_reliable=True, inferred_orientation=False)
        observed['rotation'] = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        self.assertAlmostEqual(mixed_state_error(observed, target)['rotation_error_rad'], np.pi/2)

    def test_two_support_overlaps_and_cylinder_gap(self):
        targets = graph_targets(self.nodes, self.models, [0., 0., .4], [0., 0.], .03)
        for value in targets.values():
            value['model'] = {'radius': .015}
        for color in ('blue', 'yellow'):
            relation = mixed_support_relation(targets['green'], targets[color])
            self.assertAlmostEqual(relation['signed_surface_gap_m'], 0.)
            self.assertGreater(relation['projected_overlap_area_m2'], 0.)
        cap = mixed_support_relation(targets['red'], targets['green'])
        self.assertAlmostEqual(cap['signed_surface_gap_m'], 0.)
        self.assertFalse(cap['contact_or_stability_certified'])


if __name__ == '__main__':
    unittest.main()
