"""Three-valued geometric predicates distinguish unknown evidence from false."""
import json
import unittest
import numpy as np
from aspire_so100.geometric_state import GeometricState
from aspire_so100.symbolic_geometry import geometric_predicate, symbolic_packet, resolve_reference
from test_geometric_state import fit


class SymbolicGeometryTests(unittest.TestCase):
    def state(self):
        memory = GeometricState()
        memory.observe('blue', fit(), 0.)
        memory.observe('red', fit((0., 0., .445)), 0.)
        return memory

    def test_support_candidate_is_not_contact(self):
        state = self.state()
        self.assertEqual(geometric_predicate(state, 'support_candidate', ['red','blue'])['truth'], 'true')
        self.assertEqual(geometric_predicate(state, 'contact', ['red','blue'])['truth'], 'unknown')
        self.assertFalse(geometric_predicate(state, 'support_candidate', ['red','blue'])['contact_or_stability_certified'])

    def test_partial_observation_can_establish_height_but_not_xy_alignment(self):
        state = self.state()
        for name, z in [('blue', .415), ('red', .445)]:
            state.observe(name, fit((.1, .1, z), axes=(False,False,True)), 1.)
        self.assertEqual(geometric_predicate(state, 'above', ['red','blue'])['truth'], 'true')
        self.assertEqual(geometric_predicate(state, 'aligned_xy', ['red','blue'])['truth'], 'unknown')
        self.assertEqual(geometric_predicate(state, 'support_candidate', ['red','blue'])['truth'], 'unknown')

    def test_observed_large_gap_is_false_not_unknown(self):
        state = self.state()
        state.observe('red', fit((0.,0.,.47)), 0.)
        self.assertEqual(geometric_predicate(state, 'support_candidate', ['red','blue'])['truth'], 'false')

    def test_memory_and_prediction_do_not_certify_support(self):
        state = self.state()
        self.assertEqual(geometric_predicate(state, 'support_candidate', ['red','blue'], time=1.)['truth'], 'unknown')
        state.predict('red', np.eye(4), 1., 'if grasp remains attached')
        self.assertEqual(geometric_predicate(state, 'above', ['red','blue'])['truth'], 'unknown')

    def test_typed_face_and_point_references_resolve_to_geometry(self):
        state = self.state()
        top = resolve_reference(state, 'blue.top_face')
        bottom = resolve_reference(state, 'red.bottom_face')
        self.assertEqual(top['type'], 'plane_patch')
        np.testing.assert_allclose(top['center_m'], bottom['center_m'])
        np.testing.assert_allclose(top['normal'], -bottom['normal'])
        self.assertEqual(top['vertices_m'].shape, (4, 3))
        self.assertEqual(resolve_reference(state, 'red.corner_0')['type'], 'point')
        with self.assertRaisesRegex(ValueError, 'Unknown geometric reference'):
            resolve_reference(state, 'red.original_material_face')

    def test_packet_is_typed_finite_and_explicit(self):
        packet = symbolic_packet(self.state())
        json.dumps(packet, allow_nan=False)
        self.assertEqual(packet['objects']['red']['type'], 'rigid_cube')
        self.assertEqual(packet['objects']['red']['references']['bottom_face'], 'red.bottom_face')
        self.assertEqual({f['truth'] for f in packet['facts']}, {'true', 'false'})
        with self.assertRaisesRegex(ValueError, 'Unknown geometric predicate'):
            geometric_predicate(self.state(), 'on', ['red','blue'])
