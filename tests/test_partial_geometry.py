"""Partial evidence cannot become a current pose, stale state, or Base geometry."""
import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from build_partial_geometry_session import library
from build_visual_input_session import library as visual_library


class PartialGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = {}; exec(library(visual_library('symbolic')), cls.ns)
        cls.packet = json.loads((ROOT/'tests/fixtures/partial_cuboid_failure_rgbd.json').read_text())
        with np.load(ROOT/'tests/fixtures/partial_cuboid_failure_rgbd.npz') as data:
            cls.frames = {view: {k[len(view)+1:]: data[k] for k in data.files if k.startswith(view+'_')}
                          for view in ('side', 'wrist')}

    def session(self):
        packet, frames = self.packet, self.frames
        class API:
            def __init__(self): self.calls = []; self.records = []
            def gripper_geometry(self): self.calls.append('robot'); return packet['robot_geometry']
            def camera(self, view): self.calls.append(view); return frames[view]
            def proprioception(self): return {'time': packet['time']}
            def record_geometry(self, row): self.records.append(row)
        session = self.ns['GeometryStepSession'].__new__(self.ns['GeometryStepSession'])
        session.api = API(); session.colors = ['red']; session.ledger = self.ns['MetricScene']()
        session.backend = self.ns['MixedPnPObservation']([dict(color='red', **packet['descriptor'])], correspondence=True)
        return session

    def fail_observation(self, session):
        with self.assertRaisesRegex(ValueError, 'wide hidden position intervals'):
            session.current_observation(self.frames['side'], 'red', self.packet['visual_model'], self.packet['table_plane'])

    def test_real_failure_retains_constraints_without_a_point_pose_or_extra_sensing(self):
        session = self.session(); self.fail_observation(session)
        self.assertEqual(session.api.calls, ['robot', 'wrist'])
        with self.assertRaises(ValueError): session.get('red')
        record = session.checkpoint('red', 'place', action_status='exception: visual pose unavailable')
        self.assertEqual(record['status'], 'unavailable'); self.assertNotIn('geometry', record)
        constraints = record['constraints']
        self.assertFalse(constraints['pose_available']); self.assertNotIn('center', constraints)
        self.assertEqual(constraints['face_constrained_axes'], [True, True, False])
        widths = np.diff(constraints['bounds_m'], axis=1).ravel()
        self.assertAlmostEqual(widths[2], .001973, places=6)
        self.assertLess(len(json.dumps(record, separators=(',', ':'))), 1000)
        self.assertEqual(session.api.calls, ['robot', 'wrist'])

    def test_missing_current_evidence_clears_old_constraints(self):
        session = self.session(); self.fail_observation(session)
        blank = dict(self.frames['side'], valid_depth=np.zeros_like(self.frames['side']['valid_depth']))
        session.api.camera = lambda view: dict(self.frames['wrist'], valid_depth=np.zeros_like(self.frames['wrist']['valid_depth']))
        with self.assertRaises(ValueError):
            session.current_observation(blank, 'red', self.packet['visual_model'], self.packet['table_plane'])
        self.assertIsNone(session.partial_constraints)
        self.assertNotIn('constraints', session.checkpoint('red', 'retry'))

    def test_stale_or_other_target_constraints_never_transfer(self):
        session = self.session(); self.fail_observation(session)
        current = self.ns['current_partial_geometry']; packet = session.partial_constraints
        self.assertIsNone(current(packet, 'blue', packet['time']))
        self.assertIsNone(current(packet, 'red', packet['time']+1))
        self.assertIsNone(current(packet, 'red', float('nan')))
        copy = current(packet, 'red', packet['time']); copy['bounds_m'][0][0] = 99
        self.assertNotEqual(packet['bounds_m'][0][0], 99)

    def test_world_constraint_region_follows_calibration_translation(self):
        packet = self.packet; delta = np.array([.031, -.017, .012])
        frames = []
        for view in ('side', 'wrist'):
            frame = self.ns['filter_robot_visual_geometry'](self.frames[view], packet['robot_geometry'])[0]
            frames.append((view, frame))
        original = self.ns['partial_geometry_packet']('red', frames, packet['descriptor'],
            packet['visual_model'], packet['table_plane'])
        shifted = []
        for view, frame in frames:
            matrix = np.array(frame['camera_to_world'], copy=True)
            matrix[:3, 3] += delta
            shifted.append((view, dict(frame, camera_to_world=matrix)))
        plane = np.array(packet['table_plane'], copy=True)
        plane[2] += delta[2] - plane[:2] @ delta[:2]
        moved = self.ns['partial_geometry_packet']('red', shifted, packet['descriptor'],
            packet['visual_model'], plane)
        def corners(record):
            bounds = np.asarray(record['bounds_m'])
            local = np.array([[bounds[0, i], bounds[1, j], bounds[2, k]]
                              for i in range(2) for j in range(2) for k in range(2)])
            return np.linalg.solve(np.asarray(record['basis_world']), local.T).T
        expected = corners(original) + delta
        actual = corners(moved)
        # Compare regions rather than fitting-axis labels; axis signs/order are
        # not persistent material identities. Allow micrometer text rounding.
        distances = np.linalg.norm(actual[:, None, :] - expected[None, :, :], axis=2)
        self.assertLess(max(distances.min(axis=0).max(), distances.min(axis=1).max()), 5e-6)
        np.testing.assert_allclose(moved['visual_dimensions_m'], original['visual_dimensions_m'], atol=1e-6)
        self.assertFalse(moved['pose_available'])

    def test_visual_base_cannot_query_constraints(self):
        session = self.session(); self.fail_observation(session)
        base = self.ns['VisualInputSession'](session)
        result = base.checkpoint('red', 'place', action_status='exception: visual pose unavailable')
        self.assertEqual(result['status'], 'visual_input_only')
        self.assertNotIn('constraints', result); self.assertFalse(hasattr(base, 'partial_constraints'))
        self.assertFalse(hasattr(base, 'observe')); self.assertFalse(hasattr(base, 'get'))


if __name__ == '__main__': unittest.main()
