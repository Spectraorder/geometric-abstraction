"""Protect evidence, freshness, schema boundaries and the actor transport contract."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
import numpy as np
from unittest.mock import MagicMock, patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from online_compact_geometry import compact_geometry, expand_geometry, compact_prompt, wire, CONTRACT, ENCODING
from online_compact_actor import CompactBoundActor
from online_bound_actor import BoundActor
import run_online_bound_development as development


def geometry():
    return {'time': 2., 'semantics': 'Current sensor evidence', 'objects': [
        {'object': name, 'references': {'center': {
            'type': 'point', 'position_m': [0.12345678901234567, -0., .4],
            'reference': name + '.center', 'coordinate_frame': 'world', 'units': 'meters',
            'observation_time': 1., 'source': 'mixed_rigid_rgbd_observation',
            'position_fully_observed': True, 'inferred_orientation': False}}}
        for name in ('red', 'blue', 'green', 'yellow')]}


def prompt(arm, value=None):
    packet = {'time': 2., 'last_result': {'status': 'unconfirmed'}, 'holding': None}
    if value is not None:
        packet['geometry'] = value
    return 'Interface: ' + arm + '\nCurrent feedback:\n' + json.dumps(packet)


class CompactGeometryTests(unittest.TestCase):
    def test_roundtrip_preserves_precision_and_does_not_mutate_input(self):
        source = geometry(); before = wire(source)
        encoded = compact_geometry(source)
        self.assertEqual(encoded['encoding'], ENCODING)
        self.assertLess(len(wire(encoded)), len(before))
        self.assertEqual(wire(expand_geometry(encoded)), before)
        self.assertEqual(wire(source), before)
        self.assertEqual(encoded['reference_defaults']['observation_time'], 1.)
        self.assertEqual(encoded['geometry']['time'], 2.)

    def test_unavailable_partial_and_unknown_metadata_never_promoted(self):
        source = geometry()
        refs = source['objects'][0]['references']
        refs['top_face'] = {'status': 'unavailable', 'reason': 'Occluded', 'bounds': [1., 2.]}
        refs['bottom_face'] = {'position_fully_observed': False, 'source': 'conditional_shape'}
        refs['center']['future_evidence'] = {'inferred_vertices': [0, 1]}
        encoded = compact_geometry(source)
        self.assertEqual(wire(expand_geometry(encoded)), wire(source))
        for role in ('top_face', 'bottom_face'):
            ref = encoded['geometry']['objects'][0]['references'][role]
            self.assertNotIn('$shared', ref)
            self.assertEqual(ref, refs[role])

    def test_mixed_times_frames_and_evidence_are_retained(self):
        source = geometry()
        other = source['objects'][1]['references']['center']
        other.update(observation_time=1.5, coordinate_frame='camera', inferred_orientation=True)
        encoded = compact_geometry(source)
        self.assertEqual(wire(expand_geometry(encoded)), wire(source))
        for key in ('observation_time', 'coordinate_frame', 'inferred_orientation'):
            self.assertNotIn(key, encoded['reference_defaults'])

    def test_no_observations_and_small_records_fall_back_without_growth(self):
        for source in ({'time': 2., 'objects': []}, {'time': 2., 'objects': [
                {'object': 'red', 'references': {'center': {'status': 'unavailable'}}}]}):
            self.assertEqual(compact_geometry(source), source)
        source = {'time': 2., 'objects': [{'object': 'red', 'references': {
            'center': {'position_fully_observed': True}}}]}
        self.assertEqual(compact_geometry(source), source)

    def test_invalid_envelopes_and_nonfinite_values_rejected(self):
        source = geometry(); source['time'] = float('nan')
        with self.assertRaises(ValueError): compact_geometry(source)
        encoded = compact_geometry(geometry())
        encoded['geometry']['objects'][0]['references']['center']['status'] = 'unavailable'
        with self.assertRaises(ValueError): expand_geometry(encoded)
        with self.assertRaises(ValueError): expand_geometry({'encoding': 'future'})
        source = geometry(); source['objects'][0]['references']['center']['$shared'] = False
        with self.assertRaises(ValueError): compact_geometry(source)

    def test_opt_in_runner_selects_adapter_and_archives_its_sources(self):
        env = MagicMock()
        env.data = SimpleNamespace(qpos=np.zeros(1), qvel=np.zeros(1))
        env.episode_started = None
        env.structure_status.return_value = {}
        api = SimpleNamespace(task=lambda: {}, camera_count=0, camera_seconds=0.,
                              action_seconds=0., depth_query_count=0, depth_query_seconds=0.)
        skills = SimpleNamespace(provenance={})
        loop = SimpleNamespace(actor=None, completed_operations=0, dispatched_operations=0,
                               feedback_seconds=0., run=MagicMock(side_effect=TimeoutError))
        with TemporaryDirectory() as name, \
             patch.object(development, 'ContactCompleteEnv', return_value=env), \
             patch.object(development, 'TargetSensorAPI', return_value=api), \
             patch.object(development, 'ExplicitSpatialSkills', return_value=skills), \
             patch.object(development, 'BoundDecisionLoop', return_value=loop) as factory:
            output = Path(name) / 'episode'
            result = development.run(output, 2028100101, 'symbolic', compact_text=True)
            self.assertIs(factory.call_args.kwargs['actor_factory'], CompactBoundActor)
            self.assertEqual(result['geometry_serialization'], ENCODING)
            sources = json.loads((output / 'sources.json').read_text())
            for filename in ('online_compact_geometry.py', 'online_compact_actor.py'):
                self.assertIn(filename, sources)
                self.assertTrue((output / 'sources' / filename).exists())
            env.start_episode.assert_called_once_with(200., realtime=True)

    def test_actor_preserves_deadline_images_decisions_and_base_payload(self):
        images = ['initial.png']; advance = lambda: None
        with TemporaryDirectory() as name:
            actor = CompactBoundActor(Path(name)); response = {'action': 'wait'}
            with patch.object(BoundActor, 'request', return_value=response) as request:
                for index in range(2):
                    result = actor.request(prompt('symbolic', geometry()), images, slow=True,
                                           advance=advance, deadline=123.)
                    self.assertIs(result, response)
                    sent = request.call_args.args[0]
                    self.assertEqual(CONTRACT in sent, index == 0)
                    self.assertIs(request.call_args.args[1], images)
                    self.assertEqual(request.call_args.kwargs, dict(slow=True, advance=advance, deadline=123.))
                    packet = json.loads(sent.split('Current feedback:\n')[1])
                    self.assertEqual(wire(expand_geometry(packet['geometry'])), wire(geometry()))
                    self.assertEqual(packet['last_result'], {'status': 'unconfirmed'})
                base = prompt('basic')
                actor.request(base, images, slow=True, advance=advance, deadline=123.)
                self.assertEqual(request.call_args.args[0], base)
                with self.assertRaises(ValueError): compact_prompt(prompt('basic', geometry()), explain=True)


if __name__ == '__main__':
    unittest.main()
