"""Learning-protocol isolation, equal starting code and evaluation freeze gates."""
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest
from unittest.mock import patch
import json
import numpy as np
from test_rgbd_pnp_base import PixelsOnlyAPI, ROOT
sys.path.insert(0, str(ROOT/'scripts'))
import run_mixed_scene_experiment as experiment


class API(PixelsOnlyAPI):
    def proprioception(self):
        return {'tool_position': self.position.copy(), 'time': float(self.frame['time'])}
    def gripper_geometry(self):
        return {'time': float(self.frame['time']), 'coordinate_frame': 'world', 'units': 'meters',
                'source': 'robot_model_and_joint_encoders', 'visual_spheres': []}


class MixedExperimentTests(unittest.TestCase):
    def test_public_sessions_do_not_expose_mutable_dimension_models(self):
        for arm in experiment.ARMS:
            namespace = {}; exec(compile(experiment.library(arm), '<library>', 'exec'), namespace)
            session = namespace['GEOMETRY'].session(API(), [{'color': 'blue', 'shape': 'cube'}])
            first = session.observe('blue'); original = first['height']
            first['model']['height'] = 10.
            first['center'][:] = 99.
            second = session.observe('blue')
            self.assertAlmostEqual(second['height'], original)
            self.assertLess(np.max(abs(second['center'])), 1.)

    def test_actor_helpers_cannot_replace_functions_inside_the_library(self):
        namespace = {}; exec(compile(experiment.library('numeric'), '<library>', 'exec'), namespace)
        actor = {'geometry': namespace['GEOMETRY']}
        exec('Session = geometry.session\ndef color_mask(*args):\n    raise RuntimeError("replacement")\n', actor)
        session = actor['Session'](API(), [{'color': 'blue', 'shape': 'cube'}])
        self.assertEqual(session.observe('blue')['shape'], 'cube')

    def test_validation_rejects_executor_mutation_and_accepts_starter(self):
        experiment.validate_planner((ROOT/'policies/support_graph_starter.py').read_text())
        for source in ['def run(api):\n    api.move_tool([0,0,0])\n',
                       'def run(api):\n    session = Session(api, [])\n    session.pnp.place([0,0,0])\n',
                       'def run(api):\n    Session = lambda *args: None\n',
                       'def run(api):\n    session = Session(api, [], grasp_fraction=.9)\n',
                       'def run(api):\n    geometry.session = None\n']:
            with self.assertRaises(ValueError): experiment.validate_planner(source)

    def test_scipy_helpers_are_allowed_in_the_isolated_actor_namespace(self):
        experiment.validate_planner('from scipy import ndimage\ndef run(api):\n    mask = ndimage.label([[1, 0]])\n')
        experiment.validate_planner('import scipy.ndimage as ndimage\ndef run(api):\n    mask = ndimage.label([[1, 0]])\n')

    def test_numerical_geometry_helper_cannot_replace_the_isolated_backend(self):
        source = 'def geometry(f):\n    return f["center"]\ndef run(api):\n    return geometry({"center": [1, 2, 3]})\n'
        experiment.validate_planner(source)
        library_namespace = {}; exec(compile(experiment.library('numeric'), '<library>', 'exec'), library_namespace)
        actor = {'geometry': library_namespace['GEOMETRY']}
        exec('Session = geometry.session\n'+source, actor)
        self.assertEqual(actor['run'](None), [1, 2, 3])
        self.assertEqual(actor['Session'](API(), [{'color':'blue','shape':'cube'}]).observe('blue')['shape'], 'cube')
        for forbidden in ['def run(api):\n    return geometry.session(api, [])\n',
                          'def run(api):\n    f = geometry\n    return f.session(api, [])\n']:
            with self.assertRaises(ValueError): experiment.validate_planner(forbidden)

    def test_frozen_library_changes_are_detected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)/'experiment'; manifest = experiment.prepare(root, 1, [91001], [92001])
            experiment.assert_frozen(root, manifest)
            (root/'numeric_library.py').write_text('changed')
            with self.assertRaisesRegex(RuntimeError, 'Library changed'):
                experiment.assert_frozen(root, manifest)

    def test_all_candidates_freeze_before_any_reserved_trial(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)/'experiment'; manifest = experiment.prepare(root, 1, [91001], [92001])
            events = []
            def propose(directory, prompt, images, manifest):
                directory.mkdir(); source = 'def run(api):\n    pass\n'
                (directory/'candidate.py').write_text(source)
                experiment.write(directory/'response.json', {'code': source, 'hypothesis': 'test'})
                experiment.write(directory/'generation.json', {'candidate_sha256': experiment.digest(directory/'candidate.py')})
            def execute(root, arm, candidate, directory, seeds, manifest, error=None):
                if directory.name == 'heldout':
                    self.assertTrue((root/'all_candidates_frozen.json').exists())
                    for name in experiment.ARMS:
                        self.assertTrue((root/name/'frozen.py').exists())
                        for revision in range(2):
                            self.assertTrue((root/name/'frozen_revisions'/f'revision_{revision:03d}.py').exists())
                    events.append(('reserved', arm))
                else: events.append(('train', arm))
                return [{'seed': s, 'success': directory.name == 'heldout' or 'round_001' in str(directory),
                         'policy_error': None} for s in seeds]
            with patch.object(experiment, 'propose', propose), patch.object(experiment, 'execute', execute), patch.object(experiment, 'context', return_value=('', [])):
                experiment.campaign(root, manifest)
            boundary = next(i for i, event in enumerate(events) if event[0] == 'reserved')
            self.assertTrue(all(kind == 'reserved' for kind, arm in events[boundary:]))
            self.assertEqual(len(events[boundary:]), 6)
            report = json.loads((root/'summary.json').read_text())
            self.assertTrue(all(len(report['heldout_curves'][arm]) == 2 for arm in experiment.ARMS))
            self.assertEqual({arm: report['frozen'][arm]['revision'] for arm in experiment.ARMS}, dict.fromkeys(experiment.ARMS, 1))
