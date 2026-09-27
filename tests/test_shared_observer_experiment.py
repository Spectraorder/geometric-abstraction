"""Variant isolation, matched libraries and explicit frozen representation contrast."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
from test_mixed_scene_contract import RecordedSceneAPI, OBJECTS
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_mixed_scene_experiment as base
import run_shared_observer_experiment as variant
import summarize_shared_observer_experiment as summary


class API(RecordedSceneAPI):
    def task(self):
        return {'nodes': [dict(x, orientation='upright') for x in OBJECTS]}


class SharedObserverVariantTests(unittest.TestCase):
    def test_scoped_hooks_restore_the_original_runner_even_on_error(self):
        before = (base.library, base.sources, base.COMMON, base.BASIC)
        with self.assertRaises(RuntimeError):
            with variant.configured():
                self.assertIs(base.library, variant.library)
                raise RuntimeError('test')
        self.assertEqual((base.library, base.sources, base.COMMON, base.BASIC), before)

    def test_libraries_differ_only_in_projection_switch_and_match_measured_fields(self):
        sources = [variant.library(a) for a in variant.ARMS]
        self.assertEqual(sources[0].replace('SURFACE_ONLY = True', 'SURFACE_ONLY = False'), sources[1])
        fits = []
        descriptors = [{'shape': x['shape'], 'color': x['color']} for x in reversed(OBJECTS)]
        for source in sources:
            namespace = {}; exec(compile(source, '<variant>', 'exec'), namespace)
            session = namespace['GEOMETRY'].session(API(), descriptors)
            fits.append([session.observe(x['color']) for x in OBJECTS])
        for surface, full in zip(*fits):
            for key, value in surface.items():
                np.testing.assert_equal(value, full[key])
            self.assertNotIn('vertices', surface)
            self.assertIn('vertices', full)

    def test_manifest_freezes_adapters_and_describes_the_actual_contrast(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)/'pilot'
            manifest = variant.prepare(root, [11], [21], rounds=1)
            self.assertEqual(manifest['arms'], ['basic','symbolic'])
            self.assertEqual(manifest['arm_labels']['basic'], 'shared surface')
            self.assertTrue(set(variant.EXTRA_SOURCES) <= set(manifest['source_sha256']))
            self.assertFalse(any('differ in observation implementation' in s for s in manifest['limitations']))
            variant.check(root, manifest)
            report = summary.audit(root, allow_partial=True)
            self.assertEqual(report['exact_actor_contexts_verified'], 0)
            self.assertEqual(report['starter_parity_audit'], 'pending')
            with patch.object(variant, 'sources', return_value={}):
                with self.assertRaisesRegex(RuntimeError, 'Implementation changed'):
                    variant.check(root, manifest)
            report = summary.audit(root, allow_partial=True)
            self.assertEqual(report['exact_actor_contexts_verified'], 0)
            self.assertEqual(report['starter_parity_audit'], 'pending')
            self.assertFalse(list(root.glob('*/round_*/invocation.json')))


if __name__ == '__main__':
    unittest.main()
