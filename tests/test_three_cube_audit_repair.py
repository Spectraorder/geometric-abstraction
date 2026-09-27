"""Regression coverage for the three-cube audit repair and explicit resumption."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'tests'))
import resume_three_cube_geometry_study as repair
import summarize_mixed_scene_learning as mixed
import summarize_visual_input_experiment as visual
import test_mixed_scene_learning_audit as fixtures


class ThreeCubeAuditRepairTests(unittest.TestCase):
    def fixture(self, root, count):
        fixtures.LearningAuditTests().fixture(root, arms=('basic', 'symbolic'))
        manifest = mixed.read(root/'manifest.json')
        manifest['object_count'] = count
        repair.write_json(root/'manifest.json', manifest)
        for path in root.glob('*/**/result.json'):
            result = mixed.read(path)
            result['scene']['object_count'] = count
            repair.write_json(path, result)
        for name in ('summary.json', 'training_summary.json'):
            record = mixed.read(root/name)
            record['manifest_sha256'] = repair.runner.base.digest(root/'manifest.json')
            repair.write_json(root/name, record)

    def test_complete_three_cube_audit_passes_without_changing_legacy_source(self):
        before = repair.runner.base.digest(repair.AUDITOR)
        original = mixed.audit
        alias = visual.base_audit
        with TemporaryDirectory() as folder:
            root = Path(folder)/'fixture'
            self.fixture(root, 3)
            with self.assertRaisesRegex(ValueError, 'Scene mismatch'):
                original(root)
            with repair.installed():
                self.assertIs(mixed.audit, visual.base_audit)
                self.assertEqual(Path(repair.runner.study.__file__), Path(repair.__file__))
                report = mixed.audit(root)
                self.assertEqual(report['valid_rollouts'], 12)
                self.assertEqual(report['global_freeze_audit'], 'passed')
                self.assertEqual(report['auditor_correction']['after'], repair.NEW)
            self.assertIs(mixed.audit, original)
            self.assertIs(visual.base_audit, alias)
        self.assertEqual(repair.runner.base.digest(repair.AUDITOR), before)

    def test_mismatched_count_and_other_provenance_checks_still_fail(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)/'fixture'
            self.fixture(root, 3)
            path = root/'basic/revision_000/trials/seed_91001/result.json'
            original = path.read_text()
            changes = (
                (('scene', 'object_count'), 4, 'Scene mismatch'),
                (('scene', 'color_map'), {}, 'Scene mismatch'),
                (('candidate_sha256',), 'wrong', 'Trial provenance mismatch'),
                (('physics',), {}, 'Simulation settings mismatch'),
                (('success',), 'false', 'literal boolean'),
            )
            for keys, value, message in changes:
                with self.subTest(keys=keys):
                    result = json.loads(original)
                    target = result
                    for key in keys[:-1]:
                        target = target[key]
                    target[keys[-1]] = value
                    repair.write_json(path, result)
                    with self.assertRaisesRegex(ValueError, message):
                        repair.corrected_auditor()(root)
                    path.write_text(original)

    def test_legacy_four_object_records_remain_valid(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)/'fixture'
            self.fixture(root, 4)
            previous = mixed.audit(root)
            corrected = repair.corrected_auditor()(root)
            corrected.pop('auditor_correction')
            self.assertEqual(corrected, previous)

    def test_resume_preserves_original_stop_and_rejects_new_failures(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root/repair.REPAIR).mkdir()
            repair.write_json(root/repair.REPAIR/'registration.json', {})
            repair.write_json(root/'STOP.json', {'error': 'Scene mismatch'})
            original = (root/'STOP.json').read_bytes()
            receipt = {'stop_sha256': repair.runner.base.digest(root/'STOP.json')}
            repair.release_original_stop(root, receipt)
            self.assertFalse((root/'STOP.json').exists())
            self.assertEqual((root/repair.REPAIR/'STOP.resumed.json').read_bytes(), original)
            repair.write_json(root/'STOP.json', {'error': 'Different infrastructure failure'})
            with self.assertRaisesRegex(ValueError, 'New stop requires separate inspection'):
                repair.release_original_stop(root, receipt)
            self.assertTrue((root/'STOP.json').exists())

    def test_incomplete_attempts_are_never_retried(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            stage = root/'replica_01/basic/round_001'
            stage.mkdir(parents=True)
            (stage/'invocation.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Incomplete model request'):
                repair.require_complete_attempts(root)
            (stage/'generation.json').write_text('{}')
            trial = stage/'trials/seed_1'
            trial.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'Incomplete rollout'):
                repair.require_complete_attempts(root)


if __name__ == '__main__':
    unittest.main()
