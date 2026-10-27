"""Archive-only recovery must preserve raw errors and reject changed evidence."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import resume_online_extension_terminal as repair


class RecoveryTests(unittest.TestCase):
    def fixture(self, path):
        path.mkdir(parents=True)
        result = dict(status='success', success=True, archive_error=repair.ARCHIVE_ERROR,
                      budget_s=200., time_to_success_s=65., camera_acquisitions=1,
                      timing=dict(reset_count=1, max_simulation_lag_seconds=.7),
                      private_final_status=dict(on_peg=True),
                      private_success_evidence=dict(end_sim_time=68.))
        repair.write(path / 'result.json', result)
        np.savez(path / 'private_terminal_state.npz', time=68.)
        np.savez(path / 'camera_000.npz', time=2.)
        (path / 'private_model.mjb').write_bytes(b'test-only model')
        repair.write(path / 'private_monitor.json', {})
        return result

    def test_other_errors_or_mismatched_physics_cannot_be_recovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'episode'
            result = self.fixture(path)
            for change in (dict(archive_error='unrelated error'), dict(success=False),
                           dict(time_to_success_s=201.)):
                repair.write(path / 'result.json', dict(result, **change))
                with self.assertRaises(ValueError):
                    repair.validate_recoverable(path)
            repair.write(path / 'result.json', result)
            np.savez(path / 'private_terminal_state.npz', time=69.)
            with self.assertRaises(ValueError):
                repair.validate_recoverable(path)

    def test_derived_view_preserves_raw_result_and_requires_independent_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            evidence, root = Path(tmp) / 'evidence', Path(tmp) / 'repair'
            slot = dict(index=164, task='ring', path='episodes/1/symbolic')
            path = repair.study.episode_path(evidence, slot)
            raw = self.fixture(path)
            original_bytes = (path / 'result.json').read_bytes()
            (evidence / 'sources').mkdir()

            def audit(view, slots, batch):
                derived = repair.study.episode_path(view, slot) / 'result.json'
                data = repair.read(derived)
                self.assertEqual(data, {k: v for k, v in raw.items() if k != 'archive_error'})
                self.assertEqual((derived.parent / 'private_terminal_state.npz').resolve(),
                                 path / 'private_terminal_state.npz')
                return [dict(source=str(derived), source_sha256=repair.digest(derived),
                             max_simulation_lag_seconds=.7)]

            with patch.object(repair.study, 'audit_batch', side_effect=audit) as audited:
                rows = repair.recover_archives(root, evidence, [slot])
            audited.assert_called_once()
            self.assertEqual((path / 'result.json').read_bytes(), original_bytes)
            self.assertEqual(rows[0]['source'], str(path / 'result.json'))
            self.assertTrue(rows[0]['archive_recovery_verified'])
            self.assertEqual(rows[0]['original_archive_error'], repair.ARCHIVE_ERROR)
            self.assertFalse(rows[0]['terminal_image_available'])

    def test_failed_independent_audit_never_produces_verified_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            evidence, root = Path(tmp) / 'evidence', Path(tmp) / 'repair'
            slot = dict(index=164, task='ring', path='episodes/1/symbolic')
            self.fixture(repair.study.episode_path(evidence, slot))
            (evidence / 'sources').mkdir()
            with patch.object(repair.study, 'audit_batch', side_effect=ValueError('audit failed')):
                with self.assertRaisesRegex(ValueError, 'audit failed'):
                    repair.recover_archives(root, evidence, [slot])
            self.assertFalse((root / 'archive_recovery/recovery.json').exists())


if __name__ == '__main__':
    unittest.main()
