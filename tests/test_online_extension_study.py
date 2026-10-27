"""Guard the authorized sample extension, pairing, and failure-stop behavior."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_online_extension_study as study


class ExtensionTests(unittest.TestCase):
    def test_only_new_pairs_are_scheduled_and_global_batches_are_balanced(self):
        slots = study.schedule()
        study.validate_schedule(slots)
        self.assertEqual(len(slots), 270)
        for task in study.STUDIES:
            rows = [s for s in slots if s['task'] == task]
            self.assertEqual(len({s['seed'] for s in rows}), 45)
            self.assertEqual(sum(s['arm'] == 'basic' for s in rows), 45)
            old = study.read(study.ROOT / 'outputs' / study.STUDIES[task] / 'protocol.json')
            self.assertTrue(set(old['seeds']).isdisjoint(s['seed'] for s in rows))
        with self.assertRaises(ValueError):
            study.validate_schedule(slots + [slots[0]])

    def test_existing_partial_attempt_or_stop_prevents_dispatch(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            slots = study.schedule()
            study.ensure_unstarted(root, slots)
            path = study.episode_path(root, slots[3])
            path.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'Existing attempts'):
                study.ensure_unstarted(root, slots)
            path.rmdir()
            (root / 'STOP.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Stopped campaign'):
                study.ensure_unstarted(root, slots)

    def test_one_worker_failure_stops_before_next_batch(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with patch.object(study, 'check', return_value={'slots': study.schedule()}), \
                 patch('evaluate_geometry_learning_curves.hardware', return_value={}), \
                 patch.object(study.subprocess, 'call', return_value=1) as launch, \
                 patch.object(study, 'audit_batch') as audit:
                with self.assertRaisesRegex(ValueError, 'Worker failed'):
                    study.execute(root)
                self.assertEqual(launch.call_count, 4)
                audit.assert_not_called()
                self.assertFalse(study.read(root / 'STOP.json')['automatic_retry'])
                self.assertEqual(study.read(root / 'progress.json')['status'], 'stopped')

    def test_pairing_rejects_missing_duplicate_or_different_initial_states(self):
        base = dict(task='tower', seed=1, arm='Base', success=0, capped_seconds=200.,
                    initial_state_sha256='same')
        full = dict(base, arm='Full', success=1, capped_seconds=100.)
        self.assertEqual(study.paired([base, full])[0]['full_minus_base_seconds'], -100.)
        for rows in ([base], [base, full, full], [base, dict(full, initial_state_sha256='other')]):
            with self.assertRaises(ValueError):
                study.paired(rows)

    def test_summary_includes_timeouts_and_preserves_unknown_usage(self):
        rows = []
        for task in study.STUDIES:
            for arm in ('Base', 'Full'):
                for seed in (1, 2):
                    rows.append(dict(task=task, arm=arm, seed=seed, success=int(seed == 1),
                        capped_seconds=100. if seed == 1 else 200., total_tokens=1000,
                        all_model_seconds=30., completed_requests=2, images=1,
                        camera_acquisitions=5, action_seconds=50., usage_complete=seed == 1,
                        initial_state_sha256=str(seed)))
        summary = study.summaries(rows)
        for row in summary['arms']:
            self.assertEqual(row['episodes'], 2)
            self.assertEqual(row['mean_capped_seconds'], 150.)
            self.assertEqual(row['successes'], 1)
            self.assertFalse(row['usage_complete'])
        for row in summary['paired_differences']:
            self.assertEqual(row['ci95'], [0., 0.])


if __name__ == '__main__':
    unittest.main()
