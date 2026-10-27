"""Protect old evidence, paired sensitivity rules, and fail-closed continuation."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import resume_online_extension_gpu as resume
import online_gpu_preflight as gpu


class GPUResumeTests(unittest.TestCase):
    def test_pending_plan_excludes_all_48_attempts_and_rejects_partial_or_gaps(self):
        slots = resume.study.schedule()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for slot in slots[:48]:
                path = resume.study.episode_path(root, slot)
                path.mkdir(parents=True)
                (path / 'result.json').write_text(json.dumps(dict(status='timeout', **slot)))
            completed, pending = resume.pending_slots(root, slots)
            self.assertEqual(completed, slots[:48])
            self.assertEqual(pending, slots[48:])
            self.assertEqual(len(pending), 222)
            path = resume.study.episode_path(root, slots[50])
            path.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'Partial attempt'):
                resume.pending_slots(root, slots)
            (path / 'result.json').write_text(json.dumps(dict(status='success', **slots[50])))
            with self.assertRaisesRegex(ValueError, 'contiguous'):
                resume.pending_slots(root, slots)

    def test_quality_flags_keep_raw_outcomes_but_drop_whole_pair_from_sensitivity(self):
        rows = []
        for seed in (1, 2, 3):
            for arm in ('Base', 'Full'):
                rows.append(dict(task='ring', seed=seed, arm=arm, initial_state_sha256=str(seed),
                                 success=1, capped_seconds=100., max_simulation_lag_seconds=.5))
        rows[1]['max_simulation_lag_seconds'] = 3.3
        flagged = resume.annotate_quality(rows, gpu_ok=True, epoch='before_gpu_repair')
        flagged[2]['gpu_dispatch_query_ok'] = False
        self.assertEqual(len(flagged), 6)
        self.assertEqual(sum(r['success'] for r in flagged), 6)
        self.assertFalse(flagged[1]['timing_valid'])
        valid = resume.valid_pairs(flagged)
        self.assertEqual([(r['seed'], r['arm']) for r in valid], [(3, 'Base'), (3, 'Full')])

    def test_kernel_version_mismatch_rejected_before_loading_private_libraries(self):
        with patch.object(gpu, 'read', return_value={'version': '580.173.02'}), \
             patch.object(Path, 'read_text', return_value='NVRM version: 580.178.04'), \
             patch.object(gpu, 'digest') as hash_file:
            with self.assertRaisesRegex(ValueError, 'Loaded NVIDIA module differs'):
                gpu.environment(Path('manifest.json'))
            hash_file.assert_not_called()

    def test_gpu_failure_stops_before_any_paid_worker(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'resume'
            root.mkdir()
            evidence = Path(d) / 'evidence'
            evidence.mkdir()
            reg = dict(evidence_root=str(evidence), completed_slots=[],
                       pending_slots=resume.study.schedule(), remaining=270)
            (root / 'baseline_records.json').write_text('[]')
            with patch.object(resume, 'check', return_value=reg), \
                 patch.object(resume, 'gpu_check', side_effect=ValueError('GPU unavailable')), \
                 patch.object(resume.subprocess, 'call') as launch:
                with self.assertRaisesRegex(ValueError, 'GPU unavailable'):
                    resume.execute(root)
                launch.assert_not_called()
                self.assertEqual(resume.read(root / 'progress.json')['status'], 'stopped')
                self.assertFalse(resume.read(root / 'STOP.json')['automatic_retry'])

    def test_existing_pending_attempt_prevents_restart(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'resume'
            root.mkdir()
            evidence = Path(d) / 'evidence'
            slots = resume.study.schedule()
            resume.study.episode_path(evidence, slots[48]).mkdir(parents=True)
            reg = dict(evidence_root=str(evidence), completed_slots=slots[:48], pending_slots=slots[48:])
            with patch.object(resume, 'check', return_value=reg), \
                 patch.object(resume.subprocess, 'call') as launch:
                with self.assertRaisesRegex(ValueError, 'already has an attempt'):
                    resume.execute(root)
                launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
