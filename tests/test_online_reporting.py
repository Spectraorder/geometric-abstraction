"""Offline reporting invariants, independent of private experiment archives."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import export_online_extension_results as cli
from online_reporting import evidence
from online_reporting.analysis import ARMS, TASKS, aggregate, cohort_rows, curves_for


def episodes():
    """One successful layout and one timeout for each task and condition."""
    return [dict(task=task, seed=seed, arm=arm, cohort='new_45',
                 initial_state_sha256=f'{task}/{seed}', success=int(seed == 1),
                 success_seconds=100. if seed == 1 else None,
                 capped_seconds=100. if seed == 1 else 200.,
                 total_tokens=1000, receipt_total_tokens=1000, usage_complete=seed == 1,
                 completed_requests=1, incomplete_requests=int(seed == 2),
                 all_model_seconds=10., images=2, camera_acquisitions=3,
                 action_seconds=40., pair_in_quality_sensitivity=True)
            for task in TASKS for seed in (1, 2) for arm in ARMS]


class AnalysisTests(unittest.TestCase):
    def test_archive_sensitivity_excludes_both_arms(self):
        rows = episodes()
        rows[0]['archive_recovery_verified'] = True
        subsets = cohort_rows(rows)
        self.assertEqual(len(subsets['pooled_50']), 12)
        self.assertEqual(len(subsets['sensitivity_valid_pairs_new']), 12)
        excluded = subsets['sensitivity_without_archive_recoveries_new']
        self.assertEqual(len(excluded), 10)
        self.assertFalse(any(r['task'] == 'tower' and r['seed'] == 1 for r in excluded))

    def test_passing_repeats_preserve_historical_exclusions(self):
        rows = episodes()
        for row in rows[:2]:
            row.update(attempt_replaced=True, timing_valid=True,
                       gpu_dispatch_query_ok=True, pair_in_quality_sensitivity=False)
        subsets = cohort_rows(rows)
        self.assertEqual(len(subsets['pooled_50']), 12)
        self.assertEqual(len(subsets['sensitivity_valid_pairs_pooled']), 10)
        self.assertNotIn('original_attempts_50', subsets)

    def test_timeout_never_becomes_success_at_deadline(self):
        curves = curves_for(episodes())
        for task in TASKS:
            for arm in ARMS:
                points = [r for r in curves if r['task'] == task and r['arm'] == arm]
                self.assertEqual([r['budget_seconds'] for r in points], [0., 100., 200.])
                self.assertEqual([r['sr_percent'] for r in points], [0., 50., 50.])

    def test_aggregate_keeps_timeout_costs_and_unknown_usage(self):
        summary = aggregate(episodes())
        for row in summary['arms']:
            self.assertEqual(row['mean_success'], .5)
            self.assertEqual(row['mean_capped_seconds'], 150.)
            self.assertEqual(row['mean_total_tokens'], 1000)
            self.assertFalse(row['usage_complete'])
            self.assertEqual(row['unknown_token_requests'], 1)
        for difference in summary['paired_differences']:
            self.assertEqual(difference['mean'], 0.)
            self.assertEqual(difference['ci95_low'], 0.)
            self.assertEqual(difference['ci95_high'], 0.)


class EvidenceTests(unittest.TestCase):
    def test_collectors_do_not_share_source_ledgers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'record.json').write_text('{"value": 1}')
            with patch.object(evidence, 'ROOT', root):
                first, second = evidence.EvidenceCollector(), evidence.EvidenceCollector()
                self.assertEqual(first.read('record.json'), {'value': 1})
                self.assertEqual(set(first.sources), {'record.json'})
                self.assertEqual(second.sources, {})

    def test_changed_evidence_fails_before_entering_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'record.json'
            source.write_text('{"value": 1}')
            expected = evidence.digest(source)
            source.write_text('{"value": 2}')
            with patch.object(evidence, 'ROOT', root):
                collector = evidence.EvidenceCollector()
                with self.assertRaisesRegex(ValueError, 'Changed evidence'):
                    collector.read('record.json', expected)
                self.assertEqual(collector.sources, {})


class CommandTests(unittest.TestCase):
    def test_check_verifies_without_creating_exports(self):
        frozen = dict(quality=dict(timing_flagged=0, gpu_dispatch_query_failed=0),
                      runtime_change={}, archive_recovery={},
                      infrastructure_overhead=dict(total_recorded_tokens=0))
        usage = dict(unknown_requests=6, recovered_requests=0, recovered_total_tokens=0)
        collected = (frozen, episodes(), [], [], {}, {}, usage, dict(superseded_recorded_tokens=0))
        stream = io.StringIO()
        with patch.object(cli.EvidenceCollector, 'collect', return_value=collected) as collect, \
             patch.object(cli, 'write_csv') as write, \
             patch.object(cli.Path, 'mkdir') as mkdir, redirect_stdout(stream):
            cli.main(['--check'])
        collect.assert_called_once()
        write.assert_not_called()
        mkdir.assert_not_called()
        self.assertIn('no files written', stream.getvalue())
        self.assertEqual(json.JSONDecoder().raw_decode(stream.getvalue())[0]['total_recorded_tokens'], 12000)


if __name__ == '__main__':
    unittest.main()
