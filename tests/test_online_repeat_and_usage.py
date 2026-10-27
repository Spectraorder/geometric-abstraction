"""Selected repeats and terminal accounting must not alter original outcomes."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import online_interrupted_usage as usage
import run_online_targeted_repeat as repeat


class RepeatSelectionTests(unittest.TestCase):
    def test_selects_whole_flagged_pair_but_not_ordinary_timeout(self):
        rows = []
        for seed in (1, 2, 3):
            for arm in ('Base', 'Full'):
                rows.append(dict(task='tower', seed=seed, arm=arm, success=0,
                    capped_seconds=200., usage_complete=False, initial_state_sha256=str(seed),
                    source=f'{seed}/{arm}', source_sha256='hash',
                    timing_valid=not (seed == 1 and arm == 'Full'), gpu_dispatch_query_ok=True,
                    archive_recovery_verified=seed == 3))
        selected = repeat.selected_pairs(rows, 'quality')
        self.assertEqual([(r['seed'], r['arm']) for r in selected], [(1, 'Base'), (1, 'Full')])
        self.assertEqual(len(repeat.selected_pairs(rows, 'quality_and_archive')), 4)
        self.assertEqual(len(repeat.selected_pairs(rows, 'interrupted_usage')), 6)
        slots = repeat.make_slots(selected)
        self.assertEqual({r['arm'] for r in slots}, {'basic', 'symbolic'})
        self.assertEqual([r['index'] for r in slots], [0, 1])


class UsageTests(unittest.TestCase):
    def fixture(self, root, completed=True, mismatch=False):
        directory = root / 'context_000/decision_001'
        directory.mkdir(parents=True)
        previous = directory.parent / 'decision_000'
        previous.mkdir()
        session = directory.parent / 'actor_home/sessions/test-session.jsonl'
        session.parent.mkdir(parents=True)
        before = dict(input_tokens=100, cached_input_tokens=40, output_tokens=10, total_tokens=110)
        after = dict(input_tokens=150, cached_input_tokens=60, output_tokens=15, total_tokens=165)
        delta = {k: after[k] - before[k] for k in before}
        count = lambda total, last: dict(type='event_msg', payload=dict(type='token_count', info=dict(
            total_token_usage=total, last_token_usage=last)))
        baseline = [dict(type='session_meta', payload=dict(id='test-session')), count(before, before)]
        encoded = lambda values: ''.join(json.dumps(v) + '\n' for v in values)
        (previous / 'session.jsonl').write_text(encoded(baseline))
        session.write_text(encoded(baseline + [count(after, delta)]))
        (previous / 'receipt.json').write_text(json.dumps(dict(raw_cumulative_usage=before)))
        (directory / 'invocation.json').write_text(json.dumps(dict(session_before='test-session')))
        (directory / 'attempt_failure.json').write_text(json.dumps(dict(type='TimeoutError')))
        cli = [dict(type='thread.started', thread_id='test-session')]
        if completed:
            cli.append(dict(type='turn.completed', usage=dict(after, input_tokens=151) if mismatch else after))
        (directory / 'model.jsonl').write_text(encoded(cli))
        request = dict(task='ring', cohort='new_45', seed=1, arm='Base', context='context_000',
                       decision='decision_001', status='TimeoutError', usage_complete=False,
                       source=str((directory / 'attempt_failure.json').relative_to(root)))
        return request

    def test_verified_delta_changes_tokens_not_outcome_or_completed_decisions(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(usage, 'ROOT', Path(tmp)):
            request = self.fixture(Path(tmp))
            report = usage.audit_interrupted_requests([request])
            self.assertEqual(report['recovered_total_tokens'], 55)
            episode = dict(task='ring', seed=1, arm='Base', success=0, capped_seconds=200.,
                completed_requests=1, incomplete_requests=1, input_tokens=100, cached_input_tokens=40,
                output_tokens=10, total_tokens=110, usage_complete=False)
            usage.apply_recoveries([episode], [request], report)
            self.assertEqual(episode['total_tokens'], 165)
            self.assertEqual(episode['receipt_total_tokens'], 110)
            self.assertEqual((episode['success'], episode['capped_seconds'], episode['completed_requests']), (0, 200., 1))
            self.assertEqual(request['status'], 'TimeoutError')
            self.assertFalse(request['decision_completed_in_episode'])
            self.assertTrue(request['usage_complete'])

    def test_no_cli_completion_is_not_assumed_zero_or_complete(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(usage, 'ROOT', Path(tmp)):
            report = usage.audit_interrupted_requests([self.fixture(Path(tmp), completed=False)])
            self.assertEqual(report['unknown_requests'], 1)
            self.assertIsNone(report['records'][0]['usage'])

    def test_disagreeing_cli_and_session_counters_fail(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(usage, 'ROOT', Path(tmp)):
            request = self.fixture(Path(tmp), mismatch=True)
            with self.assertRaisesRegex(ValueError, 'CLI usage differs'):
                usage.audit_interrupted_requests([request])


if __name__ == '__main__':
    unittest.main()
