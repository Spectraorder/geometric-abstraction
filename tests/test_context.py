"""Offline checks for retained failure evidence and model-call spending guards."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from aspire_so100 import agent_loop
from aspire_so100.context import build_prompt, feedback, read_usage, select_prior
from aspire_so100.agent_loop import BudgetStop, budgeted_call, resolve_codex
from aspire_so100.skill_report import build_report


class ContextTests(unittest.TestCase):
    def test_grouped_failures_keep_distinct_errors_and_complete_code(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = root / 'candidate.py'
            code = 'def run(api):\n    api.wait(.5)\n'
            policy.write_text(code)
            trace = root / 'trace.json'
            trace.write_text(json.dumps({'primitives': [], 'stability_checks': [
                {'stable': False, 'reasons': ['red: tilted']}]}))
            results = [{'seed': n, 'error': None, 'metrics': {'success': False, 'score': 0},
                        'trace': str(trace)} for n in (51, 52)]
            results.append({'seed': 53, 'error': 'trial timeout', 'metrics': {'success': False, 'score': 0}})
            entry = {'candidate': str(policy), 'successes': 0, 'trials': 3,
                     'mean_score': 0, 'results': results}
            context = feedback(entry)
            self.assertEqual(context['code'], code)
            self.assertEqual([g['seeds'] for g in context['outcome_groups']], [[51, 52], [53]])
            prompt = build_prompt('API', [entry], root)
            self.assertIn('red: tilted', prompt)
            self.assertIn('trial timeout', prompt)
            with self.assertRaisesRegex(ValueError, 'No model call'):
                build_prompt('API', [entry], root, max_chars=10)

    def test_recent_failure_survives_selection(self):
        best = {'candidate': 'best.py', 'successes': 2, 'mean_score': .8}
        latest = {'candidate': 'latest.py', 'successes': 0, 'mean_score': 0,
                  'generation_error': 'syntax error'}
        self.assertEqual(select_prior([best, latest]), [best, latest])
        self.assertEqual(select_prior([best]), [best])

    def test_budget_and_unknown_usage_prevent_calls(self):
        args = SimpleNamespace(max_prompt_chars=1000, max_reported_tokens=10)
        with patch('aspire_so100.agent_loop.ask_llm') as llm:
            for prompt, state in [('x' * 1001, {}), ('x', {'llm_calls': [{'usage': None}]}),
                                  ('x', {'llm_calls': [{'usage': {'input_tokens': 9, 'output_tokens': 1}}]})]:
                with self.assertRaises(BudgetStop):
                    budgeted_call(prompt, Path('/unused'), ('code',), args, state)
            llm.assert_not_called()

    def test_call_records_usage_without_double_counting_reasoning(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(output=root, max_prompt_chars=1000, max_reported_tokens=10,
                                   model='requested-model', llm_timeout=1, codex_bin=None)
            state = {}
            def fake_call(prompt, directory, fields, model, timeout, codex_bin, image_paths):
                directory.mkdir()
                (directory / 'agent.jsonl').write_text(json.dumps({'type': 'turn.completed', 'usage': {
                    'input_tokens': 6, 'cached_input_tokens': 2, 'output_tokens': 3,
                    'reasoning_output_tokens': 1}}) + '\n')
                return {'code': 'pass'}
            with patch('aspire_so100.agent_loop.ask_llm', side_effect=fake_call) as llm:
                for i in range(2):
                    budgeted_call('x', root / str(i), ('code',), args, state)
                with self.assertRaises(BudgetStop):
                    budgeted_call('x', root / '2', ('code',), args, state)
                self.assertEqual(llm.call_count, 2)
            self.assertTrue((root / '0/usage.json').exists())
            self.assertEqual(len(json.loads((root / 'campaign.json').read_text())['llm_calls']), 2)

    def test_missing_or_partial_usage_is_unknown(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'events.jsonl'
            self.assertIsNone(read_usage(path))
            path.write_text('{"type":"turn.started"}\n{"unfinished":')
            self.assertIsNone(read_usage(path))
            path.write_text('{"type":"turn.completed","usage":{}}\n')
            self.assertIsNone(read_usage(path))

    def test_codex_override_and_installed_fallback(self):
        with TemporaryDirectory() as tmp:
            executable = Path(tmp) / 'codex'
            executable.write_text('#!/bin/sh\n')
            executable.chmod(0o700)
            self.assertEqual(resolve_codex(str(executable)), executable.resolve())
            with self.assertRaisesRegex(FileNotFoundError, 'not usable'):
                resolve_codex(str(executable) + '-missing')
        self.assertTrue(resolve_codex().is_file())

    def test_continuation_stops_proposals_at_three_of_three(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / 'old.py'
            old.write_text('def run(api):\n    api.wait(.1)\n')
            source = root / 'old_campaign.json'
            source.write_text(json.dumps({'history': [{
                'candidate': str(old), 'hypothesis': 'partial', 'successes': 0,
                'trials': 3, 'mean_score': .3, 'results': []}]}))
            output = root / 'continued'
            args = SimpleNamespace(output=output, continue_from=source, rounds=0,
                                   model='test-model', codex_bin=None, llm_timeout=1,
                                   trial_timeout=1, max_prompt_chars=32000,
                                   max_reported_tokens=1000)
            calls = []
            def fake_model(prompt, directory, fields, model, timeout, codex_bin, image_paths):
                calls.append(fields)
                directory.mkdir(parents=True)
                (directory / 'agent.jsonl').write_text(json.dumps({
                    'type': 'turn.completed',
                    'usage': {'input_tokens': 10, 'output_tokens': 5}}) + '\n')
                if fields == ('hypothesis', 'code'):
                    return {'hypothesis': 'fixed', 'code': 'def run(api):\n    api.wait(.1)\n'}
                return {field: 'validated fixture' for field in fields}
            def passing_trials(candidate, seeds, directory, timeout):
                return {'successes': len(seeds), 'trials': len(seeds), 'mean_score': 1.,
                        'results': [{'seed': seed, 'error': None,
                                     'metrics': {'success': True, 'score': 1.}} for seed in seeds]}
            with patch.object(agent_loop, 'ROOT', root), \
                 patch.object(agent_loop, 'ask_llm', side_effect=fake_model), \
                 patch.object(agent_loop, 'run_trials', side_effect=passing_trials):
                agent_loop.campaign(args)
            state = json.loads((output / 'campaign.json').read_text())
            self.assertEqual(len(state['history']), 2)
            self.assertEqual(state['history'][-1]['successes'], 3)
            self.assertEqual(state['source_campaign'], str(source.resolve()))
            self.assertTrue((output / 'round_001/candidate.py').exists())
            self.assertFalse((output / 'round_002').exists())
            self.assertEqual((output / 'best_policy.py').read_text(),
                             (output / 'round_001/candidate.py').read_text())
            self.assertEqual(state['best_policy'], str(output / 'best_policy.py'))
            self.assertEqual(calls, [('hypothesis', 'code'),
                                     ('failure_signature', 'when_to_apply', 'repair', 'limitations')])
            self.assertEqual(state['status'], 'completed')
            self.assertEqual(state['round_mode'], 'until_pass_or_interrupt')

    def test_zero_token_threshold_disables_cumulative_stop(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(output=root, max_prompt_chars=1000,
                                   max_reported_tokens=0, model='test',
                                   llm_timeout=1, codex_bin=None)
            state = {'llm_calls': [{'usage': {'input_tokens': 999, 'output_tokens': 999}}]}
            def fake_call(prompt, directory, fields, model, timeout, codex_bin, image_paths):
                directory.mkdir()
                (directory / 'agent.jsonl').write_text(json.dumps({
                    'type': 'turn.completed', 'usage': {'input_tokens': 1, 'output_tokens': 1}}) + '\n')
                return {'code': 'pass'}
            with patch.object(agent_loop, 'ask_llm', side_effect=fake_call):
                self.assertEqual(agent_loop.budgeted_call('x', root / 'call', ('code',), args, state),
                                 {'code': 'pass'})

    def test_skill_report_distinguishes_partial_from_promoted(self):
        with TemporaryDirectory() as tmp:
            trace = Path(tmp) / 'trace.json'
            trace.write_text(json.dumps({'stability_checks': [{
                'stable': True, 'blocks': ['blue_block', 'small_red_block'],
                'duration_s': 2, 'reasons': []}, {
                'stable': False, 'blocks': ['blue_block', 'small_red_block', 'green_block'],
                'duration_s': 5, 'reasons': ['green_block: missing contact with small_red_block']}]}))
            report = build_report({'status': 'running', 'history': [{
                'candidate': 'candidate.py', 'hypothesis': 'place green more carefully',
                'successes': 0, 'trials': 3, 'mean_score': 1/3,
                'results': [{'seed': 52, 'error': None,
                             'metrics': {'success': False, 'score': .5, 'placed_blocks': 1},
                             'trace': str(trace)}]}]})
            item = report['rounds'][0]
            self.assertEqual(item['candidate_status'], 'partial')
            self.assertEqual(item['demonstrated_capabilities'][0]['seeds'], [52])
            self.assertIsNone(report['validated_skill_promoted'])


if __name__ == '__main__':
    unittest.main()
