"""Transport-only recovery retains request semantics and incomplete accounting."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import resume_ring_peg_http as http


class HttpRecoveryTests(unittest.TestCase):
    def test_actor_archives_http_config_without_changing_budget_or_prompt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stage = root/'basic/round_001'
            stage.parent.mkdir()
            state = stage.parent/'.actor_session'
            state.mkdir()
            manifest = {'model': 'gpt-6-astra', 'model_timeout_s': 900, 'max_candidate_characters': 50000}
            http.previous.write(root/'manifest.json', manifest)
            with patch.dict(os.environ, {'CODEX_HOME': str(root/'empty_home')}), \
                    patch.object(http.previous.study.actor.subprocess, 'run', side_effect=subprocess.TimeoutExpired('mock', 900)) as invoke:
                with self.assertRaises(subprocess.TimeoutExpired):
                    http.http_actor()[0](stage, 'unchanged prompt', [], manifest, root, 'basic', 1, state)
            invocation = http.previous.read(stage/'invocation.json')
            command = invocation['command']
            offset = command.index('model_provider="openai-http"')-1
            self.assertEqual(command[offset:offset+4], http.HTTP_ARGS)
            self.assertEqual(command[command.index('--model')+1], 'gpt-6-astra')
            self.assertIn('model_reasoning_effort="high"', command)
            self.assertEqual(invoke.call_args.kwargs['timeout'], 900)
            self.assertEqual(invoke.call_args.kwargs['input'], 'unchanged prompt')
            self.assertEqual(command[command.index('--sandbox')+1], 'read-only')
            self.assertFalse((state/'codex_home/auth.json').exists())

    def test_specialization_keeps_24_workers_and_two_prior_failures(self):
        adapter = http.build_adapter()
        _, _, source = adapter['corrected_functions']()
        self.assertEqual(adapter['REPAIR'], 'resume_02_http')
        self.assertIn("'incomplete_model_attempts': 2, 'model_attempt_budget': 32", source)
        self.assertIn("'rollout_workers': 24", source)
        self.assertIn("manifest['train_seeds']", source)

    def test_resource_freeze_keeps_both_incomplete_attempts(self):
        adapter = http.build_adapter()
        usage = [{'usage_complete': False, 'wall_seconds': 900}, {'usage_complete': False, 'wall_seconds': 901}]
        def freeze(root):
            return http.previous.study.resources.summarize(root/'replica_03', 3)
        with patch.dict(adapter, {'check': lambda root: {'incomplete_usage': usage},
                                  'recovered_lease_audit': lambda *args: {}}), \
                patch.object(http.previous.study, 'check', return_value={'limits': {'model': 4}}), \
                patch.object(http.previous.study, 'freeze_training', side_effect=freeze), \
                patch.object(http.previous.study.resources, 'summarize', return_value={'notes': []}):
            result = adapter['freeze_training'](Path('/unused'))
        self.assertEqual(len(result['incomplete_attempts']), 2)
        self.assertFalse(result['all_attempt_usage_complete'])
        self.assertEqual(sum(u['wall_seconds'] for u in result['incomplete_attempts']), 1801)


if __name__ == '__main__':
    unittest.main()
