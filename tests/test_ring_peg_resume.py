"""Recovery reuses complete work and rejects unregistered incomplete attempts."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import resume_ring_peg_comparison as recovery


class RecoveryTests(unittest.TestCase):
    def test_saved_proposal_never_calls_model_and_checks_candidate(self):
        with tempfile.TemporaryDirectory() as temp:
            stage = Path(temp)
            (stage/'candidate.py').write_text('saved policy')
            (stage/'prompt.txt').write_text('same prompt')
            recovery.write(stage/'invocation.json', {'images': []})
            recovery.write(stage/'generation.json', {'candidate_sha256': recovery.digest(stage/'candidate.py')})
            with patch.object(recovery.study.actor, 'audit_session'), patch.object(recovery.study.actor, 'propose') as call:
                recovery.reuse_or_propose(stage, 'same prompt', [], {})
                call.assert_not_called()
                (stage/'candidate.py').write_text('changed')
                with self.assertRaisesRegex(RuntimeError, 'candidate changed'):
                    recovery.reuse_or_propose(stage, 'same prompt', [], {})

    def test_incomplete_proposal_and_trial_are_not_retried(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stage = recovery.study.stage_path(root/'replica_01', 'basic', 1)
            stage.mkdir(parents=True)
            with patch.object(recovery.study.actor, 'propose') as call:
                with self.assertRaisesRegex(RuntimeError, 'Incomplete model'):
                    recovery.reuse_or_propose(stage, '', [], {})
                call.assert_not_called()
            (stage/'trials/seed_1').mkdir(parents=True)
            with patch.object(recovery.study, 'rollout') as call:
                with self.assertRaisesRegex(RuntimeError, 'Incomplete trial'):
                    recovery.reusable_rollout(root, root/'replica_01', 'basic', 1, 1)
                call.assert_not_called()

    def test_unchanged_usage_broadcast_does_not_establish_zero_consumption(self):
        with tempfile.TemporaryDirectory() as temp:
            previous, live = Path(temp)/'previous', Path(temp)/'live'
            counts = {'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15}
            event = {'type': 'event_msg', 'payload': {'type': 'token_count', 'info': {
                'total_token_usage': counts, 'last_token_usage': counts}}}
            line = json.dumps(event)+'\n'
            previous.write_text(line)
            live.write_text(line+line+line)
            result = recovery.inspect_interrupted(previous, live)
            self.assertEqual(result['recorded_token_delta']['total_tokens'], 0)
            self.assertFalse(result['usage_complete'])
            live.write_text(line+json.dumps({'type': 'response_item', 'payload': {'role': 'assistant'}})+'\n')
            with self.assertRaisesRegex(RuntimeError, 'assistant response'):
                recovery.inspect_interrupted(previous, live)

    def test_only_registered_failure_is_allowed_in_lease_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            event = root/'scheduler/model/events/one.json'
            recovery.write(event, {'status': 'failed', 'queued_at': 0, 'started_at': 1, 'finished_at': 3})
            recovery.write(root/recovery.REPAIR/'registration.json', {
                'failed_leases': {str(event.relative_to(root)): recovery.digest(event)}, 'model_attempt_budget': 31})
            report = recovery.recovered_lease_audit(root, {'model': 4, 'rollout': 24})
            self.assertEqual(report['model']['interrupted'], 1)
            recovery.write(event.with_name('two.json'), recovery.read(event))
            with self.assertRaisesRegex(RuntimeError, 'Unregistered failed'):
                recovery.recovered_lease_audit(root, {'model': 4, 'rollout': 24})

    def test_resume_chain_reuses_generated_revision_and_completed_trials(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root/'replica_01'
            directory.mkdir()
            seeds = [11, 12, 13, 14, 15]
            recovery.write(directory/'manifest.json', {'arms': ['basic'], 'train_seeds': seeds})
            (root/'starter.py').write_text('starter')
            (root/'basic_contract.txt').write_text('contract')
            protocol = {'replicas': [{'name': 'replica_01'}], 'limits': {'model': 4}}
            r0 = recovery.study.stage_path(directory, 'basic', 0)
            r0.mkdir(parents=True)
            (r0/'candidate.py').write_text('starter')
            def save_trial(revision, seed):
                row = {'seed': seed, 'success': revision > 0 and seed == 11, 'status': 'completed'}
                path = recovery.study.stage_path(directory, 'basic', revision)/'trials'/f'seed_{seed}'
                recovery.write(path/'result.json', row)
                recovery.write(path/'actor_feedback.json', row)
                return row
            for seed in seeds:
                save_trial(0, seed)
            r1 = recovery.study.stage_path(directory, 'basic', 1)
            r1.mkdir()
            (r1/'candidate.py').write_text('saved R1')
            (r1/'prompt.txt').write_text(recovery.study.prompt_for(root, directory, 'basic', 1, 0, seeds))
            recovery.write(r1/'generation.json', {'candidate_sha256': recovery.digest(r1/'candidate.py')})
            recovery.write(r1/'invocation.json', {'images': []})
            def propose(stage, prompt, images, manifest):
                stage.mkdir()
                (stage/'candidate.py').write_text('new '+stage.name)
                recovery.write(stage/'generation.json', {})
                recovery.write(stage/'invocation.json', {})
            def rollout(root, directory, arm, revision, seed, reserved, workers):
                return save_trial(revision, seed)
            with patch.object(recovery.study, 'check', return_value=protocol), \
                    patch.object(recovery.study.actor, 'audit_session'), \
                    patch.object(recovery.study.actor, 'propose', side_effect=propose) as model, \
                    patch.object(recovery.study, 'audit_trial', side_effect=lambda path, *a: recovery.read(path/'result.json')), \
                    patch.object(recovery.study, 'rollout', side_effect=rollout) as dispatch:
                chain, _, _ = recovery.corrected_functions()
                chain(root, 'replica_01', 'basic')
            self.assertEqual(model.call_count, 2)
            self.assertEqual(dispatch.call_count, 15)
            self.assertEqual(recovery.read(directory/'basic/summary.json')['selected_revision'], 1)
            self.assertEqual((r1/'candidate.py').read_text(), 'saved R1')


if __name__ == '__main__':
    unittest.main()
