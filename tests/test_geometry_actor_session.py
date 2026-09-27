"""Verify actor continuity, isolation, failure accounting and incremental prompts offline."""
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import geometry_actor_session as actor
import run_constraint_geometry_experiment as experiment


class ActorSessionTests(unittest.TestCase):
    def manifest(self, root):
        manifest = {'arms': ['basic', 'symbolic'], 'actor_context_mode': 'persistent',
                    'model': 'gpt-6-astra', 'model_timeout_s': 10, 'max_candidate_characters': 30000}
        actor.write(root / 'manifest.json', manifest)
        return manifest

    def fake_run(self, command, **kwargs):
        home = Path(kwargs['env']['CODEX_HOME'])
        session_id = command[command.index('resume') + 1] if 'resume' in command else str(uuid4())
        sessions = home / 'sessions'
        sessions.mkdir(exist_ok=True)
        path = sessions / f'rollout-{session_id}.jsonl'
        with path.open('a') as transcript:
            if path.stat().st_size == 0:
                transcript.write(json.dumps({'type': 'session_meta', 'payload': {'id': session_id}}) + '\n')
            transcript.write(json.dumps({'type': 'response_item', 'payload': {'role': 'user', 'text': kwargs['input']}}) + '\n')
        response = {'code': 'def run(api):\n    pass\n', 'hypothesis': 'Synthetic response'}
        actor.write(Path(command[command.index('--output-last-message') + 1]), response)
        for event in ({'type': 'thread.started', 'thread_id': session_id},
                      {'type': 'turn.completed', 'usage': {'input_tokens': 100, 'cached_input_tokens': 60,
                                                         'output_tokens': 20, 'reasoning_output_tokens': 5}}):
            kwargs['stdout'].write(json.dumps(event) + '\n')
        return subprocess.CompletedProcess(command, 0)

    def invoke(self, root, arm, revision, manifest, prompt='Synthetic feedback', images=()):
        directory = root / arm / f'round_{revision:03d}'
        actor.propose(directory, prompt, list(images), manifest)
        invocation, generation = actor.read(directory / 'invocation.json'), actor.read(directory / 'generation.json')
        actor.audit_session(directory, manifest, invocation, generation)
        return directory, invocation, generation

    def test_resume_uses_own_session_and_keeps_usage_and_original_history(self):
        with TemporaryDirectory() as tmp, patch.object(actor.subprocess, 'run', side_effect=self.fake_run):
            root = Path(tmp)
            manifest = self.manifest(root)
            first, initial, a = self.invoke(root, 'basic', 1, manifest, 'Original visual grounding')
            _, _, other = self.invoke(root, 'symbolic', 1, manifest)
            second, continued, b = self.invoke(root, 'basic', 2, manifest, 'Only new feedback')
            self.assertEqual(a['session_id'], b['session_id'])
            self.assertNotEqual(a['session_id'], other['session_id'])
            self.assertTrue(initial['fresh_context'])
            self.assertFalse(continued['fresh_context'])
            self.assertEqual(continued['resumed_session_id'], a['session_id'])
            self.assertEqual(continued['session_before_sha256'], a['session_sha256'])
            self.assertIn('Original visual grounding', (second / 'session.jsonl').read_text())
            self.assertNotIn('Original visual grounding', (second / 'prompt.txt').read_text())
            self.assertEqual(b['usage']['input_tokens'], 100)
            self.assertEqual(b['usage']['cached_input_tokens'], 60)
            self.assertEqual(b['usage']['reasoning_output_tokens'], 5)
            self.assertNotIn('--ephemeral', continued['command'])
            self.assertEqual((first / 'session.jsonl').read_bytes(),
                             (second / 'session.jsonl').read_bytes()[:(first / 'session.jsonl').stat().st_size])

    def test_full_campaign_audit_accepts_continuous_sessions(self):
        from test_mixed_scene_learning_audit import LearningAuditTests
        from summarize_mixed_scene_learning import audit
        with TemporaryDirectory() as tmp, patch.object(actor.subprocess, 'run', side_effect=self.fake_run):
            root = Path(tmp) / 'study'
            LearningAuditTests().fixture(root, arms=('basic', 'symbolic'),
                                        train=[91001, 91002, 91003], persistent=True, rounds=2)
            report = audit(root)
            self.assertEqual(report['status'], 'completed')
            self.assertEqual(report['valid_rollouts'], 30)
            self.assertEqual(len(report['actors']), 4)
            self.assertEqual(len({row['session_id'] for row in report['actors']}), 2)
            self.assertTrue(all(row['actor_context_mode'] == 'persistent' for row in report['actors']))

    def test_history_tampering_blocks_resume_before_a_model_call(self):
        with TemporaryDirectory() as tmp, patch.object(actor.subprocess, 'run', side_effect=self.fake_run) as run:
            root = Path(tmp)
            manifest = self.manifest(root)
            _, _, generation = self.invoke(root, 'basic', 1, manifest)
            live = actor.session_file(root / 'basic/.actor_session/codex_home', generation['session_id'])
            with live.open('a') as stream:
                stream.write('{"type":"unexpected"}\n')
            with self.assertRaisesRegex(RuntimeError, 'outside the registered chain'):
                self.invoke(root, 'basic', 2, manifest)
            self.assertEqual(run.call_count, 1)

    def test_failure_is_retained_not_retried_and_credentials_are_removed(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.manifest(root)
            auth_home = root / 'test_auth'
            auth_home.mkdir()
            (auth_home / 'auth.json').write_text('{"synthetic":true}')
            def failed(command, **kwargs):
                self.assertTrue((Path(kwargs['env']['CODEX_HOME']) / 'auth.json').exists())
                raise subprocess.TimeoutExpired(command, 10)
            with patch.dict(os.environ, {'CODEX_HOME': str(auth_home)}), patch.object(actor.subprocess, 'run', side_effect=failed) as run:
                with self.assertRaises(subprocess.TimeoutExpired):
                    self.invoke(root, 'basic', 1, manifest)
                directory = root / 'basic/round_001'
                self.assertTrue((directory / 'attempt_failure.json').exists())
                self.assertFalse((directory / 'generation.json').exists())
                self.assertFalse((root / 'basic/.actor_session/codex_home/auth.json').exists())
                with self.assertRaises(FileExistsError):
                    self.invoke(root, 'basic', 1, manifest)
                self.assertEqual(run.call_count, 1)

    def test_audit_rejects_cross_arm_session_and_tampered_archive(self):
        with TemporaryDirectory() as tmp, patch.object(actor.subprocess, 'run', side_effect=self.fake_run):
            root = Path(tmp)
            manifest = self.manifest(root)
            directory, invocation, generation = self.invoke(root, 'basic', 1, manifest)
            foreign = root / 'symbolic/round_001'
            foreign.mkdir(parents=True)
            actor.write(foreign / 'generation.json', {'session_id': generation['session_id']})
            with self.assertRaisesRegex(ValueError, 'shared across'):
                actor.audit_session(directory, manifest, invocation, generation)
            (foreign / 'generation.json').unlink()
            with (directory / 'session.jsonl').open('a') as stream:
                stream.write('{"type":"unexpected"}\n')
            with self.assertRaisesRegex(ValueError, 'Archived actor session changed'):
                actor.audit_session(directory, manifest, invocation, generation)

    def test_training_images_persist_but_foreign_and_reserved_images_are_rejected(self):
        with TemporaryDirectory() as tmp, patch.object(actor.subprocess, 'run', side_effect=self.fake_run):
            root = Path(tmp)
            manifest = self.manifest(root)
            image = root / 'basic/revision_000/trials/seed_1/frame.png'
            image.parent.mkdir(parents=True)
            image.write_bytes(b'Synthetic image')
            _, invocation, _ = self.invoke(root, 'basic', 1, manifest, images=[image])
            command = invocation['command']
            copied = Path(command[command.index('--image') + 1])
            self.assertEqual(copied.read_bytes(), image.read_bytes())
            self.invoke(root, 'basic', 2, manifest)
            self.assertTrue(copied.exists())
            with self.assertRaisesRegex(ValueError, 'own training images'):
                self.invoke(root, 'symbolic', 1, manifest, images=[image])
        with TemporaryDirectory() as tmp, patch.object(actor.subprocess, 'run', side_effect=self.fake_run) as run:
            root = Path(tmp)
            manifest = self.manifest(root)
            image = root / 'basic/heldout/seed_1/frame.png'
            image.parent.mkdir(parents=True)
            image.write_bytes(b'Reserved image')
            with self.assertRaisesRegex(ValueError, 'own training images'):
                self.invoke(root, 'basic', 1, manifest, images=[image])
            run.assert_not_called()


class SessionContextTests(unittest.TestCase):
    def training(self, root, arm, revision, seeds):
        stage = 'revision_000' if revision == 0 else f'round_{revision:03d}'
        directory = root / arm / stage
        directory.mkdir(parents=True, exist_ok=True)
        if revision:
            (directory / 'candidate.py').write_text('def run(api):\n    pass\n')
            actor.write(directory / 'response.json', {'hypothesis': 'Own historical hypothesis'})
        for seed in seeds:
            trial = directory / 'trials' / f'seed_{seed}'
            trial.mkdir(parents=True)
            actor.write(trial / 'actor_feedback.json', {'success': False, 'policy_error': None,
                                                       'notes': [f'{arm}-r{revision}-s{seed}']})
            actor.write(trial / 'api_calls.json', [])
            record = {'object': 'red', 'step': 'settled', 'time': 1, 'status': 'unavailable'}
            if arm == 'symbolic':
                record['constraints'] = {'symbolic_marker': seed}
            actor.write(trial / 'geometry_records.json', [{'kind': 'geometry_step', 'record': record}])

    def test_continuation_is_incremental_text_only_and_arm_isolated(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / 'study'
            manifest = experiment.prepare(root, [11, 12, 13], [21], 2, actor_images='initial_only')
            for arm in manifest['arms']:
                for revision in (0, 1):
                    self.training(root, arm, revision, manifest['train_seeds'])
            image = root / 'basic/revision_000/trials/seed_11/synthetic.png'
            with patch.object(experiment.base, 'sensor_context', return_value=([image], [])) as sensors:
                first, images = experiment.context(root, 'basic', 1, manifest)
                second, none = experiment.context(root, 'basic', 2, manifest)
                full, _ = experiment.context(root, 'symbolic', 2, manifest)
            self.assertEqual(len(images), 2)
            self.assertEqual(sensors.call_count, 2)
            self.assertEqual(none, [])
            self.assertIn('Frozen common motion loop', first)
            self.assertNotIn('Frozen common motion loop', second)
            self.assertNotIn('Identical starter planner', second)
            self.assertNotIn('basic-r0-', second)
            for seed in manifest['train_seeds']:
                self.assertIn(f'basic-r1-s{seed}', second)
            self.assertNotIn('symbolic_marker', second)
            self.assertIn('symbolic_marker', full)
            self.assertLess(len(second), len(first))
            self.assertEqual((second, none), experiment.context(root, 'basic', 2, manifest))

    def test_fresh_and_every_revision_options_remain_explicit(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / 'study'
            manifest = experiment.prepare(root, [11, 12, 13], [21], 2,
                                          actor_context_mode='fresh', actor_images='every_revision')
            for revision in (0, 1):
                self.training(root, 'basic', revision, manifest['train_seeds'])
            with patch.object(experiment.base, 'sensor_context', return_value=([], [])) as sensors:
                prompt, _ = experiment.context(root, 'basic', 2, manifest)
            self.assertEqual(sensors.call_count, 2)
            self.assertIn('Identical starter planner', prompt)
            self.assertIn('Latest exact planner', prompt)
            self.assertIn('basic-r0-', prompt)
            experiment.check(root, manifest)


if __name__ == '__main__':
    unittest.main()
