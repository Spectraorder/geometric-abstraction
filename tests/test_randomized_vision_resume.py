"""Campaign recovery preserves incomplete trials and reuses completed trials."""

import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('vision_learner', ROOT / 'scripts/learn_randomized_vision.py')
LEARNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LEARNER)


class RandomizedVisionResumeTests(unittest.TestCase):
    def test_inventory_cache_cannot_reuse_a_different_object_count(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root/'candidate.py'
            candidate.write_text('def run(api): pass\n')
            trial = root/'trials/seed_601'
            trial.mkdir(parents=True)
            result = {'candidate_sha256': hashlib.sha256(candidate.read_bytes()).hexdigest(),
                      'task': 'inventory_prism', 'layout': 'sampled', 'seed': 601,
                      'scene': {'object_count': 6}, 'success': False,
                      'metrics': {'score': 0.}, 'policy_error': None}
            (trial/'result.json').write_text(json.dumps(result))
            with patch.object(LEARNER.subprocess, 'run') as run:
                LEARNER.evaluate(candidate, trial.parent, [601], 1, task='inventory_prism', layout='sampled', object_count=6)
                with self.assertRaisesRegex(RuntimeError, 'does not match'):
                    LEARNER.evaluate(candidate, trial.parent, [601], 1, task='inventory_prism', layout='sampled', object_count=2)
                run.assert_not_called()

    def test_resume_archives_partial_trial_and_reuses_completed_results(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / 'candidate.py'
            candidate.write_text('def run(api): pass\n')
            trials = root / 'trials'
            partial = trials / 'seed_101'
            partial.mkdir(parents=True)
            (partial / 'initial.png').write_bytes(b'previous image')
            (trials / 'seed_101.log').write_text('previous failure')
            result = {'candidate_sha256': hashlib.sha256(candidate.read_bytes()).hexdigest(),
                      'layout': 'random_xy', 'seed': 101, 'success': True,
                      'metrics': {'score': 1.0}, 'policy_error': None}

            def run_trial(command, **kwargs):
                out = Path(command[command.index('--output') + 1])
                # Match the real runner's requirement; regression used to fail here.
                out.mkdir(parents=True, exist_ok=False)
                (out / 'result.json').write_text(json.dumps(result))

            with patch.object(LEARNER.subprocess, 'run', side_effect=run_trial) as run:
                summary = LEARNER.evaluate(candidate, trials, [101], 1)
                self.assertEqual(summary['successes'], 1)
                run.assert_called_once()
            archives = list(trials.glob('seed_101.interrupted-*'))
            self.assertEqual(len(archives), 1)
            self.assertEqual((archives[0] / 'trial/initial.png').read_bytes(), b'previous image')
            self.assertEqual((archives[0] / 'seed_101.log').read_text(), 'previous failure')
            with patch.object(LEARNER.subprocess, 'run') as run:
                self.assertEqual(LEARNER.evaluate(candidate, trials, [101], 1), summary)
                run.assert_not_called()
                with self.assertRaisesRegex(RuntimeError, 'does not match'):
                    LEARNER.evaluate(candidate, trials, [101], 1, task='prism')
                run.assert_not_called()
                candidate.write_text('def run(api): api.wait(1)\n')
                with self.assertRaisesRegex(RuntimeError, 'does not match'):
                    LEARNER.evaluate(candidate, trials, [101], 1)
                run.assert_not_called()

    def test_duplicate_or_empty_seed_requests_fail_before_starting_trials(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            candidate = root/'candidate.py'
            candidate.write_text('def run(api): pass\n')
            with patch.object(LEARNER.subprocess, 'run') as run:
                for seeds in ([], [101, 101]):
                    with self.subTest(seeds=seeds), self.assertRaisesRegex(ValueError, 'unique'):
                        LEARNER.evaluate(candidate, root/'trials', seeds, 2)
                run.assert_not_called()
            self.assertFalse((root/'trials').exists())

    def test_named_recipe_cache_checks_scene_order_independently_of_goal(self):
        from aspire_so100.tower_tasks import tower_recipe
        with TemporaryDirectory() as temp:
            root = Path(temp)
            candidate = root/'candidate.py'
            candidate.write_text('def run(api): pass\n')
            out = root/'trials/seed_901'
            out.mkdir(parents=True)
            recipe = tower_recipe('cube_triple')
            result = dict(candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
                          task='inventory_tower', layout='sampled', seed=901,
                          scene=dict(object_count=3, recipe='cube_triple', order=list(reversed(recipe.order))),
                          goal=recipe.public_goal(), success=False, metrics=dict(score=0.), policy_error=None)
            path = out/'result.json'
            path.write_text(json.dumps(result))
            original = path.read_bytes()
            with patch.object(LEARNER.subprocess, 'run') as run:
                with self.assertRaisesRegex(RuntimeError, 'different tower goal'):
                    LEARNER.evaluate(candidate, out.parent, [901], 1, task='inventory_tower',
                                     layout='sampled', object_count=3, tower_recipe='cube_triple')
                run.assert_not_called()
            self.assertEqual(path.read_bytes(), original)
