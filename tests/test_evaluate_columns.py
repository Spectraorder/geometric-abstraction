"""Reproducible matrices and frozen-source boundaries for column evaluation."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
SPEC = importlib.util.spec_from_file_location('evaluate_columns', ROOT/'scripts/evaluate_columns.py')
COLUMNS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLUMNS)


class ColumnMatrixTests(unittest.TestCase):
    def test_explicit_level_and_distractor_matrix_preserves_every_goal(self):
        cases = COLUMNS.plan_cases(['cube_pair', 'random'], [901, 902],
                                   levels=[2, 3, 4], distractors=[0, 2])
        self.assertEqual(len(cases), 8)
        self.assertEqual(len({case['name'] for case in cases}), 8)
        for case in cases:
            self.assertEqual([trial['seed'] for trial in case['trials']], [901, 902])
            for trial in case['trials']:
                self.assertEqual(trial['object_count'], len(trial['order'])+case['distractors'])
                self.assertEqual(trial['level_count'], len(trial['goal']['levels']))
                self.assertEqual(trial['task'], 'inventory_tower')
                self.assertEqual(trial['sensor'], COLUMNS.COLUMN_SENSOR_PROFILE)
        self.assertEqual({case['levels'] for case in cases if case['recipe'] == 'random'}, {2, 3, 4})

    def test_defaults_preserve_existing_named_paths_and_random_three_levels(self):
        named = COLUMNS.plan_cases(['cube_pair'], [901])[0]
        self.assertEqual(named['name'], 'cube_pair')
        random = COLUMNS.plan_cases(['random'], [901, 902])[0]
        self.assertEqual(random['name'], 'random')
        self.assertEqual([trial['level_count'] for trial in random['trials']], [3, 3])

    def test_duplicate_cells_and_unavailable_scene_sizes_are_rejected(self):
        options = [dict(seeds=[901, 901]), dict(recipes=['random', 'random']),
                   dict(levels=[2, 2]), dict(levels=[5]), dict(distractors=[-1]),
                   dict(distractors=[0, 0]), dict(distractors=[19]),
                   dict(recipes=['cube_pair'], levels=[2])]
        for option in options:
            args = dict(recipes=['random'], seeds=[901]) | option
            with self.subTest(option=option), self.assertRaises(ValueError):
                COLUMNS.plan_cases(**args)

    def test_freeze_rejects_source_matrix_and_split_changes(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            candidate = root/'source.py'
            candidate.write_text('def run(api): pass\n')
            cases = COLUMNS.plan_cases(['random'], [901], levels=[2, 3, 4])
            manifest, frozen = COLUMNS.prepare_manifest(candidate, root/'run', cases)
            self.assertEqual(manifest['trial_count'], 3)
            self.assertEqual(manifest['sensor'], COLUMNS.COLUMN_SENSOR_PROFILE)
            self.assertEqual(manifest['schema_version'], 2)
            self.assertEqual(frozen.read_bytes(), candidate.read_bytes())
            self.assertEqual(json.loads((root/'run/manifest.json').read_text()), manifest)
            COLUMNS.prepare_manifest(candidate, root/'run', cases)
            changed_cases = COLUMNS.plan_cases(['random'], [902], levels=[2, 3, 4])
            with self.assertRaisesRegex(ValueError, 'different source, split, or trial matrix'):
                COLUMNS.prepare_manifest(candidate, root/'run', changed_cases)
            with self.assertRaisesRegex(ValueError, 'requires --frozen-sha256'):
                COLUMNS.prepare_manifest(candidate, root/'heldout', cases, split='heldout')
            with self.assertRaisesRegex(ValueError, 'does not match'):
                COLUMNS.prepare_manifest(candidate, root/'heldout', cases, split='heldout', frozen_sha256='wrong')
            COLUMNS.prepare_manifest(candidate, root/'heldout', cases, split='heldout',
                                     frozen_sha256=manifest['candidate_sha256'])
            candidate.write_text('def run(api): api.note("changed")\n')
            with self.assertRaisesRegex(ValueError, 'different source, split, or trial matrix'):
                COLUMNS.prepare_manifest(candidate, root/'run', cases)

    def test_sensor_changes_invalidate_manifest_without_overwriting_it(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            candidate = root/'source.py'
            candidate.write_text('def run(api): pass\n')
            cases = COLUMNS.plan_cases(['cube_pair'], [901])
            manifest, _ = COLUMNS.prepare_manifest(candidate, root/'run', cases)
            legacy = dict(manifest)
            legacy.pop('sensor')
            path = root/'run/manifest.json'
            path.write_text(json.dumps(legacy))
            previous = path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'sensor profile'):
                COLUMNS.prepare_manifest(candidate, root/'run', cases)
            self.assertEqual(path.read_bytes(), previous)

    def test_legacy_and_blackout_caches_are_rejected_before_evaluation(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            candidate = root/'candidate.py'
            candidate.write_text('def run(api): pass\n')
            case = COLUMNS.plan_cases(['cube_pair'], [901])[0]
            path = root/'results/seed_901/result.json'
            path.parent.mkdir(parents=True)
            records = [{}, {'sensor': dict(depth_sampling='legacy_multisample', offscreen_samples=4)},
                       {'sensor': dict(COLUMNS.COLUMN_SENSOR_PROFILE), 'sensor_blackout': True}]
            for record in records:
                path.write_text(json.dumps(record))
                previous = path.read_bytes()
                with self.subTest(record=record), patch.object(COLUMNS, 'evaluate') as evaluate:
                    with self.assertRaisesRegex(RuntimeError, 'sensor profile'):
                        COLUMNS.evaluate_case(candidate, root/'results', case, 1)
                    evaluate.assert_not_called()
                    self.assertEqual(path.read_bytes(), previous)

    def test_results_keep_declared_task_and_reject_wrong_count(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            candidate = root/'candidate.py'
            candidate.write_text('def run(api): pass\n')
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            case = COLUMNS.plan_cases(['random'], [901, 902], levels=[4], distractors=[2])[0]
            directory = root/'results'
            directory.mkdir()
            results = [dict(seed=trial['seed'], task=trial['task'], layout=trial['layout'],
                            candidate_sha256=digest, success=index == 0,
                            metrics=dict(success=index == 0, score=1-index, required_placements=3,
                                         placed_blocks=3 if index == 0 else 0, structure_stable=index == 0, failure=None),
                            goal=trial['goal'], scene=dict(object_count=6, seed=trial['seed'],
                                                          recipe=trial['recipe'], order=trial['order']),
                            sensor=dict(COLUMNS.COLUMN_SENSOR_PROFILE))
                       for index, trial in enumerate(case['trials'])]
            with patch.object(COLUMNS, 'evaluate', return_value=dict(sha256=digest, results=results)) as evaluate:
                summary = COLUMNS.evaluate_case(candidate, directory, case, 2)
                self.assertEqual(evaluate.call_args.kwargs['object_count'], 6)
                self.assertEqual(evaluate.call_args.kwargs['tower_levels'], 4)
                self.assertEqual(summary['successes'], 1)
                self.assertEqual(summary['trials'], 2)
                self.assertEqual(summary['score'], .5)
                self.assertEqual(summary['results'][1]['declared_trial'], case['trials'][1])
                with self.assertRaisesRegex(RuntimeError, 'no longer matches the manifest'):
                    COLUMNS.evaluate_case(candidate, directory, case, 2, expected_sha256='wrong')
                results[0].pop('sensor')
                with self.assertRaisesRegex(RuntimeError, 'sensor profile'):
                    COLUMNS.evaluate_case(candidate, directory, case, 2)
                results[0]['sensor'] = dict(COLUMNS.COLUMN_SENSOR_PROFILE)
                results[0]['scene']['object_count'] = 5
                with self.assertRaisesRegex(RuntimeError, 'declared goal and scene size'):
                    COLUMNS.evaluate_case(candidate, directory, case, 2)


    def test_cached_success_requires_complete_stable_tower_and_exact_provenance(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            candidate = root/'candidate.py'
            candidate.write_text('def run(api): pass\n')
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            case = COLUMNS.plan_cases(['cube_triple'], [901])[0]
            trial = case['trials'][0]
            result = dict(seed=901, task='inventory_tower', layout='sampled', candidate_sha256=digest,
                          goal=trial['goal'], sensor=trial['sensor'], sensor_blackout=False,
                          scene=dict(seed=901, recipe='cube_triple', object_count=3, order=trial['order']),
                          success=True, policy_error=None,
                          metrics=dict(success=True, score=1., placed_blocks=2, required_placements=2,
                                       structure_stable=True, failure=None))
            path = root/'results/seed_901/result.json'
            path.parent.mkdir(parents=True)
            changes = [('candidate_sha256', 'wrong'), ('success', False), ('policy_error', 'TimeoutError'),
                       ('metrics.structure_stable', False), ('metrics.placed_blocks', 1),
                       ('metrics.required_placements', 1), ('metrics.score', .5),
                       ('metrics.failure', 'simulation error'), ('scene.seed', 902),
                       ('scene.order', list(reversed(trial['order'])))]
            for field, value in changes:
                bad = deepcopy(result)
                owner = bad
                keys = field.split('.')
                for key in keys[:-1]:
                    owner = owner[key]
                owner[keys[-1]] = value
                path.write_text(json.dumps(bad))
                original = path.read_bytes()
                with self.subTest(field=field), patch.object(COLUMNS, 'evaluate') as evaluate:
                    with self.assertRaises(RuntimeError):
                        COLUMNS.evaluate_case(candidate, root/'results', case, 1)
                    evaluate.assert_not_called()
                    self.assertEqual(path.read_bytes(), original)
            # Full partial-placement score cannot count when the strict final
            # stability check fails, even if every layer was briefly present.
            result['success'] = False
            result['metrics'].update(success=False, structure_stable=False)
            path.write_text(json.dumps(result))
            with patch.object(COLUMNS, 'evaluate', return_value=dict(sha256=digest, results=[result])):
                summary = COLUMNS.evaluate_case(candidate, root/'results', case, 1)
            self.assertEqual(summary['successes'], 0)
            self.assertEqual(summary['trials'], 1)
            self.assertEqual(summary['score'], 1.)


if __name__ == '__main__':
    unittest.main()
