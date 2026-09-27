"""Fresh evaluation preserves policy selection and pairs every registered layout."""
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import evaluate_geometry_extension as evaluation


class FreshEvaluationTests(unittest.TestCase):
    def test_inspected_duplicate_and_wrong_size_splits_are_rejected(self):
        for seeds, inspected in [(list(range(20)), [7]), ([0]*20, []), (list(range(19)), [])]:
            with self.assertRaises(ValueError):
                evaluation.fresh_seeds(seeds, inspected)
        self.assertEqual(evaluation.fresh_seeds(list(range(20)), [21]), list(range(20)))

    def test_changed_training_evidence_blocks_evaluation(self):
        with TemporaryDirectory() as temp:
            root = Path(temp); output = root/'evaluation'; output.mkdir()
            (root/'frozen.py').write_text('original')
            registration = {'runner_sha256': evaluation.base.digest(Path(evaluation.__file__)),
                            'preserved_files': {'frozen.py': evaluation.base.digest(root/'frozen.py')}}
            (output/'registration.json').write_text(json.dumps(registration))
            (root/'frozen.py').write_text('modified')
            with self.assertRaisesRegex(ValueError, 'Preserved evidence changed'):
                evaluation.check(root, output)

    def test_runs_only_selected_policies_once_per_shared_seed(self):
        with TemporaryDirectory() as temp:
            root = Path(temp); output = root/'evaluation'; output.mkdir()
            registration = {'rounds': 5, 'workers': 4, 'required_rollouts': 40,
                            'reserved_seeds': list(range(20))}
            manifest = {'arms': ['basic', 'symbolic']}; calls = []
            def execute(root, arm, candidate, directory, seeds, manifest):
                calls.append((arm, seeds[0], candidate))
                return [{'seed': seeds[0], 'success': False}]
            with patch.object(evaluation, 'check', return_value=(registration, manifest)), \
                 patch.object(evaluation.experiment, 'configured', return_value=nullcontext()), \
                 patch.object(evaluation.base, 'execute', side_effect=execute), \
                 patch.object(evaluation, 'audit', return_value={'status': 'completed'}):
                evaluation.run(root, output)
            self.assertEqual(sorted((a, s) for a, s, _ in calls),
                             [(a, s) for a in ['basic', 'symbolic'] for s in range(20)])
            for arm, _, candidate in calls:
                self.assertEqual(candidate, root/'extension_05'/arm/'frozen.py')
            self.assertEqual(json.loads((output/'progress.json').read_text())['completed'], 40)


if __name__ == '__main__':
    unittest.main()
