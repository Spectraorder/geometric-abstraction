"""Report replacements must preserve paired layouts and retain worse repeats too."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from online_reporting.analysis import substitute_pairs


class ReportReplacementTests(unittest.TestCase):
    def fixture(self):
        original = [dict(task='tower', seed=seed, arm=arm, cohort='new_45',
            initial_state_sha256=str(seed), budget_seconds=200, success=1,
            capped_seconds=100., total_tokens=1000, usage_complete=True,
            timing_valid=seed != 1, gpu_dispatch_query_ok=True,
            source=f'original/{seed}/{arm}', source_sha256='original-hash')
            for seed in (1, 2) for arm in ('Base', 'Full')]
        repeated = [dict(r, cohort='targeted_repeat', success=0, capped_seconds=200.,
            timing_valid=True, total_tokens=2000, source='repeat/' + r['arm'],
            source_sha256='repeat-hash') for r in original if r['seed'] == 1]
        return original, repeated

    def test_keeps_regressions_and_preserves_unaffected_pair(self):
        original, repeated = self.fixture()
        merged, changes = substitute_pairs(original, repeated, {('tower', 1)})
        self.assertEqual(len(merged), 4)
        self.assertEqual([r['success'] for r in merged], [0, 0, 1, 1])
        self.assertEqual([r['total_tokens'] for r in merged], [2000, 2000, 1000, 1000])
        self.assertTrue(all(r['cohort'] == 'new_45' for r in merged))
        self.assertFalse(merged[0]['pair_in_quality_sensitivity'])
        self.assertTrue(merged[2]['pair_in_quality_sensitivity'])
        self.assertEqual(len(changes), 2)
        self.assertEqual([r['success'] for r in original], [1, 1, 1, 1])

    def test_rejects_missing_or_duplicate_repeat(self):
        original, repeated = self.fixture()
        for rows in (repeated[:1], repeated + repeated[:1]):
            with self.assertRaises(ValueError):
                substitute_pairs(original, rows, {('tower', 1)})

    def test_rejects_changed_layout_or_failed_quality(self):
        original, repeated = self.fixture()
        for change in ({'initial_state_sha256': 'different'}, {'timing_valid': False}):
            with self.assertRaises(ValueError):
                substitute_pairs(original, [dict(repeated[0], **change), repeated[1]], {('tower', 1)})


if __name__ == '__main__':
    unittest.main()
