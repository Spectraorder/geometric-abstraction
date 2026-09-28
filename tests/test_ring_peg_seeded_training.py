"""Isolation and provenance checks for the capable-start training variant."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_ring_peg_training as original
import run_ring_peg_seeded_training as seeded
from run_ring_peg_trial import validate_policy


class CapableStartTests(unittest.TestCase):
    def test_training_context_restores_neutral_start_even_on_error(self):
        before = (original.STARTER, original.CONTRACT, original.sources, original.rollout)
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            with seeded.installed():
                validate_policy(original.STARTER)
                self.assertIn('def run(api):', original.STARTER)
                self.assertNotEqual(original.STARTER, before[0])
                self.assertNotIn('R0 only observes', original.CONTRACT)
                self.assertIn('not learning from scratch', original.CONTRACT)
                self.assertIn('policies/controllers/ring_peg_starter.py', original.sources())
                raise RuntimeError('interrupted')
        self.assertEqual((original.STARTER, original.CONTRACT, original.sources, original.rollout), before)

    def test_starter_stays_inside_existing_policy_sandbox(self):
        source = seeded.STARTER_PATH.read_text()
        validate_policy(source)
        for private in ('private_ring_state', 'evaluate_on_peg', 'ring_peg_oracle', 'peg_axis_xy'):
            self.assertNotIn(private, source)


if __name__ == '__main__':
    unittest.main()
