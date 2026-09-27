"""Cumulative session counters must not be summed as independent turns."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from summarize_geometry_session_resources import turn_usage


class SessionResourceTests(unittest.TestCase):
    def test_resumed_turn_subtracts_prior_usage_and_counts_all_new_requests(self):
        previous = {'input_tokens': 100, 'output_tokens': 20, 'total_tokens': 120}
        current = {'input_tokens': 350, 'output_tokens': 70, 'total_tokens': 420}
        events = [{'last_token_usage': {'input_tokens': 150, 'output_tokens': 30, 'total_tokens': 180}},
                  {'last_token_usage': {'input_tokens': 100, 'output_tokens': 20, 'total_tokens': 120}}]
        self.assertEqual(turn_usage(previous, current, current, events),
                         {'input_tokens': 250, 'output_tokens': 50, 'total_tokens': 300})

    def test_mismatched_cli_and_event_counters_are_rejected(self):
        current = {'input_tokens': 100, 'output_tokens': 20, 'total_tokens': 120}
        with self.assertRaisesRegex(ValueError, 'CLI usage'):
            turn_usage({}, current, {'input_tokens': 101}, [])
        with self.assertRaisesRegex(ValueError, 'new token events'):
            turn_usage({}, current, current, [])

    def test_counter_reset_is_not_silently_counted(self):
        with self.assertRaisesRegex(ValueError, 'decreased'):
            turn_usage({'input_tokens': 101}, {'input_tokens': 100}, {'input_tokens': 100}, [])


if __name__ == '__main__':
    unittest.main()
