"""Token broadcasts must not inflate request consumption or hide counter errors."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from verified_token_usage import turn_usage


def usage(i,o):
    return {'input_tokens':i,'output_tokens':o,'total_tokens':i+o}


class TokenUsageTests(unittest.TestCase):
    def test_unchanged_broadcast_is_not_another_request(self):
        previous=usage(20,5); current=usage(35,8)
        events=[{'total_token_usage':previous,'last_token_usage':usage(20,5)},
                {'total_token_usage':current,'last_token_usage':usage(15,3)},
                {'total_token_usage':current,'last_token_usage':usage(15,3)}]
        self.assertEqual(turn_usage(previous,current,{'input_tokens':35,'output_tokens':8},events),usage(15,3))

    def test_multiple_requests_are_checked_individually(self):
        events=[{'total_token_usage':usage(10,2),'last_token_usage':usage(10,2)},
                {'total_token_usage':usage(30,7),'last_token_usage':usage(20,5)}]
        self.assertEqual(turn_usage({},usage(30,7),usage(30,7),events),usage(30,7))
        events[0]['last_token_usage']=usage(11,2)
        with self.assertRaisesRegex(ValueError,'transition'):
            turn_usage({},usage(30,7),usage(30,7),events)

    def test_changed_cli_and_decreasing_totals_fail(self):
        with self.assertRaisesRegex(ValueError,'CLI usage'):
            turn_usage({},usage(20,2),usage(21,2),[])
        with self.assertRaisesRegex(ValueError,'decreased'):
            turn_usage(usage(30,3),usage(20,2),usage(20,2),[{'total_token_usage':usage(20,2),'last_token_usage':usage(20,2)}])


if __name__=='__main__': unittest.main()
