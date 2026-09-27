"""Policy retention may use past training results only, with the declared tie rule."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from summarize_retained_policy_budgets import retained_revisions


class RetentionTests(unittest.TestCase):
    def rows(self,counts):
        return [{'revision':i,'successes':n,'total_slots':5} for i,n in enumerate(counts)]

    def test_regressions_and_ties_keep_the_earliest_training_winner(self):
        self.assertEqual(retained_revisions(self.rows([2,1,0,1])),[0,0,0,0])
        self.assertEqual(retained_revisions(self.rows([2,2,3,3])),[0,0,2,2])

    def test_future_improvement_cannot_change_an_earlier_budget_choice(self):
        prefix=self.rows([1,1,0]);extended=self.rows([1,1,0,5])
        self.assertEqual(retained_revisions(prefix),retained_revisions(extended)[:3])
        self.assertEqual(retained_revisions(extended)[-1],3)


if __name__=='__main__':unittest.main()
