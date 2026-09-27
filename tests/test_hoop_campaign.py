"""Continuation retains its measured baseline and isolates frozen evaluation."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('hoop_learner',ROOT/'scripts/learn_hoop_pole.py')
learner=importlib.util.module_from_spec(spec);spec.loader.exec_module(learner)


class HoopCampaignTests(unittest.TestCase):
    def test_regression_keeps_baseline_and_freezes_before_fresh_tests(self):
        with TemporaryDirectory() as td:
            root=Path(td);source=root/'source.py'
            source.write_text('def run(api): api.wait(.1)\n')
            (root/'initial.png').touch()
            folder=root/'round_001';folder.mkdir()
            (folder/'candidate.py').write_text('def run(api): api.wait(.2)\n')
            (folder/'model.jsonl').write_text(json.dumps(dict(type='turn.completed',usage=dict(input_tokens=1,output_tokens=1)))+'\n')
            a=argparse.Namespace(root=root,model='gpt-6-astra',hoop_variant='tight',train_seeds=[1,2],heldout_seeds=[3,4],
                reference_policy=[],initial_policy=source,rounds=1,max_reported_tokens=60000,max_prompt_chars=40000,
                trial_timeout=300,llm_timeout=300)
            calls=[]
            def trial(candidate,directory,seed,timeout,variant):
                calls.append((str(candidate),seed,variant))
                if seed in a.heldout_seeds:
                    state=json.loads((root/'campaign.json').read_text())
                    self.assertEqual(state['status'],'frozen_evaluation')
                    self.assertEqual(state['frozen_sha256'],hashlib.sha256(source.read_bytes()).hexdigest())
                success=candidate.read_text()==source.read_text() and seed%2==1
                result=dict(success=success,policy_error=None,metrics=dict(milestones={'stable':int(success)},stable_seconds=3. if success else 0.,center_error_m=.002 if success else .004))
                directory.mkdir(parents=True,exist_ok=True)
                (directory/'actor_feedback.json').write_text(json.dumps(dict(notes=[],success=success)))
                return result
            with patch.object(learner,'trial',side_effect=trial):learner.run(a,root)
            state=json.loads((root/'campaign.json').read_text())
            self.assertEqual((root/'best_policy.py').read_bytes(),source.read_bytes())
            self.assertEqual(state['best']['round'],0)
            self.assertEqual(state['status'],'completed')
            self.assertEqual([seed for _,seed,_ in calls],[1,2,1,2,3,4,3,4])
            self.assertTrue(all(v=='tight' for _,_,v in calls))
            self.assertNotIn('heldout', (root/'context_001.txt').read_text())
            with patch.object(learner,'trial') as trial_again:
                learner.run(a,root);trial_again.assert_not_called()
            a.heldout_seeds=[5,6]
            with self.assertRaisesRegex(RuntimeError,'configuration changed'):learner.run(a,root)

    def test_cached_standard_geometry_cannot_be_reused_for_tight(self):
        with TemporaryDirectory() as td:
            root=Path(td);candidate=root/'policy.py';candidate.write_text('def run(api): pass\n')
            result=dict(candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),seed=1,task='hoop_pole',
                scene=dict(generator='hoop_pole_learning_v1',controller=learner.CONTROLLER,evaluator='hoop_seated_v1'))
            (root/'result.json').write_text(json.dumps(result))
            with self.assertRaisesRegex(RuntimeError,'Cached trial'):
                learner.trial(candidate,root,1,300,'tight')


if __name__=='__main__':unittest.main()
