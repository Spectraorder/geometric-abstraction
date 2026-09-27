"""All-layout target feedback is bounded, arm-isolated and exactly reconstructable."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_constraint_geometry_experiment as variant


class ConstraintFeedbackTests(unittest.TestCase):
    def record(self, root, seed, revision=0, records=3, arm='symbolic'):
        name='revision_000' if revision==0 else f'round_{revision:03d}'
        trial=root/arm/name/'trials'/f'seed_{seed}';trial.mkdir(parents=True)
        (trial/'api_calls.json').write_text('[]')
        (trial/'geometry_records.json').write_text(json.dumps([
            {'kind':'geometry_step','record':{'object':'red','time':i,'status':'unavailable','constraints':{'seed_marker':seed}}}
            for i in range(records)]))
        return trial

    def test_every_layout_latest_execution_without_cross_arm_or_stale_fallback(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            for seed in range(5):self.record(root,seed)
            self.record(root,4,revision=1,records=0)
            rows=variant.target_feedback(root,'symbolic',2,list(range(5)))
            self.assertEqual([r['seed'] for r in rows],list(range(5)))
            self.assertEqual([r['time'] for r in rows[0]['checkpoints']],[1,2])
            self.assertEqual(rows[4]['revision'],1);self.assertEqual(rows[4]['checkpoints'],[])
            self.assertTrue(all(not r['checkpoints'] for r in variant.target_feedback(root,'basic',2,list(range(5)))))

    def test_budget_reduces_history_equally_without_dropping_layouts(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            for seed in range(3):self.record(root,seed)
            rows=variant.target_feedback(root,'symbolic',1,[0,1,2])
            latest=[dict(r,checkpoints=r['checkpoints'][-1:]) for r in rows]
            budget=len(json.dumps(latest,separators=(',',':')))
            self.assertEqual(variant.target_feedback(root,'symbolic',1,[0,1,2],budget),latest)
            with self.assertRaisesRegex(RuntimeError,'every training layout'):
                variant.target_feedback(root,'symbolic',1,[0,1,2],budget-1)

    def test_registration_audit_and_context_hooks_preserve_old_variant(self):
        old=(variant.parent.library,variant.parent.sources,variant.parent.context,variant.parent.RICH)
        with TemporaryDirectory() as tmp:
            root=Path(tmp)/'trial';manifest=variant.prepare(root,[11,12,13],[21],1)
            variant.check(root,manifest)
            self.assertEqual(manifest['variant'],variant.VERSION)
            result=variant.audit(root,allow_partial=True)
            self.assertEqual(result['exact_actor_contexts_verified'],0)
            for seed in (11,12,13):
                trial=self.record(root,seed)
                (trial/'actor_feedback.json').write_text(json.dumps({'seed':seed,'success':False,'policy_error':None,'notes':[]}))
            with patch.object(variant.base,'sensor_context',return_value=([],{'mock_sensor':True})):
                a,images=variant.context(root,'symbolic',1,manifest)
                b,again=variant.context(root,'symbolic',1,manifest)
            self.assertEqual(a,b);self.assertEqual(images,again)
            self.assertIn('"seed_marker":13',a)
            self.assertIn('pose_available=False',a)
            self.assertNotIn('Own recent action checkpoints',a)
            self.assertIn('Own latest target checkpoints for every training layout',a)
        self.assertEqual((variant.parent.library,variant.parent.sources,variant.parent.context,variant.parent.RICH),old)


if __name__=='__main__':unittest.main()
