"""Freeze the visual baseline contract, context reconstruction and global gate."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_visual_input_experiment as variant
import summarize_visual_input_experiment as summary
import run_visual_input_replications as study
import test_mixed_scene_replications as helpers


class VisualExperimentTests(unittest.TestCase):
    def test_registered_contract_and_scoped_hooks(self):
        before=(variant.base.library,variant.base.context,variant.base.sources)
        with TemporaryDirectory() as tmp:
            root=Path(tmp)/'trial';manifest=variant.prepare(root,[11],[21],1)
            variant.check(root,manifest)
            self.assertEqual(manifest['schema'],'visual_input_geometry_pilot_v1')
            self.assertEqual((root/'starter.py').read_text(),(variant.ROOT/'policies/support_graph_visual_starter.py').read_text())
            reports=summary.audit(root,allow_partial=True)
            self.assertEqual(reports['exact_actor_contexts_verified'],0)
            self.assertEqual(reports['starter_parity_audit'],'pending')
            self.assertFalse(list(root.glob('*/round_*/invocation.json')))
            with variant.configured():self.assertIs(variant.base.library,variant.library)
            self.assertEqual((variant.base.library,variant.base.context,variant.base.sources),before)
            for arm in variant.ARMS:
                p=root/arm/'revision_000/trials/seed_11';p.mkdir(parents=True)
                (p/'actor_feedback.json').write_text(json.dumps({'seed':11,'success':False,'policy_error':None,'notes':['own visual estimate']}))
                prompt,images=variant.context(root,arm,1,manifest)
                self.assertIn('own visual estimate',prompt)
                self.assertFalse(images)
                self.assertNotIn('Both conditions provide get(color)',prompt)
                if arm=='basic':
                    self.assertIn('Session has NO observe, get, table_z',prompt)
                    self.assertNotIn('Full additionally provides observe(color)',prompt)
                else:self.assertIn('Full additionally provides observe(color)',prompt)
            (root/'starter.py').write_text('def run(api): pass\n')
            with self.assertRaises(RuntimeError):variant.check(root,manifest)

    def test_global_freeze_precedes_every_evaluation_and_resume(self):
        helper=helpers.ReplicationTests()
        with TemporaryDirectory() as tmp:
            root,protocol=helper.prepare(Path(tmp));events=[]
            def campaign(directory,manifest,phase):
                events.append(phase)
                if phase=='train':helper.fake_train(directory,manifest,phase)
                else:
                    self.assertTrue((root/'all_candidates_frozen.json').exists())
                    self.assertTrue(all((root/r['directory']/'training_summary.json').exists() for r in protocol['replicas']))
                    variant.base.write(directory/'summary.json',{'status':'completed'})
            original=study.coordinator.campaign
            with patch.object(study,'check',lambda *a:None),patch.object(study,'campaign',campaign):
                study.run(root,protocol,'all');study.run(root,protocol,'all')
            self.assertIs(study.coordinator.campaign,original)
            self.assertEqual(events,['train']*3+['evaluate']*3)

    def test_training_runs_parity_then_context_audit(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);events=[]
            with patch.object(variant,'bootstrap',lambda *a:events.append('parity')),patch.object(variant.base,'campaign',lambda *a:events.append('train')),patch.object(study,'audit',lambda *a,**kw:events.append(('audit',kw['allow_partial'])) or {'status':'trained'}):
                study.campaign(root,{},'train')
            self.assertEqual(events,['parity','train',('audit',True)])
            self.assertTrue((root/'train_audit.json').exists())


if __name__=='__main__':unittest.main()
