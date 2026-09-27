"""Protect successful layouts, failure evidence, and bounded image escalation."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import geometry_iteration as iteration
import run_constraint_geometry_experiment as experiment
import test_geometry_actor_session as actor_tests


class IterationTests(unittest.TestCase):
    def curve(self, successes):
        return [{'revision': i, 'successes': len(values),
                 'results': [{'seed': seed, 'success': seed in values} for seed in range(5)]}
                for i, values in enumerate(successes)]

    def test_more_successes_cannot_trade_away_a_protected_case(self):
        curve = self.curve([{0}, {0,1}, {1,2,3}, {0,1,4}, {0,1,4}])
        decisions = iteration.retention(curve)
        self.assertEqual([d['selected_revision'] for d in decisions], [0,1,1,3,3])
        self.assertEqual(decisions[2]['regressed_seeds'], [0])
        self.assertEqual(decisions[2]['repaired_seeds'], [2,3])
        self.assertEqual(iteration.select_revision(curve, {'retention_rule': iteration.RULE}), 3)
        self.assertEqual(iteration.select_revision(curve, {}), 2)
        curve[-1]['results'].pop()
        with self.assertRaisesRegex(ValueError, 'seeds changed'):
            iteration.retention(curve)

    def test_rejected_policy_is_replaced_by_exact_incumbent_in_next_prompt(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)/'study'
            manifest = experiment.prepare(root, [0,1,2,3,4], [20], 3, actor_images='initial_only')
            for revision, successes in enumerate([{0}, {0,1}, {1,2,3}]):
                actor_tests.SessionContextTests().training(root, 'basic', revision, manifest['train_seeds'])
                for seed in manifest['train_seeds']:
                    path = iteration.stage(root,'basic',revision)/'trials'/f'seed_{seed}'/'actor_feedback.json'
                    row = json.loads(path.read_text()); row['success'] = seed in successes
                    path.write_text(json.dumps(row))
            incumbent = iteration.stage(root,'basic',1)/'candidate.py'
            incumbent.write_text('def run(api):\n    api.note("incumbent-marker")\n')
            prompt, images = experiment.context(root, 'basic', 3, manifest)
            self.assertIn('Parent to revise: R1', prompt)
            self.assertIn(incumbent.read_text(), prompt)
            self.assertIn('Latest proposal was rejected', prompt)
            self.assertEqual(images, [])

    def test_first_failure_survives_later_unrelated_checkpoints(self):
        records = [{'object':'blue','step':'pick','time':1,'status':'current_estimate'},
                   {'object':'blue','step':'place_exception','time':2,'action_status':'exception'},
                   {'object':'blue','step':'reobserve','time':3,'status':'unavailable'},
                   {'object':'red','step':'settled','time':4,'status':'unavailable'}]
        self.assertEqual([r['time'] for r in iteration.event_records(records)], [1,2,3])

    def test_current_revision_without_records_does_not_reuse_old_geometry(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            for revision in (0,1):
                actor_tests.SessionContextTests().training(root,'symbolic',revision,[1,2,3,4,5])
            trial=iteration.stage(root,'symbolic',1)/'trials/seed_5'
            (trial/'geometry_records.json').unlink()
            rows=iteration.event_feedback(root,'symbolic',2,[1,2,3,4,5])
            self.assertEqual(rows[-1]['checkpoints'], [])
            self.assertEqual(len(rows),5)
            with self.assertRaisesRegex(RuntimeError,'every training layout'):
                iteration.event_feedback(root,'symbolic',2,[1,2,3,4,5],budget=5)

    def test_failure_images_use_nearby_frames_and_one_layout_only(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)/'study'
            manifest=experiment.prepare(root,[1,2,3,4,5],[20],2)
            actor_tests.SessionContextTests().training(root,'basic',1,manifest['train_seeds'])
            trial=iteration.stage(root,'basic',1)/'trials/seed_1'
            calls=[]
            for i,(time,view) in enumerate([(1,'side'),(2,'side'),(2,'wrist'),(4,'side')]):
                calls.append({'method':'camera','time':time,'params':{'view':view}})
                np.savez(trial/f'camera_{i:03d}.npz',time=time,intrinsics=np.eye(3),camera_to_world=np.eye(4))
                (trial/f'camera_{i:03d}.png').write_bytes(b'fixture')
            calls.append({'method':'failed','time':3,'params':{}})
            (trial/'api_calls.json').write_text(json.dumps(calls))
            prompt,images=experiment.image_feedback(root,'basic',2,manifest)
            self.assertEqual([p.name for p in images],['camera_001.png','camera_002.png'])
            self.assertIn('unexplained_failure',prompt)
            self.assertIn('"seed": 1',prompt)
            self.assertIsNone(iteration.failure_kind({'success':True,'notes':['No usable current view']}))

    def test_success_reference_matches_task_role_across_color_permutations(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            actor_tests.SessionContextTests().training(root,'symbolic',0,[1,2])
            (root/'manifest.json').write_text(json.dumps({'color_maps': {
                '1': {'blue':'red'}, '2': {'blue':'yellow'}}}))
            trial=iteration.stage(root,'symbolic',0)/'trials/seed_2'
            (trial/'actor_feedback.json').write_text(json.dumps({'success':True,'policy_error':None}))
            (trial/'geometry_records.json').write_text(json.dumps([
                {'kind':'geometry_step','record':{'object':'yellow','step':'place','time':2}},
                {'kind':'geometry_step','record':{'object':'red','step':'place','time':3}}]))
            rows=iteration.event_feedback(root,'symbolic',1,[1,2])
            reference=rows[0]['successful_reference']
            self.assertEqual(reference['canonical_task_color'],'blue')
            self.assertEqual(reference['checkpoints'][0]['object'],'yellow')

    def test_custom_diagnostics_and_inactive_probe_reach_next_feedback(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)/'study'
            manifest=experiment.prepare(root,[1,2,3,4,5],[20],2,actor_images='initial_only')
            for revision in (0,1):
                actor_tests.SessionContextTests().training(root,'basic',revision,manifest['train_seeds'])
            trial=iteration.stage(root,'basic',1)/'trials/seed_1'
            path=trial/'geometry_records.json'
            records=json.loads(path.read_text())
            records += [{'color':'blue','step':'retry_evidence','residual_m':.039},
                        {'kind':'policy_probe','hypothesis':'test-change','triggered':False,
                         'step':'before_pick','object':'blue','reason':'threshold not reached'}]
            path.write_text(json.dumps(records))
            prompt,_=experiment.context(root,'basic',2,manifest)
            self.assertIn('retry_evidence',prompt)
            self.assertIn('test-change',prompt)
            self.assertIn('own_policy_report_unverified',prompt)
            self.assertIn('"motion_commands_identical":true',prompt)
            bounded,omitted=iteration.policy_diagnostics([{'data':'x'*5000}]*12)
            self.assertEqual(omitted,8)
            self.assertTrue(all('truncated_json_preview' in row['data'] for row in bounded))

    def test_behavior_comparison_uses_incumbent_not_latest_rejected_proposal(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            for revision,success in [(0,True),(1,False),(2,True)]:
                actor_tests.SessionContextTests().training(root,'basic',revision,[1])
                trial=iteration.stage(root,'basic',revision)/'trials/seed_1'
                (trial/'actor_feedback.json').write_text(json.dumps({'success':success,'policy_error':None}))
                calls=[{'method':'camera','params':{'view':'side'}},
                       {'method':'wait','params':{'seconds':2 if revision==1 else 1}}]
                (trial/'api_calls.json').write_text(json.dumps(calls))
            comparison=iteration.behavior_feedback(root,'basic',2,[1])[1]
            self.assertEqual(comparison['parent_revision'],0)
            self.assertTrue(comparison['motion_commands_identical'])
            self.assertEqual(comparison['candidate_command_count'],1)
            trial=iteration.stage(root,'basic',2)/'trials/seed_1'
            calls=json.loads((trial/'api_calls.json').read_text());calls[-1]['params']['seconds']=3
            (trial/'api_calls.json').write_text(json.dumps(calls))
            comparison=iteration.behavior_feedback(root,'basic',2,[1])[1]
            self.assertFalse(comparison['motion_commands_identical'])
            self.assertEqual(comparison['first_changed_command_index'],0)

    def test_guarded_campaign_and_audit_agree(self):
        from test_mixed_scene_learning_audit import LearningAuditTests
        import run_mixed_scene_experiment as base
        from summarize_mixed_scene_learning import audit
        original=base.prepare
        def prepare(*args, **kwargs):
            manifest=original(*args, **kwargs)
            manifest['retention_rule']=iteration.RULE
            manifest['evaluation_workers']=4
            base.write(args[0]/'manifest.json',manifest)
            return manifest
        with TemporaryDirectory() as tmp, patch.object(base,'prepare',side_effect=prepare):
            root=Path(tmp)/'study'
            LearningAuditTests().fixture(root,arms=('basic','symbolic'),train=[1,2,3,4,5],rounds=2)
            report=audit(root)
            self.assertEqual(report['status'],'completed')
            self.assertTrue(all(c['selected_revision']==1 for c in report['curves'].values()))


if __name__ == '__main__':
    unittest.main()
