"""Exercise complete evidence accounting with synthetic, non-executed trials.

The fixtures are temporary protocol records, not measured experiment results.
No model service, robot simulation or frozen pilot file is accessed or modified.
"""
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_mixed_scene_experiment as campaign
import summarize_mixed_scene_learning as audit


class LearningAuditTests(unittest.TestCase):
    def fixture(self, root, invalid_arm=None, arms=None, train=None):
        manifest = campaign.prepare(root, 1, [91001] if train is None else train, [92001,92002], arms=arms)
        def propose(directory, prompt, images, manifest):
            directory.mkdir(parents=True)
            source = 'def run(api):\n    pass\n'
            if directory.parent.name == invalid_arm:
                source = 'def run(api):\n    api.move_tool([0,0,0])\n'
            (directory/'candidate.py').write_text(source)
            (directory/'prompt.txt').write_text(prompt)
            usage = {'input_tokens':1,'output_tokens':1}
            campaign.write(directory/'response.json',{'code':source,'hypothesis':'Synthetic test only'})
            campaign.write(directory/'generation.json',{'candidate_sha256':campaign.digest(directory/'candidate.py'),
                                                       'usage':usage,'wall_seconds':0.})
            campaign.write(directory/'invocation.json',{'prompt_sha256':campaign.digest(directory/'prompt.txt'),
                'fresh_context':True,'tools_disabled':True,'repository_context':False,'user_config_loaded':False,
                'skills_loaded':False,'images':[], 'command':['codex','exec','--model',manifest['model'],
                '--ignore-user-config','--ephemeral','-c','model_reasoning_effort="high"']})
            (directory/'model.jsonl').write_text(json.dumps({'type':'turn.completed','usage':usage})+'\n')
        def execute(root, arm, candidate, directory, seeds, manifest, error=None):
            rows=[]
            for seed in seeds:
                out=directory/f'seed_{seed}'; out.mkdir(parents=True)
                revision_one = 'round_001' in str(candidate) or 'revision_001.py' in str(candidate) or candidate.name == 'frozen.py'
                success = bool(revision_one and not error)
                value={'seed':seed,'success':success,'policy_error':error,'candidate_sha256':campaign.digest(candidate)}
                if error:
                    value.update(invalid_candidate=True,rollout_executed=False)
                else:
                    value.update(geometry_library_sha256=manifest['library_sha256'][arm],task=manifest['task'],
                        scene={'color_map':manifest['color_maps'][str(seed)],'recipe':manifest['recipe'],'object_count':4},
                        physics=manifest['physics'],sensor=manifest['sensor'],wrist_depth=True,sensor_blackout=False)
                    (out/'initial.png').write_bytes(b'Synthetic initial image '+str(seed).encode())
                campaign.write(out/'result.json',value)
                rows.append({'seed':seed,'success':success,'policy_error':error,'rollout_executed':not bool(error)})
            return rows
        with patch.object(campaign,'propose',propose),patch.object(campaign,'execute',execute),patch.object(campaign,'context',return_value=('Synthetic test context',[])):
            campaign.campaign(root,manifest)
        return manifest

    def test_full_accounting_and_invalid_slots_are_preserved(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)/'fixture'; self.fixture(root,invalid_arm='basic')
            report=audit.audit(root)
            self.assertEqual(report['valid_rollouts'],15)
            self.assertEqual(report['invalid_slots'],3)
            self.assertEqual(len(report['actors']),3)
            self.assertEqual(report['curves']['basic']['selected_revision'],0)
            self.assertEqual(report['curves']['numeric']['selected_revision'],1)
            self.assertEqual(report['curves']['symbolic']['first_reserved_revision_at_80_percent'],1)
            self.assertEqual(report['global_freeze_audit'],'passed')

    def test_two_arm_representation_comparison_has_complete_accounting(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)/'fixture'
            self.fixture(root, arms=('numeric', 'symbolic'))
            report=audit.audit(root)
            self.assertEqual(report['valid_rollouts'], 12)
            self.assertEqual(len(report['actors']), 2)
            self.assertEqual(set(report['curves']), {'numeric', 'symbolic'})
            self.assertEqual(set(report['comparisons']), {'numeric_versus_symbolic'})
            self.assertFalse((root/'basic').exists())

    def test_same_total_but_changed_training_outcomes_are_rejected(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)/'fixture'; self.fixture(root)
            for name in ('training_summary.json','summary.json'):
                path=root/name; value=audit.read(path)
                value['curves']['basic'][1]['results'][0]['seed']=91002
                campaign.write(path,value)
            with self.assertRaisesRegex(ValueError,'Training outcomes differ'):
                audit.audit(root)

    def test_reserved_trial_start_before_freeze_is_rejected(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)/'fixture'; self.fixture(root)
            image=root/'basic/heldout/seed_92001/initial.png'
            before=(root/'all_candidates_frozen.json').stat().st_mtime-10
            os.utime(image,(before,before))
            with self.assertRaisesRegex(ValueError,'started before global freeze'):
                audit.audit(root)

    def test_foreign_model_and_non_boolean_success_are_rejected(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)/'fixture'; self.fixture(root)
            path=root/'basic/round_001/invocation.json'; value=audit.read(path)
            value['command'][3]='different-model'; campaign.write(path,value)
            with self.assertRaisesRegex(ValueError,'model mismatch'):
                audit.audit(root)
        with TemporaryDirectory() as tmp:
            root=Path(tmp)/'fixture'; self.fixture(root)
            path=root/'basic/revision_000/trials/seed_91001/result.json'; value=audit.read(path)
            value['success']='false'; campaign.write(path,value)
            with self.assertRaisesRegex(ValueError,'literal boolean'):
                audit.audit(root)

    def test_selected_source_cannot_differ_from_training_revision(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)/'fixture'; self.fixture(root)
            path=root/'basic/frozen.py'; path.write_text('changed source')
            for name in ('training_summary.json','summary.json','all_candidates_frozen.json'):
                file=root/name; value=audit.read(file)
                frozen=value['selected'] if name=='all_candidates_frozen.json' else value['frozen']
                frozen['basic']['sha256']=campaign.digest(path); campaign.write(file,value)
            # Preserve freeze ordering after changing the synthetic metadata.
            stamp=(root/'all_candidates_frozen.json').stat().st_mtime+1
            for file in root.glob('*/**/heldout/seed_*/*'):
                os.utime(file,(stamp,stamp))
            with self.assertRaisesRegex(ValueError,'Selected policy differs'):
                audit.audit(root)

    def test_pairing_rejects_duplicate_seeds(self):
        with self.assertRaisesRegex(ValueError,'duplicate'):
            audit.paired([{'seed':1,'success':True}]*2,[{'seed':1,'success':False}])
        rows=lambda flags:[{'seed':i,'success':success} for i,success in enumerate(flags)]
        paired=audit.paired(rows([True,True,False,False]),rows([True,False,True,False]))
        self.assertEqual([paired[k] for k in ('both_success','first_only','second_only','neither')],[1,1,1,1])
        self.assertEqual(paired['exact_mcnemar_p_unadjusted'],1.)
