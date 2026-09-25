"""Selection and resume tests do not invoke an LLM or simulator."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from aspire_so100.repair_learning import (
    annotate, build_context, diagnostics, select_candidates, source_for_prompt,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import learn_inventory_repair as learner


def entry(name, stage=None, *, success=False, safe=True, controls=True, version=2):
    results=[]
    for case in ('centered','offset'):
        results.append({'success': controls if case=='centered' else success,
            'scene':{'repair_case':case}, 'policy_error':None,
            'repair':{'collapse_seen':not safe,'max_base_drift_m':0.,
                      'initial_offset_m':[.007,0.], 'final_center_error_m':.007},
            'repair_progress':{'version':version,'verified':{
                s:stage is not None and i<=('identify','plan','grasp').index(stage)
                for i,s in enumerate(('identify','plan','grasp'))}}})
    return annotate({'candidate':name,'sha256':name,'results':results,
                     'successes':sum(r['success'] for r in results),'trials':2,'score':.8})


class RepairLearningTests(unittest.TestCase):
    def test_noop_is_not_repair_and_tie_keeps_champion(self):
        original=entry('original')
        recent=entry('recent')
        recent['score']=.99
        champion,parent,archive,focus=select_candidates([original,recent])
        self.assertEqual(champion['candidate'],'original')
        self.assertEqual(parent['candidate'],'recent')
        self.assertEqual(archive,{})
        self.assertEqual(focus,'identify')
        self.assertEqual(champion['case_metrics']['offset']['passed'],0)

    def test_intermediate_progress_survives_without_becoming_best_repair(self):
        champion,parent,archive,focus=select_candidates([
            entry('original'),entry('planner','plan'),entry('later_noop')])
        self.assertEqual(champion['candidate'],'original')
        self.assertEqual(parent['candidate'],'planner')
        self.assertEqual(archive['plan']['candidate'],'planner')
        self.assertEqual(focus,'grasp')

    def test_safety_and_centered_regressions_cannot_win(self):
        for bad in (entry('collapsed','grasp',safe=False),
                    entry('disturbed','grasp',controls=False)):
            champion,parent,archive,_=select_candidates([entry('original'),bad])
            self.assertEqual(parent['candidate'],'original')
            self.assertFalse(archive)

    def test_legacy_stage_claims_and_ungrounded_partial_gain_ignored(self):
        old=entry('old','grasp',version=1)
        old['results'][1]['repair']['final_center_error_m']=.001
        old=annotate(old)
        self.assertEqual(old['case_metrics']['offset']['verified']['grasp'],0)
        self.assertEqual(old['case_metrics']['offset']['mean_verified_improvement_m'],0)
        valid=entry('valid','grasp')
        valid['results'][1]['repair']['final_center_error_m']=.004
        self.assertAlmostEqual(annotate(valid)['case_metrics']['offset']['mean_verified_improvement_m'],.002)

    def test_success_outweighs_milestones(self):
        champion,parent,_,_=select_candidates([entry('grasp','grasp'),entry('repair',success=True)])
        self.assertEqual(champion['candidate'],'repair')
        self.assertEqual(parent['candidate'],'repair')

    def test_legacy_missing_ik_response_remains_unknown(self):
        result=diagnostics([{'method':'plan_tool','params':{'position':[0,0,1]}},
                            {'method':'gripper','params':{'angle':-.174}},
                            {'method':'camera','params':{}}])
        self.assertIsNone(result['ik_previews'][0]['result'])
        self.assertEqual(result['gripper_close_requests'],1)
        self.assertEqual(result['camera_calls'],{'side':1})

    def test_source_compaction_is_display_only(self):
        source='def legacy(api):\n    api.wait(1)\ndef helper(api):\n    api.wait(2)\ndef run(api):\n    helper(api)\n'
        compact=source_for_prompt(source)
        self.assertNotIn('def legacy',compact)
        self.assertIn('def helper',compact)
        self.assertIn('def legacy',source)
        compile(compact,'<preview>','exec')

    def test_prompt_contains_actual_failed_edits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            history=[]
            for i in range(2):
                folder=root/str(i);folder.mkdir()
                source=folder/'candidate.py'
                source.write_text(f'def run(api):\n    api.wait({i+1})\n')
                candidate=entry(str(source))
                for j,result in enumerate(candidate['results']):
                    trial=folder/str(j);trial.mkdir()
                    result['directory']=str(trial)
                    (trial/'actor_feedback.json').write_text(json.dumps({'success':result['success'],'notes':[]}))
                history.append(candidate)
            (root/'1'/'response.json').write_text(json.dumps({'hypothesis':'Try waiting'}))
            (root/'1'/'invocation.json').write_text(json.dumps({'parent_sha256':history[0]['sha256']}))
            prompt=build_context(history[0],history,'identify','CONTRACT')
            self.assertIn('actual_code_diff',prompt)
            self.assertIn('+    api.wait(2)',prompt)
            self.assertIn('CURRENT FOCUS: identify',prompt)
            with self.assertRaisesRegex(RuntimeError,'no model call'):
                build_context(history[0],history,'identify','CONTRACT',limit=1)

    def test_report_contract_explains_camera_aliases_and_no_online_truth(self):
        self.assertIn('ALIASES of the SAME FIXED CAMERA',learner.CONTRACT)
        self.assertIn('return None',learner.CONTRACT)

    def test_heartbeat_and_failure_are_written_directly(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(learner,'EVENT_DIR',Path(tmp)):
            learner.run_logged([sys.executable,'-c','pass'],phase='test',timeout=10)
            events=[json.loads(line) for line in (Path(tmp)/'progress.log').read_text().splitlines()]
            self.assertEqual([e['message'] for e in events],['Starting test','Finished test'])
            with self.assertRaises(subprocess.CalledProcessError):
                learner.run_logged([sys.executable,'-c','raise SystemExit(2)'],phase='failure',timeout=10)

    def test_timeout_terminates_process(self):
        with patch.object(learner,'EVENT_DIR',None):
            with self.assertRaises(subprocess.TimeoutExpired):
                learner.run_logged([sys.executable,'-c','import time; time.sleep(20)'],phase='timeout',timeout=.05)

    def test_usage_requires_both_fields_and_does_not_double_count_reasoning(self):
        self.assertEqual(learner.usage_total([{'input_tokens':100,'output_tokens':20,
                                               'reasoning_output_tokens':5}]),120)
        for usage in ([],[{}],[{'input_tokens':1,'output_tokens':None}]):
            with self.assertRaises(ValueError): learner.usage_total(usage)

    def test_stagnation_stops_without_another_model_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);history=[]
            for i in range(3):
                candidate=root/f'candidate_{i}.py'
                candidate.write_text('def run(api):\n    api.wait(6)\n')
                history.append(entry(str(candidate)))
            learner.write(root/'campaign.json',{'model':'gpt-6-astra','status':'executing',
                'reported_tokens':0,'history':history,'v2_start_history_index':1})
            args=['learner','--root',str(root),'--rounds','20']
            with patch.object(sys,'argv',args),patch.object(learner,'EVENT_DIR',None), \
                 patch.object(learner,'propose') as propose,patch.object(learner,'evaluate') as evaluate:
                learner.main()
                propose.assert_not_called();evaluate.assert_not_called()
            self.assertEqual(json.loads((root/'campaign.json').read_text())['status'],'stalled')

    def test_failed_holdout_is_frozen_and_not_reused_for_learning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);candidate=root/'initial_policy.py'
            candidate.write_text('def run(api):\n    api.wait(6)\n')
            best=entry(str(candidate),success=True)
            best['results'][1]['directory']='example-offset-trial'
            learner.write(root/'campaign.json',{'model':'gpt-6-astra','status':'executing',
                'reported_tokens':0,'history':[best]})
            args=['learner','--root',str(root)]
            with patch.object(sys,'argv',args),patch.object(learner,'EVENT_DIR',None), \
                 patch.object(learner,'propose') as propose, \
                 patch.object(learner,'evaluate',return_value=entry('heldout')) as evaluate:
                learner.main();learner.main()
                propose.assert_not_called();evaluate.assert_called_once()
            self.assertEqual(json.loads((root/'campaign.json').read_text())['status'],'heldout_failed')

    def test_paid_pending_candidate_is_tested_despite_budget_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            initial=root/'initial_policy.py'; initial.write_text('def run(api):\n    api.wait(6)\n')
            baseline=entry(str(initial)); baseline['sha256']=learner.digest(initial)
            state={'model':'gpt-6-astra','status':'interrupted','reported_tokens':0,'history':[baseline]}
            learner.write(root/'campaign.json',state)
            round_dir=root/'round_001';round_dir.mkdir()
            candidate=round_dir/'candidate.py';candidate.write_text(initial.read_text())
            learner.write(round_dir/'usage.json',[{'input_tokens':123,'output_tokens':4}])
            learner.write(round_dir/'invocation.json',{'parent_sha256':baseline['sha256']})
            args=['learner','--root',str(root),'--rounds','1','--max-reported-tokens','1']
            with patch.object(sys,'argv',args),patch.object(learner,'EVENT_DIR',None), \
                 patch.object(learner,'propose') as propose, \
                 patch.object(learner,'evaluate',side_effect=KeyboardInterrupt) as evaluate:
                with self.assertRaises(KeyboardInterrupt):
                    learner.main()
                propose.assert_not_called()
                evaluate.assert_called_once()
            saved=json.loads((root/'campaign.json').read_text())
            self.assertEqual(saved['reported_tokens'],127)
            self.assertEqual(saved['status'],'interrupted')
            self.assertFalse(json.loads((root/'best_policy.json').read_text())['repair_demonstrated'])


if __name__=='__main__':
    unittest.main()
