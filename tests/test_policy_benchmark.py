import importlib.util,json,unittest
from pathlib import Path
R=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('benchmark',R/'scripts/benchmark_policy.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
S=json.loads((R/'benchmarks/policy_suite.json').read_text())
def record(i,cell='tower_nominal',**extra):
    return dict({'cell':cell,'case_id':str(i),'repeat':0,'status':'completed','passed_endpoint':True,'split':'heldout','policy_sha256':'a','harness_sha256':'b','evaluator_sha256':'c','configuration_sha256':'config','unsafe_actions':0},**extra)
class TestBenchmark(unittest.TestCase):
    def test_empty_not_complete(self):self.assertNotEqual(m.score(S,'all_supported',[],{})['status'],'meets_declared_profile')
    def test_duplicates_rejected(self):
        with self.assertRaises(ValueError):m.score(S,'tower_recovery',[record(1),record(1)],{})
    def test_development_cannot_release(self):
        with self.assertRaises(ValueError):m.score(S,'tower_recovery',[record(1,split='development')],{})
    def test_versions_cannot_mix(self):
        with self.assertRaises(ValueError):m.score(S,'tower_recovery',[record(1),record(2,policy_sha256='changed')],{})
    def test_infra_not_silent_exclusion(self):
        rr=[record(1,status='infrastructure_error',passed_endpoint=False)]
        c=m.score(S,'tower_recovery',rr,{})['cells'][0];self.assertEqual(c['n'],1);self.assertEqual(c['rate'],0)
    def test_repeats_do_not_inflate_coverage(self):
        rr=[record(1,repeat=i) for i in range(50)]
        row=m.score(S,'tower_recovery',rr,{})['cells'][0]
        self.assertEqual(row['n'],1);self.assertEqual(row['attempts'],50);self.assertFalse(row['coverage_complete'])
    def test_missing_safety_count_rejected(self):
        r=record(1);r.pop('unsafe_actions')
        with self.assertRaises(ValueError):m.score(S,'tower_recovery',[r],{})
    def test_config_cannot_mix(self):
        with self.assertRaises(ValueError):m.score(S,'tower_recovery',[record(1),record(2,configuration_sha256='changed')],{})
    def test_confidence_not_demo_rate(self):self.assertLess(m.wilson(2,2)[0],.5)
    def test_all_cells_and_gates(self):
        rr=[record(i,c) for c in S['profiles']['tower_recovery'] for i in range(50)]
        self.assertEqual(m.score(S,'tower_recovery',rr,{g:True for g in S['gates']})['status'],'meets_declared_profile')
    def test_unsafe_action_blocks(self):
        rr=[record(i,c,unsafe_actions=int(i==0)) for c in S['profiles']['tower_recovery'] for i in range(50)]
        self.assertNotEqual(m.score(S,'tower_recovery',rr,{g:True for g in S['gates']})['status'],'meets_declared_profile')
