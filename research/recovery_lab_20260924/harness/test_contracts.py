import copy,unittest
from contracts import validate_belief,route,validate_action,propose_memory

def fixture():
    return {'version':4,'time_s':10.,'source_frames':['sensor_12.npz'],'objects':[{'id':'blue_0','visible':True,'last_seen_s':10.,'association':'unique','pose_sigma_m':.001,'geometry_origin':'fitted','support_margin_lower_m':.008}]}

class ContractTests(unittest.TestCase):
    def test_ready(self):self.assertEqual(route(fixture(),'blue_0'),'continue')
    def test_hidden(self):
        b=fixture();b['objects'][0]['visible']=False;self.assertEqual(route(b,'blue_0'),'reobserve')
    def test_bounded_reobserve(self):self.assertEqual(route(fixture(),'missing',reobservations=2),'escalate')
    def test_stale(self):
        b=fixture();b['objects'][0]['last_seen_s']=9.;self.assertEqual(route(b,'blue_0'),'reobserve')
    def test_ambiguous(self):
        b=fixture();b['objects'][0]['association']='ambiguous';self.assertEqual(route(b,'blue_0'),'reobserve')
    def test_unknown_support(self):
        b=fixture();del b['objects'][0]['support_margin_lower_m'];self.assertEqual(route(b,'blue_0'),'reobserve')
    def test_failed_grasp(self):
        b=fixture();b['objects'][0]['grasp_confirmed']=False;self.assertEqual(route(b,'blue_0'),'retry_grasp');self.assertEqual(route(b,'blue_0',retries=1),'escalate')
    def test_nested_oracle(self):
        b=fixture();b['objects'][0]['debug']={'true_pose':[0,0,0]}
        with self.assertRaises(ValueError):validate_belief(b)
    def test_uncertainty(self):
        b=fixture();b['objects'][0]['pose_sigma_m']=float('nan')
        with self.assertRaises(ValueError):validate_belief(b)
    def test_id_collision(self):
        b=fixture();b['objects'].append(copy.deepcopy(b['objects'][0]))
        with self.assertRaises(ValueError):validate_belief(b)
    def test_delayed_agent(self):
        a={'kind':'continue','belief_version':4,'expires_at_s':11.,'target_id':'blue_0','reason':'support observed'}
        self.assertEqual(validate_action(a,fixture(),10.1),a)
        for b,t in [(dict(fixture(),version=5),10.1),(fixture(),12.),(fixture(),10.6)]:
            with self.assertRaises(ValueError):validate_action(a,b,t)
    def test_unsafe_action(self):
        a={'kind':'continue','belief_version':4,'expires_at_s':11.,'target_id':'blue_0','reason':'guess'};b=fixture();b['objects'][0]['visible']=False
        with self.assertRaises(ValueError):validate_action(a,b,10.1)
    def test_no_heldout_memory(self):
        r={'split':'heldout','policy_hash':'abc','regression_suite_passed':True,'applicability':['transient initial occlusion'],'counterexamples':['persistent occlusion']}
        with self.assertRaises(ValueError):propose_memory(r)
        r['split']='development';self.assertEqual(propose_memory(r)['policy_hash'],'abc')
    def test_reject_arbitrary_code_action(self):
        a={'kind':'continue','belief_version':4,'expires_at_s':11.,'target_id':'blue_0','reason':'x','code':'exec(...)'}
        with self.assertRaises(ValueError):validate_action(a,fixture(),10.1)

if __name__=='__main__':unittest.main()
