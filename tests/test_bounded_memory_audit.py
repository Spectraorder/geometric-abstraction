"""Memory must remain conditional and be rejected by contradictory current bounds."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_bounded_memory import predict


def observation(time,z,rank,lower,upper):
    return {'time':time,'color':'object','center_m':[0.,0.,z],'observed_rank':rank,
            'observed_projector':np.diag([1.,1.,float(rank==3)]).tolist(),
            'rotation':np.eye(3).tolist(),'bounds_available':True,
            'center_local_lower_m':[-.001,-.001,lower],'center_local_upper_m':[.001,.001,upper]}


class BoundedMemoryTests(unittest.TestCase):
    def test_motion_along_hidden_axis_invalidates_static_anchor(self):
        packet={'trial':'synthetic','rows':[observation(0,.03,3,.029,.031),observation(1,.07,2,.068,.072)]}
        output=predict(packet)['rows'];naive=next(r for r in output if r['time']==1 and r['mode']=='static_memory')
        gated=next(r for r in output if r['time']==1 and r['mode']=='bounded_static_memory')
        self.assertAlmostEqual(naive['center_m'][2],.03)
        self.assertAlmostEqual(gated['center_m'][2],.07)
        self.assertTrue(gated['memory_rejected']);self.assertFalse(gated['memory_used'])
        self.assertIsNone(gated['full_anchor_time'])

    def test_compatible_memory_does_not_become_current_full_observation(self):
        packet={'trial':'synthetic','rows':[observation(0,.03,3,.029,.031),observation(1,.032,2,.029,.034)]}
        value=next(r for r in predict(packet)['rows'] if r['time']==1 and r['mode']=='bounded_static_memory')
        self.assertAlmostEqual(value['center_m'][2],.03)
        self.assertEqual(value['observed_rank'],2)
        self.assertTrue(value['memory_used']);self.assertEqual(value['full_anchor_time'],0)
        self.assertIsNotNone(value['memory_condition'])


if __name__=='__main__':unittest.main()
