"""Secure transport is rigid motion with the tool, not just jaw contact."""
import math
import runpy
from pathlib import Path
import unittest
import numpy as np
from aspire_so100.grasp_quality import GraspQuality


class GraspQualityTests(unittest.TestCase):
    def sample(self, monitor, *, seconds=.1, x=0., slip=0., tilt=0., yaw=0., **flags):
        defaults=dict(airborne=True,touching=True,both_jaws=True,unsupported=True,engaged=False)
        defaults.update(flags)
        c,s=math.cos(yaw),math.sin(yaw)
        tool_rotation=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
        c,s=math.cos(tilt),math.sin(tilt)
        relative=np.array([[1.,0.,0.],[0.,c,-s],[0.,s,c]])
        tool=np.array([x,0.,.46])
        nut=tool+tool_rotation@np.array([slip,0.,-.02])
        monitor.update(seconds,nut,tool_rotation@relative,tool,tool_rotation,**defaults)

    def test_rigid_transport_and_tool_rotation_are_not_slip(self):
        m=GraspQuality()
        for i in range(15):self.sample(m,x=i*.003,yaw=i*.05)
        s=m.status()
        self.assertTrue(s['stable_grasp']);self.assertTrue(s['stable_transport'])
        self.assertLess(s['max_relative_slip_m'],1e-12)
        self.assertLess(s['max_relative_rotation_deg'],1e-5)
        self.assertEqual(s['slip_events'],0)

    def test_hover_is_not_transport(self):
        m=GraspQuality()
        for _ in range(15):self.sample(m)
        self.assertTrue(m.status()['stable_grasp'])
        self.assertFalse(m.status()['stable_transport'])

    def test_flip_while_both_jaws_touch_is_not_secure(self):
        m=GraspQuality();self.sample(m)
        for a in (39,90,150,150,150):self.sample(m,seconds=.4,tilt=math.radians(a))
        s=m.status()
        self.assertFalse(s['stable_grasp']);self.assertFalse(s['stable_transport'])
        self.assertEqual(s['airborne_two_jaw_fraction'],1.)
        self.assertAlmostEqual(s['max_relative_rotation_deg'],150.)
        self.assertEqual(s['slip_events'],1)

    def test_translation_slip_resets_window(self):
        m=GraspQuality()
        for _ in range(8):self.sample(m)
        self.sample(m,slip=.004)
        self.assertEqual(m.status()['current_stable_seconds'],0.)
        self.assertEqual(m.status()['slip_events'],1)
        self.assertFalse(m.status()['stable_grasp'])

    def test_contact_loss_support_and_threading_excluded(self):
        for flags in ({'both_jaws':False},{'unsupported':False},{'engaged':True},{'airborne':False}):
            with self.subTest(flags=flags):
                m=GraspQuality()
                for _ in range(8):self.sample(m)
                self.sample(m,**flags)
                for _ in range(8):self.sample(m)
                self.assertFalse(m.status()['stable_grasp'])
        m=GraspQuality()
        for i in range(20):self.sample(m,engaged=True,tilt=i*.1)
        self.assertEqual(m.status()['slip_events'],0)
        self.assertFalse(m.status()['stable_grasp'])

    def test_quality_ranking_prioritizes_score_then_secure_handling(self):
        root=Path(__file__).resolve().parents[1]
        rank=runpy.run_path(str(root/'scripts/learn_fasten_screws.py'))['rank']
        def entry(score=0,secure=False,count=1):
            return dict(stage='fasten',grasp_quality='grasp_quality_v1',results=[dict(success=False,
                policy_error=None,metrics=dict(score=score,grasp_quality={'red':dict(stable_grasp=secure,stable_transport=secure)},
                milestones=dict(approach=count,grasp=count,lift=count,align=0,engage=0,rotate=0,depth=0,fasten=0)))])
        self.assertGreater(rank(entry(secure=True)),rank(entry(count=3)))
        self.assertGreater(rank(entry(score=20)),rank(entry(secure=True)))


if __name__=='__main__':unittest.main()
