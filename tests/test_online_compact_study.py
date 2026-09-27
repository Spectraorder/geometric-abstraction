"""Check fresh paired scheduling, codec selection and independent prompt audit."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_online_compact_study as study
from audit_online_compact_prompts import check_feedback
from online_compact_geometry import compact_prompt


class CompactStudyTests(unittest.TestCase):
    def test_registration_uses_fixed_fresh_pairs(self):
        self.assertTrue(set(study.SEEDS).isdisjoint(study.shared.SEEDS))
        slots = study.shared.slots(study.SEEDS)
        self.assertEqual(len(slots), 10)
        for offset in range(0, 10, 4):
            batch = slots[offset:offset+4]
            for seed in {s['seed'] for s in batch}:
                self.assertEqual({s['arm'] for s in batch if s['seed'] == seed}, {'basic', 'symbolic'})
        with patch.object(study.shared, 'prepare') as prepare:
            study.prepare(Path('/unused'))
            prepare.assert_called_once_with(Path('/unused'), seeds=study.SEEDS, compact_text=True)

    def test_worker_passes_same_budget_and_compact_adapter_to_both_arms(self):
        for arm in ('basic', 'symbolic'):
            slot = {'seed': study.SEEDS[0], 'arm': arm, 'path': arm}
            protocol = {'slots': [slot], 'budget_s': 200., 'compact_text': True}
            with patch.object(study.shared, 'verify_sources', return_value=protocol) as verify, \
                 patch.object(study.shared, '_run_episode', return_value={'status':'timeout','success':False}) as run:
                study.shared.worker(Path('/unused'), 0)
                run.assert_called_once_with(Path('/unused')/arm, study.SEEDS[0], arm, 200.,
                                            kind='explicit_target_reserved', compact_text=True)
                self.assertEqual(verify.call_count, 2)

    def test_independent_audit_rejects_changed_geometry_and_base_leak(self):
        geometry = {'time': 2., 'objects': [
            {'object': str(i), 'references': {'center': {'position_m': [0., 1., 2.],
              'position_fully_observed': True, 'coordinate_frame': 'world', 'units': 'meters',
              'source': 'mixed_rigid_rgbd_observation', 'observation_time': 1., 'inferred_orientation': False}}}
            for i in range(5)]}
        packet = {'time':2., 'geometry':geometry}
        prompt = 'Interface: symbolic\nCurrent feedback:\n' + json.dumps(packet)
        compact, used = compact_prompt(prompt, explain=True)
        self.assertTrue(used)
        self.assertTrue(check_feedback(compact, packet, 'symbolic', explanation_needed=True))
        packet['geometry']['objects'][0]['references']['center']['position_m'][0] = 9.
        with self.assertRaises(ValueError): check_feedback(compact, packet, 'symbolic', explanation_needed=True)
        with self.assertRaises(ValueError): check_feedback(compact, packet, 'basic', explanation_needed=True)


if __name__ == '__main__':
    unittest.main()
