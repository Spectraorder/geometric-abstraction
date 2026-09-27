"""Pilot pairing, failure-inclusive timing and layout-level summary checks."""
import json
import hashlib
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from run_online_bridge_pilot import make_slots, SEEDS
import run_online_bridge_pilot as pilot
from summarize_online_bridge_pilot import summarize


class PilotTests(unittest.TestCase):
    def test_registration_seals_sources_and_rejects_later_edits(self):
        with TemporaryDirectory() as folder:
            home = Path(folder)
            source = home/'scripts'/'example.py'
            source.parent.mkdir()
            source.write_text('value = 1\n')
            development = home/'development'
            for index, seed in enumerate(range(2028100101, 2028100104)):
                case = development/str(seed)
                case.mkdir(parents=True)
                (case/'result.json').write_text(json.dumps({
                    'seed': seed, 'arm': 'basic' if index == 0 else 'symbolic',
                    'success': True, 'model_requests': [{'model': 'gpt-6-luna'}]}))
            root = home/'registration'
            with patch.object(pilot, 'ROOT', home), patch.object(pilot, 'source_paths', return_value=[source]), \
                 patch.object(pilot, 'online_library', return_value='fixed geometry'):
                registered = pilot.prepare(root, development)
                self.assertEqual(len(registered['slots']), 20)
                pilot.verify_sources(root)
                source.write_text('value = 2\n')
                with self.assertRaisesRegex(ValueError, 'Frozen source changed'):
                    pilot.verify_sources(root)

    def test_five_shared_layouts_and_balanced_arm_order(self):
        slots = make_slots()
        self.assertEqual(len(slots), 20)
        self.assertEqual(len({r['seed'] for r in slots}), 5)
        for seed in SEEDS:
            for repetition in range(2):
                selected = [r for r in slots if r['seed'] == seed and r['repetition'] == repetition]
                self.assertEqual({r['arm'] for r in selected}, {'basic', 'symbolic'})
            first = [r['arm'] for r in slots if r['seed'] == seed and r['repetition'] == 0]
            second = [r['arm'] for r in slots if r['seed'] == seed and r['repetition'] == 1]
            self.assertEqual(first, list(reversed(second)))

    def test_timeouts_count_at_budget_and_repeats_remain_grouped_by_layout(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            slots = make_slots(seeds=(10, 11), repetitions=2)
            protocol = {'slots': slots, 'budget_s': 300., 'seeds': [10, 11],
                        'geometry_sha256': 'geometry_hash', 'shared_skills': {},
                        'source_sha256': {'scripts/example.py': hashlib.sha256(b'example').hexdigest()},
                        'max_timing_lag_for_realtime_claim_s': 2.}
            (root/'protocol.json').write_text(json.dumps(protocol))
            for slot in slots:
                out = root/slot['path']
                out.mkdir(parents=True)
                success = slot['arm'] == 'symbolic' or slot['seed'] == 10
                result = {'seed': slot['seed'], 'arm': slot['arm'], 'budget_s': 300.,
                          'success': success, 'status': 'success' if success else 'timeout',
                          'time_to_success_s': 100. if success else None,
                          'capped_completion_seconds': 100. if success else 300.,
                          'private_final_status': {'stable': success},
                          'private_success_evidence': {'duration_sim_seconds': 5.,
                                                       'maximum_final_displacement_m': .001},
                          'initial_physics_state_sha256': str(slot['seed']),
                          'decisions': 5, 'timing': {'reset_count': 1, 'max_simulation_lag_seconds': .1,
                                                   'physics_during_model_wait': True, 'paced_actions': True}}
                (out/'result.json').write_text(json.dumps(result))
                (out/'skill_provenance.json').write_text(json.dumps({'geometry_sha256': 'geometry_hash'}))
                (out/'sources').mkdir()
                (out/'sources/example.py').write_bytes(b'example')
                (out/'source_hashes.json').write_text(json.dumps({'example.py': hashlib.sha256(b'example').hexdigest()}))
            summary = summarize(root)
            self.assertEqual(summary['aggregate']['basic']['success_rate'], .5)
            self.assertEqual(summary['aggregate']['basic']['mean_capped_completion_seconds'], 200.)
            self.assertEqual(summary['aggregate']['symbolic']['mean_capped_completion_seconds'], 100.)
            self.assertEqual(len(summary['paired_layouts']), 2)
            self.assertEqual(summary['paired_layouts'][1]['full_minus_base_success_rate'], 1.)


if __name__ == '__main__':
    unittest.main()
