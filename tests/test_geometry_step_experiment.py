"""Compact two-arm feedback, source freezing and scoped orchestration hooks."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_geometry_step_experiment as variant
from summarize_mixed_scene_learning import audit


class StepExperimentTests(unittest.TestCase):
    def test_bounded_feedback_excludes_legacy_global_packets(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'geometry_records.json'
            rows = [{'kind':'planner_state', 'state':{'secret_legacy_marker':'x'*50000}}]
            rows += [{'kind':'geometry_step', 'record':{'object':'red','step':str(i)}} for i in range(12)]
            path.write_text(json.dumps(rows))
            selected = variant.compact_checkpoints(path)
            self.assertEqual([r['record']['step'] for r in selected], list(map(str,range(4,12))))
            self.assertNotIn('secret_legacy_marker', json.dumps(selected))
            rows[-1]['record']['oversized'] = 'x'*17000
            path.write_text(json.dumps(rows))
            self.assertLessEqual(len(json.dumps(variant.compact_checkpoints(path),separators=(',',':'))),16000)

    def test_registration_freezes_new_starter_and_scoped_hooks_restore(self):
        before = (variant.base.library,variant.base.context,variant.base.validate_planner)
        with TemporaryDirectory() as tmp:
            root=Path(tmp)/'pilot'; manifest=variant.prepare(root,[31],[41],rounds=1)
            self.assertEqual(manifest['arms'],['basic','symbolic'])
            self.assertTrue(set(variant.EXTRA) <= set(manifest['source_sha256']))
            self.assertEqual((root/'starter.py').read_text(),(variant.ROOT/'policies/support_graph_step_starter.py').read_text())
            variant.check(root,manifest)
            self.assertEqual(audit(root,allow_partial=True)['valid_rollouts'],0)
        self.assertEqual(before,(variant.base.library,variant.base.context,variant.base.validate_planner))
        with self.assertRaisesRegex(ValueError,'target-only'):
            variant.validate_planner('def run(api):\n    Session(api, []).snapshot()\n')

    def test_prompt_protocol_does_not_request_scene_packets(self):
        self.assertNotIn("'state':session.snapshot()",variant.COMMON)
        self.assertIn('checkpoint(color, step',variant.COMMON)
        self.assertNotIn('state() includes',variant.RICH)


if __name__=='__main__':unittest.main()
