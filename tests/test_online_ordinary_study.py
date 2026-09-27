"""Check paired scheduling and stop behavior without simulation or model calls."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_online_ordinary_study as study


class OrdinaryScheduleTests(unittest.TestCase):
    def test_each_batch_contains_whole_pairs(self):
        slots = study.slots()
        self.assertEqual(len(slots), 10)
        self.assertEqual(len({(row['seed'], row['arm']) for row in slots}), 10)
        for offset in range(0, 10, 4):
            batch = slots[offset:offset+4]
            for seed in {row['seed'] for row in batch}:
                self.assertEqual({row['arm'] for row in batch if row['seed']==seed}, set(study.ARMS))

    def test_failure_stops_after_current_batch_and_cannot_resume(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            protocol = {'slots':study.slots(), 'concurrency':4}
            with patch.object(study, 'verify_sources', return_value=protocol), \
                 patch.object(study.subprocess, 'call', return_value=1) as launch, \
                 patch.object(study, 'summarize') as summarize:
                with self.assertRaisesRegex(RuntimeError, 'Worker failed'):
                    study.execute(root)
                self.assertEqual(launch.call_count, 4)
                summarize.assert_not_called()
                self.assertEqual(json.loads((root/'progress.json').read_text())['status'], 'stopped')
                with self.assertRaisesRegex(FileExistsError, 'Previous attempts'):
                    study.execute(root)
                self.assertEqual(launch.call_count, 4)


if __name__ == '__main__':
    unittest.main()
