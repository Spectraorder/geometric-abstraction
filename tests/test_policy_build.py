"""Generated candidates cannot overwrite user code or drift on a repeated build."""
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from build_inventory_policy import write_policy


class PolicyBuildTests(unittest.TestCase):
    def test_existing_candidate_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'candidate.py'
            path.write_text('def run(api):\n    api.note("my candidate")\n')
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                write_policy(path, 'column_policy')
            self.assertEqual(path.read_bytes(), before)

    def test_repeated_build_reuses_identical_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'generated'/'candidate.py'
            write_policy(path, 'column_policy')
            before, stamp = path.read_bytes(), path.stat().st_mtime_ns
            write_policy(path, 'column_policy')
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(path.stat().st_mtime_ns, stamp)
            with self.assertRaises(ValueError):
                write_policy(path, 'cube_alignment_policy')


if __name__ == '__main__':
    unittest.main()
