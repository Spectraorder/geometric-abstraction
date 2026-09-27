"""Validate source provenance independently of mutable working files."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_online_reasoning_development import audit


class SnapshotAuditTests(unittest.TestCase):
    def fixture(self, root):
        episode = root/'episodes/123/basic'
        (episode/'sources').mkdir(parents=True)
        (episode/'sources/example.py').write_text('archived = True\n')
        expected = hashlib.sha256((episode/'sources/example.py').read_bytes()).hexdigest()
        (episode/'sources.json').write_text(json.dumps({'example.py':expected}))
        (episode/'result.json').write_text(json.dumps(dict(
            usage={}, incomplete_requests=0, model_requests_completed=0,
            timing={'reset_count':1}, status='success')))
        return episode

    def test_episode_snapshot_has_priority(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.fixture(root)
            (root/'sources').mkdir()
            (root/'sources/example.py').write_text('different batch snapshot\n')
            result = audit(root, ['episodes/123/basic'])
            self.assertEqual(result['status'], 'passed')
            self.assertEqual(result['episodes'][0]['episode'], 'episodes/123/basic')

    def test_corrupted_episode_snapshot_cannot_fall_back(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            episode = self.fixture(root)
            (root/'sources').mkdir()
            (root/'sources/example.py').write_bytes((episode/'sources/example.py').read_bytes())
            (episode/'sources/example.py').write_text('changed\n')
            with self.assertRaisesRegex(ValueError, 'Unavailable source version'):
                audit(root, ['episodes/123/basic'])


if __name__ == '__main__':
    unittest.main()
