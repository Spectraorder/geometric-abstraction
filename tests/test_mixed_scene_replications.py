"""Protect study-wide freezing and resumption across independent actor chains."""
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import run_mixed_scene_experiment as experiment
import run_mixed_scene_replications as study
import summarize_mixed_scene_replications as report


class ReplicationTests(unittest.TestCase):
    def prepare(self, path):
        pilot = path/'pilot'
        manifest = experiment.prepare(pilot)
        # Model the previous runner revision; runtime libraries remain identical.
        manifest['source_sha256']['scripts/run_mixed_scene_experiment.py'] = 'previous-runner'
        experiment.write(pilot/'manifest.json', manifest)
        root = path/'replications'
        protocol = study.initialize(root, pilot)
        return root, protocol

    def fake_train(self, directory, manifest, phase):
        self.assertEqual(phase, 'train')
        selected = {}; revisions = {}
        for arm in manifest['arms']:
            (directory/arm/'frozen_revisions').mkdir(parents=True)
            (directory/arm/'frozen.py').write_text('def run(api): pass\n')
            selected[arm] = {'revision': 0, 'sha256': experiment.digest(directory/arm/'frozen.py')}
            revisions[arm] = []
            for revision in range(manifest['rounds']+1):
                p = directory/arm/'frozen_revisions'/f'revision_{revision:03d}.py'
                p.write_text('def run(api): pass\n')
                revisions[arm].append({'revision': revision, 'sha256': experiment.digest(p), 'validation_error': None})
        experiment.write(directory/'training_summary.json', {'manifest_sha256': experiment.digest(directory/'manifest.json'),
            'frozen': selected, 'revision_frozen': revisions})
        experiment.write(directory/'all_candidates_frozen.json', {'selected': selected, 'revisions': revisions})

    def test_no_reserved_execution_before_all_chains_freeze_and_resume_keeps_timestamp(self):
        with TemporaryDirectory() as tmp:
            root, protocol = self.prepare(Path(tmp)); events=[]
            def campaign(directory, manifest, phase):
                events.append(phase)
                if phase == 'train':
                    self.fake_train(directory, manifest, phase)
                else:
                    self.assertTrue((root/'all_candidates_frozen.json').exists())
                    self.assertTrue(all((root/r['directory']/'training_summary.json').exists() for r in protocol['replicas']))
                    experiment.write(directory/'summary.json', {'status': 'completed'})
            with patch.object(study, 'campaign', campaign):
                study.run(root, protocol, 'all')
                stamp=(root/'all_candidates_frozen.json').stat().st_mtime_ns
                study.run(root, protocol, 'all')
                self.assertEqual((root/'all_candidates_frozen.json').stat().st_mtime_ns, stamp)
            self.assertEqual(events, ['train']*3+['evaluate']*3)

    def test_modified_unselected_revision_blocks_evaluation(self):
        with TemporaryDirectory() as tmp:
            root, protocol = self.prepare(Path(tmp))
            with patch.object(study, 'campaign', self.fake_train):
                study.run(root, protocol, 'train')
            target=root/'replica_02/symbolic/frozen_revisions/revision_003.py'
            target.write_text('changed')
            with patch.object(study, 'campaign') as mocked:
                with self.assertRaisesRegex(RuntimeError, 'Frozen revision changed'):
                    study.run(root, protocol, 'evaluate')
                mocked.assert_not_called()

    def test_study_audit_rejects_reserved_initial_image_before_global_gate(self):
        with TemporaryDirectory() as tmp:
            root, protocol = self.prepare(Path(tmp))
            with patch.object(study, 'campaign', self.fake_train):
                study.run(root, protocol, 'train')
            image=root/'replica_01/numeric/heldout/seed_79001/initial.png'
            image.parent.mkdir(parents=True); image.write_bytes(b'Synthetic image')
            stamp=(root/'all_candidates_frozen.json').stat().st_mtime-1
            os.utime(image,(stamp,stamp))
            with self.assertRaisesRegex(ValueError, 'started before study-wide freeze'):
                report.summarize(root, allow_partial=True)

    def test_evaluation_without_all_training_is_rejected(self):
        with TemporaryDirectory() as tmp:
            root, protocol = self.prepare(Path(tmp))
            with patch.object(study, 'campaign') as mocked:
                with self.assertRaises(FileNotFoundError):
                    study.run(root, protocol, 'evaluate')
                mocked.assert_not_called()


if __name__ == '__main__':
    unittest.main()
