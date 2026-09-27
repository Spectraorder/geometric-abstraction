"""Full study accounting on synthetic artifacts, never measured experiment data."""
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import test_mixed_scene_learning_audit as fixtures
import run_mixed_scene_experiment as campaign
import summarize_mixed_scene_replications as study


class ReplicationAuditTests(unittest.TestCase):
    def fixture(self, root):
        root.mkdir()
        replicas=[]; gate={}
        for i in range(3):
            directory=root/f'replica_{i+1:02d}'
            manifest=fixtures.LearningAuditTests().fixture(directory, arms=('numeric','symbolic'),train=[91001+i])
            replicas.append({'directory':directory.name,'train_seeds':manifest['train_seeds'],
                             'manifest_sha256':campaign.digest(directory/'manifest.json')})
            gate[directory.name]=study.read(directory/'all_candidates_frozen.json')
        (root/'coordinator_snapshot.py').write_text('# Synthetic coordinator evidence\n')
        protocol={'arms':['numeric','symbolic'],'replicas':replicas,'heldout_seeds':[92001,92002],
                  'training_slots':12,'reserved_slots':24,'model_generations':6,
                  'coordinator_sha256':campaign.digest(root/'coordinator_snapshot.py')}
        campaign.write(root/'protocol.json',protocol)
        campaign.write(root/'all_candidates_frozen.json',{'protocol_sha256':campaign.digest(root/'protocol.json'),'replicas':gate})
        # Fixtures are constructed offline; explicitly synthesize legal event ordering.
        stamp=(root/'all_candidates_frozen.json').stat().st_mtime+1
        for file in root.glob('replica_*/*/**/heldout/seed_*/*'):
            os.utime(file,(stamp,stamp))

    def test_completed_study_reports_three_chains_without_pooled_inference(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)/'study'; self.fixture(root)
            report=study.summarize(root)
            self.assertEqual(report['status'],'completed')
            self.assertEqual(report['valid_rollouts'],36)
            self.assertEqual(report['completed_actor_generations'],6)
            self.assertEqual(len(report['per_chain_comparisons']),3)
            self.assertEqual(report['mean_paired_curve_sr_difference'],0.)
            self.assertNotIn('pooled_p_value',report)

    def test_equal_total_cannot_hide_lost_replica_evidence(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)/'study'; self.fixture(root)
            result=root/'replica_02/numeric/heldout/seed_92001/result.json'
            result.unlink()
            with self.assertRaisesRegex(ValueError,'Reserved.*incomplete|Missing|differ|count'):
                study.summarize(root)

    def test_correction_equivalence_receipt_cannot_be_rewritten(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'repair_equivalence.json').write_text('{}')
            protocol={'validator_repair':{'equivalence_sha256':'not-the-recorded-hash'}}
            with self.assertRaisesRegex(ValueError,'Repair equivalence changed'):
                study.audit_validator_repair(root,protocol)

    def test_late_training_rewrite_is_rejected_even_without_content_changes(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)/'study'; self.fixture(root)
            path=root/'replica_03/training_summary.json'
            stamp=(root/'all_candidates_frozen.json').stat().st_mtime+2
            os.utime(path,(stamp,stamp))
            with self.assertRaisesRegex(ValueError,'Training changed after study-wide freeze'):
                study.summarize(root)


if __name__=='__main__':unittest.main()
