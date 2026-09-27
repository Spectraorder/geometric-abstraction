"""Reject premature evaluation even if an individual child looks valid."""
from pathlib import Path
import os
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import summarize_visual_input_replications as report
from run_mixed_scene_experiment import write


class GlobalAuditTests(unittest.TestCase):
    def prepare(self,root):
        for name in ('coordinator_snapshot.py','shared_coordinator_snapshot.py'):(root/name).write_text('frozen coordinator')
        child=root/'replica_01';child.mkdir()
        manifest={'arms':['basic','symbolic'],'train_seeds':[11],'heldout_seeds':[21],'rounds':3}
        write(child/'manifest.json',manifest)
        protocol={'schema':'visual_input_replications_v1','arms':manifest['arms'],'heldout_seeds':[21],
                  'coordinator_sha256':report.digest(root/'coordinator_snapshot.py'),
                  'shared_coordinator_sha256':report.digest(root/'shared_coordinator_snapshot.py'),
                  'replicas':[{'directory':'replica_01','train_seeds':[11],'manifest_sha256':report.digest(child/'manifest.json')}]}
        write(root/'protocol.json',protocol)
        return child

    def test_initial_reserved_image_without_global_gate_is_rejected(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);child=self.prepare(root)
            image=child/'basic/heldout/seed_21/initial.png';image.parent.mkdir(parents=True);image.write_bytes(b'probe')
            with patch.object(report,'audit') as child_audit:
                with self.assertRaisesRegex(ValueError,'before global freeze'):report.summarize(root,True)
                child_audit.assert_not_called()

    def test_initial_reserved_image_predating_existing_gate_is_rejected(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);child=self.prepare(root)
            local={'selected':{},'revisions':{}}
            write(child/'all_candidates_frozen.json',local);write(child/'training_summary.json',{})
            write(child/'train_audit.json',{'exact_actor_contexts_verified':6})
            gate=root/'all_candidates_frozen.json'
            write(gate,{'protocol_sha256':report.digest(root/'protocol.json'),'replicas':{'replica_01':local}})
            image=child/'symbolic/heldout_revisions/revision_000/heldout/seed_21/initial.png'
            image.parent.mkdir(parents=True);image.write_bytes(b'probe')
            old=gate.stat().st_mtime_ns-1000000000;os.utime(image,ns=(old,old))
            with patch.object(report,'audit') as child_audit:
                with self.assertRaisesRegex(ValueError,'before global freeze'):report.summarize(root,True)
                child_audit.assert_not_called()


if __name__=='__main__':unittest.main()
