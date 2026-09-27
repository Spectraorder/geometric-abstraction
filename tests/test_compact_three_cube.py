"""Budget amendments must preserve every arm/revision and frozen evidence."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import evaluate_compact_three_cube as compact


class CompactStudyTests(unittest.TestCase):
    def test_registration_pairs_all_revisions_without_changing_original_protocol(self):
        with TemporaryDirectory() as folder:
            root=Path(folder)
            protocol={'heldout_seeds':list(range(100,200)), 'replicas':[{'directory':f'replica_{i:02d}'} for i in range(1,6)]}
            compact.write_json(root/'protocol.json',protocol)
            original=(root/'protocol.json').read_bytes()
            for row in protocol['replicas']:
                directory=root/row['directory']; revisions={}
                compact.write_json(directory/'manifest.json',{})
                compact.write_json(directory/'all_candidates_frozen.json',{})
                for arm in ('basic','symbolic'):
                    revisions[arm]=[]
                    for revision in range(4):
                        source='def run(api):\n    pass\n'
                        p=directory/arm/'frozen_revisions'/f'revision_{revision:03d}.py'
                        p.parent.mkdir(parents=True,exist_ok=True);p.write_text(source)
                        old=directory/arm/('revision_000' if revision==0 else f'round_{revision:03d}')/'bundled.py'
                        old.parent.mkdir(parents=True);old.write_text(source)
                        revisions[arm].append({'revision':revision,'sha256':compact.base.digest(p)})
                compact.write_json(directory/'training_summary.json',{'revision_frozen':revisions})
            with patch.object(compact.repair,'check'), patch.object(compact.runner.study,'check',return_value=protocol), \
                 patch.object(compact.runner.variant,'audit',return_value={'exact_actor_contexts_verified':6,'valid_rollouts':40,'invalid_slots':0}), \
                 patch.object(compact.resources,'summarize',return_value={}):
                registration=compact.prepare(root)
                self.assertEqual((root/'protocol.json').read_bytes(),original)
                self.assertEqual(registration['slots'],800)
                self.assertEqual(registration['model_calls'],0)
                for replica in protocol['replicas']:
                    for arm in ('basic','symbolic'):
                        for revision in range(4):
                            rows=[r for r in registration['records'] if (r['replica'],r['arm'],r['revision'])==(replica['directory'],arm,revision)]
                            self.assertEqual([r['seed'] for r in rows],list(range(100,120)))
                compact.check(root)
                candidate=root/registration['records'][0]['candidate']
                candidate.write_text('changed')
                with self.assertRaisesRegex(ValueError,'Frozen compact evidence changed'):
                    compact.check(root)


if __name__=='__main__':unittest.main()
