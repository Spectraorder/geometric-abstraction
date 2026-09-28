"""Expanded evaluation must reuse all prior trials without changing training."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import extend_three_cube_contract_evaluation as extension


class ExtensionTests(unittest.TestCase):
    def test_matrix_reuses_all_old_trials_and_pairs_new_layouts(self):
        old=[{'replica':f'replica_{rep:02d}','arm':arm,'revision':revision,'seed':seed,
              'trial':f'old/{rep}/{arm}/{revision}/{seed}','candidate':'frozen.py',
              'candidate_sha256':'frozen','validation_error':None}
             for rep in range(1,6) for arm in ('basic','symbolic') for revision in range(4) for seed in range(20)]
        rows=extension.matrix(old,list(range(20,100)))
        self.assertEqual(len(rows),4000)
        self.assertEqual(sum(r['reused'] for r in rows),800)
        self.assertEqual([r['trial'] for r in rows[:800]],[r['trial'] for r in old])
        for rep in range(1,6):
            for arm in ('basic','symbolic'):
                for revision in range(4):
                    selected=[r['seed'] for r in rows if (r['replica'],r['arm'],r['revision'])==(f'replica_{rep:02d}',arm,revision)]
                    self.assertEqual(sorted(selected),list(range(100)))

    def test_manifest_expansion_is_nonmutating_and_keeps_runtime_hashes(self):
        manifest={'heldout_seeds':[1],'train_seeds':[9],'color_maps':{'1':{'red':'blue'}},'source_sha256':{'runtime.py':'hash'}}
        result=extension.expanded_manifest(manifest,[1,2])
        self.assertEqual(manifest['heldout_seeds'],[1])
        self.assertNotIn('2',manifest['color_maps'])
        self.assertEqual(result['train_seeds'],[9])
        self.assertEqual(result['source_sha256'],manifest['source_sha256'])
        self.assertIn('2',result['color_maps'])


if __name__=='__main__':unittest.main()
