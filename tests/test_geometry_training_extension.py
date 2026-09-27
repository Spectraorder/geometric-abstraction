"""Round extensions preserve prior evidence and execute only additional budgets."""
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import extend_geometry_training as extension


class ExtensionTests(unittest.TestCase):
    def test_effective_manifest_changes_only_round_budget(self):
        original={'rounds':2,'actor_context_mode':'persistent','train_seeds':[1,2],
                  'heldout_seeds':[3,4],'arms':['basic','symbolic']}
        before=deepcopy(original)
        effective=extension.effective_manifest(original,5)
        self.assertEqual(original,before)
        self.assertEqual({k:v for k,v in effective.items() if k!='rounds'},
                         {k:v for k,v in original.items() if k!='rounds'})
        effective['train_seeds'].append(99)
        self.assertEqual(original,before)
        for bad in [2,1,True,5.5]:
            with self.assertRaises(ValueError):extension.effective_manifest(original,bad)
        with self.assertRaises(ValueError):
            extension.effective_manifest(dict(original,actor_context_mode='fresh'),5)

    def test_tampered_original_evidence_blocks_extension(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);directory=root/'extension_05';directory.mkdir()
            path=root/'manifest.json';path.write_text('{}')
            amendment={'total_rounds':5,'runner_sha256':extension.base.digest(Path(extension.__file__)),
                       'preserved_files':{'manifest.json':extension.base.digest(path)}}
            (directory/'amendment.json').write_text(json.dumps(amendment))
            path.write_text('{"modified":true}')
            with self.assertRaisesRegex(ValueError,'Original evidence changed'):
                extension.check(root,5)

    def test_only_three_new_rounds_execute_and_old_freezes_are_preserved(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);directory=root/'extension_05';directory.mkdir()
            manifest={'rounds':5,'actor_context_mode':'persistent','arms':['basic','symbolic'],
                      'train_seeds':list(range(5))}
            (root/'manifest.json').write_text(json.dumps(dict(manifest,rounds=2)))
            (directory/'amendment.json').write_text('{}')
            amendment={'original_rounds':2}
            curves={arm:[] for arm in manifest['arms']}
            source='def run(api):\n    pass\n'
            for arm in manifest['arms']:
                for revision in range(3):
                    stage=extension.stage(root,arm,revision);stage.mkdir(parents=True)
                    (stage/'bundled.py').write_text('Session = geometry.session\n'+source)
                    rows=[]
                    for seed in manifest['train_seeds']:
                        trial=stage/'trials'/f'seed_{seed}';trial.mkdir(parents=True)
                        row={'seed':seed,'success':False,'policy_error':None}
                        (trial/'actor_feedback.json').write_text(json.dumps(row))
                        (trial/'api_calls.json').write_text('[]');rows.append(row)
                    curves[arm].append({'revision':revision,'successes':0,'trials':5,'results':rows})
                (root/arm/'frozen.py').write_text('original frozen bytes')
            (root/'training_summary.json').write_text(json.dumps({'curves':curves}))
            original=(root/'training_summary.json').read_bytes()
            calls=[]
            def propose(output,prompt,images,settings):
                calls.append((output.parent.name,output.name))
                output.mkdir();(output/'candidate.py').write_text(source)
                (output/'generation.json').write_text(json.dumps({'candidate_sha256':extension.base.digest(output/'candidate.py')}))
            def execute(root,arm,candidate,trials,seeds,settings,error):
                revision=int(candidate.parent.name.removeprefix('round_'))
                rows=[]
                for seed in seeds:
                    trial=trials/f'seed_{seed}';trial.mkdir(parents=True)
                    row={'seed':seed,'success':seed < (2 if revision==5 else 1),'policy_error':None}
                    (trial/'actor_feedback.json').write_text(json.dumps(row))
                    (trial/'api_calls.json').write_text('[]');rows.append(row)
                return rows
            with patch.object(extension,'check',return_value=(directory,amendment,manifest)), \
                 patch.object(extension,'audit',return_value={'status':'completed_training_extension'}), \
                 patch.object(extension.experiment,'configured',return_value=nullcontext()), \
                 patch.object(extension.experiment,'context',return_value=('own training only',[])), \
                 patch.object(extension.base,'propose',side_effect=propose), \
                 patch.object(extension.base,'execute',side_effect=execute):
                extension.run(root,5)
            self.assertEqual(sorted(calls),[(a,f'round_{r:03d}') for a in ['basic','symbolic'] for r in [3,4,5]])
            self.assertEqual((root/'training_summary.json').read_bytes(),original)
            for arm in manifest['arms']:
                self.assertEqual((root/arm/'frozen.py').read_text(),'original frozen bytes')
            summary=json.loads((directory/'training_summary.json').read_text())
            self.assertTrue(all(row['revision']==5 for row in summary['frozen'].values()))
            self.assertEqual(len(summary['curves']['basic']),6)
            self.assertFalse(summary['reserved_evaluation_started'])


if __name__=='__main__':unittest.main()
