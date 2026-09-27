"""Prior code can seed generation, but cannot import historical scores."""
import hashlib
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
import unittest

ROOT=Path(__file__).resolve().parents[1]
reference_context=runpy.run_path(str(ROOT/'scripts/learn_fasten_screws.py'))['reference_context']


class ReferenceTests(unittest.TestCase):
    def test_snapshots_are_exact_and_do_not_create_a_best_policy(self):
        with TemporaryDirectory() as td:
            root=Path(td);source=root/'prior.py'
            source.write_text('def run(api):\n    api.wait(.1)\n')
            state={'history':[]}
            context=reference_context(root,state,[source])
            self.assertIn(source.read_text(),context)
            self.assertNotIn('best',state)
            self.assertEqual(state['history'],[])
            self.assertEqual(len(state['references']),1)
            snapshot=Path(state['references'][0]['snapshot'])
            self.assertEqual(snapshot.read_bytes(),source.read_bytes())
            self.assertEqual(state['references'][0]['sha256'],hashlib.sha256(source.read_bytes()).hexdigest())
            source.unlink()  # Resumption relies on the saved snapshot.
            self.assertEqual(reference_context(root,state,[]),context)

    def test_resume_rejects_changed_source_or_snapshot(self):
        with TemporaryDirectory() as td:
            root=Path(td);source=root/'prior.py';source.write_text('def run(api): pass\n')
            state={'history':[]};reference_context(root,state,[source])
            source.write_text('def run(api): api.wait(.1)\n')
            with self.assertRaisesRegex(ValueError,'changed'):
                reference_context(root,state,[source])
            Path(state['references'][0]['snapshot']).write_text(source.read_text())
            with self.assertRaisesRegex(ValueError,'checksum'):
                reference_context(root,state,[])

    def test_cannot_insert_references_into_existing_history(self):
        with TemporaryDirectory() as td:
            root=Path(td);source=root/'prior.py';source.write_text('def run(api): pass\n')
            with self.assertRaisesRegex(ValueError,'new campaign'):
                reference_context(root,{'history':[{'round':1}]},[source])


if __name__=='__main__': unittest.main()
