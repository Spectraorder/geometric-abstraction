"""The information-only starter change must preserve executed R0 behavior."""
import ast
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_ring_peg_guided_training as guided
from run_ring_peg_trial import validate_policy


class GuidanceTests(unittest.TestCase):
    def test_only_module_documentation_changes_in_starter(self):
        old = ast.parse(guided.scaffold.STARTER_PATH.read_text())
        new = ast.parse(guided.STARTER_PATH.read_text())
        self.assertEqual(ast.dump(ast.Module(body=old.body[1:], type_ignores=[])),
                         ast.dump(ast.Module(body=new.body[1:], type_ignores=[])))
        validate_policy(guided.STARTER_PATH.read_text())

    def test_guidance_is_in_prompt_and_frozen_sources_without_strong_policy(self):
        with guided.installed():
            self.assertIn(guided.GUIDANCE_PATH.read_text(), guided.original.CONTRACT)
            sources = guided.original.sources()
            self.assertIn('docs/RING_PEG_STARTER_GUIDANCE.md', sources)
            self.assertNotIn('policies/controllers/ring_peg_starter.py', sources)
            self.assertIn('No grasp search', guided.original.CONTRACT)


if __name__ == '__main__':
    unittest.main()
