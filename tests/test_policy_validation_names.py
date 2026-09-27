"""Ordinary load/save values are not file operations in numerical policy code."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_vision_trial import validate_source
from run_geometry_step_experiment import validate_planner


class ValidationNameTests(unittest.TestCase):
    def test_object_named_load_is_a_legal_local_value(self):
        validate_planner("def run(api):\n    load = api.task()['nodes'][-1]\n    api.note(load['color'])\n")
        validate_source('def choose(save, load):\n    return load if save else None\n')

    def test_file_operations_and_alias_imports_remain_rejected(self):
        for source in ('import numpy as np\nnp.load("x")',
                       'load("x")', 'save("x", [1])',
                       'from numpy import load as reader\nreader("x")',
                       'from numpy import save as writer\nwriter("x", [1])',
                       'reader = open\nreader("x")'):
            with self.subTest(source=source), self.assertRaises(ValueError):
                validate_source(source)


if __name__=='__main__':unittest.main()
