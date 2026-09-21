"""Access-control checks, independent of task success or image heuristics."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('vision_runner',ROOT/'scripts/run_vision_trial.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class VisionBoundaryTests(unittest.TestCase):
    def test_policy_cannot_import_simulator_or_open_files(self):
        for source in ['import mujoco\ndef run(api): pass', 'def run(api): open("secret")',
                       'def run(api): return api._env', 'import numpy as np\ndef run(api): np.load("secret")']:
            with self.subTest(source=source):
                with self.assertRaises(ValueError):runner.validate_source(source)
        runner.validate_source('import numpy as np\ndef run(api):\n s=api.camera()\n api.note(str(np.mean(s["rgb"])))')

    @unittest.skipUnless(sys.platform=='darwin','Mac sandbox-exec check')
    def test_os_denies_old_policy_and_allows_numerical_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=str(Path(tmp).resolve())
            source='import numpy; print("numpy-ok",flush=True); open('+repr(str(ROOT/'README.md'))+').read()'
            result=subprocess.run(['/usr/bin/sandbox-exec','-p',runner.sandbox_profile(tmp),sys.executable,'-I','-c',source],capture_output=True,text=True)
            self.assertIn('numpy-ok',result.stdout)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('PermissionError',result.stderr)
