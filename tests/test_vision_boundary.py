"""Access-control checks, independent of task success or image heuristics."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('vision_runner',ROOT/'scripts/run_vision_trial.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class VisionBoundaryTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == 'linux' and shutil.which('bwrap'), 'Linux bwrap check')
    def test_linux_worker_launches_with_runtime_but_cannot_read_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp).resolve()
            shutil.copy(ROOT / 'scripts/vision_policy_worker.py', directory / 'worker.py')
            candidate = directory / 'candidate.py'
            candidate.write_text('import numpy as np\nfrom scipy import ndimage\n'
                                 'def run(api):\n'
                                 ' assert ndimage.label(np.ones((2, 2)))[1] == 1\n')
            command = runner.policy_worker_command(directory, candidate)
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['method'], 'done')

            # Probe OS access directly, independently of the source validator.
            probe = (
                'from pathlib import Path\n'
                f'checkout = Path({str(ROOT / "README.md")!r})\n'
                'try:\n checkout.read_text()\n'
                'except OSError:\n pass\n'
                'else:\n raise AssertionError("checkout is readable")\n'
                f'runtime = Path({str(Path(sys.prefix) / "pyvenv.cfg")!r})\n'
                'assert runtime.read_text()\n'
                'try:\n runtime.open("r+")\n'
                'except OSError:\n pass\n'
                'else:\n raise AssertionError("runtime is writable")\n'
                f'Path({str(directory / "scratch.txt")!r}).write_text("ok")\n'
            )
            result = subprocess.run(command[:-3] + ['-I', '-c', probe],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((directory / 'scratch.txt').read_text(), 'ok')

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
