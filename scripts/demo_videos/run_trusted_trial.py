"""Run scripts/run_vision_trial.py for a trusted, hand-written repo controller.

The repo wraps every policy in bwrap because study policies are LLM-generated.
This demo uses canonical controllers from policies/controllers, and bwrap cannot
create its network namespace inside the current sandbox, so the worker runs as
a plain isolated-mode subprocess instead. Physics, sensors and evaluation are the
unchanged run_vision_trial code.
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]

sys.path.insert(0, str(Path(__file__).parent))
import demo_recipes  # noqa: E402,F401  registers demo-only recipes
import run_vision_trial  # noqa: E402


def trusted_worker_command(directory, candidate):
    return [sys.executable, '-I', str(directory / 'worker.py'), str(candidate)]


run_vision_trial.policy_worker_command = trusted_worker_command
sys.argv = [str(ROOT / 'scripts/run_vision_trial.py'), *sys.argv[1:]]
run_vision_trial.main()
