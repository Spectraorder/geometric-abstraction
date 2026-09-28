"""Bounded, zero-model development checks for the repaired ring interface."""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import ring_peg_development as repair
import run_ring_peg_trial as trial
from ring_peg_scene import RingPegEnv
from ring_peg_runtime import plain


def run(output, seed, candidate=None):
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError('Preserve existing attempts; choose a new development output')
    if candidate is not None:
        with repair.installed(), patch.object(trial, 'RingSensors', repair.DevelopmentSensors):
            result = trial.run(output, candidate, seed)
    else:
        output.mkdir()
        env = RingPegEnv(output/'env', scene_seed=seed)
        env.reset()
        result = {'seed': seed, 'initialization_complete': False}
        try:
            with repair.installed():
                api = repair.DevelopmentSensors(env, output)
                result['observations'] = {name: api.observe(name) for name in ('ring', 'peg')}
                result['initialization_complete'] = True
        except (ValueError, RuntimeError) as error:
            result['error'] = str(error)
        finally:
            env.close()
    result['scope'] = 'Development only; no model requests or reserved success-rate estimate.'
    (output/'result.json').write_text(json.dumps(plain(result), indent=2)+'\n')
    source = Path(repair.__file__)
    (output/source.name).write_bytes(source.read_bytes())
    (output/'repair_receipt.json').write_text(json.dumps({
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'resolution': [640, 480], 'seed': seed,
        'candidate_sha256': hashlib.sha256(candidate.read_bytes()).hexdigest() if candidate else None,
    }, indent=2)+'\n')
    return {k: v for k, v in result.items() if k != 'observations' and k != 'private_evaluation'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--candidate', type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.seed, args.candidate)), flush=True)
