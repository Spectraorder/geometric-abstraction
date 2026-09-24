"""Improve an existing RGB-D policy on reproducible random layouts using the LLM."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TRAIN = [101, 102, 103, 104, 105]
HELDOUT = [201, 202, 203, 204, 205]


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def evaluate(candidate, directory, seeds, jobs, *, task='tower', layout='random_xy', object_count=None,
             tower_recipe=None, tower_levels=None):
    if object_count is not None and (task not in ('inventory_prism', 'inventory_tower') or not 2 <= object_count <= 20):
        raise ValueError('object_count requires an inventory task and must be 2..20')
    if task != 'inventory_tower' and (tower_recipe is not None or tower_levels is not None):
        raise ValueError('Recipe options require inventory_tower')
    candidate = candidate.resolve()
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    directory.mkdir(exist_ok=True, parents=True)
    def trial(seed):
        out = directory / f'seed_{seed}'
        if not (out / 'result.json').exists():
            if out.exists():
                # A killed/timed-out trial leaves partial artifacts. Keep them
                # for diagnosis, but give the runner a fresh output path.
                archive = Path(tempfile.mkdtemp(prefix=f'seed_{seed}.interrupted-',
                                                dir=directory))
                out.rename(archive / 'trial')
                previous_log = directory / f'seed_{seed}.log'
                if previous_log.exists():
                    previous_log.rename(archive / previous_log.name)
            worker_env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
            if sys.platform == 'linux' and not os.environ.get('DISPLAY'):
                worker_env.setdefault('MUJOCO_GL', 'egl')
            with (directory / f'seed_{seed}.log').open('w') as log:
                subprocess.run([sys.executable, str(ROOT / 'scripts/run_vision_trial.py'),
                    '--candidate', str(candidate), '--seed', str(seed), '--layout', layout, '--task', task,
                    '--output', str(out)] + ([] if object_count is None else ['--object-count', str(object_count)])
                    + ([] if tower_recipe is None else ['--tower-recipe', tower_recipe])
                    + ([] if tower_levels is None else ['--tower-levels', str(tower_levels)]),
                    stdout=log, stderr=subprocess.STDOUT, check=True,
                    env=worker_env,
                    timeout=240)
        result = json.loads((out / 'result.json').read_text())
        if (result['candidate_sha256'] != digest or result.get('layout') != layout
                or result.get('task', 'tower') != task
                or (task in ('inventory_prism', 'inventory_tower') and result.get('scene', {}).get('object_count') != (object_count or 6))):
            raise RuntimeError(f'Existing result does not match this experiment: {out}')
        if task == 'inventory_tower':
            from aspire_so100.tower_tasks import tower_recipe as recipe_for
            expected = recipe_for(tower_recipe or 'cube_triple', seed=seed, levels=tower_levels)
            if result.get('goal') != expected.public_goal() or result.get('scene', {}).get('recipe') != expected.name:
                raise RuntimeError(f'Existing result has a different tower goal: {out}')
        print(f'{directory.name}, seed {seed}: success={result["success"]}, '
              f'score={result["metrics"]["score"]}, error={result["policy_error"]}', flush=True)
        return result | {'directory': str(out)}
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(trial, seeds))
    summary = {'candidate': str(candidate), 'sha256': digest,
               'successes': sum(r['success'] for r in results), 'trials': len(results),
               'score': sum(r['metrics']['score'] for r in results) / len(results),
               'results': results}
    write(directory / 'summary.json', summary)
    return summary


def actor_context(best, latest):
    entries = [best] if best['candidate'] == latest['candidate'] else [best, latest]
    sections = ['Random initial XY positions: all three colors share x=[-0.14,0.14], '
                'y=[-0.32,-0.21] metres, with at least 25 mm edge clearance. '
                'Yaw remains zero. No color has a fixed location. Tower task, sensors, '
                'hardware and evaluator are unchanged. Improve reliability on all five '
                'training layouts. Obtain target locations through the RGB-D camera. '
                'Do not key actions to seeds, recorded poses, or memorized coordinates. '
                'Only the supplied existing policy and feedback may be reused.']
    for label, entry in zip(['Best policy', 'Latest policy'], entries):
        trials = []
        for result in entry['results']:
            evidence = json.loads((Path(result['directory']) / 'actor_feedback.json').read_text())
            notes = evidence.pop('notes', [])
            evidence['notes'] = [str(x)[:300] for x in (notes[:3] + notes[-6:])]
            trials.append({'seed': result['seed'], **evidence})
        sections.extend([label + ':', Path(entry['candidate']).read_text(),
                         'Sensor-derived notes and post-trial feedback:', json.dumps(trials)])
    # Choose up to two representative failures, otherwise successes. The files
    # contain rendered sensor pixels only, never ground-truth object overlays.
    selected = sorted(latest['results'], key=lambda r: r['success'])[:2]
    images = []
    for result in selected:
        folder = Path(result['directory'])
        for phase in ('initial', 'outcome'):
            sections.append(f'Image {len(images)+1}: latest candidate seed {result["seed"]}, {phase}.')
            images.append(folder / f'{phase}.png')
    prompt = '\n'.join(sections)
    if len(prompt) > 28000 and len(entries) > 1:
        # Keep the complete best policy. A large unsuccessful revision can be
        # represented by its stated hypothesis plus its actual trial feedback.
        response = Path(latest['candidate']).parent / 'response.json'
        hypothesis = json.loads(response.read_text())['hypothesis'] if response.exists() else ''
        latest_label = sections.index('Latest policy:')
        sections[latest_label + 1] = ('Latest source omitted to preserve the context budget. '
                                    'Its proposed changes were: ' + hypothesis[:1500])
        prompt = '\n'.join(sections)
    if len(prompt) > 28000:
        raise RuntimeError('Feedback exceeded compact context budget; inspect before spending tokens')
    return prompt, images


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--initial-policy', type=Path, default=ROOT / 'policies/stack_three_blocks_vision.py')
    p.add_argument('--model', default='gpt-6-astra')
    p.add_argument('--rounds', type=int, default=0, help='0: keep proposing until all training layouts pass')
    p.add_argument('--jobs', type=int, default=2)
    p.add_argument('--baseline-only', action='store_true')
    a = p.parse_args()
    if a.rounds < 0 or a.jobs < 1:
        p.error('rounds must be nonnegative and jobs positive')
    root = a.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / 'campaign.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        state.pop('error', None)
    else:
        state = {'created_at': datetime.now(timezone.utc).isoformat(), 'model': a.model,
                 'layout': 'random_xy', 'train_seeds': TRAIN, 'heldout_seeds': HELDOUT,
                 'history': [], 'status': 'baseline', 'every_attempt_rollout': True}
        shutil.copyfile(a.initial_policy, root / 'initial_policy.py')
        write(state_path, state)
    try:
        if not state['history']:
            baseline = evaluate(root / 'initial_policy.py', root / 'baseline', TRAIN, a.jobs)
            state['history'].append(baseline)
            write(state_path, state)
        used = 0
        while True:
            best = max(state['history'], key=lambda e: (e['successes'], e['score']))
            state['best'] = best
            shutil.copyfile(best['candidate'], root / '.best_policy.py.tmp')
            (root / '.best_policy.py.tmp').replace(root / 'best_policy.py')
            successes = [r for r in best['results'] if r['success']]
            if successes:
                (root / 'best_trial.txt').write_text(successes[0]['directory'] + '\n')
            write(state_path, state)
            if a.baseline_only:
                return
            if best['successes'] == len(TRAIN):
                state['status'] = 'validating_frozen_policy'
                write(state_path, state)
                # Held-out evidence is evaluated only after freezing and is never
                # sent back to the actor or used to select another candidate.
                state['heldout'] = evaluate(root / 'best_policy.py', root / 'heldout', HELDOUT, a.jobs)
                state['status'] = 'completed'
                write(state_path, state)
                print(f'Frozen policy: {best["successes"]}/5 train, '
                      f'{state["heldout"]["successes"]}/5 held-out. {root}', flush=True)
                return
            if a.rounds and used >= a.rounds:
                state['status'] = 'round_limit'
                write(state_path, state)
                return
            index = len(state['history'])
            directory = root / f'round_{index:03d}'
            context, images = actor_context(best, state['history'][-1])
            context_file = root / f'context_{index:03d}.txt'
            context_file.write_text(context)
            state.update(status='proposing', active_round=index)
            write(state_path, state)
            if not (directory / 'candidate.py').exists():
                command = [sys.executable, str(ROOT / 'scripts/propose_vision_policy.py'),
                           '--round-dir', str(directory), '--context-file', str(context_file), '--model', a.model]
                for path in images:
                    command += ['--image', str(path)]
                print(f'Round {index}: asking {a.model}; {len(context)} context chars, {len(images)} images', flush=True)
                subprocess.run(command, check=True)
            events = [json.loads(line) for line in (directory / 'model.jsonl').read_text().splitlines() if line.startswith('{')]
            usage = [event['usage'] for event in events if event.get('type') == 'turn.completed' and 'usage' in event]
            if usage:
                state.setdefault('usage_by_round', {})[str(index)] = usage
                print(f'Round {index} reported usage: {usage}', flush=True)
            if any(event.get('item', {}).get('type') in
                   {'command_execution','mcp_tool_call','web_search','file_change'} for event in events):
                raise RuntimeError('Actor used tools; model isolation audit failed')
            state['status'] = 'executing'
            write(state_path, state)
            result = evaluate(directory / 'candidate.py', directory / 'trials', TRAIN, a.jobs)
            state['history'].append(result)
            write(state_path, state)
            used += 1
    except KeyboardInterrupt:
        state['status'] = 'interrupted'
        write(state_path, state)
        raise
    except Exception as exc:
        state.update(status='error', error=f'{type(exc).__name__}: {exc}')
        write(state_path, state)
        raise


if __name__ == '__main__':
    main()
