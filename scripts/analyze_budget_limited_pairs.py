"""Describe matched outcomes in the stopped study; never run or select policies.

This is a post-hoc descriptive analysis of one learning chain. No population
significance test is justified by treating its 20 layouts as 20 learning chains.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze():
    report_path = ROOT / 'docs/evaluations/constraint_geometry_budget_limited.json'
    report = json.loads(report_path.read_text())
    if report['status'] != 'user_budget_stopped':
        raise ValueError('Expected user-stopped evaluation')
    for name, digest in report['source_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Source changed: ' + name)
    root = ROOT / 'outputs/constraint_geometry_learning_replications_01'
    progress = json.loads((root / 'reserved_progress.json').read_text())
    replica = root / 'replica_01'
    manifest = json.loads((replica / 'manifest.json').read_text())
    gate = json.loads((replica / 'all_candidates_frozen.json').read_text())
    seeds = manifest['heldout_seeds']
    batches = {}
    sources = {str(report_path.relative_to(ROOT)): sha(report_path)}
    sources.update(report['source_sha256'])
    for batch in report['batches']:
        if batch['replica'] != 'replica_01' or not batch['batch_complete']:
            continue
        arm, revision = batch['arm'], batch['revision']
        folder = replica / arm
        if revision != gate['selected'][arm]['revision']:
            folder = folder / 'reserved_revisions' / f'revision_{revision:03d}'
        rows = {}
        for seed in seeds:
            path = folder / 'heldout' / f'seed_{seed}' / 'result.json'
            digest = sha(path)
            if digest != progress['result_sha256'][str(path.relative_to(root))]:
                raise ValueError('Retained outcome changed')
            result = json.loads(path.read_text())
            if result['seed'] != seed or type(result['success']) is not bool:
                raise ValueError('Invalid paired outcome')
            calls_path = path.parent / 'api_calls.json'
            calls = json.loads(calls_path.read_text())
            notes = [call['params']['text'] for call in calls if call['method'] == 'note']
            sources[str(calls_path.relative_to(ROOT))] = sha(calls_path)
            rows[seed] = {
                'last_policy_note': notes[-1] if notes else None,
                'success': result['success'],
                'terminal_policy_error': result['policy_error'],
                'offline_placed_blocks': result['metrics']['placed_blocks'],
            }
            sources[str(path.relative_to(ROOT))] = digest
        if sum(row['success'] for row in rows.values()) != batch['successes']:
            raise ValueError('Complete-batch accounting differs')
        batches[f'{arm}_r{revision}'] = rows
    starter = batches['basic_r0']
    if any(starter[s]['success'] != batches['symbolic_r0'][s]['success'] for s in seeds):
        raise ValueError('Common starter outcomes differ')

    def compare(before, after):
        groups = {'both_success': [], 'repaired': [], 'regressed': [], 'both_failure': []}
        for seed in seeds:
            a, b = before[seed]['success'], after[seed]['success']
            key = ('both_success' if a else 'repaired') if b else ('regressed' if a else 'both_failure')
            groups[key].append(seed)
        return {'counts': {k: len(v) for k, v in groups.items()}, 'seeds': groups}

    output = {
        'scope': __doc__,
        'new_rollouts': 0,
        'independent_learning_chains_with_reserved_results': 1,
        'matched_layout_count': len(seeds),
        'r0_to_r1': {arm: compare(starter, batches[f'{arm}_r1']) for arm in ('basic', 'symbolic')},
        'r1_basic_to_symbolic': compare(batches['basic_r1'], batches['symbolic_r1']),
        'full_r1_to_training_selected_r3': compare(batches['symbolic_r1'], batches['symbolic_r3']),
        'terminal_errors_on_failed_trials': {
            name: dict(Counter(row['terminal_policy_error'] or 'No terminal exception recorded'
                               for row in rows.values() if not row['success']))
            for name, rows in batches.items()
        },
        'last_notes_on_failed_trials': {
            name: dict(Counter(row['last_policy_note'] or 'No policy note recorded'
                               for row in rows.values() if not row['success']))
            for name, rows in batches.items()
        },
        'per_layout': [{'seed': seed, **{name: rows[seed] for name, rows in batches.items()}} for seed in seeds],
        'limitations': [
            'Post-hoc descriptive analysis; not a new registered causal endpoint.',
            'Terminal exceptions are not a complete failure-cause taxonomy: policies may catch exceptions.',
            'The user stopped evaluation after interim outcomes were visible.',
            'No held-out feedback is supplied to an actor or used to select a revision.',
            'The package includes supplied perception and geometry; this does not isolate symbolic reasoning.',
        ],
        'source_sha256': sources,
        'analyzer_sha256': sha(Path(__file__)),
    }
    destination = ROOT / 'docs/evaluations/constraint_geometry_paired_outcomes.json'
    destination.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({k: output[k] for k in ('r0_to_r1', 'r1_basic_to_symbolic',
          'full_r1_to_training_selected_r3', 'terminal_errors_on_failed_trials', 'last_notes_on_failed_trials')}, indent=2))


if __name__ == '__main__':
    analyze()
