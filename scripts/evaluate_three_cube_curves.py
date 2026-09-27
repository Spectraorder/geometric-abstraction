"""Materialize and evaluate the three-cube curve plan registered before training."""
from contextlib import contextmanager
import csv
from pathlib import Path
import shutil
import time
from unittest.mock import patch

import evaluate_geometry_learning_curves as curves
from experiment_parallel import write_json
from summarize_mixed_scene_learning import require

ROOT = curves.ROOT
base = curves.base
read = curves.read
SCOPE = 'Planned before training: all frozen R0-R3 policies on the same reserved layouts; no test feedback or test-based selection.'


def prepare(root, output, data_root, workers):
    protocol = curves.study.check(root)
    curves.study.check_gate(root, protocol)
    require(protocol['rounds'] == 3 and read(root/'summary.json')['status'] == 'completed', 'Selected stage must be audited')
    require(workers == protocol['limits']['rollout'], 'Worker budget differs from registration')
    require(protocol['learning_curve_plan']['revisions'] == [0, 1, 2, 3], 'Curve plan changed')
    require(not output.exists() and not data_root.exists(), 'Use new curve directories')
    records = []
    preserved = {name: base.digest(root/name) for name in ('protocol.json', 'all_candidates_frozen.json', 'summary.json')}
    for replica in protocol['replicas']:
        name = replica['directory']
        directory = root/name
        training = read(directory/'training_summary.json')
        audit = read(directory/'selected_evaluation/audit.json')
        preserved[str((directory/'selected_evaluation/audit.json').relative_to(root))] = base.digest(directory/'selected_evaluation/audit.json')
        for arm in ('basic', 'symbolic'):
            require([entry['revision'] for entry in training['revision_frozen'][arm]] == [0, 1, 2, 3], 'Missing frozen revision')
            for entry in training['revision_frozen'][arm]:
                revision = entry['revision']
                source = directory/arm/'frozen_revisions'/f'revision_{revision:03d}.py'
                require(base.digest(source) == entry['sha256'], 'Frozen revision changed')
                preserved[str(source.relative_to(root))] = entry['sha256']
                reuse = revision == training['frozen'][arm]['revision']
                for seed in protocol['heldout_seeds']:
                    trial = (directory/'selected_evaluation'/arm/'heldout'/f'seed_{seed}' if reuse else
                             directory/'curve_rollouts'/arm/f'revision_{revision:03d}'/f'seed_{seed}')
                    if reuse:
                        for relative, digest in audit['result_hashes'].items():
                            if relative.startswith(f'{arm}/heldout/seed_{seed}/'):
                                preserved[str((directory/'selected_evaluation'/relative).relative_to(root))] = digest
                    records.append({'replica': name, 'arm': arm, 'revision': revision, 'seed': seed,
                                    'candidate': str(source.relative_to(root)), 'candidate_sha256': entry['sha256'],
                                    'validation_error': entry.get('validation_error'), 'reused': reuse,
                                    'trial': str(trial.relative_to(root))})
    require(len(records) == protocol['learning_curve_plan']['total_slots'], 'Curve budget mismatch')
    data_root.mkdir(parents=True)
    output.mkdir(parents=True)
    for replica in protocol['replicas']:
        destination = data_root/replica['directory']
        destination.mkdir()
        (root/replica['directory']/'curve_rollouts').symlink_to(destination, target_is_directory=True)
    registration = {'schema': 'preregistered_three_cube_curves_v1', 'registered_at': time.time(),
        'plan_registered_at': protocol['registered_at'], 'plan_protocol_sha256': base.digest(root/'protocol.json'),
        'root': str(root), 'data_root': str(data_root), 'max_workers': workers, 'pilot_rollouts': 48,
        'minimum_free_disk_bytes': 15*1024**3, 'minimum_available_ram_bytes': 6*1024**3,
        'model_calls': 0, 'records': records, 'slots': len(records),
        'reused_slots': sum(row['reused'] for row in records), 'new_slots': sum(not row['reused'] for row in records),
        'preserved_files': preserved, 'runner_sha256': base.digest(Path(__file__)),
        'primary_threshold': .8, 'secondary_thresholds': [.6, .7],
        'curves': ['Raw proposal at each revision', 'Training-retained policy at each proposal budget'],
        'scope': SCOPE, 'ramp_rule': 'Up to 48 action-taking pilot slots; fallback to eight workers below 6 GiB RAM or above 90 percent GPU memory.',
        'hardware_at_registration': curves.hardware()}
    write_json(output/'registration.json', registration)
    shutil.copyfile(Path(__file__), output/'runner_snapshot.py')
    curves.check(root, output)
    return registration


def plot(output, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for axis, kind, token_axis in zip(axes, ('raw', 'retained', 'retained'), (False, False, True)):
        for arm, label, color in (('basic', 'Base', '#2463a5'), ('symbolic', 'Full geometry', '#c75c24')):
            for replica in report['replicas'].values():
                row = replica[arm]
                x = [v/1000 for v in row['cumulative_generation_tokens']] if token_axis else range(4)
                axis.plot(x, [100*v for v in row[kind]], color=color, alpha=.18)
            mean = report['mean_curves'][arm]
            x = [v/1000 for v in mean['cumulative_generation_tokens']] if token_axis else range(4)
            axis.plot(x, [100*v for v in mean[kind]], '-o', color=color, label=label, linewidth=2.5)
        axis.axhline(80, color='gray', linestyle=':', linewidth=1)
        axis.set(ylim=(0, 100), ylabel='Test success rate (%)',
                 xlabel='Cumulative generation tokens (thousands)' if token_axis else 'Proposal iteration')
        if not token_axis:
            axis.set_xticks(range(4))
        axis.grid(alpha=.2)
        axis.legend()
    for axis, title in zip(axes, ('Raw proposals', 'Training-retained policies', 'Retained policies vs token budget')):
        axis.set_title(title)
    fig.suptitle('Three-cube stacking: five paired training chains; 100 shared test layouts\nEvaluation of every revision planned before training; thin lines show individual chains.')
    for suffix in ('png', 'pdf'):
        fig.savefig(output/('learning_curves.'+suffix), dpi=180)
    plt.close(fig)
    with (output/'learning_curves.csv').open('w') as handle:
        writer = csv.writer(handle)
        writer.writerow(['replica', 'arm', 'iteration', 'raw_sr', 'retained_sr', 'retained_revision', 'cumulative_tokens'])
        for name, replica in report['replicas'].items():
            for arm in ('basic', 'symbolic'):
                row = replica[arm]
                for i in range(4):
                    writer.writerow([name, arm, i, row['raw'][i], row['retained'][i],
                                     row['retained_revisions'][i], row['cumulative_generation_tokens'][i]])


@contextmanager
def installed():
    with patch.object(curves, '__file__', str(Path(__file__).resolve())), patch.object(curves, 'plot', plot):
        yield
