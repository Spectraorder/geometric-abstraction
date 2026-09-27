"""Export a completed pilot's measured learning curves without feeding them to actors."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def export(root, output, figure):
    summary = json.loads((root/'summary.json').read_text())
    manifest = json.loads((root/'manifest.json').read_text())
    if summary['status'] != 'completed':
        raise ValueError('Only completed experiments can be exported')
    if hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest() != summary['manifest_sha256']:
        raise ValueError('Manifest changed after the campaign completed')
    if hashlib.sha256((root/'geometry.py').read_bytes()).hexdigest() != manifest['geometry_sha256']:
        raise ValueError('Frozen geometry was modified')
    overview = {}
    for arm, curve in summary['curves'].items():
        usage = {}
        for revision in curve:
            for key, value in revision['generation']['usage'].items():
                if isinstance(value, (int, float)):
                    usage[key] = usage.get(key, 0) + value
        heldout = summary['heldout'][arm]
        overview[arm] = {
            'revision_numbers': [r['revision'] for r in curve],
            'train_successes_by_revision': [r['successes'] for r in curve],
            'train_trials_per_revision': [r['trials'] for r in curve],
            'candidate_validation_failures_by_revision': [
                r.get('candidate_validation_failures', r['trials'] if r['generation'].get('source_budget_exceeded') else 0)
                for r in curve],
            'proposed_policy_rollouts_by_revision': [
                r.get('executed_rollouts', 0 if r['generation'].get('source_budget_exceeded') else r['trials'])
                for r in curve],
            'heldout_successes': sum(r['success'] for r in heldout),
            'heldout_trials': len(heldout), 'frozen_revision': summary['frozen'][arm]['revision'],
            'total_generation_wall_seconds': sum(r['generation']['wall_seconds'] for r in curve),
            'total_model_usage': usage,
        }
        for field in ('host_trial_wall_seconds', 'camera_calls'):
            measured = [r[field] for r in heldout if r.get(field) is not None]
            overview[arm]['heldout_'+field+'_median'] = float(np.median(measured)) if measured else None
        compute = [r['cpu_profile']['policy_compute_cpu_seconds'] for r in heldout if r.get('cpu_profile')]
        overview[arm]['heldout_policy_compute_cpu_seconds_median'] = float(np.median(compute)) if compute else None
        overview[arm]['timing_scope'] = 'host trial includes private evaluation and IO; CPU compute excludes RPC; neither measures online VLM checks'
    result = {'manifest': manifest, 'overview': overview, **summary}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), gridspec_kw={'width_ratios': [1.5, 1]})
    colors = {'raw': '#64748b', 'geometry': '#168a65'}
    for arm, curve in summary['curves'].items():
        x = [r['cumulative_rollout_slots'] for r in curve]
        y = np.array([r['successes']/r['trials']*100 for r in curve])
        axes[0].plot(x, y, 'o-', color=colors[arm], label=arm+' (latest)')
        axes[0].plot(x, np.maximum.accumulate(y), '--', alpha=.5, color=colors[arm])
    axes[0].set(xlabel='Training rollout slots', ylabel='Training success (%)',
                title=f'{manifest["rounds"]} revisions from '+('shared RGB-D base' if manifest.get('base_policy_sha256') else 'an empty policy'), ylim=(-4, 104))
    axes[0].legend(loc='upper left', fontsize=8)
    labels = list(overview)
    values = [overview[a]['heldout_successes']/overview[a]['heldout_trials']*100 for a in labels]
    axes[1].bar(labels, values, color=[colors[a] for a in labels], width=.55)
    for i, arm in enumerate(labels):
        row = overview[arm]
        axes[1].text(i, values[i]+4, f'{row["heldout_successes"]}/{row["heldout_trials"]}', ha='center')
    axes[1].set(title='Frozen policies on reserved layouts', ylabel='Success (%)', ylim=(0, 115))
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle('Geometry-abstraction pilot: one campaign per arm', fontsize=12)
    fig.text(.5, .015, 'Small pilot, not a significance claim. Dashed lines: best-so-far training success.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, .95))
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    print(json.dumps(overview, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--figure', type=Path, required=True)
    args = parser.parse_args()
    export(args.root, args.output, args.figure)
