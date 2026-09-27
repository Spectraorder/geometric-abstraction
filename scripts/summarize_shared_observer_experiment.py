"""Audit the matched surface/rigid-symbolic pilot with exact actor contexts."""
import argparse
import json
from pathlib import Path
import run_shared_observer_experiment as variant
from summarize_mixed_scene_learning import audit as base_audit, paired, read, require, digest, plot
from summarize_retained_policy_budgets import retained_revisions


def audit(root, allow_partial=False):
    manifest = read(root/'manifest.json')
    # Exact context reconstruction uses the registered source version. Refuse to
    # silently reconstruct with later code; use its snapshot/version for reruns.
    variant.check(root, manifest)
    first = (root/'basic_library.py').read_text()
    second = (root/'symbolic_library.py').read_text()
    require(first.replace('SURFACE_ONLY = True', 'SURFACE_ONLY = False') == second,
            'Treatment libraries differ beyond the public projection switch')
    parity_path = root/'starter_parity.json'
    if list(root.glob('*/round_*/invocation.json')):
        require(parity_path.exists(), 'Actor generation started without starter parity')
    parity = None
    if parity_path.exists():
        parity = read(parity_path)
        require(parity['status'] == 'passed' and [r['seed'] for r in parity['pairs']] == manifest['train_seeds'],
                'Incomplete starter parity')
        for row in parity['pairs']:
            require(row['identical_physics'] and row['identical_surface_state'], 'Failed starter pair')
            for relative, sha in row['source_sha256'].items():
                require(digest(root/relative) == sha, 'Starter parity artifact changed')
    result = base_audit(root, allow_partial)
    verified_contexts = 0
    with variant.configured():
        for arm in variant.ARMS:
            for revision in range(1, manifest['rounds']+1):
                directory = root/arm/f'round_{revision:03d}'
                if not (directory/'invocation.json').exists():
                    continue
                prompt, images = variant.base.context(root, arm, revision, manifest)
                invocation = read(directory/'invocation.json')
                require(prompt == (directory/'prompt.txt').read_text(), 'Reconstructed actor prompt differs')
                require([digest(p) for p in images] == [r['sha256'] for r in invocation['images']],
                        'Reconstructed actor images differ')
                verified_contexts += 1
    result.update({'arm_labels': manifest['arm_labels'], 'contrasts': manifest['contrasts'],
                   'identical_library_except_projection': True,
                   'starter_parity_audit': 'passed' if parity else 'pending',
                   'exact_actor_contexts_verified': verified_contexts,
                   'auditor_sha256': digest(Path(__file__))})
    if result['status'] == 'completed':
        expected = len(variant.ARMS)*(manifest['rounds']+1)*(len(manifest['train_seeds'])+len(manifest['heldout_seeds']))
        require(result['valid_rollouts']+result['invalid_slots'] == expected, 'Slot accounting differs')
        require(verified_contexts == len(variant.ARMS)*manifest['rounds'], 'Generation accounting differs')
        rows = {arm: [read(root/arm/'heldout'/f'seed_{seed}'/'result.json') for seed in manifest['heldout_seeds']]
                for arm in variant.ARMS}
        result['selected_surface_vs_rigid_symbolic'] = paired(rows['basic'], rows['symbolic'])
        result['retained_budget_secondary'] = {}
        for arm, curve in result['curves'].items():
            chosen = retained_revisions(curve['training'])
            require(chosen[-1] == curve['selected_revision'], 'Secondary final selection differs from training rule')
            result['retained_budget_secondary'][arm] = [
                {'budget': budget, 'selected_revision': revision,
                 'reserved_successes': curve['reserved'][revision]['successes'],
                 'reserved_trials': curve['reserved'][revision]['trials'],
                 'reserved_sr': curve['reserved'][revision]['sr']}
                for budget, revision in enumerate(chosen)]
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--figure', type=Path)
    args = parser.parse_args()
    result = audit(args.root.resolve(), args.allow_partial)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    if args.figure:
        require(result['status'] == 'completed', 'Only completed curves may be plotted')
        displayed = dict(result, arm_labels={'basic': 'Base', 'symbolic': 'Full geometry'})
        plot(displayed, args.figure)
    print(json.dumps({k: result[k] for k in ('status','valid_rollouts','invalid_slots','exact_actor_contexts_verified','starter_parity_audit')}, indent=2))
