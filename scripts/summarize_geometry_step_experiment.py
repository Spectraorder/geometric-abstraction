"""Audit target-only geometry evolution and exact bounded actor feedback."""
import argparse
import ast
import difflib
import json
from pathlib import Path
import run_geometry_step_experiment as variant
from summarize_mixed_scene_learning import audit as base_audit, paired, read, require, digest, plot
from summarize_retained_policy_budgets import retained_revisions



def audit_validator_correction(root, manifest):
    lineage = manifest.get('io_name_validator_correction')
    if lineage is None:
        return None
    parent = Path(lineage['parent_directory'])
    require(digest(parent/'manifest.json') == lineage['parent_manifest_sha256'], 'Correction parent changed')
    original = read(parent/'manifest.json')
    changed = [name for name, sha in original['source_sha256'].items()
               if manifest['source_sha256'].get(name) != sha]
    require(changed == ['scripts/run_vision_trial.py'], 'Correction changed more than the validator host')
    before = (parent/'source_snapshot/scripts/run_vision_trial.py').read_text()
    after = (root/'source_snapshot/scripts/run_vision_trial.py').read_text()
    trees = [ast.parse(text) for text in (before, after)]
    for tree in trees:
        tree.body = [node for node in tree.body if getattr(node, 'name', None) != 'validate_source']
    require(ast.dump(trees[0]) == ast.dump(trees[1]), 'Non-validator runtime changed')
    require(digest(root/'validator_patch.diff') == lineage['validator_patch_sha256'], 'Reviewed validator patch changed')
    expected_patch = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                            fromfile='registered_validator', tofile='corrected_validator'))
    require((root/'validator_patch.diff').read_text() == expected_patch, 'Patch does not describe the frozen source change')
    for key in ('library_sha256','starter_sha256','executor_sha256','train_seeds','heldout_seeds','color_maps','model','rounds','physics','sensor','feedback'):
        require(manifest[key] == original[key], 'Correction changed frozen setting: '+key)
    receipt = read(root/'retained_artifacts.json')
    require(digest(root/'retained_artifacts.json') == lineage['retention_receipt_sha256'], 'Retention receipt changed')
    for relative, sha in receipt.items():
        require(digest(root/relative) == sha == digest(parent/relative), 'Retained evidence changed: '+relative)
    for arm in manifest['arms']:
        require(digest(parent/arm/'round_003/invocation.json') == lineage['interrupted_invocation_sha256'][arm],
                'Interrupted invocation evidence changed')
    require(not list(parent.glob('*/**/heldout/seed_*/result.json')), 'Reserved execution in withdrawn parent')
    retained_trials = [name for name in receipt if name.endswith('/result.json')]
    retained_generations = [name for name in receipt if name.endswith('/generation.json')]
    rejected = list((parent/'basic/round_002/trials').glob('seed_*/result.json'))
    require(len(retained_trials) == lineage['retained_valid_rollouts'] == 25, 'Retained trial accounting changed')
    require(all(not read(parent/name).get('invalid_candidate', False) for name in retained_trials), 'Invalid trial retained')
    require(len(retained_generations) == lineage['retained_completed_generations'] == 4, 'Retained model accounting changed')
    require(len(rejected) == lineage['withdrawn_invalid_slots'] == 5, 'Rejected slot accounting changed')
    require(all(read(p).get('invalid_candidate') and not read(p).get('rollout_executed') for p in rejected), 'Rejected slot executed')
    require(all(not (parent/arm/'round_003/generation.json').exists() for arm in manifest['arms']), 'Interrupted generation was complete')
    return {'status':'passed','retained_valid_rollouts':len(retained_trials),'retained_completed_generations':len(retained_generations),
            'withdrawn_invalid_slots':len(rejected),'interrupted_model_invocations':len(manifest['arms']),
            'retained_artifact_files':len(receipt),
            'scope':'Submitted policies and completed evidence unchanged; only host name validation corrected.'}


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
            require(row['identical_physics'] and row['matching_focus_estimates'], 'Failed starter pair')
            for relative, sha in row['source_sha256'].items():
                require(digest(root/relative) == sha, 'Starter parity artifact changed')
    result = base_audit(root, allow_partial)
    correction = audit_validator_correction(root, manifest)
    if correction is not None:
        result['io_name_validator_correction'] = correction
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
    administrative = root/'administrative_status.json'
    if administrative.exists() and read(administrative).get('status', '').startswith('withdrawn'):
        result['protocol_phase_at_withdrawal'] = result['status']
        result['status'] = read(administrative)['status']
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
