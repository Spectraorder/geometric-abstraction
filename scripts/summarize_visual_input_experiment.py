"""Audit visual-input versus supplied geometry without model or simulator calls."""
import argparse
from pathlib import Path
import run_visual_input_experiment as variant
from summarize_mixed_scene_learning import audit as base_audit, paired, read, require, digest, plot
from summarize_retained_policy_budgets import retained_revisions


def audit(root, allow_partial=False):
    manifest=read(root/'manifest.json');variant.check(root,manifest)
    require((root/'basic_library.py').read_text().replace('SURFACE_ONLY = True','SURFACE_ONLY = False')==(root/'symbolic_library.py').read_text(), 'Libraries differ beyond public projection')
    parity_path=root/'starter_parity.json'
    if list(root.glob('*/round_*/invocation.json')):require(parity_path.exists(),'Actor started without starter parity')
    if parity_path.exists():
        parity=read(parity_path)
        require(parity['status']=='passed' and [r['seed'] for r in parity['pairs']]==manifest['train_seeds'],'Starter parity incomplete')
        for row in parity['pairs']:
            require(row['identical_physics'] and row['same_action_boundaries'] and row['base_supplied_geometry_absent'],'Starter parity failed')
            for relative,sha in row['source_sha256'].items():require(digest(root/relative)==sha,'Starter artifact changed')
    result=base_audit(root,allow_partial);contexts=0
    with variant.configured():
        for arm in variant.ARMS:
            for revision in range(1,manifest['rounds']+1):
                directory=root/arm/f'round_{revision:03d}'
                if not (directory/'invocation.json').exists():continue
                prompt,images=variant.context(root,arm,revision,manifest)
                invocation=read(directory/'invocation.json')
                require(prompt==(directory/'prompt.txt').read_text(),'Reconstructed prompt differs')
                require([digest(p) for p in images]==[r['sha256'] for r in invocation['images']],'Reconstructed images differ')
                contexts+=1
    result.update(arm_labels=manifest['arm_labels'],contrasts=manifest['contrasts'],
                  identical_library_except_projection=True,exact_actor_contexts_verified=contexts,
                  starter_parity_audit='passed' if parity_path.exists() else 'pending',auditor_sha256=digest(Path(__file__)))
    if result['status']=='completed':
        expected=len(variant.ARMS)*(manifest['rounds']+1)*(len(manifest['train_seeds'])+len(manifest['heldout_seeds']))
        require(result['valid_rollouts']+result['invalid_slots']==expected,'Rollout accounting mismatch')
        require(contexts==len(variant.ARMS)*manifest['rounds'],'Generation accounting mismatch')
        rows={arm:[read(root/arm/'heldout'/f'seed_{seed}'/'result.json') for seed in manifest['heldout_seeds']] for arm in variant.ARMS}
        result['selected_visual_vs_geometry']=paired(rows['basic'],rows['symbolic'])
        result['retained_budget_secondary']={}
        for arm,curve in result['curves'].items():
            selected=retained_revisions(curve['training'])
            require(selected[-1]==curve['selected_revision'],'Training selection differs')
            result['retained_budget_secondary'][arm]=[{'budget':budget,'selected_revision':revision,'reserved_sr':curve['reserved'][revision]['sr']} for budget,revision in enumerate(selected)]
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-partial',action='store_true')
    parser.add_argument('--figure',type=Path)
    args=parser.parse_args();report=audit(args.root.resolve(),args.allow_partial)
    args.output.parent.mkdir(parents=True,exist_ok=True);variant.base.write(args.output,report)
    if args.figure:
        require(report['status']=='completed','Cannot plot incomplete reserved curves');plot(report,args.figure)
    print({k:report[k] for k in ('status','valid_rollouts','invalid_slots','exact_actor_contexts_verified','starter_parity_audit')})
