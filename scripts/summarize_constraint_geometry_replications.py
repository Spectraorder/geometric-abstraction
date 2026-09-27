"""Audit new constraint-feedback learning curves without pooling old studies."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import run_constraint_geometry_experiment as variant
import summarize_visual_input_replications as inherited



def audit_interruptions(root):
    from archive_quota_interruption import file_hashes, quota_failure
    rows=[]
    for path in sorted(root.glob('interrupted_actor_attempts/batch_*/receipt.json')):
        receipt=inherited.read(path)
        inherited.require(receipt['status']=='archived','Interrupted attempt archival incomplete')
        inherited.require(receipt['protocol_sha256']==inherited.digest(root/'protocol.json'),'Interrupted protocol changed')
        inherited.require(receipt['source_sha256']==inherited.digest(path.parent/'archive_driver.py'),'Archive driver changed')
        gate=root/'all_candidates_frozen.json'
        if gate.exists():
            inherited.require(path.stat().st_mtime_ns<=gate.stat().st_mtime_ns,'Interruption archive postdates global freeze')
        for row in receipt['attempts']:
            archived=root/row['archive_directory']
            inherited.require(file_hashes(archived)==row['files_sha256'],'Interrupted files changed')
            inherited.require(quota_failure(archived)==row['errors'],'Interrupted event reason changed')
            replica_name,arm,stage=Path(row['original_directory']).parts
            replica=root/replica_name;manifest=inherited.read(replica/'manifest.json')
            prompt,images=variant.context(replica,arm,int(stage.split('_')[1]),manifest)
            inherited.require(prompt==(archived/'prompt.txt').read_text(),'Interrupted prompt cannot be reconstructed')
            inherited.require([inherited.digest(p) for p in images]==row['image_sha256'],'Interrupted images changed')
            metadata=inherited.read(archived/'invocation.json')
            inherited.require(metadata['prompt_sha256']==row['prompt_sha256']==inherited.digest(archived/'prompt.txt'),'Interrupted prompt digest differs')
            inherited.require([r['sha256'] for r in metadata['images']]==row['image_sha256'],'Interrupted invocation images differ')
            rows.append({'path':str((archived/'invocation.json').relative_to(root)),
                'sha256':inherited.digest(archived/'invocation.json'),'completed':False,
                'infrastructure_failure':'pre_response_usage_limit','original_stage':row['original_directory'],
                'receipt_sha256':inherited.digest(path),'exact_context_verified':True})
    return rows


def summarize(root, allow_partial=False):
    protocol=inherited.read(root/'protocol.json')
    inherited.require(protocol.get('experiment_variant')==variant.VERSION,'Wrong experiment variant')
    with patch.object(inherited,'audit',variant.audit):
        result=inherited.summarize(root,allow_partial)
    interrupted=audit_interruptions(root)
    result['model_invocation_attempts'].extend(interrupted)
    result.update(interrupted_actor_attempts=len(interrupted),
        experiment_variant=variant.VERSION,variant_auditor_sha256=inherited.digest(Path(__file__)))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-partial',action='store_true');args=parser.parse_args()
    result=summarize(args.root.resolve(),args.allow_partial)
    args.output.parent.mkdir(parents=True,exist_ok=True);variant.base.write(args.output,result)
    print({k:result[k] for k in ('status','valid_rollouts','invalid_slots','completed_actor_generations','study_freeze_audit')})
