"""Preserve pre-response quota failures before an identical-context future retry.

No model calls or candidate edits. Refuse generated responses, completed turns,
non-quota errors, changed contexts, or an experiment already globally frozen.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import run_constraint_geometry_experiment as variant


def file_hashes(directory):
    return {str(p.relative_to(directory)):variant.base.digest(p) for p in sorted(directory.rglob('*')) if p.is_file()}


def quota_failure(directory):
    if any((directory/name).exists() for name in ('generation.json','response.json','candidate.py','bundled.py','trials')):
        raise ValueError('Refuse to archive a generated candidate or any rollout')
    events=[json.loads(line) for line in (directory/'model.jsonl').read_text().splitlines() if line.startswith('{')]
    if any(e.get('type')=='turn.completed' or e.get('item',{}).get('type')=='agent_message' for e in events):
        raise ValueError('Refuse to retry a completed or emitted actor response')
    errors=[e.get('error',{}).get('message','') if e.get('type')=='turn.failed' else e.get('message','')
            for e in events if e.get('type') in ('error','turn.failed')]
    if not errors or not all('usage limit' in e.lower() for e in errors):
        raise ValueError('Only explicit pre-response quota failures are supported')
    return errors


def archive(root):
    if (root/'all_candidates_frozen.json').exists():raise ValueError('Cannot alter completed training layout')
    candidates=[]
    for invocation in sorted(root.glob('replica_*/*/round_*/invocation.json')):
        stage=invocation.parent
        if (stage/'generation.json').exists():continue
        errors=quota_failure(stage);replica=stage.parent.parent;arm=stage.parent.name
        revision=int(stage.name.split('_')[1]);manifest=json.loads((replica/'manifest.json').read_text())
        variant.check(replica,manifest)
        prompt,images=variant.context(replica,arm,revision,manifest)
        metadata=json.loads(invocation.read_text())
        if prompt!=(stage/'prompt.txt').read_text() or [variant.base.digest(p) for p in images]!=[r['sha256'] for r in metadata['images']]:
            raise ValueError('Interrupted context no longer reconstructs exactly')
        candidates.append((stage,{'original_directory':str(stage.relative_to(root)), 'errors':errors,
            'files_sha256':file_hashes(stage),'prompt_sha256':metadata['prompt_sha256'],
            'image_sha256':[r['sha256'] for r in metadata['images']]}))
    if not candidates:raise ValueError('No interrupted quota attempts')
    parent=root/'interrupted_actor_attempts';parent.mkdir(exist_ok=True)
    number=1
    while (parent/f'batch_{number:03d}').exists():number+=1
    batch=parent/f'batch_{number:03d}';batch.mkdir()
    receipt={'scope':__doc__,'created_at_utc':datetime.now(timezone.utc).isoformat(),
             'source_sha256':variant.base.digest(Path(__file__)),'protocol_sha256':variant.base.digest(root/'protocol.json'),
             'status':'prepared','attempts':[row for _,row in candidates],
             'resume_policy':'Retry only after service availability is restored; same frozen model/context. Count interrupted attempts separately from completed generations and policy failures.'}
    for stage,row in candidates:row['archive_directory']=str((batch/row['original_directory']).relative_to(root))
    variant.base.write(batch/'receipt.json',receipt)
    shutil.copy2(Path(__file__),batch/'archive_driver.py')
    for stage,row in candidates:
        target=root/row['archive_directory'];target.parent.mkdir(parents=True,exist_ok=True)
        shutil.move(str(stage),str(target))
        if file_hashes(target)!=row['files_sha256']:raise ValueError('Interrupted artifacts changed during archival')
    receipt['status']='archived';variant.base.write(batch/'receipt.json',receipt)
    print({'archived_pre_response_quota_failures':len(candidates),'receipt':str(batch/'receipt.json')})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    archive(parser.parse_args().root.resolve())
