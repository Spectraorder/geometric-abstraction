"""Check completed reserved slots while the final per-chain audit is pending.

Does not modify candidates, select policies, call models or start rollouts. Slot
counts are separate from the full audit, which counts a chain's reserved phase only
once its complete summary exists. Partial batches never receive an SR estimate.
"""
import argparse
import hashlib
import json
from pathlib import Path


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def require(condition, message):
    if not condition: raise ValueError(message)


def inspect(root):
    protocol=read(root/'protocol.json');gate_path=root/'all_candidates_frozen.json'
    gate=read(gate_path)
    require(gate['protocol_sha256']==sha(root/'protocol.json'),'Global protocol changed')
    rows=[];hashes={};initial={};seen=set();valid=invalid=0
    for spec in protocol['replicas']:
        replica=root/spec['directory'];manifest=read(replica/'manifest.json')
        require(sha(replica/'manifest.json')==spec['manifest_sha256'],'Manifest changed')
        child=read(replica/'all_candidates_frozen.json')
        require(child==gate['replicas'][spec['directory']],'Global and child freeze differ')
        for arm in protocol['arms']:
            selected=child['selected'][arm]['revision']
            library=replica/(arm+'_library.py')
            require(sha(library)==manifest['library_sha256'][arm],'Frozen geometry library changed')
            for entry in child['revisions'][arm]:
                revision=entry['revision'];candidate=replica/arm/'frozen_revisions'/f'revision_{revision:03d}.py'
                require(sha(candidate)==entry['sha256'],'Frozen candidate changed')
                if revision==selected:
                    require(sha(replica/arm/'frozen.py')==entry['sha256'],'Selected source differs from its revision')
                    folder=replica/arm/'heldout'
                else:folder=replica/arm/'reserved_revisions'/f'revision_{revision:03d}'/'heldout'
                completed=successes=0
                for seed in manifest['heldout_seeds']:
                    path=folder/f'seed_{seed}'/'result.json'
                    if not path.exists():continue
                    require(path not in seen,'Duplicate slot path');seen.add(path)
                    result=read(path);hashes[str(path.relative_to(root))]=sha(path)
                    require(type(result['success']) is bool,'Nonboolean outcome')
                    require(result['seed']==seed and result['candidate_sha256']==entry['sha256'],'Trial provenance differs')
                    require(path.stat().st_mtime_ns>=gate_path.stat().st_mtime_ns,'Reserved result predates global freeze')
                    if result.get('invalid_candidate'):
                        require(not result['success'] and not result['rollout_executed'],'Invalid slot counted as executed success')
                        invalid+=1
                    else:
                        require(result['geometry_library_sha256']==manifest['library_sha256'][arm],'Trial library differs')
                        require(result['task']==manifest['task'] and result['scene']['recipe']==manifest['recipe'],'Task differs')
                        require(result['scene']['color_map']==manifest['color_maps'][str(seed)] and result['scene']['object_count']==4,'Scene differs')
                        require(result['physics']==manifest['physics'] and result['sensor']==manifest['sensor'],'Simulation configuration differs')
                        require(result['wrist_depth'] and not result['sensor_blackout'],'Sensor availability differs')
                        image=path.parent/'initial.png'
                        require(image.stat().st_mtime_ns>=gate_path.stat().st_mtime_ns,'Reserved scene predates global freeze')
                        fingerprint=sha(image)
                        if seed in initial:require(fingerprint==initial[seed],'Initial RGB differs across matched reserved layouts')
                        initial[seed]=fingerprint;valid+=1
                    completed+=1;successes+=result['success']
                rows.append({'replica':spec['directory'],'arm':arm,'revision':revision,
                    'selected_from_training':revision==selected,'completed_slots':completed,
                    'planned_slots':len(manifest['heldout_seeds']),
                    'batch_complete':completed==len(manifest['heldout_seeds']),
                    'successes':successes if completed==len(manifest['heldout_seeds']) else None,
                    'sr':successes/completed if completed==len(manifest['heldout_seeds']) else None})
    require(sum(r['planned_slots'] for r in rows)==protocol['reserved_slots'],'Registered reserved budget differs')
    require(valid+invalid==len(seen),'Completed slot count differs')
    return {'scope':__doc__,'status':'all_slots_present' if len(seen)==protocol['reserved_slots'] else 'in_progress',
        'completed_reserved_slots':len(seen),'planned_reserved_slots':protocol['reserved_slots'],
        'valid_executed_reserved_slots':valid,'invalid_reserved_slots':invalid,
        'full_final_audit_required':True,'frozen_candidate_and_settings_checks':'passed',
        'protocol_sha256':sha(root/'protocol.json'),'global_freeze_sha256':sha(gate_path),
        'inspector_sha256':sha(Path(__file__)),'batches':rows,'result_sha256':hashes}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=inspect(args.root.resolve())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print({k:report[k] for k in ('status','completed_reserved_slots','planned_reserved_slots','invalid_reserved_slots','frozen_candidate_and_settings_checks')})
    for row in report['batches']:
        if row['batch_complete']:print({k:row[k] for k in ('replica','arm','revision','selected_from_training','successes','planned_slots')})
