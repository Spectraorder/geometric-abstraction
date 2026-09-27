"""Verify matched R0 physics and numeric state across registered training splits."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def audit(root):
    protocol=read(root/'protocol.json');rows=[]
    for replica in protocol['replicas']:
        directory=root/replica['directory'];manifest=read(directory/'manifest.json')
        if digest(directory/'manifest.json')!=replica['manifest_sha256']:
            raise ValueError('Replica manifest changed')
        for seed in manifest['train_seeds']:
            paths=[directory/arm/'revision_000/trials'/f'seed_{seed}' for arm in ('numeric','symbolic')]
            results=[read(p/'result.json') for p in paths]
            for field in ('candidate_sha256','scene','physics','sensor','wrist_depth','camera_calls','success','policy_error'):
                if results[0][field]!=results[1][field]:
                    raise ValueError(f'R0 mismatch: {replica["directory"]}, {seed}, {field}')
            if digest(paths[0]/'initial.png')!=digest(paths[1]/'initial.png'):
                raise ValueError('Initial sensor image differs')
            with np.load(paths[0]/'rollout.npz') as a,np.load(paths[1]/'rollout.npz') as b:
                if a.files!=b.files or not all(a[k].dtype==b[k].dtype and a[k].shape==b[k].shape and a[k].tobytes()==b[k].tobytes() for k in a.files):
                    raise ValueError(f'R0 physical arrays differ for seed {seed}')
            packets=[[r for r in read(p/'geometry_records.json') if r.get('kind')=='planner_state'] for p in paths]
            if len(packets[0])!=len(packets[1]):raise ValueError('Checkpoint count differs')
            for first,second in zip(*packets):
                projection=json.loads(json.dumps(second['state']))
                projection['schema']='mixed_metric_scene_v1'
                projection.pop('facts',None);projection.pop('contact_state',None)
                for obj in projection['objects'].values():obj.pop('references',None)
                if first['label']!=second['label'] or first['state']!=projection:
                    raise ValueError(f'R0 numeric checkpoint differs for seed {seed}')
            rows.append({'replica':replica['directory'],'seed':seed,'identical_physical_arrays':True,
                         'identical_numeric_checkpoints':True,'checkpoint_count':len(packets[0]),
                         'source_sha256':{str(p/f):digest(p/f) for p in paths for f in ('result.json','initial.png','rollout.npz','geometry_records.json')}})
    return {'status':'passed','purpose':'Offline starter equivalence across numeric and symbolic interfaces; no evolved policies or reserved layouts.',
            'protocol_sha256':digest(root/'protocol.json'),'auditor_sha256':digest(Path(__file__)),
            'projection':'Remove only symbolic schema, facts, contact_state and object reference names; compare all remaining state exactly.',
            'pairs':rows,'pair_count':len(rows),'not_demonstrated':'Policy learning advantage. Physical arrays used in this audit are never supplied to actors.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=audit(args.root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','pair_count')},indent=2))
