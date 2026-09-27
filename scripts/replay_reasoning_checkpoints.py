"""Replay recorded sensor checkpoints; no private truth or success labels are read."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from reasoning_checkpoint_state import ReasoningCheckpoints


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(paths):
    pairs=[];events=[];example=None
    for path in paths:
        records=[r for r in json.loads(path.read_text()) if r.get('kind')=='planner_state']
        episode=str(path.resolve());memory=ReasoningCheckpoints(episode,capacity=4);previous=None
        for index,record in enumerate(records):
            label=f'{index}:{record["label"]}';packet=record['state']
            if previous is not None and packet['time']>memory.checkpoints[previous]['time']+1e-6:
                result=memory.compare(previous,packet,episode=episode)
                changes=result['checkpoint_changes']['objects']
                pairs.append({'source':str(path),'from_label':previous,'to_label':label,
                              'available_object_pairs':sum(x['status']=='estimated_change' for x in changes.values())})
                for name,change in changes.items():
                    if change.get('estimated_distance_m',0)<=.01:
                        continue
                    vector=change.get('jointly_constrained_translation_m')
                    events.append({'source':str(path),'from_label':previous,'to_label':label,'object':name,
                        'estimated_distance_m':change['estimated_distance_m'],
                        'joint_constraint_rank':change['joint_constraint_rank'],
                        'jointly_constrained_distance_m':None if vector is None else float(np.linalg.norm(vector)),
                        'both_estimates_fully_observed':change['both_estimates_fully_observed']})
                    if example is None:example={'source':str(path),'to_label':label,'packet':result}
            memory.remember(label,packet,episode=episode);previous=label
    return {'scope':'Offline recorded sensor checkpoints only; no private poses, outcome labels, reserved data, or policy execution.',
            'source_sha256':{str(p):digest(p) for p in (Path(__file__),Path(__file__).with_name('reasoning_checkpoint_state.py'),Path(__file__).with_name('checkpoint_geometry_prototype.py'))},
            'inputs_sha256':{str(p):digest(p) for p in paths},
            'checkpoint_pairs':len(pairs),'available_object_pairs':sum(p['available_object_pairs'] for p in pairs),
            'estimated_changes_above_10mm':len(events),
            'large_changes_with_jointly_constrained_component_above_10mm':sum(e['jointly_constrained_distance_m'] is not None and e['jointly_constrained_distance_m']>.01 for e in events),
            'large_changes_joint_rank_histogram':{str(rank):sum(e['joint_constraint_rank']==rank for e in events) for rank in (None,0,1,2,3)},
            'conditional_effects_in_replay':False,'pairs':pairs,'large_changes':events,'example':example,
            'not_demonstrated':['Certified physical motion','Calibrated uncertainty','Physical prediction accuracy','Policy learning benefit','Reasoning acceleration']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=replay(args.records)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('checkpoint_pairs','available_object_pairs','estimated_changes_above_10mm','large_changes_with_jointly_constrained_component_above_10mm','large_changes_joint_rank_histogram')},indent=2))
