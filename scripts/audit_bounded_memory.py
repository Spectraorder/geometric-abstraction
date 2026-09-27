"""Test whether current geometric bounds can reject stale hidden-axis memory.

All modes receive the same frozen RGB-D fits. This is a development diagnostic,
not a policy experiment. Missing objects are not silently made visible. Only
previously fully observed centers may anchor a static-memory hypothesis.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from audit_rigid_containment import estimate,score

MODES=('instantaneous','static_memory','bounded_static_memory')


def predict(packet):
    memories={mode:{} for mode in MODES[1:]};output=[]
    for row in packet['rows']:
        current=np.asarray(row['center_m']);projector=np.asarray(row['observed_projector'])
        for mode in MODES:
            item=copy.deepcopy(row);item['mode']=mode;item['memory_used']=False;item['memory_rejected']=False
            if mode!='instantaneous':
                prior=memories[mode].get(row['color']);center=current.copy()
                anchor=row['observed_rank']==3;origin=row['time'] if anchor else None
                if not anchor and prior is not None and prior['anchor']:
                    candidate=projector@current+(np.eye(3)-projector)@prior['center']
                    local=candidate@np.asarray(row['rotation'])
                    compatible=bool(row['bounds_available'] and np.all(local>=np.asarray(row['center_local_lower_m']))
                                    and np.all(local<=np.asarray(row['center_local_upper_m'])))
                    if mode=='static_memory' or compatible:
                        center=candidate;anchor=True;origin=prior['origin'];item['memory_used']=True
                    else:item['memory_rejected']=True
                memories[mode][row['color']]={'center':center,'anchor':anchor,'origin':origin}
                item['center_m']=center.tolist();item['full_anchor_time']=origin
                item['memory_condition']='object remained static along presently unobserved directions' if item['memory_used'] else None
            output.append(item)
    return {'trial':packet['trial'],'rows':output}


def summarize(packets):
    rows=[row for packet in packets for row in packet['rows']];groups={}
    for mode in MODES:
        groups[mode]={}
        for subset in ['all','partial']:
            chosen=[row for row in rows if row['mode']==mode and (subset=='all' or row['observed_rank']<3)]
            errors=[row['current_center_error_mm'] for row in chosen]
            groups[mode][subset]={'count':len(chosen),'median_mm':float(np.median(errors)),
                'p95_mm':float(np.percentile(errors,95)),'max_mm':max(errors),
                'memory_used':sum(row['memory_used'] for row in chosen),
                'memory_rejected':sum(row['memory_rejected'] for row in chosen)}
    return {'groups':groups,'limits':['Identical correlated development checkpoints, no policy-success comparison.',
        'Static retention is explicitly conditional; bounds can reject large stale errors but do not prove absence of motion.',
        'Only previously fully observed centers anchor memory; no output is invented for an unavailable current pose.',
        'Ideal simulated depth, unique colors, measured cube dimensions; no calibrated real-sensor guarantee.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replica',type=Path,required=True);parser.add_argument('--trials',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--robot-mask',action='store_true');args=parser.parse_args()
    manifest=json.loads((args.replica/'manifest.json').read_text());path=args.replica/'geometry.py'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['geometry_sha256']:raise ValueError('Library mismatch')
    library={};exec(compile(path.read_text(),str(path),'exec'),library)
    args.output.mkdir(parents=True,exist_ok=False)
    for name in ['audit_bounded_memory.py','audit_rigid_containment.py','rigid_containment_prototype.py']:
        source=Path(__file__).with_name(name);(args.output/name).write_bytes(source.read_bytes())
    packets=[];mask_reports=[]
    if args.robot_mask:
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        from view_scene import get_so100_spec
        from aspire_so100.robot_visual_geometry import exclude_robot_spheres
        radius=float(get_so100_spec().site('ee_site').size[0])
        source=Path(__file__).resolve().parents[1]/'src/aspire_so100/robot_visual_geometry.py'
        (args.output/source.name).write_bytes(source.read_bytes())
    for path in args.trials:
        callback=None;events=[]
        if args.robot_mask:
            calls=json.loads((path/'api_calls.json').read_text())
            own=[call for call in calls if call['method']=='proprioception']
            def callback(frame):
                matches=[call for call in own if abs(call['time']-float(frame['time']))<1e-6]
                if not matches:
                    events.append({'time':float(frame['time']),'robot_pose_available':False});return frame
                observation=matches[-1]['observation']
                filtered,event=exclude_robot_spheres(frame,[{'center_m':observation['tool_position'],'radius_m':radius}],observation['time'])
                events.append(dict(event,robot_pose_available=True));return filtered
        packets.append(predict(estimate(path.resolve(),library,callback)))
        mask_reports.append({'trial':str(path),'events':events})
    if args.robot_mask:
        (args.output/'robot_mask_inputs.json').write_text(json.dumps({'radius_m':radius,'source':'get_so100_spec().site(ee_site); poses from logged robot-only proprioception','trials':mask_reports},indent=2)+'\n')
    (args.output/'sensor_only_estimates.json').write_text(json.dumps(packets,indent=2)+'\n')
    scored=[score(packet) for packet in packets]
    (args.output/'offline_scored_estimates.json').write_text(json.dumps(scored,indent=2)+'\n')
    summary=summarize(scored);(args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
