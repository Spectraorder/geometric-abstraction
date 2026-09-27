"""Compare robot-mask diagnostics on matched keys and report lost coverage separately."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def read(root):
    packets=json.loads((root/'offline_scored_estimates.json').read_text())
    return {(packet['trial'],row['time'],row['color'],row['mode']):row for packet in packets for row in packet['rows']}


def stats(values):
    return {'count':len(values),'median_mm':float(np.median(values)),'p95_mm':float(np.percentile(values,95)),'max_mm':max(values)}


def compare(original,masked):
    old=read(original);new=read(masked)
    before={key for key in old if key[-1]=='instantaneous'};after={key for key in new if key[-1]=='instantaneous'}
    common=sorted(before&after)
    groups={}
    for group,keys in [('all',common),('original_partial',[key for key in common if old[key]['observed_rank']<3])]:
        groups[group]={}
        for name,data,mode in [('original_instantaneous',old,'instantaneous'),('robot_mask_instantaneous',new,'instantaneous'),
                               ('robot_mask_bounded_memory',new,'bounded_static_memory')]:
            values=[data[(*key[:3],mode)]['current_center_error_mm'] for key in keys]
            groups[group][name]=stats(values)
    return {'original':str(original),'masked':str(masked),'matched_groups':groups,
            'coverage':{'original':len(before),'masked':len(after),'matched':len(common),'new_only':len(after-before),
                'lost':[{ 'trial':key[0],'time':key[1],'color':key[2],
                    'original_center_error_mm':old[key]['current_center_error_mm']} for key in sorted(before-after)]},
            'sensor_only_sha256':{str(root):hashlib.sha256((root/'sensor_only_estimates.json').read_bytes()).hexdigest() for root in [original,masked]}}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();value={'schema':'robot_mask_memory_development_report_v1','comparisons':[
        compare(Path('outputs/bounded_memory_audit_'+suffix),Path('outputs/bounded_memory_robot_mask_'+suffix)) for suffix in ['01','02']],
        'limits':['Robot radius comes from its model and centers from same-time encoder-derived proprioception; no object truth is used by the mask.',
                  'Private physics is used only after all sensor-only estimates are saved, for scoring.',
                  'Already inspected development trajectories, not a held-out policy-success or real-sensor result.',
                  'This masks one known robot visual sphere, not full robot meshes or arbitrary same-color objects.']}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value,indent=2))
