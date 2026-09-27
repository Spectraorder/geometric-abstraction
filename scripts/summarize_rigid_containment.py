"""Summarize development interval coverage and symbolic information availability."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from rigid_containment_prototype import above_interval


def summarize(roots):
    reports=[]
    for root in roots:
        packets=json.loads((root/'sensor_only_estimates.json').read_text())
        before=Counter();after=Counter();transitions=Counter()
        for packet in packets:
            records={row['time']:row for row in json.loads((Path(packet['trial'])/'geometry_records.json').read_text())}
            groups={}
            for row in packet['rows']:groups.setdefault(row['time'],{})[row['color']]=row
            for time,objects in groups.items():
                for up,low in [('red','blue'),('green','red')]:
                    if up not in objects or low not in objects:continue
                    old=next((f for f in records[time]['facts'] if f['predicate']=='above' and f['args']==[up,low]),None)
                    if old is None:continue
                    value=above_interval(objects[up],objects[up]['rotation'],objects[low],objects[low]['rotation'])
                    before[old['truth']]+=1;after[value['truth']]+=1
                    transitions[old['truth']+' -> '+value['truth']]+=1
        reports.append({'audit':str(root),'coverage_and_accuracy':json.loads((root/'summary.json').read_text()),
            'same_relation_queries':sum(before.values()),'current_above':dict(before),'interval_above':dict(after),
            'transitions':dict(transitions),'sensor_only_sha256':hashlib.sha256((root/'sensor_only_estimates.json').read_bytes()).hexdigest()})
    return {'schema':'rigid_containment_development_report_v1','audits':reports,
        'interpretation':'Information availability and conditional interval coverage only; no policy-success or improved point-estimate claim.',
        'prototype_sha256':hashlib.sha256(Path(__file__).with_name('rigid_containment_prototype.py').read_bytes()).hexdigest()}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--audits',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();value=summarize(args.audits)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in row.items() if k!='coverage_and_accuracy'} for row in value['audits']],indent=2))
