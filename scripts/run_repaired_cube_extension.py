"""Resume the path repair while explicitly auditing its failed log-only leases."""
import argparse
from pathlib import Path
from unittest.mock import patch
import resume_three_cube_contract_extension as repair


def audit_leases(output,limits):
    receipt=repair.read(output/repair.REPAIR/'registration.json')
    inspected=receipt['preserved_results'];root=output.parent;report={}
    for resource,limit in limits.items():
        records=[repair.read(p) for p in (output/'scheduler'/resource/'events').glob('*.json')]
        timeline=[];repaired=0
        for record in records:
            repair.require('finished_at' in record,'Unfinished lease')
            if record['status']!='completed':
                path=record['label']+'/result.json'
                repair.require(record['status']=='failed' and 'is not in the subpath' in record.get('error','')
                    and path in inspected and repair.base.digest(root/path)==inspected[path],
                    'Uninspected failed lease')
                repaired+=1
            timeline.extend([(record['started_at'],1),(record['finished_at'],-1)])
        active=maximum=0
        for _,change in sorted(timeline):
            active+=change;maximum=max(maximum,active)
        repair.require(active==0 and maximum<=limit,'Concurrency audit failed')
        report[resource]={'jobs':len(records),'max_concurrent':maximum,'limit':limit,
            'verified_results_with_repaired_log_failures':repaired,
            'queue_seconds':sum(r['started_at']-r['queued_at'] for r in records)}
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();output=root/repair.extension.DIRECTORY
    receipt=repair.read(output/repair.REPAIR/'lease_audit_adapter.json')
    repair.require(repair.base.digest(Path(__file__))==receipt['sha256']==repair.base.digest(output/repair.REPAIR/Path(__file__).name),'Lease auditor changed')
    with repair.extension.runner.installed(),patch.object(repair.extension,'audit_leases',audit_leases):
        print(repair.run(root)['mean_curves'])
