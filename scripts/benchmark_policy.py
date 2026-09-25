"""Audit coverage of a declared policy profile; missing evidence cannot pass.

Consumes normalized private-evaluator records, never model self-reports.
A test distribution and model-independent labels must be frozen separately.
"""
import argparse,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def wilson(k,n):
    if not n:return [0.,1.]
    z=1.95996398454;p=k/n;den=1+z*z/n
    c=(p+z*z/(2*n))/den;d=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0,c-d),min(1,c+d)]

def score(suite,profile,records,attestations):
    selected=suite['profiles'][profile];seen=set();groups={c:[] for c in selected}
    for r in records:
        if r['cell'] not in groups:raise ValueError('Record outside declared profile')
        key=(r['cell'],r['case_id'],r['repeat'])
        if key in seen:raise ValueError('Duplicate attempt')
        seen.add(key)
        if r['status'] not in {'completed','policy_failed','infrastructure_error','timeout'}:raise ValueError('Unknown terminal status')
        if type(r['passed_endpoint']) is not bool:raise ValueError('Boolean private-evaluator endpoint required')
        if r['status']!='completed' and r['passed_endpoint']:raise ValueError('An incomplete attempt cannot pass')
        if r.get('split')!='heldout':raise ValueError('Release assessment requires heldout records')
        if not r.get('policy_sha256') or not r.get('harness_sha256') or not r.get('evaluator_sha256') or not r.get('configuration_sha256'):raise ValueError('Missing frozen provenance')
        if type(r.get('unsafe_actions')) is not int or r['unsafe_actions']<0:raise ValueError('Explicit nonnegative unsafe-action count required')
        if type(r['repeat']) is not int or r['repeat']<0:raise ValueError('Invalid repeat index')
        groups[r['cell']].append(r)
    # A profile describes one frozen implementation, not the best candidate per case.
    fingerprints={(r['policy_sha256'],r['harness_sha256'],r['evaluator_sha256'],r['configuration_sha256']) for r in records}
    if len(fingerprints)>1:raise ValueError('Mixed policy/harness/evaluator versions')
    rows=[]
    for cell in suite['cells']:
        if cell['id'] not in groups:continue
        rr=groups[cell['id']];cases={}
        for r in rr:cases.setdefault(r['case_id'],[]).append(r)
        n=len(cases);k=sum(all(r['passed_endpoint'] for r in attempts) for attempts in cases.values())
        ci=wilson(k,n);unsafe=sum(r.get('unsafe_actions',0) for r in rr)
        covered=n>=cell['minimum_n']
        rows.append({'cell':cell['id'],'endpoint':cell['endpoint'],'n':n,'attempts':len(rr),'passed':k,'rate':k/n if n else None,
            'wilson_95':ci,'coverage_complete':covered,'unsafe_actions':unsafe,
            'infrastructure_errors':sum(r['status']=='infrastructure_error' for r in rr),
            'passes_release_target':covered and ci[0]>=cell['wilson_lower_target'] and unsafe==0})
    missing=[g for g in suite['gates'] if attestations.get(g) is not True]
    return {'profile':profile,'status':'meets_declared_profile' if not missing and all(r['passes_release_target'] for r in rows) else 'incomplete_or_below_target','cells':rows,'missing_gates':missing,'scope':'Frozen task distributions only. Repeats share a scene cluster; a scene passes only when all repeats pass. Not universal policy completeness.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--profile',choices=['tower_recovery','inventory','all_supported'],default='all_supported');p.add_argument('--records',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Output must be new')
    suite=json.loads((ROOT/'benchmarks/policy_suite.json').read_text())
    payload=json.loads(a.records.read_text()) if a.records else {'records':[],'attestations':{}}
    result=score(suite,a.profile,payload['records'],payload.get('attestations',{}))
    result['protocol_sha256']=hashlib.sha256((ROOT/'benchmarks/policy_suite.json').read_bytes()).hexdigest()
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(result['status'])
if __name__=='__main__':main()
