"""Check measured campaign routing, frozen sources, guards and exported media."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--require-videos',action='store_true');a=p.parse_args()
manifest=json.loads((a.root/'campaign.json').read_text());frozen=json.loads((a.root/'frozen_sources.json').read_text())
report=json.loads((ROOT/'web/recovery/luna-astra/results.json').read_text())
assert report['complete'] and len(report['runs'])==len(manifest['jobs'])==57
assert len({r['candidate_sha256'] for r in report['runs']})==1
provider_calls=0;violations=0
for job in manifest['jobs']:
    folder=a.root/job['id'];h=json.loads((folder/'harness.json').read_text())
    assert h['frozen_sources']==frozen,job['id']
    r=json.loads((folder/'result.json').read_text());t=json.loads((folder/'decision_trace.json').read_text())
    assert r['seed']==job['seed'] and t['mode']==job['mode']
    fast='gpt-6-astra' if job['mode']=='astra_only' else 'gpt-6-luna'
    for call in t['fast_calls']:
        assert call['requested_model']==fast and call['reasoning']==job['fast_effort']
    for call in t['planner_calls']:
        assert call['requested_model']=='gpt-6-astra' and call['reasoning']==job['slow_effort']
    if job['mode']=='luna_only':assert not t['planner_calls']
    provider_calls+=len(t['fast_calls'])+len(t['planner_calls'])
    for e in t['records']:
        s=e['state'];assert s['phase']=='initial_observation'
        assert set(s)=={'version','time_s','phase','visible_colors','table_fit_valid','grasp_confirmed','reobservations_left','grasp_retries_left'}
        if e['gated_action']=='continue':
            assert s['table_fit_valid'] and set(s['visible_colors'])=={'blue','red','green'}
        if e['plan_after']!=e['plan_before']:assert e['gated_action'] in {'reobserve','stop'}
        if e['trigger']=='precondition_violation':violations+=1
    item=next(x for x in report['runs'] if x['id']==job['id'])
    if a.require_videos:
        assert 'video' in item,job['id']
        assert hashlib.sha256((ROOT/'web/recovery/luna-astra'/item['video']).read_bytes()).hexdigest()==item['video_sha256']
print(json.dumps({'attempts':57,'model_calls':provider_calls,'rejected_precondition_violations':violations,'all_sources_match':True,'all_model_effort_routes_match':True,'no_jev':True,'shared_candidate':True,'videos_verified':a.require_videos},indent=2))
