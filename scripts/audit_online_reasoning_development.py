"""Recompute development request usage from archived cumulative session counters."""
import argparse
import hashlib
import json
from pathlib import Path
from online_bridge_actor import audit_decision_events
from verified_token_usage import turn_usage


def read(path):return json.loads(path.read_text())
def records(path):return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def archived_source(root, episode, name):
    """Prefer the episode snapshot; never hide corruption with a live fallback."""
    candidates = (episode/'sources'/name, root/'sources'/name,
                  root/'sources/scripts'/name, Path(__file__).with_name(name))
    return next((path for path in candidates if path.exists()), candidates[-1])


def audit(root, episode_paths=None):
    reports=[]
    paths = root.glob('*/result.json') if episode_paths is None else (
        root/path/'result.json' for path in episode_paths)
    for path in sorted(paths):
        d=path.parent;result=read(path);computed=[];failures=0
        for context in sorted(d.glob('context_*')):
            previous=[];totals={};transcript_before=b''
            if (context/'actor_home/auth.json').exists():raise ValueError('Temporary credentials retained')
            for turn in sorted(context.glob('decision_*')):
                if (turn/'attempt_failure.json').exists():
                    failures+=1;continue
                receipt=read(turn/'receipt.json');invocation=read(turn/'invocation.json')
                raw=audit_decision_events(records(turn/'model.jsonl'),receipt['model'],bool(previous))
                transcript=(turn/'session.jsonl').read_bytes()
                if not transcript.startswith(transcript_before):raise ValueError('Rewritten context')
                archive=[json.loads(line) for line in transcript.decode().splitlines() if line.strip()]
                counts=[e['payload']['info'] for e in archive if e.get('type')=='event_msg'
                        and e.get('payload',{}).get('type')=='token_count' and e['payload'].get('info') is not None]
                if counts[:len(previous)]!=previous:raise ValueError('Rewritten usage history')
                next_totals=counts[-1]['total_token_usage']
                usage=turn_usage(totals,next_totals,raw,counts[len(previous):])
                if usage!=receipt['usage']:raise ValueError('Usage mismatch')
                for image in invocation['images']:
                    if digest(Path(image['source']))!=image['sha256']:raise ValueError('Image changed')
                computed.append(usage);previous=counts;totals=next_totals;transcript_before=transcript
        sums={key:sum(r.get(key,0) for r in computed) for key in result['usage']}
        if sums!=result['usage'] or failures!=result['incomplete_requests']:
            raise ValueError('Episode resource aggregate mismatch')
        if len(computed)!=result['model_requests_completed']:raise ValueError('Request count mismatch')
        if result['timing']['reset_count']!=1:raise ValueError('Unexpected reset')
        for name,expected in read(d/'sources.json').items():
            source=archived_source(root,d,name)
            if digest(source)!=expected:raise ValueError('Unavailable source version: '+name)
        reports.append({'episode':str(d.relative_to(root)),'status':result['status'],'requests':len(computed),
                        'incomplete_requests':failures,'verified_usage':sums,'result_sha256':digest(path)})
    if not reports:raise ValueError('No completed episodes')
    report={'status':'passed','kind':'development_resource_and_source_audit',
            'episodes':reports,'auditor_sha256':digest(Path(__file__)),
            'scope':'Sources, cumulative token deltas, image file hashes, cleanup and reset counters; '
                    'this does not retrospectively certify unnamed-mesh robot contact exclusion.'}
    (root/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    report=audit(p.parse_args().root);print(json.dumps({'status':report['status'],'episodes':len(report['episodes'])}))
