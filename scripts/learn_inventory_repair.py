"""Bounded LLM edit/execute/repair campaign, seeded by the best inventory policy.

The learner produces exact source edits, avoiding retransmission of the entire
58 KB policy as model output. No human-authored recovery controller is inserted.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import fcntl
import signal

from aspire_so100.policy_validation import validate_source
from aspire_so100.repair_learning import VERSION, annotate, build_context, select_candidates, progress_key

ROOT = Path(__file__).resolve().parents[1]
TRAIN = (1501, 1502, 1503, 1504)
HELDOUT = (2501, 2502, 2503, 2504)
CASES = ('centered', 'offset')
EVENT_DIR = None
CONTRACT = '''You are the policy learner for a simulated human-placement REPAIR curriculum.
Adapt the supplied best inventory policy. Return JSON hypothesis and edits [{old,new}].
Each old string must occur exactly once in the CURRENT supplied source; apply edits in order.
Use only supplied code, feedback and images. Do not use tools, files, networking or introspection.
Only numpy/scipy/math imports; code must define run(api). All logging through api.note.

TASK: a blue cube is already on the table and a red slender prism is ALREADY ON that cube.
It is either centered or standing with an unknown 6-8 mm lateral offset. Random
same-color and different-shape distractors come from the 20-object inventory. Observe
RGB-D to identify the actual supported red slender prism, estimate its geometry and
support relation, and recenter it if needed. Do not assume each object bottom is the table.
Do not confuse tabletop red distractors with the supported target. Preserve the foundation:
never move the blue cube more than 3 mm, never drop/lower the target onto the table.
You may gently regrasp/lift/reseat the red target. No relocation of the whole tower.
Centered controls: leave the sound stack alone (target displacement <=4 mm).
Offsets: final horizontal center error <=4 mm, with correct base/table and target/base
contacts, upright within 8 degrees, COM support margin >=3 mm, no robot contact,
linear speed <=.015 m/s, angular speed <=.1 rad/s, and <=3 mm drift over a 5-second
final hold. Withdraw and wait at least 6 seconds before returning. The private evaluator
runs AFTER the policy; cannot be queried online. Corrected poses are not injected.
Simulator gravity runs continuously during all movements and waits.

API: camera() or camera('side') -> rgb, depth_m (optical Z), valid_depth, intrinsics,
camera_to_world, time. camera('wrist') -> RGB, intrinsics, camera_to_world, time; NO depth.
IMPORTANT: camera() and camera('side') are ALIASES of the SAME FIXED CAMERA.
They are not two viewpoints; repeated identical views do not resolve occlusion.
Moving the arm can clear its occlusion; it does not move the side camera.
Wrist RGB is a distinct view but has no depth. Do not assume a hardcoded upward
retreat is reachable. Preview smaller/lateral alternatives using measured robot pose.
proprioception() -> joint_angles, tool_position, tool_rotation, time (robot state only).
plan_tool(xyz,orientation=None) -> joint_angles, tool_position, tool_rotation, position_error;
robot-only IK, no collision/objects oracle. move_tool(xyz,seconds=1.5,orientation=None),
move_joints(5 angles,seconds), gripper(angle,seconds), wait(seconds), note(text).
All positions/dimensions must be measured from pixels/calibration, never memorized poses,
seed-specific branches, or catalog dimensions. Geometry priors already in the supplied
fitter can be retained, but do not substitute known true sizes for camera measurement.
Depth backprojection is world=R@(inv(K)@[u,v,1]*depth[v,u])+t.
Jaw -.174 closes, .8 opens; allowed [-.174,1.75]. Motion .1..10 seconds, waits <=10.
120 wall seconds and 600 RPC calls. Blackout must stop before moving.
The prior policy's one-target tabletop assumptions and base relocation are inappropriate
for this task. You may edit perception helpers and controller; existing full source follows.
Prefer compact, testable repairs and retain sensor gating. Do not relax the evaluator.
Any legacy-function omission comments are prompt display only, not actual source;
do not target those comments in edits. All retained code is literal saved source.

STAGED DIAGNOSTICS (no truth or feedback is returned online):
api.report('identify',base_center=[x,y,z],target_center=[x,y,z],target_dimensions=[dx,dy,dz])
logs your measured supported-object geometry; private checks after the trial verify
center errors <=10 mm and dimension errors <=8 mm. Do not guess ground-truth sizes.
api.report('plan',role='approach'|'grasp'|'lift',position=[x,y,z]) immediately after
api.plan_tool on that same position logs a local preflight. All three previews must
have error <=3 mm and tool tilt <=15 degrees, remain within 25 mm XY of the target,
and use approach/lift heights 5..60 mm above the proposed grasp. The grasp site
is 5..70 mm above target center; measure finger geometry to select the actual offset.
This milestone certifies local IK only, not a collision-free grasp or trajectory.
After these milestones a private monitor verifies a >=5 mm lift with both jaw
contacts for >=0.1 s. Closing an empty gripper earns no grasp credit. Final repair
still requires all original stability checks and preserved centered controls.
api.report('reject',check='support_height',measured=0.012,threshold=0.003,reason='...')
or api.report('observe',...) records your numeric sensor-derived diagnostics.
Reports use JSON values and return None. Never treat a report as confirmation.
Log each rejected candidate pair, measured bottom/top gap, dimensions, confidence,
IK error and proposed alternative so the next revision can diagnose the failed gate.
Focus on the specified earliest missing milestone, while preserving stages already
verified. Full task success is never replaced by self-reported milestones.
'''


def write(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def usage_total(usage):
    if not usage or any(not isinstance(u,dict) or not all(
            type(u.get(k)) is int and u[k]>=0 for k in ('input_tokens','output_tokens')) for u in usage):
        raise ValueError('Model usage is missing/incomplete; inspect before further paid calls')
    return sum(u['input_tokens']+u['output_tokens'] for u in usage)


def emit(message, **details):
    event = {'at': datetime.now(timezone.utc).isoformat(), 'message':message, **details}
    print(message,flush=True)
    if EVENT_DIR is not None:
        # Direct writes survive Slurm stdout/PTY routing problems.
        with (EVENT_DIR/'progress.log').open('a') as stream:
            stream.write(json.dumps(event)+'\n')
        write(EVENT_DIR/'progress.json',event)


def run_logged(command, *, phase, timeout, input=None, **kwargs):
    started = time.monotonic()
    emit('Starting '+phase,phase=phase)
    proc = subprocess.Popen(command, stdin=subprocess.PIPE if input is not None else None,
                            start_new_session=True, **kwargs)
    try:
        pending = input
        while True:
            remaining = timeout-(time.monotonic()-started)
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command,timeout)
            try:
                proc.communicate(input=pending,timeout=min(15.,remaining))
                break
            except subprocess.TimeoutExpired:
                pending = None
                emit('Still running '+phase,phase=phase,elapsed_seconds=round(time.monotonic()-started,1))
        if proc.returncode:
            raise subprocess.CalledProcessError(proc.returncode,command)
    finally:
        if proc.poll() is None:
            # Stop this trial/model's process group, including sandbox children.
            try: os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            try: proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL); proc.wait()
    emit('Finished '+phase,phase=phase,elapsed_seconds=round(time.monotonic()-started,1))


def apply_edits(source, edits):
    if not edits:
        raise ValueError('Learner returned no edits')
    previous_end = 0
    for index, edit in enumerate(edits):
        old, new = edit['old'], edit['new']
        if not old:
            raise ValueError('An edit cannot have empty source')
        start = source.find(old)
        if source.count(old) != 1:
            # Ordered edits provide context too. A repeated line is acceptable
            # only when the previous applied edit and the next unique source
            # anchor bracket exactly ONE occurrence. Never guess the first match.
            upper = len(source)
            for following in edits[index+1:]:
                anchor = following['old']
                if anchor and source.count(anchor) == 1:
                    upper = source.index(anchor)
                    break
            region = source[previous_end:upper]
            if region.count(old) != 1:
                raise ValueError('Edit is ambiguous even with neighboring anchors: '+repr(old[:120]))
            start = previous_end+region.index(old)
        if start < 0:
            raise ValueError('Edit source not found: '+repr(old[:120]))
        source = source[:start]+new+source[start+len(old):]
        previous_end = start+len(new)
    validate_source(source)
    return source


def evaluate(candidate, directory, seeds):
    directory.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in seeds:
        for case in CASES:
            out = directory/f'seed_{seed}_{case}'
            if not (out/'result.json').exists():
                if out.exists():
                    archive = Path(tempfile.mkdtemp(prefix=out.name+'.interrupted-', dir=directory))
                    out.rename(archive/'trial')
                    if out.with_suffix('.log').exists():
                        out.with_suffix('.log').rename(archive/'trial.log')
                with out.with_suffix('.log').open('w') as log:
                    run_logged([sys.executable, str(ROOT/'scripts/run_vision_trial.py'),
                        '--candidate', str(candidate), '--task', 'inventory_repair',
                        '--layout', 'sampled', '--object-count', '6', '--seed', str(seed),
                        '--repair-case', case, '--output', str(out)],
                        stdout=log, stderr=subprocess.STDOUT, timeout=240,
                        phase=f'{directory.parent.name}/{directory.name} seed {seed} {case}')
            result = json.loads((out/'result.json').read_text())
            if (result['candidate_sha256'] != digest(candidate)
                    or result['task'] != 'inventory_repair'
                    or result['scene']['object_count'] != 6
                    or result['scene']['repair_case'] != case or result['seed'] != seed):
                raise RuntimeError('Cached trial does not match: '+str(out))
            results.append(result | {'directory': str(out)})
            emit(f'{directory.name} {seed} {case}: success={result["success"]}',
                 phase='trial_complete',seed=seed,case=case,success=result['success'])
    summary = {'candidate': str(candidate), 'sha256': digest(candidate),
               'successes': sum(r['success'] for r in results), 'trials': len(results),
               'score': sum(r['metrics']['score'] for r in results)/len(results),
               'results': results}
    summary = annotate(summary)
    write(directory/'summary.json', summary)
    emit('Candidate evaluated',phase='evaluation_complete',candidate=str(candidate),cases=summary['case_metrics'])
    return summary


def propose(best, history, directory, args, focus='identify'):
    directory.mkdir(parents=True, exist_ok=False)
    source = Path(best['candidate']).read_text()
    prompt = build_context(best, history, focus, CONTRACT)
    (directory/'prompt.txt').write_text(prompt)
    selected = []
    for case in CASES:
        group = [r for r in history[-1]['results'] if r['scene']['repair_case'] == case]
        representative = min(group, key=lambda r: (r['success'], r['metrics']['score']))
        selected.extend(Path(representative['directory'])/f'{phase}.png' for phase in ('initial','outcome'))
    schema = {'type':'object', 'properties':{'hypothesis':{'type':'string'},
        'edits':{'type':'array','items':{'type':'object','properties':{
            'old':{'type':'string'},'new':{'type':'string'}},
            'required':['old','new'],'additionalProperties':False}}},
        'required':['hypothesis','edits'],'additionalProperties':False}
    executable = shutil.which('codex')
    if not executable:
        raise RuntimeError('codex is not on PATH')
    with tempfile.TemporaryDirectory(prefix='aspire-repair-learner-') as temp:
        work = Path(temp)
        fresh_home = work/'config'
        fresh_home.mkdir(mode=0o700)
        auth = Path(os.environ.get('CODEX_HOME', str(Path.home()/'.codex')))/'auth.json'
        if auth.exists():
            shutil.copyfile(auth, fresh_home/'auth.json')
            (fresh_home/'auth.json').chmod(0o600)
        write(work/'schema.json', schema)
        command = [executable,'exec','--ignore-user-config','--ephemeral','--sandbox','read-only',
            '--skip-git-repo-check','--cd',str(work),'--model',args.model,
            '-c','project_doc_max_bytes=0','--enable','skip_host_skill_discovery',
            '--disable','shell_tool','--disable','unified_exec','--disable','apps',
            '--disable','browser_use','--disable','computer_use','--disable','skill_search',
            '--output-schema',str(work/'schema.json'),'--output-last-message',str(work/'response.json'),'--json']
        for index, path in enumerate(selected):
            dest = work/f'image_{index}.png'
            shutil.copyfile(path, dest)
            command.extend(['--image', str(dest)])
        command.append('-')
        write(directory/'invocation.json', {'model': args.model, 'images': list(map(str,selected)),
              'prompt_chars':len(prompt),'parent_sha256':digest(Path(best['candidate'])),
              'parent_candidate':best['candidate'], 'loop_version':VERSION, 'focus':focus,
              'started_at':datetime.now(timezone.utc).isoformat(), 'tools_disabled':True})
        emit(f'Asking {args.model}: {len(prompt)} prompt characters, {len(selected)} images',phase='proposing',focus=focus)
        with (directory/'model.jsonl').open('w') as log, (directory/'model.stderr').open('w') as err:
            run_logged(command, input=prompt, text=True, stdout=log, stderr=err,
                env=dict(os.environ, CODEX_HOME=str(fresh_home)), timeout=600, phase='LLM proposal')
        events = [json.loads(line) for line in (directory/'model.jsonl').read_text().splitlines()
                  if line.startswith('{')]
        usage = [e['usage'] for e in events if e.get('type') == 'turn.completed' and 'usage' in e]
        write(directory/'usage.json', usage)
        usage_total(usage)
        if any(e.get('item',{}).get('type') in {'command_execution','mcp_tool_call','web_search','file_change'} for e in events):
            raise RuntimeError('Learner isolation audit failed')
        response = json.loads((work/'response.json').read_text())
        write(directory/'response.json', response)
        (directory/'candidate.py').write_text(apply_edits(source, response['edits']))
        emit(response['hypothesis'],phase='proposal_received')
    return usage


def main():
    global EVENT_DIR
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--model', default='gpt-6-astra')
    p.add_argument('--rounds', type=int, default=4)
    p.add_argument('--max-reported-tokens', type=int, default=80000)
    p.add_argument('--baseline-only', action='store_true')
    p.add_argument('--inspect-only', action='store_true', help='Audit saved evidence and preview the next prompt; no model or simulation calls')
    p.add_argument('--stagnation-rounds', type=int, default=2,
                   help='Stop after this many v2 revisions without verified progress (0 disables)')
    args = p.parse_args()
    if args.rounds < 1 or args.max_reported_tokens < 1 or args.stagnation_rounds < 0:
        p.error('rounds/budget must be positive; stagnation-rounds must be nonnegative')
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root/'campaign.json'
    if args.inspect_only:
        state = json.loads(path.read_text())
        champion,parent,archive,focus = select_candidates(state['history'])
        prompt = build_context(parent,state['history'],focus,CONTRACT)
        report = {'loop_version':VERSION,'champion':champion['candidate'],
                  'champion_cases':champion['case_metrics'],'exploration_parent':parent['candidate'],
                  'parent_cases':parent['case_metrics'],'focus':focus,'milestone_archive':archive,
                  'prompt_chars':len(prompt),'model_calls':0,'simulation_calls':0,
                  'history':[{'candidate':e['candidate'],'cases':annotate(e)['case_metrics']} for e in state['history']]}
        write(root/'loop_v2_audit.json',report)
        (root/'loop_v2_prompt_preview.txt').write_text(prompt)
        print(json.dumps(report,indent=2))
        return
    # flock is released by the OS on normal exit, interruption, or process death.
    lock = (root/'.learner.lock').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        p.error('Another learner already owns this campaign')
    with lock:
        run_campaign(args,p,root,path)


def run_campaign(args,p,root,path):
    global EVENT_DIR
    EVENT_DIR = root
    if path.exists():
        state = json.loads(path.read_text())
        state.pop('error', None)
        if state['model'] != args.model:
            p.error('Resuming with a different model requires a new campaign directory')
        if state['status'] in ('completed','heldout_failed'):
            print('Campaign frozen after held-out validation:', root, state['status'])
            return
        state.setdefault('resumes', []).append({
            'at': datetime.now(timezone.utc).isoformat(),
            'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
            'previous_status': state['status'],
            'reported_tokens_at_resume': state['reported_tokens'],
            'previous_max_reported_tokens': state.get('max_reported_tokens'),
            'round_limit': args.rounds,
            'max_reported_tokens': args.max_reported_tokens,
        })
        state.update(round_limit=args.rounds, max_reported_tokens=args.max_reported_tokens)
        write(path, state)
    else:
        shutil.copyfile(ROOT/'policies/stack_inventory_cpu.py', root/'initial_policy.py')
        state = {'task':'inventory_repair', 'created_at':datetime.now(timezone.utc).isoformat(),
                 'model':args.model, 'train_seeds':TRAIN, 'reserved_seeds':HELDOUT,
                 'cases':CASES, 'history':[], 'status':'baseline', 'reported_tokens':0,
                 'round_limit':args.rounds, 'max_reported_tokens':args.max_reported_tokens}
        write(path,state)
    # Import old trials without inventing unavailable milestone evidence.
    state['history']=[annotate(e) for e in state['history']]
    state['loop_version']=VERSION
    state['stagnation_rounds']=args.stagnation_rounds
    state.setdefault('v2_start_history_index',max(1,len(state['history'])))
    def account_usage():
        # A paid response may be rejected before becoming executable source.
        unknown=[]
        for usage_path in sorted(root.glob('round_*/usage.json')):
            index=str(int(usage_path.parent.name.split('_')[-1]))
            if index not in state.setdefault('usage_by_round',{}):
                try:
                    usage=json.loads(usage_path.read_text())
                    count=usage_total(usage)
                except (ValueError,TypeError):
                    unknown.append(index)
                    continue
                state['usage_by_round'][index]=usage
                state['reported_tokens']+=count
        state['usage_unknown_rounds']=unknown
    try:
        account_usage()
        if state['usage_unknown_rounds']:
            raise RuntimeError('Unaccounted model usage; inspect rounds '+str(state['usage_unknown_rounds']))
        if not state['history']:
            state['history'].append(evaluate(root/'initial_policy.py', root/'baseline', TRAIN))
            write(path,state)
        while True:
            best,parent,archive,focus = select_candidates(state['history'])
            state['best'] = best
            state.update(exploration_parent=parent,focus=focus,milestone_archive=archive)
            shutil.copyfile(best['candidate'],root/'.best_policy.py.tmp')
            (root/'.best_policy.py.tmp').replace(root/'best_policy.py')
            shutil.copyfile(parent['candidate'],root/'.exploration_policy.py.tmp')
            (root/'.exploration_policy.py.tmp').replace(root/'exploration_policy.py')
            write(root/'milestones.json',archive)
            write(root/'best_policy.json',{'candidate':best['candidate'],'cases':best['case_metrics'],
                  'repair_demonstrated':best['case_metrics']['offset']['passed']>0,
                  'exploration_parent':parent['candidate'],'focus':focus})
            # Never advertise an untouched centered control as a repair replay.
            successes = [r for r in best['results'] if r['success'] and r['scene']['repair_case']=='offset']
            if successes:
                (root/'best_repair_trial.txt').write_text(successes[0]['directory']+'\n')
            write(path,state)
            emit('Selected policies',phase='selection',focus=focus,cases=best['case_metrics'],parent=parent['candidate'])
            if args.baseline_only:
                state['status']='baseline_complete'; write(path,state); return
            if best['successes'] == best['trials']:
                state['status']='validating_frozen_policy'; write(path,state)
                state['heldout']=evaluate(root/'best_policy.py',root/'heldout',HELDOUT)
                state['status']='completed' if state['heldout']['successes']==state['heldout']['trials'] else 'heldout_failed'
                write(path,state); return
            index = len(state['history'])
            directory=root/f'round_{index:03d}'
            pending_proposal = directory.exists()
            new_history=state['history'][state['v2_start_history_index']:]
            if not pending_proposal and args.stagnation_rounds and len(new_history)>=args.stagnation_rounds:
                earlier=state['history'][:-args.stagnation_rounds]
                _,before,_,_=select_candidates(earlier or state['history'][:1])
                if progress_key(parent)<=progress_key(before):
                    state['status']='stalled'; write(path,state)
                    emit('Stopped: no verified progress; inspect diagnostics before more model calls',phase='stalled')
                    return
            prior_usage=[sum(u.get('input_tokens',0)+u.get('output_tokens',0) for u in usage)
                         for usage in state.get('usage_by_round',{}).values()]
            reserve=max([20000]+prior_usage[-2:])
            if not pending_proposal and (index > args.rounds or state['reported_tokens']+reserve > args.max_reported_tokens):
                state['status']='round_limit' if index > args.rounds else 'token_budget_limit'
                write(path,state); emit('Stopped: '+state['status'],phase=state['status']); return
            # An already charged proposal must retain its recorded parent when
            # resuming, even if candidate-selection logic has since changed.
            if (directory/'invocation.json').exists():
                invocation=json.loads((directory/'invocation.json').read_text())
                matches=[e for e in state['history'] if e['sha256']==invocation['parent_sha256']]
                if not matches:
                    raise RuntimeError('Saved proposal parent is missing from campaign history')
                parent=matches[-1]
            state.update(status='proposing',active_round=index); write(path,state)
            if not (directory/'candidate.py').exists():
                if directory.exists():
                    # Resume a saved response after an infrastructure/formatting
                    # repair without charging another model invocation.
                    if not (directory/'response.json').exists() or not (directory/'usage.json').exists():
                        raise RuntimeError('Incomplete proposal exists; inspect before another model call: '+str(directory))
                    invocation=json.loads((directory/'invocation.json').read_text())
                    if invocation['parent_sha256'] != digest(Path(parent['candidate'])):
                        raise RuntimeError('Saved proposal belongs to a different parent policy')
                    response=json.loads((directory/'response.json').read_text())
                    source=apply_edits(Path(parent['candidate']).read_text(),response['edits'])
                    (directory/'candidate.py').write_text(source)
                    usage=json.loads((directory/'usage.json').read_text())
                    emit('Reused saved LLM response; no additional model call',phase='reuse_proposal')
                else:
                    usage=propose(parent,state['history'],directory,args,focus)
            else:
                usage=json.loads((directory/'usage.json').read_text())
            account_usage()
            state['status']='executing'; write(path,state)
            state['history'].append(evaluate(directory/'candidate.py', directory/'trials',TRAIN))
            write(path,state)
    except BaseException as error:
        account_usage()
        state.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'error',
                     error=f'{type(error).__name__}: {error}')
        write(path,state)
        emit(state['error'],phase=state['status'])
        raise


if __name__ == '__main__':
    main()
