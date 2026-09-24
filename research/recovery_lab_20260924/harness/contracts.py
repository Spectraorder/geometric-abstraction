"""Executable DESIGN contract, not connected to the robot controller yet."""
import json
from math import isfinite

FORBIDDEN = {'qpos','qvel','body_xpos','geom_xpos','contact','contacts','segmentation_id','true_pose','success','reward','evaluator','fault_schedule'}

def reject_oracles(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FORBIDDEN: raise ValueError('Forbidden actor field: '+key)
            reject_oracles(item)
    elif isinstance(value, list):
        for item in value: reject_oracles(item)

def validate_belief(belief):
    reject_oracles(belief)
    for key in ('version','time_s','source_frames','objects'):
        if key not in belief: raise ValueError('Missing '+key)
    if not belief['source_frames']: raise ValueError('No sensor provenance')
    if not isfinite(belief['time_s']): raise ValueError('Nonfinite time')
    if not isinstance(belief['version'],int) or belief['version']<0:raise ValueError('Invalid version')
    for obj in belief['objects']:
        for key in ('id','visible','last_seen_s','association','pose_sigma_m','geometry_origin'):
            if key not in obj: raise ValueError('Missing object '+key)
        if obj['geometry_origin'] not in ('observed','fitted','propagated'):raise ValueError('Unlabeled geometry source')
        if not isfinite(obj['last_seen_s']) or obj['last_seen_s']>belief['time_s']:raise ValueError('Invalid observation time')
        if not isfinite(obj['pose_sigma_m']) or obj['pose_sigma_m']<0:raise ValueError('Invalid uncertainty')
    ids=[o['id'] for o in belief['objects']]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate instance ID')
    return belief

def route(belief, target_id, *, retries=0, reobservations=0):
    """Illustrative deterministic fast path; thresholds await calibration."""
    validate_belief(belief)
    target=next((o for o in belief['objects'] if o['id']==target_id),None)
    uncertain=(target is None or not target['visible'] or target['association']!='unique'
        or belief['time_s']-target['last_seen_s']>.5 or target['pose_sigma_m']>.004)
    if uncertain:
        return 'reobserve' if reobservations<2 else 'escalate'
    if target.get('grasp_confirmed') is False:
        return 'retry_grasp' if retries<1 else 'escalate'
    margin=target.get('support_margin_lower_m')
    if margin is None or not isfinite(margin):return 'reobserve' if reobservations<2 else 'escalate'
    if margin<.004:return 'escalate'
    return 'continue'

def validate_action(action, belief, now):
    """Reject stale/malformed slow-agent proposals before any executor sees them."""
    validate_belief(belief);reject_oracles(action)
    required={'kind','belief_version','expires_at_s','target_id','reason'}
    allowed=required|{'evidence_frames'}
    if not required<=action.keys() or set(action)-allowed:raise ValueError('Action schema mismatch')
    if action['kind'] not in {'reobserve','retry_grasp','continue','escalate','stop'}:raise ValueError('Unimplemented action')
    if action['belief_version']!=belief['version']:raise ValueError('Stale belief version')
    if not isfinite(now) or not isfinite(action['expires_at_s']) or now>action['expires_at_s']:raise ValueError('Expired action')
    if now<belief['time_s'] or now-belief['time_s']>.5:raise ValueError('Belief needs fresh observation')
    if action['kind'] in {'retry_grasp','continue'}:
        recommendation=route(belief,action['target_id'])
        if recommendation!=action['kind']:raise ValueError('Action violates perception precondition')
    return action

def propose_memory(record):
    """Design promotion gate: validation/heldout labels never enter retrieval memory."""
    if record.get('split')!='development':raise ValueError('Only development evidence enters repair memory')
    if not record.get('policy_hash') or not record.get('regression_suite_passed'):raise ValueError('Repair unvalidated')
    if not record.get('applicability') or 'counterexamples' not in record:raise ValueError('Missing applicability or negative evidence')
    return {k:record[k] for k in ('policy_hash','applicability','counterexamples','split')}
