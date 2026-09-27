"""Visual Base versus compact geometry with partial constraints and complete coverage.

This is a new experiment variant. Historical studies and actor contexts are never
rewritten. Both arms share the same actuator; only supplied target evidence differs.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from unittest.mock import patch
import run_visual_input_experiment as parent
from build_partial_geometry_session import library as extend
from geometry_actor_session import propose as propose_actor
from geometry_iteration import (RULE, retention, training_curve, stage as revision_stage, event_feedback,
                                failure_kind, failure_images, behavior_feedback)

ROOT = parent.ROOT
base = parent.base
ARMS = parent.ARMS
PARENT_LIBRARY = parent.library
PARENT_SOURCES = parent.sources
VERSION = 'diagnostic_geometry_actor_sessions_v4'
EXTRA = ('policies/partial_geometry_state.py', 'scripts/build_partial_geometry_session.py',
         'scripts/run_constraint_geometry_experiment.py', 'scripts/geometry_actor_session.py',
         'scripts/summarize_mixed_scene_learning.py', 'scripts/geometry_iteration.py',
         'scripts/summarize_visual_input_experiment.py')
COMMON = parent.COMMON.replace(
    'Each selected training example supplies at most the last eight checkpoint records,\nwithin 16000 characters, alongside the same initial/final image selection in both arms.',
    'Every training layout supplies its latest one or two target checkpoints, within\na shared 16000-character cap. Images still use the same two-example selection rule.')
COMMON += '''
Revise the selected incumbent, not a rejected candidate. Choose one falsifiable
failure hypothesis and identify its affected task phase and functions. You may
change your own perception, worksite, action order or recovery when the evidence
supports it; these are not frozen. Avoid unrelated refactors, but do not preserve
an ineffective action sequence merely to keep the patch small. Keep the shared
actuator and all documented safety/observation checks unchanged.
Return the complete source. Acceptance still requires preserving every incumbent
training success and repairing at least one failure; ties retain the incumbent.
For the proposed branch, emit api.record_geometry with kind='policy_probe',
hypothesis (short text), step, object (when relevant), triggered (boolean), reason,
and concise sensor-derived evidence. Record both activation and nonactivation.
Record the relevant before/after error if your permitted observations establish
it; otherwise explicitly mark it unknown. Do not hardcode training seed identities.
The host compares issued motion/gripper/wait commands against the proposal parent.
If they are identical on the targeted failed scene, the intended action change was
not exercised; do not count extra diagnostic code as a successful intervention.
Own custom diagnostics return as unverified policy reports, not supplied truth.
After a release, an unavailable follow-up fit is reported separately; released is
still not proof of stable placement or whole-task success. Check fresh evidence.
Feedback covers every training layout: first explicit failure, nearby same-object
checkpoints, and a same-arm successful reference when available within 16000 chars.
Unavailable geometry is missing evidence, not proof of task failure or object loss.
After waiting or moving, Full should explicitly refresh the target before a settled
checkpoint; Base should inspect fresh RGB-D and record its own measured diagnostics.
For object/tool mismatch, do not relax the executor's 0.12 m association check or
assume held=True proves a grasp. For missing current views, do not turn stored poses
or partial bounds into measurements. Prefer observing objects before occlusion.
If recovery is justified, acquire fresh permitted sensor evidence with a bounded
retry (at most one); never blindly repeat a failed transport or reset held state.
If current evidence remains insufficient, checkpoint the failed action with
exception status, note the unresolved cause, and stop the rollout. Catch observation
errors for diagnostics without claiming task success. A changed transfer sequence
requires an explicit hypothesis and runtime evidence. Camera images are historical
training evidence only.
'''
RICH = parent.RICH + '''\nWhen a current rectangular-prism fit has unique correspondence but no accepted
pose, checkpoint may retain constraints while status stays unavailable. Rows of
constraints.basis_world project a world center: bounds_m[:,0] <= basis_world @
center <= bounds_m[:,1]. face_constrained_axes indicates which directions have
current face-plane evidence. visual_dimensions_m comes from depth fitting;
dimension_refinements counts model updates. These bounds depend on correct visual
association, fitted orientation and measured dimensions, not calibrated confidence.
pose_available=False: no center, grasp, contact, or stable placement is certified.
The unchanged executor cannot use this packet as a point pose. New camera evidence
may resolve uncertainty; a stored or predicted point must not become an observation.
Constraints are only for the same target and timestamp, not a full-scene history.
'''


def library(arm):
    return extend(PARENT_LIBRARY(arm))


def sources():
    result = PARENT_SOURCES()
    result.update({name: base.digest(ROOT/name) for name in EXTRA})
    return result


def target_feedback(root, arm, revision, seeds, budget=16000):
    rows = []
    for seed in seeds:
        selected = None
        # Match the established image selector's latest executed revision rule.
        for previous in range(revision-1, -1, -1):
            stage = 'revision_000' if previous == 0 else f'round_{previous:03d}'
            trial = root/arm/stage/'trials'/f'seed_{seed}'
            if (trial/'api_calls.json').exists():
                selected = (previous, trial); break
        records = []
        if selected is not None:
            path = selected[1]/'geometry_records.json'
            if path.exists():
                records = [r['record'] for r in json.loads(path.read_text()) if r.get('kind') == 'geometry_step'][-2:]
        rows.append({'seed': seed, 'revision': None if selected is None else selected[0],
                     'status': 'recorded' if records else 'no_target_checkpoint', 'checkpoints': records})
    size = lambda: len(json.dumps(rows, separators=(',', ':')))
    if size() > budget:
        for row in rows: row['checkpoints'] = row['checkpoints'][-1:]
    if size() > budget:
        raise RuntimeError('Latest target records for every training layout exceed the feedback budget')
    return rows


def image_feedback(root, arm, revision, manifest):
    if manifest['actor_images'] == 'on_failure' and revision > 1:
        for seed in manifest['train_seeds']:
            trial = revision_stage(root, arm, revision-1) / 'trials' / f'seed_{seed}'
            feedback = base.feedback(trial)
            kind = failure_kind(feedback)
            if kind not in ('object_tool_mismatch', 'insufficient_current_view', 'unexplained_failure'):
                continue
            images, metadata = failure_images(trial)
            if images:
                return ('\nSupplemental training images: ' + json.dumps({
                    'seed': seed, 'revision': revision-1, 'trigger': kind,
                    'attachment_cap': 2, 'frames': metadata}) + '\n'), images
        return '\nNo eligible unresolved visual failure; no supplemental images.\n', []
    if manifest['actor_images'] == 'initial_only' and revision > 1:
        return '\nNo new images are attached to this revision.\n', []
    examples = []
    for seed in manifest['train_seeds']:
        for previous in range(revision - 1, -1, -1):
            stage = 'revision_000' if previous == 0 else f'round_{previous:03d}'
            trial = root / arm / stage / 'trials' / f'seed_{seed}'
            if (trial / 'api_calls.json').exists():
                examples.append((seed, previous, trial, base.feedback(trial)['success']))
                break
    chosen = []
    for success in (False, True):
        eligible = [row for row in examples if row[3] == success]
        if eligible:
            chosen.append(eligible[0])
    for row in examples:
        if len(chosen) == 2:
            break
        if row not in chosen:
            chosen.append(row)
    text, images = '', []
    for seed, previous, trial, success in chosen:
        attached, metadata = base.sensor_context(trial)
        images.extend(attached)
        text += ('\nTraining images: seed=' + str(seed) + ', revision=' + str(previous)
                 + ', success=' + str(success) + '. First/last side RGB then depth, calibration:\n'
                 + json.dumps(metadata))
    return text, images


def context(root, arm, revision, manifest):
    persistent = manifest['actor_context_mode'] == 'persistent'
    guarded = manifest.get('retention_rule') == RULE
    if arm not in manifest['arms'] or not 1 <= revision <= manifest['rounds']:
        raise ValueError('Invalid actor context request')
    if not persistent or revision == 1:
        common = COMMON.replace('Images still use the same two-example selection rule.',
                                'Images follow the registered actor image schedule.')
        common = common.replace(
            'Initial\nand final SIDE RGB/depth for two training trials are attached, preferring one failure\nand one success; same selection rule in every condition.',
            'When the image schedule permits, initial/final SIDE RGB/depth for at most two\ntraining trials are attached, preferring one failure and one success.')
        common = common.replace('Every training layout supplies its latest one or two target checkpoints, within\na shared 16000-character cap. Images follow the registered actor image schedule.', 'Every training layout supplies event-centered evidence within a shared 16000-character cap. Images follow the registered actor image schedule.')
        prompt = common + (parent.BASIC if arm == 'basic' else RICH)
        prompt += ('\nActor context mode: ' + manifest['actor_context_mode']
                   + '. Image schedule: ' + manifest['actor_images'] + '.\n'
                   'Historical images, positions and inferred relations are not current observations.\n'
                   'Include useful visual observations and unresolved uncertainty concisely in your hypothesis.\n')
        if persistent:
            prompt += ('This is the first turn of your own persistent learning session. Later turns\n'
                       'append new training feedback; do not expect the full contract or policy to be resent.\n')
        prompt += '\nFrozen common motion loop:\n' + (root / 'executor.py').read_text()
        prompt += '\nIdentical starter planner:\n' + (root / 'starter.py').read_text()
    else:
        previous = root / arm / f'round_{revision - 1:03d}'
        prompt = (f'Continue your own learning session: revision {revision}.\n'
                  f'The latest submitted policy is revision {revision - 1}, SHA256='
                  + base.digest(previous / 'candidate.py') + '.\n'
                  'Use the existing interface and code budget. Return a complete revised policy.\n'
                  'Earlier visual evidence is historical, not a current observation.\n')
    history = [revision - 1] if persistent else range(revision)
    for previous in history:
        stage = 'revision_000' if previous == 0 else f'round_{previous:03d}'
        directory = root / arm / stage
        rows = [dict(seed=seed, **base.feedback(directory / 'trials' / f'seed_{seed}'))
                for seed in manifest['train_seeds']]
        prompt += '\nOwn revision ' + str(previous) + ' feedback:\n' + json.dumps(rows)
        if previous and not persistent:
            response = json.loads((directory / 'response.json').read_text())
            prompt += '\nOwn hypothesis: ' + response['hypothesis'][:2000]
    if not persistent and revision > 1:
        prompt += '\nLatest exact planner:\n' + (root / arm / f'round_{revision - 1:03d}' / 'candidate.py').read_text()
    if guarded:
        decisions = retention(training_curve(root, arm, revision, manifest['train_seeds']))
        incumbent = decisions[-1]['selected_revision']
        path = root/'starter.py' if incumbent == 0 else revision_stage(root, arm, incumbent)/'candidate.py'
        prompt += '\nTraining-only retention decisions:\n' + json.dumps(decisions)
        prompt += f'\nParent to revise: R{incumbent}, SHA256={base.digest(path)}.\n'
        if incumbent != revision-1:
            prompt += '\nLatest proposal was rejected; restore this exact incumbent:\n' + path.read_text()
        elif not persistent and revision > 1:
            prompt += '\nRevise the supplied incumbent.\n'
    image_text, images = image_feedback(root, arm, revision, manifest)
    prompt += image_text
    feedback = (event_feedback if guarded else target_feedback)(root, arm, revision, manifest['train_seeds'])
    prompt += '\nOwn latest target checkpoints for every training layout:\n' + json.dumps(feedback, separators=(',', ':'))
    prompt += '\nReturn the next planner. No manual policy fixes will be inserted.'
    if len(prompt) > manifest['max_prompt_characters']:
        raise RuntimeError('Prompt budget exceeded')
    return prompt, images


@contextmanager
def installed():
    with patch.multiple(parent, library=library, sources=sources, context=context, COMMON=COMMON, RICH=RICH), \
            patch.object(base, 'propose', propose_actor):
        yield


@contextmanager
def configured():
    with installed(), parent.configured(): yield


def prepare(root, train, reserved, rounds=3, actor_context_mode='persistent', actor_images='on_failure', evaluation_workers=4):
    if actor_context_mode not in ('fresh', 'persistent') or actor_images not in ('initial_only', 'every_revision', 'on_failure'):
        raise ValueError('Unsupported actor context or image schedule')
    if evaluation_workers not in (1, 2, 4, 8):
        raise ValueError('Evaluation workers must be 1, 2, 4 or 8')
    with installed(): manifest = parent.prepare(root, train, reserved, rounds)
    manifest.update(hypothesis_feedback=True, evaluation_workers=evaluation_workers, retention_rule=RULE, selection='Preserve all incumbent training successes and strictly improve; ties retain incumbent.',
        supplemental_images={'trigger': ['object_tool_mismatch', 'insufficient_current_view', 'unexplained_failure'], 'max_layouts': 1, 'max_images': 2, 'views': ['side', 'wrist'], 'extra_model_calls': 0},
        variant=VERSION, actor_context_mode=actor_context_mode, actor_images=actor_images,
        feedback='All training layouts: event-centered geometry, bounded own-policy diagnostics, parent command comparisons and successful references within 16000 characters. Same image trigger and cap in both arms.',
        partial_constraints='Unique current-view rectangular-prism center bounds remain separate from unavailable point poses; no controller acceptance change.',
        contrasts=['Visual Base versus supplied full geometry package, including target-only partial constraints. Identical hidden RGB-D actuator. Not a pure symbolic or partial-constraint ablation.'])
    manifest['retained_budget_secondary'] = 'Guarded incumbent at each proposal budget; rejected proposals remain in the raw curve.'
    manifest['limitations'].append('Persistent history can grow or be compacted; fewer new image attachments do not guarantee fewer billed tokens or lower latency.')
    base.write(root/'manifest.json', manifest)
    return manifest


def check(root, manifest):
    if manifest.get('variant') != VERSION: raise ValueError('Wrong constraint-feedback variant')
    if manifest.get('actor_context_mode') not in ('fresh', 'persistent') or manifest.get('actor_images') not in ('initial_only', 'every_revision', 'on_failure'):
        raise ValueError('Unregistered actor context or image schedule')
    with installed(): parent.check(root, manifest)


def bootstrap(root, manifest):
    check(root, manifest)
    with installed(): parent.bootstrap(root, manifest)


def audit(root, allow_partial=False):
    from summarize_visual_input_experiment import audit as audit_parent
    manifest = json.loads((root/'manifest.json').read_text()); check(root, manifest)
    with installed(): report = audit_parent(root, allow_partial)
    reviews = 0
    if manifest.get('hypothesis_feedback'):
        for arm in manifest['arms']:
            for revision in range(1, manifest['rounds']+1):
                directory = revision_stage(root, arm, revision)
                if not all((directory/'trials'/f'seed_{seed}'/'result.json').exists() for seed in manifest['train_seeds']):
                    continue
                expected = {'revision': revision, 'command_comparisons': behavior_feedback(
                    root, arm, revision, manifest['train_seeds'])}
                actual = json.loads((directory/'hypothesis_review.json').read_text())
                if actual != json.loads(json.dumps(expected)):
                    raise ValueError('Hypothesis command comparison differs from recorded training calls')
                reviews += 1
    report.update(experiment_variant=VERSION, hypothesis_reviews_verified=reviews,
                  variant_auditor_sha256=base.digest(Path(__file__)))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--train-seeds',nargs='+',type=int);parser.add_argument('--reserved-seeds',nargs='+',type=int)
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--eval-workers', type=int, choices=(1, 2, 4, 8), default=None)
    parser.add_argument('--actor-context', choices=('fresh', 'persistent'), default=None)
    parser.add_argument('--actor-images', choices=('initial_only', 'every_revision', 'on_failure'), default=None)
    parser.add_argument('--phase',choices=('prepare','bootstrap','train','evaluate','audit'),default='prepare')
    args=parser.parse_args();root=args.output.resolve()
    if not args.resume and (not args.train_seeds or not args.reserved_seeds):parser.error('Explicit seeds required')
    if args.resume and (args.actor_context is not None or args.actor_images is not None or args.eval_workers is not None):
        parser.error('Actor settings are frozen; set them only during preparation')
    manifest=json.loads((root/'manifest.json').read_text()) if args.resume else prepare(
        root,args.train_seeds,args.reserved_seeds,args.rounds,
        args.actor_context or 'persistent', args.actor_images or 'on_failure', args.eval_workers or 4)
    check(root,manifest)
    if args.phase=='bootstrap':bootstrap(root,manifest)
    elif args.phase in ('train','evaluate'):
        if not (root/'starter_parity.json').exists():bootstrap(root,manifest)
        with configured():base.campaign(root,manifest,args.phase)
    if args.phase=='audit':
        report=audit(root,allow_partial=True);base.write(root/'audit.json',report)
        print({k:report[k] for k in ('status','valid_rollouts','invalid_slots','exact_actor_contexts_verified')})
