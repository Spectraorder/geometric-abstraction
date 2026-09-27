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

ROOT = parent.ROOT
base = parent.base
ARMS = parent.ARMS
PARENT_LIBRARY = parent.library
PARENT_SOURCES = parent.sources
PARENT_CONTEXT = parent.context
VERSION = 'partial_constraints_all_training_checkpoints_v1'
EXTRA = ('policies/partial_geometry_state.py', 'scripts/build_partial_geometry_session.py',
         'scripts/run_constraint_geometry_experiment.py')
COMMON = parent.COMMON.replace(
    'Each selected training example supplies at most the last eight checkpoint records,\nwithin 16000 characters, alongside the same initial/final image selection in both arms.',
    'Every training layout supplies its latest one or two target checkpoints, within\na shared 16000-character cap. Images still use the same two-example selection rule.')
COMMON += '''\nTarget feedback covers every training layout, independently of image selection.
At most two final checkpoints per layout are provided from its latest execution.
If the combined cap is exceeded, use only the final checkpoint of every layout.
Absent checkpoints are explicitly marked. No episode is omitted based on outcome.
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


def context(root, arm, revision, manifest):
    # Preserve image selection and outcome history, but do not tie text coverage
    # to the two image examples. Scoped hooks never affect another process.
    with patch.multiple(parent, COMMON=COMMON, RICH=RICH), patch.object(parent.step, 'compact_checkpoints', lambda path: []):
        prompt, images = PARENT_CONTEXT(root, arm, revision, manifest)
    marker = '\nReturn the next planner. No manual policy fixes will be inserted.'
    if not prompt.endswith(marker): raise ValueError('Unexpected inherited prompt boundary')
    feedback = target_feedback(root, arm, revision, manifest['train_seeds'])
    prompt = prompt[:-len(marker)] + '\nOwn latest target checkpoints for every training layout:\n' + json.dumps(feedback, separators=(',', ':')) + marker
    if len(prompt) > manifest['max_prompt_characters']: raise RuntimeError('Prompt budget exceeded')
    return prompt, images


@contextmanager
def installed():
    with patch.multiple(parent, library=library, sources=sources, context=context, COMMON=COMMON, RICH=RICH):
        yield


@contextmanager
def configured():
    with installed(), parent.configured(): yield


def prepare(root, train, reserved, rounds=3):
    with installed(): manifest = parent.prepare(root, train, reserved, rounds)
    manifest.update(variant=VERSION,
        feedback='All training layouts: latest two target checkpoints, reduce to latest one each if needed for 16000 characters. Same two image examples and raw sensors in both arms.',
        partial_constraints='Unique current-view rectangular-prism center bounds remain separate from unavailable point poses; no controller acceptance change.',
        contrasts=['Visual Base versus supplied full geometry package, including target-only partial constraints. Identical hidden RGB-D actuator. Not a pure symbolic or partial-constraint ablation.'])
    base.write(root/'manifest.json', manifest)
    return manifest


def check(root, manifest):
    if manifest.get('variant') != VERSION: raise ValueError('Wrong constraint-feedback variant')
    with installed(): parent.check(root, manifest)


def bootstrap(root, manifest):
    check(root, manifest)
    with installed(): parent.bootstrap(root, manifest)


def audit(root, allow_partial=False):
    from summarize_visual_input_experiment import audit as audit_parent
    manifest = json.loads((root/'manifest.json').read_text()); check(root, manifest)
    with installed(): report = audit_parent(root, allow_partial)
    report.update(experiment_variant=VERSION, variant_auditor_sha256=base.digest(Path(__file__)))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--train-seeds',nargs='+',type=int);parser.add_argument('--reserved-seeds',nargs='+',type=int)
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--phase',choices=('prepare','bootstrap','train','evaluate','audit'),default='prepare')
    args=parser.parse_args();root=args.output.resolve()
    if not args.resume and (not args.train_seeds or not args.reserved_seeds):parser.error('Explicit seeds required')
    manifest=json.loads((root/'manifest.json').read_text()) if args.resume else prepare(root,args.train_seeds,args.reserved_seeds,args.rounds)
    check(root,manifest)
    if args.phase=='bootstrap':bootstrap(root,manifest)
    elif args.phase in ('train','evaluate'):
        if not (root/'starter_parity.json').exists():bootstrap(root,manifest)
        with configured():base.campaign(root,manifest,args.phase)
    if args.phase=='audit':
        report=audit(root,allow_partial=True);base.write(root/'audit.json',report)
        print({k:report[k] for k in ('status','valid_rollouts','invalid_slots','exact_actor_contexts_verified')})
