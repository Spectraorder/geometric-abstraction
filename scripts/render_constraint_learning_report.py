"""Render complete constraint-feedback evidence without selecting test winners."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics

VARIANT = 'partial_constraints_all_training_checkpoints_v1'


def percent(value):
    return f'{100*value:.2f}%'


def signed_points(value):
    return f'{100*value:+.2f}'


def relative_link(path, output):
    return Path(os.path.relpath(path, output.parent)).as_posix()


def render(audit_path, output, figure):
    report=json.loads(audit_path.read_text())
    if (report.get('status')!='completed' or report.get('study_freeze_audit')!='passed'
            or report.get('experiment_variant')!=VARIANT):
        raise ValueError('Require a complete globally frozen audit of this experiment variant')
    figure_meta=figure.with_suffix('.json')
    image_receipt=json.loads(figure_meta.read_text())
    audit_hash=hashlib.sha256(audit_path.read_bytes()).hexdigest()
    if image_receipt['audit_sha256']!=audit_hash or image_receipt['figure_sha256']!=hashlib.sha256(figure.read_bytes()).hexdigest():
        raise ValueError('Figure does not match the supplied complete audit')
    replicas=report['replicas']; chains=len(replicas)
    train_counts={}; reserved_counts={}; selected={arm:[] for arm in ('basic','symbolic')}
    means={arm:[] for arm in selected}; train_lines=[]; test_lines=[]; selection_lines=[]; threshold_lines=[]
    for name,result in replicas.items():
        curves=result['curves']; label=name.replace('replica_','')
        tr=[]; te=[]; sel=[]; thresholds=[]
        for arm in selected:
            curve=curves[arm]; train=curve['training']; test=curve['reserved']
            if any(r['slots_completed']!=r['total_slots'] for r in train):
                raise ValueError('Incomplete training row')
            if any(abs(r['sr']-r['successes']/r['trials'])>1e-12 for r in test):
                raise ValueError('Reserved SR does not match its denominator')
            train_counts[name]=sorted({r['total_slots'] for r in train})
            reserved_counts[name]=sorted({r['trials'] for r in test})
            tr.append(' → '.join(f"{r['successes']}/{r['total_slots']}" for r in train))
            te.append(' → '.join(f"{100*r['sr']:.0f}%" for r in test))
            chosen=curve['selected_revision']
            expected=max(train,key=lambda r:(r['successes']/r['total_slots'],-r['revision']))['revision']
            if chosen!=expected:
                raise ValueError('Selection differs from training SR with earliest tie breaking')
            selected[arm].append(curve['selected_reserved']['sr'])
            means[arm].append(curve['mean_reserved_sr_over_revisions'])
            sel.append((chosen,curve['selected_reserved']['sr'],curve['mean_reserved_sr_over_revisions']))
            first=curve['first_reserved_revision_at_80_percent']
            thresholds.append('Not reached' if first is None else 'R'+str(first))
        train_lines.append(f'| {label} | {tr[0]} | {tr[1]} |')
        test_lines.append(f'| {label} | {te[0]} | {te[1]} |')
        selection_lines.append(f'| {label} | R{sel[0][0]} / R{sel[1][0]} | {percent(sel[0][1])} / {percent(sel[1][1])} | {percent(sel[0][2])} / {percent(sel[1][2])} |')
        threshold_lines.append(f'| {label} | {thresholds[0]} | {thresholds[1]} |')
    selected_diffs=[b-a for a,b in zip(selected['basic'],selected['symbolic'])]
    curve_diffs=[b-a for a,b in zip(means['basic'],means['symbolic'])]
    training_total=sum(r['total_slots'] for replica in replicas.values() for curve in replica['curves'].values() for r in curve['training'])
    reserved_total=sum(r['trials'] for replica in replicas.values() for curve in replica['curves'].values() for r in curve['reserved'])
    if training_total+reserved_total!=report['valid_rollouts']+report['invalid_slots']:
        raise ValueError('Total slot accounting differs')
    contexts=sum(r['exact_actor_contexts_verified'] for r in replicas.values())
    interrupts=report.get('interrupted_actor_attempts',0)
    attempts=len(report['model_invocation_attempts'])
    text=f'''# Visual Base versus full target geometry: constraint-feedback study

The completed study measures training-selected reserved success of
**{percent(statistics.mean(selected['basic']))} for Visual Base** and
**{percent(statistics.mean(selected['symbolic']))} for Full geometry**, averaged
across {chains} chain pairs. The paired differences are
{', '.join(signed_points(v) for v in selected_diffs)} percentage points.
These descriptive results must be read alongside every learning curve below;
they are not a pooled significance test or evidence for symbolic operators alone.

The independent audit verifies **{report['valid_rollouts']} valid rollouts and
{report['invalid_slots']} invalid slots**, covering {training_total} training and
{reserved_total} reserved slots. There are {report['completed_actor_generations']}
completed model generations from {attempts} invocation attempts, including
{interrupts} archived pre-response quota interruptions. All {contexts} submitted
training contexts are verified. Every candidate version in every chain froze
before any reserved evaluation; no test result selected or repaired a candidate.

## Every registered revision

R0 is the common visual starter; R1–R3 are model-generated revisions. Training
layouts are disjoint across chains. Reserved layouts are shared across chains
and revisions, so repeated episodes are not independent actor replications.

| Chain | Base training R0 → R3 | Full training R0 → R3 |
| --- | --- | --- |
{chr(10).join(train_lines)}

| Chain | Base reserved R0 → R3 | Full reserved R0 → R3 |
| --- | --- | --- |
{chr(10).join(test_lines)}

![All registered learning curves]({relative_link(figure,output)})

Open circles mark versions selected by training SR. Every regression remains
visible. No test-best version replaces the registered training selection.

## Training selection and iteration outcomes

Selection maximizes training SR and keeps the earliest tied revision.

| Chain | Selected Base / Full | Selected reserved Base / Full | Mean reserved R0–R3 Base / Full |
| --- | --- | --- | --- |
{chr(10).join(selection_lines)}

The paired training-selected difference averages
**{signed_points(statistics.mean(selected_diffs))} percentage points**, ranging
from {signed_points(min(selected_diffs))} to {signed_points(max(selected_diffs))}.
Mean reserved SR over all revisions averages
{percent(statistics.mean(means['basic']))} for Base and
{percent(statistics.mean(means['symbolic']))} for Full. The paired mean-curve
differences are {', '.join(signed_points(v) for v in curve_diffs)} points,
with mean {signed_points(statistics.mean(curve_diffs))} and range
{signed_points(min(curve_diffs))} to {signed_points(max(curve_diffs))}.

| Chain | First Base revision at 80% reserved SR | First Full revision at 80% reserved SR |
| --- | --- | --- |
{chr(10).join(threshold_lines)}

These are the registered iteration summaries. The retained training-best curve
is secondary and remains in the complete audit. Neither repeated revisions nor
common test layouts increase the number of independent learning chains. A claim
of repeatable faster improvement must account for all chain pairs and the small
number of independent replications; a favorable single chain is insufficient.

## What the treatment includes

Both arms receive raw side/wrist RGB-D and calibration, public object/task labels,
robot proprioception and CAD, and the same hidden visual PnP actuator. Base has no
supplied object center, dimensions, vertices, or goal residuals through its public
session interface; its policy may compute visual estimates itself. Qualitative
actuator/perception failure messages remain visible to both arms. This is not
a pixels-to-torques or perception-free Base comparison. Full additionally
receives the CPU geometry package and compact active-target feedback. A failed
unique cuboid correspondence can retain current center constraints without granting
an executable point pose. The last two checkpoints from every training layout are
eligible under the same bounded feedback-selection rule in both arms.

Runtime policy inputs exclude simulator object poses/dimensions, contacts and
evaluator state. Final binary success is allowed as training feedback and offline
scoring. Geometry and policy computation run on CPU; simulation may use GPU.
The task is a low-rise bridge with two cube foundations, a flat thin prism and a
cylinder, with varying layouts and unique colors. This is not demonstrated transfer
to new task families, arbitrary shapes, same-color recognition, or real sensors.

The treatment is supplied perception plus geometric abstraction, not an isolated
symbolic-versus-numeric comparison. Partial constraints and full-layout feedback
are part of the package; their individual contributions are not identified here.
Character counts and separate CPU replay timings do not prove reduced agent
latency or end-to-end realtime performance.

## Development work and evidence separation

Conditional point/region memory and explicit opening/reacquisition were tested in
separate development replays. They are not components of this frozen comparison,
and their human-authored repairs never enter actor feedback. Preserve previous
completed comparisons, including negative or mixed results, as separate studies.

- [Protocol](PARTIAL_GEOMETRY_LEARNING_PROTOCOL.md)
- [Complete independent audit]({relative_link(audit_path,output)})
- [Previous visual-input study](VISUAL_INPUT_LEARNING_RESULTS.md)
- [Conditional memory evidence](CONDITIONAL_ATTACHMENT_EVIDENCE.md)
- [Explicit opening development](EXPLICIT_OPEN_GRIPPER_PROTOTYPE.md)

This document is generated from audited results. Its adjacent JSON records source
and figure hashes. Interpretation of any benefit requires the complete curves and
limitations above; the renderer does not declare the research objective achieved.
'''
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(text)
    metadata={'scope':__doc__,'audit_sha256':audit_hash,
              'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'report_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
              'figure_sha256':image_receipt['figure_sha256'],
              'training_layout_counts_by_chain':train_counts,
              'reserved_layout_counts_by_chain':reserved_counts}
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--figure',type=Path,required=True)
    args=parser.parse_args();render(args.audit.resolve(),args.output.resolve(),args.figure.resolve())
