"""Evaluate all frozen three-cube revisions on 20 layouts under a budget amendment."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import time
from unittest.mock import patch

import resume_three_cube_geometry_study as repair
import summarize_geometry_session_resources as resources
from verified_token_usage import turn_usage
from experiment_parallel import bounded_map, lease, stop, check_stop, write_json, audit_leases
from summarize_mixed_scene_learning import require, paired

runner = repair.runner
base = runner.base
read = runner.read
curves = runner.planned_curves.curves
DIRECTORY = 'compact_evaluation_20'


@contextmanager
def installed():
    with repair.installed(), patch.object(resources, 'turn_usage', turn_usage):
        yield


def prepare(root):
    output = root/DIRECTORY
    require(not output.exists(), 'Compact evaluation already registered')
    with runner.study.exclusive(root/'pipeline.lock'), runner.study.exclusive(root/'coordinator.lock'):
        repair.check(root)
        protocol = runner.study.check(root)
        require(not list(root.glob('replica_*/selected_evaluation/*/heldout/seed_*/result.json')),
                'Budget amendment must precede reserved outcomes')
        require(not list(root.glob('replica_*/curve_rollouts/*/revision_*/seed_*/result.json')),
                'Curve evaluation already started')
        seeds = protocol['heldout_seeds'][:20]
        require(len(seeds) == 20 and len(set(seeds)) == 20, 'Expected 20 shared reserved layouts')
        reports = {}; accounting = {}; records = []; preserved = {'protocol.json': base.digest(root/'protocol.json')}
        for row in protocol['replicas']:
            name = row['directory']; directory = root/name
            report = runner.variant.audit(directory, allow_partial=True)
            require(report['exact_actor_contexts_verified'] == 6 and report['valid_rollouts']+report['invalid_slots'] == 40,
                    'Training is incomplete')
            reports[name] = report
            accounting[name] = resources.summarize(directory, 3)
            trained = read(directory/'training_summary.json')
            for file in ('manifest.json', 'training_summary.json', 'all_candidates_frozen.json'):
                preserved[name+'/'+file] = base.digest(directory/file)
            for arm in ('basic', 'symbolic'):
                for entry in trained['revision_frozen'][arm]:
                    revision = entry['revision']
                    candidate = directory/arm/'frozen_revisions'/f'revision_{revision:03d}.py'
                    original = directory/arm/('revision_000' if revision == 0 else f'round_{revision:03d}')/'bundled.py'
                    require(base.digest(candidate) == entry['sha256'] == base.digest(original), 'Policy freeze mismatch')
                    preserved[str(candidate.relative_to(root))] = entry['sha256']
                    for seed in seeds:
                        trial = directory/'compact_rollouts'/arm/f'revision_{revision:03d}'/f'seed_{seed}'
                        records.append({'replica': name, 'arm': arm, 'revision': revision, 'seed': seed,
                                        'candidate': str(candidate.relative_to(root)), 'candidate_sha256': entry['sha256'],
                                        'validation_error': entry.get('validation_error'), 'trial': str(trial.relative_to(root))})
        require(len(records) == 800, 'Compact budget mismatch')
        output.mkdir()
        for name in reports:
            write_json(output/'training'/name/'audit.json', reports[name])
            write_json(output/'training'/name/'resource_accounting.json', accounting[name])
            for file in ('audit.json', 'resource_accounting.json'):
                relative = str((output/'training'/name/file).relative_to(root))
                preserved[relative] = base.digest(root/relative)
        sources = [Path(__file__), Path(repair.__file__), Path(resources.__file__),
                   Path(__file__).with_name('verified_token_usage.py')]
        for source in sources:
            shutil.copyfile(source, output/source.name)
        registration = {'schema': 'compact_three_cube_curves_v1', 'registered_at': time.time(),
            'authorization': 'User requested reducing single-policy stacking cost while developing the online study.',
            'selection': 'First 20 reserved seeds in the original order, before any reserved outcome; all revisions and replicas.',
            'scope': 'Budget-amended descriptive study; original 100-layout protocol retained as provenance.',
            'heldout_seeds': seeds, 'slots': 800, 'original_slots': 4000, 'model_calls': 0,
            'max_workers': 24, 'minimum_free_disk_bytes': 15*1024**3,
            'source_sha256': {str(p.relative_to(runner.ROOT)): base.digest(p) for p in sources},
            'preserved_files': preserved, 'records': records}
        write_json(output/'registration.json', registration)
        write_json(root/'evaluation_budget_amendment.json', {'registration': DIRECTORY+'/registration.json',
            'sha256': base.digest(output/'registration.json'), 'new_test_slots': 800,
            'original_pipeline_status': 'Stopped; do not resume its superseded 4000-slot evaluation.'})
        write_json(output/'progress.json', {'status': 'registered', 'completed': 0, 'total': 800})
        return registration


def check(root):
    output = root/DIRECTORY
    registration = read(output/'registration.json')
    require(registration['schema'] == 'compact_three_cube_curves_v1' and registration['slots'] == 800,
            'Wrong compact registration')
    require(base.digest(output/'registration.json') == read(root/'evaluation_budget_amendment.json')['sha256'],
            'Budget registration changed')
    repair.check(root)
    for relative, digest in registration['source_sha256'].items():
        require(base.digest(runner.ROOT/relative) == digest == base.digest(output/Path(relative).name), 'Compact runner changed')
    for relative, digest in registration['preserved_files'].items():
        require(base.digest(root/relative) == digest, 'Frozen compact evidence changed: '+relative)
    return registration


def audit(root):
    registration = check(root); output = root/DIRECTORY
    rows = {}; hashes = {}; initial = {}
    for row in registration['records']:
        trial = root/row['trial']
        fingerprint = base.digest(trial/'initial.png') if not row['validation_error'] else None
        if fingerprint is not None:
            initial.setdefault(row['seed'], fingerprint)
        result = curves.audit_trial(root, row, read(root/row['replica']/'manifest.json'), initial.get(row['seed']))
        require((trial/'result.json').stat().st_mtime >= registration['registered_at'], 'Trial predates amendment')
        rows.setdefault((row['replica'], row['arm'], row['revision']), []).append(result)
        hashes[row['trial']+'/result.json'] = base.digest(trial/'result.json')
    report = {'status': 'completed', 'slots': 800, 'layouts': 20, 'scope': registration['scope'],
              'replicas': {}, 'mean_curves': {}, 'result_hashes': hashes,
              'registration_sha256': base.digest(output/'registration.json'),
              'scheduler': audit_leases(output, {'rollout': registration['max_workers']})}
    for name in sorted({row['replica'] for row in registration['records']}):
        training = read(root/name/'training_summary.json')
        accounting = read(output/'training'/name/'resource_accounting.json')
        report['replicas'][name] = {}
        for arm in ('basic', 'symbolic'):
            raw = [statistics.mean(r['success'] for r in rows[name, arm, revision]) for revision in range(4)]
            tokens = [0]
            for turn in accounting['arms'][arm]['turns']:
                tokens.append(tokens[-1]+turn['usage']['total_tokens'])
            report['replicas'][name][arm] = curves.curves(training['curves'][arm], raw, tokens, [.8, .6, .7])
        report['replicas'][name]['paired_by_raw_revision'] = [paired(rows[name,'basic',r], rows[name,'symbolic',r]) for r in range(4)]
    for arm in ('basic', 'symbolic'):
        report['mean_curves'][arm] = {key: [statistics.mean(v[arm][key][i] for v in report['replicas'].values()) for i in range(4)]
                                    for key in ('raw','retained','cumulative_generation_tokens')}
    write_json(output/'summary.json', report)
    plot(output, report)
    return report


def plot(output, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for axis, kind, cost in zip(axes, ('raw','retained','retained'), (False,False,True)):
        for arm, label in (('basic','Base'),('symbolic','Full')):
            mean = report['mean_curves'][arm]
            x = [v/1000 for v in mean['cumulative_generation_tokens']] if cost else range(4)
            axis.plot(x, mean[kind], '-o', label=label)
        axis.set(ylim=(0,1), ylabel='Test success rate', xlabel='Cumulative tokens (thousands)' if cost else 'Revision', title=kind.capitalize())
        axis.legend(); axis.grid(alpha=.2)
    fig.suptitle('Three-cube stacking: 5 paired chains, 20 shared layouts; budget amended before testing')
    fig.savefig(output/'learning_curves.png', dpi=180); fig.savefig(output/'learning_curves.pdf')
    plt.close(fig)


def run(root):
    output = root/DIRECTORY
    with runner.study.exclusive(root/'pipeline.lock'), runner.study.exclusive(root/'coordinator.lock'):
        registration = check(root); check_stop(output)
        require(not (output/'summary.json').exists(), 'Compact evaluation already complete')
        pending = [r for r in registration['records'] if not (root/r['trial']/'result.json').exists()]
        pending.sort(key=lambda r:(r['revision']==0,r['seed'],r['revision'],r['replica'],r['arm']))
        completed = 800-len(pending); started = time.time(); samples = []
        manifests = {r['replica']: read(root/r['replica']/'manifest.json') for r in pending}
        def execute(row):
            with lease(output, 'rollout', registration['max_workers'], row['trial']):
                require(shutil.disk_usage(root).free >= registration['minimum_free_disk_bytes'], 'Disk reserve reached')
                return runner.variant.execute(root/row['replica'], row['arm'], root/row['candidate'],
                    (root/row['trial']).parent, [row['seed']], manifests[row['replica']], row['validation_error'])
        def batch(jobs, workers):
            nonlocal completed
            for _, _ in bounded_map(execute, jobs, workers, on_error=lambda e:stop(output,e)):
                completed += 1
                write_json(output/'progress.json', {'status':'evaluating','pid':os.getpid(),'completed':completed,
                    'total':800,'workers':workers,'elapsed_seconds':time.time()-started,'time':time.time()})
        write_json(root/'progress.json', {'status':'compact_evaluation','pid':os.getpid(),'details':DIRECTORY+'/progress.json',
                                         'training_slots':200,'test_slots':800})
        try:
            with curves.telemetry(output,samples), runner.variant.configured():
                batch(pending[:48],24)
                workers = curves.pilot_workers(samples,24)
                batch(pending[48:],workers)
            report = audit(root)
            write_json(output/'progress.json', {'status':'completed','completed':800,'total':800,'elapsed_seconds':time.time()-started})
            write_json(root/'progress.json', {'status':'compact_evaluation_completed','training_slots':200,'test_slots':800,
                                             'details':DIRECTORY+'/summary.json'})
            return report
        except BaseException as error:
            stop(output,error)
            write_json(output/'progress.json', {'status':'failed','error':str(error),'completed':completed,'total':800})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--phase',choices=('prepare','run','audit','check'),required=True)
    args=parser.parse_args()
    with installed():
        result = {'prepare':prepare,'run':run,'audit':audit,'check':check}[args.phase](args.root.resolve())
    print({k:result[k] for k in ('status','slots','mean_curves') if k in result})


if __name__ == '__main__':
    main()
