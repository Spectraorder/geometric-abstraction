"""Independent paired agent campaigns, with all training frozen before evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from run_online_geometry_experiment import prepare, campaign, digest, write, assert_frozen


def initialize(root):
    root.mkdir(parents=True,exist_ok=False)
    replicas=[]
    for index,start in enumerate([71001,71101,71201],1):
        directory=root/f'replica_{index:02d}'
        manifest=prepare(directory,3,list(range(start,start+5)),list(range(72001,72031)))
        replicas.append({'directory':directory.name,'train_seeds':manifest['train_seeds'],
                         'manifest_sha256':digest(directory/'manifest.json')})
    protocol={'schema':'online_geometry_replications_v2','replicas':replicas,
        'heldout_seeds':list(range(72001,72031)),'training_concurrency':2,'evaluation_concurrency':1,
        'method':'consistency-gated instantaneous online state plus derived feedback; shared PnP execution',
        'training_revisions_per_arm':3,'all_training_frozen_before_any_reserved':True,
        'independence':'Fresh isolated model contexts and distinct training layouts per replica; common reserved layouts never enter prompts.',
        'primary_outcomes':['per-replica learning curves','iterations to 80% training success','selected-policy reserved success'],
        'analysis':'Report each replica. Shared reserved scenes and repeated policies induce dependence; pooled counts are descriptive, not independent Bernoulli trials.',
        'coordinator_sha256':digest(Path(__file__)),
        'prior_evidence':'Do not pool the earlier online v1 or feedback-only pilots with this method version.'}
    write(root/'protocol.json',protocol)
    (root/'coordinator_snapshot.py').write_bytes(Path(__file__).read_bytes())
    return protocol


def check(root,protocol):
    if digest(Path(__file__))!=protocol['coordinator_sha256']:
        raise RuntimeError('Coordinator changed after protocol freeze')
    for row in protocol['replicas']:
        directory=root/row['directory']
        if digest(directory/'manifest.json')!=row['manifest_sha256']:
            raise RuntimeError('Replica manifest changed')
        assert_frozen(directory,json.loads((directory/'manifest.json').read_text()))


def run(root,protocol):
    check(root,protocol)
    write(root/'progress.json',{'status':'training','replicas':protocol['replicas']})
    with ThreadPoolExecutor(max_workers=protocol['training_concurrency']) as pool:
        pending={}
        for row in protocol['replicas']:
            directory=root/row['directory']
            if not (directory/'training_summary.json').exists():
                manifest=json.loads((directory/'manifest.json').read_text())
                pending[pool.submit(campaign,directory,manifest,'train')]=row['directory']
        for future in as_completed(pending):
            future.result()
            print('TRAINING FROZEN '+pending[future],flush=True)
    check(root,protocol)
    gate={}
    for row in protocol['replicas']:
        directory=root/row['directory'];trained=json.loads((directory/'training_summary.json').read_text())
        if trained['manifest_sha256']!=row['manifest_sha256']:raise RuntimeError('Training provenance mismatch')
        for arm,frozen in trained['frozen'].items():
            if digest(directory/arm/'frozen.py')!=frozen['sha256']:raise RuntimeError('Selected policy changed')
        gate[row['directory']]=trained['frozen']
    write(root/'all_candidates_frozen.json',gate)
    write(root/'progress.json',{'status':'evaluating','frozen':gate})
    for row in protocol['replicas']:
        check(root,protocol)
        directory=root/row['directory']
        if not (directory/'summary.json').exists():
            campaign(directory,json.loads((directory/'manifest.json').read_text()),'evaluate')
    write(root/'progress.json',{'status':'completed','frozen':gate,'protocol_sha256':digest(root/'protocol.json')})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();root=args.output.resolve()
    protocol=json.loads((root/'protocol.json').read_text()) if args.resume else initialize(root)
    run(root,protocol)
