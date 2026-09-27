"""Three isolated visual-input/full-geometry chains with a global evaluation gate."""
import argparse
from pathlib import Path
import shutil
import time
from unittest.mock import patch
import run_visual_input_experiment as variant
import run_mixed_scene_replications as coordinator
from summarize_visual_input_experiment import audit

ROOT=variant.ROOT


def initialize(root):
    root.mkdir(parents=True,exist_ok=False)
    reserved=list(range(991001,991021));replicas=[]
    authorization=ROOT/'outputs/geometry_step_learning_pilot_01_corrected/external_model_authorization.json'
    if not authorization.is_file():raise RuntimeError('Missing model-transmission authorization receipt')
    for index,start in enumerate((990101,990201,990301),1):
        directory=root/f'replica_{index:02d}'
        manifest=variant.prepare(directory,list(range(start,start+5)),reserved,3)
        shutil.copyfile(authorization,directory/'external_model_authorization.json')
        replicas.append({'directory':directory.name,'train_seeds':manifest['train_seeds'],'manifest_sha256':coordinator.digest(directory/'manifest.json')})
    protocol={'schema':'visual_input_replications_v1',
        'registered_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'arms':list(variant.ARMS),'replicas':replicas,'heldout_seeds':reserved,
        'training_revisions_per_arm':3,'training_slots':120,'reserved_slots':480,'model_generations':18,
        'all_training_frozen_before_any_reserved':True,
        'coordinator_sha256':coordinator.digest(Path(__file__)),
        'shared_coordinator_sha256':coordinator.digest(Path(coordinator.__file__)),
        'authorization_sha256':coordinator.digest(authorization),
        'method':'Raw-visual high-level planning versus supplied compact geometry. Identical hidden RGB-D actuator and common policy-owned visual starter.',
        'independence':'Three isolated actor chains per arm, disjoint training layouts and common fresh reserved layouts; no cross-chain feedback.',
        'primary_outcomes':['Per-chain reserved SR at each revision','Per-chain mean reserved SR over R0-R3','First revision reaching 80 percent reserved SR','Training-selected reserved SR'],
        'analysis':'Report all paired chains and revisions, including invalid slots/regressions; mean and range of paired differences. Shared reserved layouts and revisions are not independent policy replications. No pooling with historical strong-baseline studies.',
        'selection':'Highest training SR, earliest tie. No reserved selection or early stopping on SR.',
        'limitations':['Only three chain pairs; limited inferential power.','Treatment includes supplied perception/geometry, not solely symbolic representation.','No latency benefit inferred from character counts. Rollout duration is not an objective.'],
        'exclusions':'No prior evolved policy, inspected failure diagnosis, hidden-pose prediction, new actuator, manual recovery solution or reserved outcome enters actor contexts.'}
    coordinator.write(root/'protocol.json',protocol)
    shutil.copyfile(Path(__file__),root/'coordinator_snapshot.py')
    shutil.copyfile(Path(coordinator.__file__),root/'shared_coordinator_snapshot.py')
    shutil.copyfile(authorization,root/'external_model_authorization.json')
    return protocol


def check(root,protocol):
    if protocol['schema']!='visual_input_replications_v1':raise RuntimeError('Wrong study protocol')
    for current,archived,key in ((Path(__file__),root/'coordinator_snapshot.py','coordinator_sha256'),(Path(coordinator.__file__),root/'shared_coordinator_snapshot.py','shared_coordinator_sha256')):
        if coordinator.digest(current)!=protocol[key] or coordinator.digest(archived)!=protocol[key]:raise RuntimeError('Coordinator changed')
    if not (root/'all_candidates_frozen.json').exists() and list(root.glob('replica_*/*/heldout*')):raise RuntimeError('Reserved execution precedes global freeze')
    for row in protocol['replicas']:
        directory=root/row['directory'];manifest=coordinator.read(directory/'manifest.json')
        if coordinator.digest(directory/'manifest.json')!=row['manifest_sha256']:raise RuntimeError('Manifest changed')
        if manifest['arms']!=protocol['arms'] or manifest['heldout_seeds']!=protocol['heldout_seeds'] or manifest['train_seeds']!=row['train_seeds']:raise RuntimeError('Replica split changed')
        variant.check(directory,manifest)
        if (directory/'training_summary.json').exists():
            receipt=coordinator.read(directory/'train_audit.json')
            if receipt['exact_actor_contexts_verified']!=len(protocol['arms'])*manifest['rounds'] or receipt['starter_parity_audit']!='passed' or receipt['valid_rollouts']+receipt['invalid_slots']!=len(protocol['arms'])*(manifest['rounds']+1)*len(manifest['train_seeds']):raise RuntimeError('Training context audit incomplete')


def campaign(root,manifest,phase):
    if phase=='train' and not (root/'starter_parity.json').exists():variant.bootstrap(root,manifest)
    with variant.configured():variant.base.campaign(root,manifest,phase)
    coordinator.write(root/(phase+'_audit.json'),audit(root,allow_partial=phase=='train'))


def run(root,protocol,phase):
    with patch.multiple(coordinator,check=check,campaign=campaign):coordinator.run(root,protocol,phase)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--phase',choices=('prepare','train','evaluate','all'),default='all')
    args=parser.parse_args();root=args.output.resolve()
    if args.phase=='evaluate' and not args.resume:parser.error('Evaluation requires frozen training')
    protocol=coordinator.read(root/'protocol.json') if args.resume else initialize(root)
    check(root,protocol)
    if args.phase!='prepare':run(root,protocol,args.phase)
