"""Exercise coordinator dependencies without model calls or simulator rollouts."""
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_parallel_geometry_study as study
from experiment_parallel import audit_leases, write_json


class StudyTests(unittest.TestCase):
    def test_training_uses_global_leases_and_reuses_completed_training(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root/'replica_01'
            directory.mkdir()
            manifest = {'train_seeds': [1, 2, 3, 4, 5], 'arms': ['basic', 'symbolic']}
            write_json(directory/'manifest.json', manifest)
            protocol = {'replicas': [{'directory': 'replica_01'}], 'rounds': 1,
                        'limits': {'model': 1, 'rollout': 2}, 'minimum_free_disk_bytes': 0}
            def raw_execute(root, arm, candidate, target, seeds, manifest, error=None):
                seed = seeds[0]
                write_json(target/f'seed_{seed}'/'result.json', {'seed': seed})
                return [{'seed': seed}]
            def bootstrap(root, manifest):
                for arm in manifest['arms']:
                    rows = study.neutral.execute(root, arm, root/'candidate.py', root/arm/'revision_000/trials', manifest['train_seeds'], manifest)
                    self.assertEqual([row['seed'] for row in rows], manifest['train_seeds'])
                write_json(root/'starter_parity.json', {})
            calls = []
            @contextmanager
            def configured():
                with patch.object(study.base, 'execute', study.neutral.execute), \
                     patch.object(study.base, 'propose', lambda *args: calls.append(args[0])):
                    yield
            def campaign(root, manifest, phase):
                for arm in manifest['arms']:
                    study.base.propose(root/arm/'round_001', 'prompt', [], manifest)
                    rows = study.base.execute(root, arm, root/'candidate.py', root/arm/'round_001/trials', manifest['train_seeds'], manifest)
                    self.assertEqual([row['seed'] for row in rows], manifest['train_seeds'])
                write_json(root/'training_summary.json', {})
            with patch.object(study, 'check', return_value=protocol), \
                 patch.object(study.neutral, 'execute', raw_execute), \
                 patch.object(study.variant, 'bootstrap', bootstrap), \
                 patch.object(study.variant, 'configured', configured), \
                 patch.object(study.base, 'campaign', campaign), \
                 patch.object(study.variant, 'audit', return_value={'exact_actor_contexts_verified': 2}), \
                 patch.object(study, 'summarize', return_value={}):
                study.train_replica(root, 'replica_01')
                self.assertEqual(len(calls), 2)
                before = audit_leases(root, protocol['limits'])
                self.assertEqual(before['rollout']['jobs'], 20)
                study.train_replica(root, 'replica_01')
                self.assertEqual(audit_leases(root, protocol['limits']), before)
                self.assertEqual(len(calls), 2)

    def test_global_gate_rejects_missing_replica_and_changed_evidence(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            protocol = {'replicas': [{'directory': 'replica_01'}, {'directory': 'replica_02'}]}
            write_json(root/'protocol.json', protocol)
            gate = {'protocol_sha256': study.base.digest(root/'protocol.json'),
                    'replicas': {'replica_01': {'evidence': {}}}}
            write_json(root/'all_candidates_frozen.json', gate)
            with self.assertRaisesRegex(ValueError, 'Incomplete global freeze'):
                study.check_gate(root, protocol)
            proof = root/'replica_02'/'proof.json'
            write_json(proof, {'value': 1})
            gate['replicas']['replica_02'] = {'evidence': {'proof.json': study.base.digest(proof)}}
            write_json(root/'all_candidates_frozen.json', gate)
            study.check_gate(root, protocol)
            write_json(proof, {'value': 2})
            with self.assertRaisesRegex(ValueError, 'Frozen evidence changed'):
                study.check_gate(root, protocol)

    def test_shared_test_registration_is_idempotent_and_keeps_selected_policies(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            protocol = {'heldout_seeds': list(range(100, 200)), 'replicas': [{'directory': 'replica_01'}], 'limits': {'rollout': 4}}
            write_json(root/'protocol.json', protocol)
            selected = {'basic': {'revision': 2, 'sha256': 'a'}, 'symbolic': {'revision': 1, 'sha256': 'b'}}
            gate = {'protocol_sha256': study.base.digest(root/'protocol.json'),
                    'replicas': {'replica_01': {'evidence': {}, 'selected': selected}}}
            write_json(root/'all_candidates_frozen.json', gate)
            study.register_evaluation(root, protocol)
            path = root/'replica_01/selected_evaluation/registration.json'
            before = path.stat().st_mtime_ns
            study.register_evaluation(root, protocol)
            self.assertEqual(path.stat().st_mtime_ns, before)
            self.assertEqual(study.read(path)['selected'], selected)
            self.assertEqual(study.read(path)['required_rollouts'], 200)


if __name__ == '__main__':
    unittest.main()
