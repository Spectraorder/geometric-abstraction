"""Check bounded paired dispatch and the separation from development seeds."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
import numpy as np
from unittest.mock import MagicMock, patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_online_bound_study as study
import run_online_bound_development as development


class BoundStudyTests(unittest.TestCase):
    def test_default_budget_reaches_environment_deadline(self):
        env=MagicMock()
        env.data=SimpleNamespace(qpos=np.zeros(1),qvel=np.zeros(1))
        env.episode_started=None
        env.start_episode.side_effect=TimeoutError
        env.structure_status.return_value={}
        with TemporaryDirectory() as name, \
             patch.object(development,'snapshot_sources'), \
             patch.object(development,'ContactCompleteEnv',return_value=env):
            result=development.run(Path(name)/'episode',2028100101,'basic')
        env.reset.assert_called_once_with(2028100101)
        env.start_episode.assert_called_once_with(200.,realtime=True)
        self.assertEqual(result['budget_s'],200.)
        self.assertEqual(result['capped_completion_seconds'],200.)
        self.assertEqual(result['status'],'timeout')

    def test_invalid_budget_rejected_before_creating_episode(self):
        with TemporaryDirectory() as name:
            output=Path(name)/'episode'
            for budget in (0.,-1.,201.,float('nan'),float('inf')):
                with self.subTest(budget=budget), self.assertRaises(ValueError):
                    development.run(output,2028100101,'basic',budget)
                self.assertFalse(output.exists())

    def test_fixed_batches_preserve_both_arms(self):
        slots=study.slots()
        self.assertEqual(len(slots),10)
        self.assertTrue(set(study.SEEDS).isdisjoint(development.DEVELOPMENT_SEEDS))
        for offset in range(0,10,4):
            batch=slots[offset:offset+4]
            for seed in {s['seed'] for s in batch}:
                self.assertEqual({s['arm'] for s in batch if s['seed']==seed},set(study.ARMS))

    def test_worker_failure_stops_new_batches_and_refuses_resume(self):
        with TemporaryDirectory() as name:
            root=Path(name);protocol={'slots':study.slots(),'concurrency':4}
            with patch.object(study,'verify_sources',return_value=protocol), \
                 patch.object(study.subprocess,'call',return_value=1) as launch, \
                 patch.object(study,'summarize') as summarize:
                with self.assertRaisesRegex(RuntimeError,'worker failed'):study.execute(root)
                self.assertEqual(launch.call_count,4)
                summarize.assert_not_called()
                self.assertEqual(json.loads((root/'progress.json').read_text())['status'],'stopped')
                with self.assertRaises(FileExistsError):study.execute(root)
                self.assertEqual(launch.call_count,4)

    def test_development_entrypoint_still_rejects_reserved_seed(self):
        with patch.object(development,'_run_episode') as episode:
            with self.assertRaises(ValueError):development.run(Path('/unused'),study.SEEDS[0],'basic')
            episode.assert_not_called()
            development.run(Path('/unused'),2028100101,'basic')
            episode.assert_called_once_with(Path('/unused'),2028100101,'basic',200.,scripted=False,compact_text=False)


if __name__=='__main__':unittest.main()
