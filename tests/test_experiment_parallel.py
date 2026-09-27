"""Verify shared limits, crash recovery and dispatch failure behavior."""
from concurrent.futures import ThreadPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from experiment_parallel import audit_leases, bounded_map, lease, stop


def leased_work(root, index):
    with lease(Path(root), 'rollout', 2, str(index)):
        time.sleep(0.15)


def hold_until_killed(root, ready):
    with lease(Path(root), 'model', 1, 'abandoned'):
        ready.set()
        time.sleep(60)


class ParallelTests(unittest.TestCase):
    def test_limit_is_shared_between_processes(self):
        with TemporaryDirectory() as temp:
            context = multiprocessing.get_context('spawn')
            processes = [context.Process(target=leased_work, args=(temp, index)) for index in range(6)]
            for process in processes:
                process.start()
            for process in processes:
                process.join(10)
                self.assertEqual(process.exitcode, 0)
            report = audit_leases(Path(temp), {'rollout': 2})['rollout']
            self.assertEqual(report['jobs'], 6)
            self.assertEqual(report['max_concurrent'], 2)

    def test_os_releases_crashed_process_lock_but_audit_retains_failure(self):
        with TemporaryDirectory() as temp:
            context = multiprocessing.get_context('spawn')
            ready = context.Event()
            process = context.Process(target=hold_until_killed, args=(temp, ready))
            process.start()
            self.assertTrue(ready.wait(10))
            process.terminate()
            process.join(5)
            with lease(Path(temp), 'model', 1, 'next'):
                pass
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                audit_leases(Path(temp), {'model': 1})

    def test_failure_stops_new_leases_and_releases_capacity(self):
        with TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError, 'example failure'):
                with lease(Path(temp), 'model', 1, 'failing'):
                    raise RuntimeError('example failure')
            self.assertTrue((Path(temp)/'STOP.json').exists())
            with self.assertRaisesRegex(RuntimeError, 'dispatch is stopped'):
                with lease(Path(temp), 'model', 1, 'must not run'):
                    self.fail('Stopped work ran')

    def test_bounded_dispatch_does_not_queue_all_jobs_or_continue_after_error(self):
        consumed = []
        barrier = threading.Barrier(2)
        def jobs():
            for value in range(20):
                consumed.append(value)
                yield value
        def execute(value):
            barrier.wait(timeout=5)
            if value == 0:
                raise RuntimeError('failed')
            time.sleep(.05)
            return value
        with self.assertRaisesRegex(RuntimeError, 'failed'):
            list(bounded_map(execute, jobs(), 2))
        self.assertEqual(consumed, [0, 1])


if __name__ == '__main__':
    unittest.main()
