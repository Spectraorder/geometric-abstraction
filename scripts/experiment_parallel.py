"""Process-shared resource leases and bounded dispatch for experiment workers."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import time
import uuid


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def check_stop(root):
    if (Path(root)/'STOP.json').exists():
        raise RuntimeError('Study dispatch is stopped; inspect STOP.json before resuming')


def stop(root, error):
    # Keep the first failure as the cause of the study-wide dispatch stop.
    path = Path(root)/'STOP.json'
    try:
        with path.open('x') as handle:
            json.dump({'time': time.time(), 'pid': os.getpid(), 'error': str(error)}, handle)
    except FileExistsError:
        pass


@contextmanager
def lease(root, resource, slots, label):
    """Hold one kernel-backed slot across processes and record its entire lifetime."""
    if slots < 1 or resource not in ('model', 'rollout'):
        raise ValueError('Invalid resource capacity')
    root = Path(root)
    directory = root/'scheduler'/resource
    directory.mkdir(parents=True, exist_ok=True)
    queued_at = time.time()
    handle = None
    while handle is None:
        check_stop(root)
        for index in range(slots):
            candidate = (directory/f'slot_{index}.lock').open('a')
            try:
                fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                candidate.close()
            else:
                handle = candidate
                break
        if handle is None:
            time.sleep(0.1)
    receipt = directory/'events'/(uuid.uuid4().hex + '.json')
    record = {'resource': resource, 'slot': index, 'pid': os.getpid(), 'label': label,
              'queued_at': queued_at, 'started_at': time.time(), 'status': 'running'}
    try:
        check_stop(root)
        write_json(receipt, record)
        yield
    except BaseException as error:
        record.update(status='failed', error=str(error))
        stop(root, error)
        raise
    else:
        record['status'] = 'completed'
    finally:
        record['finished_at'] = time.time()
        try:
            write_json(receipt, record)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()


def bounded_map(function, jobs, workers, on_error=None):
    """Yield completed jobs, with at most workers jobs submitted at any moment."""
    if workers < 1:
        raise ValueError('Require positive worker count')
    iterator = iter(jobs)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {}
        def fill():
            while len(pending) < workers:
                try:
                    job = next(iterator)
                except StopIteration:
                    return
                pending[pool.submit(function, job)] = job
        try:
            fill()
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                # Check the entire completed batch before submitting any more work.
                for future in done:
                    future.result()
                for future in done:
                    yield pending.pop(future), future.result()
                fill()
        except BaseException as error:
            if on_error is not None:
                on_error(error)
            for future in pending:
                future.cancel()
            raise


def audit_leases(root, limits):
    report = {}
    for resource, limit in limits.items():
        events = [json.loads(path.read_text()) for path in (Path(root)/'scheduler'/resource/'events').glob('*.json')]
        timeline = []
        for record in events:
            if record['status'] != 'completed' or 'finished_at' not in record:
                raise ValueError('Incomplete or failed resource lease requires inspection')
            timeline.extend([(record['started_at'], 1), (record['finished_at'], -1)])
        active = maximum = 0
        for _, change in sorted(timeline):
            active += change
            maximum = max(maximum, active)
        if maximum > limit:
            raise ValueError('Registered concurrency limit exceeded')
        report[resource] = {'jobs': len(events), 'max_concurrent': maximum, 'limit': limit,
                            'queue_seconds': sum(e['started_at']-e['queued_at'] for e in events)}
    return report
