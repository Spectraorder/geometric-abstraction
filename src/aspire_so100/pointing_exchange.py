"""Optional host-side RGB-only mailbox; never imported by the policy worker."""
import hashlib
import json
from pathlib import Path
import time
import uuid

import numpy as np
from PIL import Image


def checked_point_reply(reply, request):
    if reply.get('task_id') != request['task_id'] or reply.get('image_sha256') != request['image_sha256']:
        raise ValueError('Point reply is not bound to the current RGB request')
    status, point = reply.get('status'), reply.get('point_px')
    if status == 'abstain' and point is None:
        return {'status': 'abstain', 'point_px': None}
    if (status != 'point' or not isinstance(point, list) or len(point) != 2
            or any(type(v) not in (int, float) for v in point)
            or not np.all(np.isfinite(point))
            or not (0 <= point[0] < request['image_width'] and 0 <= point[1] < request['image_height'])):
        raise ValueError('Invalid point reply')
    return {'status': 'point', 'point_px': point}


class PointingExchange:
    def __init__(self, directory, timeout=240., max_requests=4):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.timeout = timeout
        self.max_requests = max_requests
        self.records = []
        self.wait_seconds = 0.

    def request(self, frame, color, shape, orientation):
        if len(self.records) >= self.max_requests:
            raise ValueError('Pointing request budget exhausted')
        task_id = uuid.uuid4().hex
        folder = self.directory / task_id
        folder.mkdir()
        image = folder / 'input_rgb.png'
        Image.fromarray(frame['rgb']).save(image)
        label = {'cube': 'cube', 'cylinder': 'cylinder', 'prism': 'rectangular prism',
                 'thin_prism': 'thin rectangular beam'}[shape]
        h, w = frame['rgb'].shape[:2]
        request = {'task_id': task_id, 'image_path': str(image.resolve()),
                   'image_width': w, 'image_height': h, 'target': color + ' ' + label,
                   'image_sha256': hashlib.sha256(image.read_bytes()).hexdigest()}
        # Publishing request.json last makes the image and manifest ready together.
        (folder / 'request.json').write_text(json.dumps(request, indent=2) + '\n')
        record = {'task_id': task_id, 'frame_time': float(frame['time']), 'request': request}
        self.records.append(record)
        started = time.monotonic()
        try:
            response = folder / 'response.json'
            while time.monotonic() - started < self.timeout:
                if response.exists():
                    # Writers must atomically rename their completed JSON file.
                    reply = json.loads(response.read_text())
                    result = checked_point_reply(reply, request)
                    record.update(reply=reply, status='completed')
                    return result
                time.sleep(.1)
            record['status'] = 'timeout'
            raise TimeoutError('RGB pointing response timed out')
        except Exception as error:
            record.setdefault('status', 'rejected')
            record['error'] = str(error)
            raise
        finally:
            elapsed = time.monotonic() - started
            self.wait_seconds += elapsed
            record['host_wait_seconds'] = elapsed
            (folder / 'completed.json').write_text(json.dumps(record, indent=2) + '\n')
