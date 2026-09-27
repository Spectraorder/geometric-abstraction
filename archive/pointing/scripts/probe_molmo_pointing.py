"""Prepare or run an optional OpenRouter pointing experiment on a saved RGB-D frame.

Default operation is local preparation. --send performs one image request;
--probe sends only generic text. Credentials stay outside the policy worker.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

import numpy as np
from PIL import Image, ImageDraw

from aspire_so100.pointing_geometry import parse_molmo2_point, locate_pointed_component

ENDPOINT = 'https://openrouter.ai/api/v1/chat/completions'
MODELS = ('allenai/molmo-2-8b', 'allenai/molmo-2-8b:free')


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def completion(body, key_file):
    """Use the credential only at the fixed OpenRouter endpoint; sanitize errors."""
    keys = set(re.findall(r'sk-or-v1-[A-Za-z0-9_-]+', key_file.read_text()))
    if len(keys) != 1:
        raise ValueError('Expected one OpenRouter token in the key file')
    key = keys.pop()
    request = urllib.request.Request(
        ENDPOINT, data=json.dumps(body).encode(),
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'},
    )
    start = time.perf_counter()
    try:
        with urllib.request.build_opener(NoRedirect).open(request, timeout=45) as response:
            status, data = response.status, json.load(response)
    except urllib.error.HTTPError as error:
        status = error.code
        try:
            data = json.loads(error.read())
        except (ValueError, UnicodeDecodeError):
            data = {'error': {'message': 'Non-JSON HTTP error'}}
    except (urllib.error.URLError, TimeoutError):
        status, data = None, {'error': {'message': 'Network request failed or timed out'}}
    safe = json.loads(json.dumps(data).replace(key, '[REDACTED]'))
    if 'error' in safe:
        safe = {'error': {k: safe['error'][k] for k in ('code', 'message') if k in safe['error']}}
    else:
        safe = {k: safe[k] for k in ('id', 'model', 'provider', 'choices', 'usage') if k in safe}
    return {'http_status': status, 'elapsed_seconds': time.perf_counter() - start,
            'response': safe}


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frame', type=Path)
    parser.add_argument('--target', help='Public color/shape description, without metric dimensions')
    parser.add_argument('--color', choices=('red', 'yellow', 'blue', 'green'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', choices=MODELS, default=MODELS[0])
    parser.add_argument('--key-file', type=Path, default=Path('.apikey'))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--probe', action='store_true', help='Send generic text only; no image')
    mode.add_argument('--send', action='store_true', help='Send the prepared RGB image to OpenRouter')
    mode.add_argument('--response', type=Path, help='Parse a saved model text response locally')
    args = parser.parse_args()
    if not args.probe and (args.frame is None or not args.target or not args.color):
        parser.error('--frame, --target and --color are required except with --probe')
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'model': args.model, 'endpoint': ENDPOINT,
              'mode': 'probe' if args.probe else 'send' if args.send else 'offline',
              'image_requests_attempted': 0}
    body = {'model': args.model, 'max_tokens': 256, 'temperature': 0,
            'provider': {'allow_fallbacks': False}}
    if args.probe:
        body.update(messages=[{'role': 'user', 'content': 'Reply with OK.'}], max_tokens=8)
        report.update(prompt='Reply with OK.', request=completion(body, args.key_file))
        report['status'] = 'available' if report['request']['http_status'] == 200 else 'probe_failed'
    else:
        # Read only sensor fields, never result.json, rollout states, or catalog data.
        with np.load(args.frame, allow_pickle=False) as archive:
            frame = {k: archive[k] for k in ('rgb', 'depth_m', 'valid_depth',
                                             'intrinsics', 'camera_to_world')}
        rgb = Image.fromarray(frame['rgb'])
        png = io.BytesIO()
        rgb.save(png, format='PNG')
        (args.output / 'input_rgb.png').write_bytes(png.getvalue())
        prompt = 'Point to the ' + args.target.rstrip('.') + '.'
        report.update(source_frame=str(args.frame), prompt=prompt,
                      source_frame_sha256=hashlib.sha256(args.frame.read_bytes()).hexdigest(),
                      input_png_sha256=hashlib.sha256(png.getvalue()).hexdigest(),
                      image_size=list(rgb.size), status='prepared',
                      payload_boundary='Only RGB and public target text leave the host. '
                      'Depth/calibration stay local; no simulator object state is read.')
        content = None
        if args.send:
            body['messages'] = [{'role': 'user', 'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,'
                                                 + base64.b64encode(png.getvalue()).decode()}},
            ]}]
            report['request'] = completion(body, args.key_file)
            report['image_requests_attempted'] = 1
            report['status'] = 'request_failed'
            data = report['request']['response']
            if report['request']['http_status'] == 200 and data.get('choices'):
                content = data['choices'][0]['message'].get('content')
        elif args.response:
            content = args.response.read_text()
            report['response_source'] = str(args.response)
        if content is not None:
            report['model_text'] = content
            try:
                point = parse_molmo2_point(content, *rgb.size)
                located = locate_pointed_component(frame, point, args.color)
                report['location'] = {k: v for k, v in located.items() if k != 'mask'}
                report['status'] = 'color_depth_component_selected'
                overlay = np.array(rgb).copy()
                mask = located['mask']
                overlay[mask] = (.5 * overlay[mask] + .5 * np.array([255, 255, 255])).astype(np.uint8)
                overlay = Image.fromarray(overlay).resize((rgb.width * 3, rgb.height * 3), Image.Resampling.NEAREST)
                x, y = point * 3
                draw = ImageDraw.Draw(overlay)
                draw.ellipse((x-6, y-6, x+6, y+6), outline='magenta', width=2)
                overlay.save(args.output / 'point_overlay.png')
            except ValueError as error:
                report.update(status='abstained', reason=str(error))
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, default=jsonable) + '\n')
    print(json.dumps({'report': str(args.output / 'report.json'),
                      'status': report.get('status'), 'http_status': report.get('request', {}).get('http_status')}))


if __name__ == '__main__':
    main()
