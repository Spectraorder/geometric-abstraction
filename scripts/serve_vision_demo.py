"""Read-only static demo server with video byte ranges; no simulation endpoints."""
import argparse
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import mimetypes
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1] / 'web/vision'


class DemoHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 30

    def __init__(self, *args, root=ROOT, **kwargs):
        self.root = Path(root).resolve()
        super().__init__(*args, **kwargs)

    def do_HEAD(self):
        self.send_asset(head=True)

    def do_GET(self):
        self.send_asset()

    def send_asset(self, head=False):
        relative = unquote(urlsplit(self.path).path).lstrip('/') or 'index.html'
        if any(part.startswith('.') for part in Path(relative).parts):
            return self.send_error(404)
        try:
            path = (self.root / relative).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                return self.send_error(404)
            if path.suffix not in {'.html', '.css', '.js', '.json', '.mp4', '.png', '.md', '.py'}:
                return self.send_error(404)
            size = path.stat().st_size
            start, end, status = 0, size - 1, 200
            request_range = self.headers.get('Range')
            if request_range:
                match = re.fullmatch(r'bytes=(\d*)-(\d*)', request_range)
                try:
                    if not match or not any(match.groups()):
                        raise ValueError
                    left, right = match.groups()
                    if left:
                        start = int(left)
                        end = min(int(right), size-1) if right else size-1
                    else:
                        start = max(0, size-int(right))
                    if start >= size or start > end or size == 0:
                        raise ValueError
                except ValueError:
                    self.send_response(416)
                    self.send_header('Content-Range', f'bytes */{size}')
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                    return
                status = 206
            self.send_response(status)
            self.send_header('Content-Type', mimetypes.guess_type(path.name)[0] or 'text/plain')
            self.send_header('Content-Length', str(max(0, end-start+1)))
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Cache-Control', 'public, max-age=60')
            if status == 206:
                self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            self.end_headers()
            if not head:
                with path.open('rb') as stream:
                    stream.seek(start)
                    remaining = end-start+1
                    while remaining > 0:
                        chunk = stream.read(min(65536, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bind', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8091)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    if not (args.root/'index.html').is_file():
        parser.error('Demo index.html is missing')
    server = ThreadingHTTPServer((args.bind, args.port), partial(DemoHandler, root=args.root))
    print(f'Vision demo: http://{args.bind}:{args.port}/', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
