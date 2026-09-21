"""Check video seeking and the public server's filesystem boundary."""
import importlib.util
from pathlib import Path
import tempfile
import threading
import unittest
from functools import partial
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('vision_server',ROOT/'scripts/serve_vision_demo.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class DemoServerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.base=Path(self.temp.name);self.root=self.base/'public';self.root.mkdir()
        (self.root/'index.html').write_text('demo')
        (self.root/'sample.mp4').write_bytes(b'0123456789')
        (self.base/'private.json').write_text('private')
        (self.root/'.private').mkdir()
        (self.root/'.private/state.json').write_text('hidden')
        (self.root/'escape.json').symlink_to(self.base/'private.json')
        self.server=ThreadingHTTPServer(('127.0.0.1',0),partial(module.DemoHandler,root=self.root))
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.url=f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join();self.temp.cleanup()

    def test_browser_video_ranges_and_head(self):
        for requested,expected in [('bytes=2-5',b'2345'),('bytes=7-',b'789'),('bytes=-3',b'789')]:
            with urlopen(Request(self.url+'/sample.mp4',headers={'Range':requested})) as response:
                self.assertEqual(response.status,206);self.assertEqual(response.read(),expected)
                self.assertEqual(response.headers['Accept-Ranges'],'bytes')
        with urlopen(Request(self.url+'/sample.mp4',method='HEAD')) as response:
            self.assertEqual(response.headers['Content-Length'],'10');self.assertEqual(response.read(),b'')
        with self.assertRaises(HTTPError) as error:
            urlopen(Request(self.url+'/sample.mp4',headers={'Range':'bytes=99-100'}))
        self.assertEqual(error.exception.code,416)

    def test_public_root_does_not_expose_parent_or_symlink(self):
        for path in ['/../private.json','/%2e%2e/private.json','/escape.json','/.git/config','/directory/','/.private/state.json']:
            with self.subTest(path=path),self.assertRaises(HTTPError) as error:urlopen(self.url+path)
            self.assertEqual(error.exception.code,404)
        with urlopen(self.url+'/') as response:self.assertEqual(response.read(),b'demo')
        with self.assertRaises(HTTPError) as error:urlopen(Request(self.url+'/',data=b'command'))
        self.assertEqual(error.exception.code,501)
