"""Pointing format, RGB-D rejection gates, and outbound sensor boundary checks."""
import base64
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from aspire_so100.instance_geometry import inventory_color_mask, inventory_points, depth_components
from aspire_so100.pointing_geometry import parse_molmo2_point, locate_pointed_component

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('probe', ROOT / 'scripts/probe_molmo_pointing.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class PointingGeometryTests(unittest.TestCase):
    def setUp(self):
        with np.load(ROOT / 'tests/fixtures/inventory_six_rgbd.npz') as archive:
            self.frame = dict(archive)

    def red_components(self):
        points, valid = inventory_points(self.frame)
        return depth_components(inventory_color_mask(self.frame['rgb'], 'red') & valid, points)

    @staticmethod
    def interior_point(mask):
        v, u = np.nonzero(mask)
        i = np.argmin((u - np.median(u))**2 + (v - np.median(v))**2)
        return np.array([u[i] + .25, v[i] + .25])

    def test_official_thousand_scale_and_axis_order(self):
        point = parse_molmo2_point('<points coords="1 1 250 750">beam</points>', 320, 240)
        np.testing.assert_allclose(point, [80, 180])

    def test_malformed_ambiguous_legacy_and_outside_points_abstain(self):
        for text in ('not visible', '<point x="50" y="50"/>',
                     '<points coords="1 1 050.0 050.0"/>',
                     '<points coords="1 1 500 500 2 600 600"/>',
                     '<points coords="2 1 500 500"/>',
                     '<points coords="1 1 500 500;2 2 600 600"/>',
                     '<points coords="1 1 1000 500"/>',
                     '<points coords="1 1 500 1001"/>',
                     '<points coords="1 1 -001 500"/>',
                     '<points coords="1 1 500 500"/>' * 2, None):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_molmo2_point(text, 320, 240)

    def test_same_color_components_stay_separate_and_surface_is_calibrated(self):
        components = self.red_components()
        self.assertGreaterEqual(len(components), 4)
        for mask in components:
            point = self.interior_point(mask)
            located = locate_pointed_component(self.frame, point, 'red')
            # Table-height filtering may remove bottom-edge pixels. Selection
            # must retain this component without leaking into another red object.
            self.assertFalse(np.any(located['mask'] & ~mask))
            self.assertGreater(located['mask'].sum(), mask.sum() / 2)
            u, v = np.floor(point).astype(int)
            ray = np.linalg.inv(self.frame['intrinsics']) @ [u, v, 1]
            camera_point = ray * self.frame['depth_m'][v, u]
            pose = self.frame['camera_to_world']
            np.testing.assert_allclose(located['surface_world_m'], pose[:3, :3] @ camera_point + pose[:3, 3])
            self.assertFalse(located['complete_object_geometry'])
            self.assertFalse(located['identity_verified'])

    def test_color_depth_and_bounds_reject_without_snapping(self):
        point = self.interior_point(self.red_components()[0])
        with self.assertRaisesRegex(ValueError, 'color'):
            locate_pointed_component(self.frame, point, 'blue')
        u, v = np.floor(point).astype(int)
        self.frame['valid_depth'][v, u] = False
        with self.assertRaisesRegex(ValueError, 'valid depth'):
            locate_pointed_component(self.frame, point, 'red')
        for bad in ([np.nan, 10], [-.01, 10], [320, 10], [10, 240]):
            with self.subTest(point=bad), self.assertRaises(ValueError):
                locate_pointed_component(self.frame, bad, 'red')

    def test_cli_sends_only_rgb_and_public_text_and_retains_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            args = ['probe', '--frame', str(ROOT / 'tests/fixtures/inventory_six_rgbd.npz'),
                    '--target', 'red thin rectangular prism', '--color', 'red',
                    '--output', directory, '--send']
            failed = {'http_status': 404, 'response': {'error': {'message': 'No endpoints'}}}
            with patch('sys.argv', args), patch.object(probe, 'completion', return_value=failed) as call:
                probe.main()
            body, _ = call.call_args.args
            self.assertEqual(len(body['messages']), 1)
            content = body['messages'][0]['content']
            self.assertEqual(content[0], {'type': 'text', 'text': 'Point to the red thin rectangular prism.'})
            self.assertEqual(len(content), 2)
            encoded = content[1]['image_url']['url'].split(',', 1)[1]
            decoded = np.array(Image.open(io.BytesIO(base64.b64decode(encoded))))
            np.testing.assert_array_equal(decoded, self.frame['rgb'])
            report = json.loads((Path(directory) / 'report.json').read_text())
            self.assertEqual(report['status'], 'request_failed')
            self.assertNotIn('location', report)


if __name__ == '__main__':
    unittest.main()
