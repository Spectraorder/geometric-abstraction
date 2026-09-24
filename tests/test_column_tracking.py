"""Visual ownership of touching instances, using analytic RGB-D scenes."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from aspire_so100.column_geometry import estimate_column_geometry
from aspire_so100.column_tracking import ColumnTracker, column_surface_distance
from aspire_so100.instance_geometry import inventory_points, inventory_color_mask, depth_components
from test_cube_geometry import cube_frame


def primitive_scene(objects):
    """Ray-cast colored boxes/cylinders; labels exist only as test assertions."""
    frame, _ = cube_frame(np.eye(3))
    camera = frame['camera_to_world'][:3, 3]
    v, u = np.indices(frame['depth_m'].shape)
    rays = np.stack((u, v, np.ones_like(u)), axis=-1) @ np.linalg.inv(frame['intrinsics']).T
    rays = rays @ frame['camera_to_world'][:3, :3].T
    depth = -camera[2] / rays[..., 2]
    labels = np.full(depth.shape, -1)
    for index, obj in enumerate(objects):
        rotation = obj.get('rotation', np.eye(3))
        origin = (camera - obj['center']) @ rotation
        direction = rays @ rotation
        if obj['shape'] == 'cylinder':
            radius = obj['dims'][0] / 2
            a = np.sum(direction[..., :2] ** 2, axis=2)
            b = 2 * (direction[..., :2] @ origin[:2])
            c = origin[:2] @ origin[:2] - radius**2
            discriminant = b**2 - 4*a*c
            candidate = np.full(depth.shape, np.inf)
            for sign in (-1, 1):
                t = (-b + sign*np.sqrt(np.maximum(discriminant, 0))) / (2*a)
                z = origin[2] + t*direction[..., 2]
                valid = (discriminant >= 0) & (abs(z) <= obj['height']/2) & (t > 0)
                candidate = np.minimum(candidate, np.where(valid, t, np.inf))
                t = (sign*obj['height']/2 - origin[2]) / direction[..., 2]
                xy = origin[:2] + t[..., None]*direction[..., :2]
                valid = (np.sum(xy**2, axis=2) <= radius**2) & (t > 0)
                candidate = np.minimum(candidate, np.where(valid, t, np.inf))
        else:
            half = np.r_[obj['dims'], obj['height']] / 2
            first, second = (-half-origin)/direction, (half-origin)/direction
            entry = np.max(np.minimum(first, second), axis=2)
            leave = np.min(np.maximum(first, second), axis=2)
            candidate = np.where((entry > 0) & (entry < leave), entry, np.inf)
        nearer = candidate < depth
        depth = np.where(nearer, candidate, depth)
        labels[nearer] = index
    frame['depth_m'] = depth
    frame['valid_depth'] = np.isfinite(depth) & (depth > 0)
    frame['rgb'][:] = [160, 120, 70]
    frame['rgb'][labels >= 0] = [30, 40, 210]
    return frame, labels


def learned_pair():
    objects = [
        {'shape': 'cylinder', 'center': np.array([-.035, -.02, .0275]),
         'dims': np.array([.032, .032]), 'height': .055},
        {'shape': 'prism', 'center': np.array([.055, -.02, .032]),
         'dims': np.array([.032, .032]), 'height': .064},
    ]
    frame, _ = primitive_scene(objects)
    tracker = ColumnTracker(plane=np.zeros(3))
    tracker.update(frame)
    keys = []
    for obj in objects:
        track = min(tracker.tracks.values(), key=lambda item: np.linalg.norm(item['center']-obj['center']))
        track['shape'] = obj['shape']
        key = track['id']
        fit = estimate_column_geometry(tracker.isolated_frame(frame, key), 'blue', obj['shape'],
                                       table_plane=tracker.plane)
        tracker.surface_models[key] = fit
        keys.append(key)
    return tracker, objects, keys


class ColumnTrackingTests(unittest.TestCase):
    def test_touching_same_color_tower_keeps_two_geometric_ids(self):
        tracker, objects, keys = learned_pair()
        old_center = objects[1]['center'].copy()
        objects[1]['center'] = objects[0]['center'] + [0, 0, (objects[0]['height']+objects[1]['height'])/2]
        displacement = objects[1]['center']-old_center
        prior = tracker.surface_models[keys[1]]
        moved = dict(prior, center=prior['center']+displacement,
                     vertices=prior['vertices']+displacement,
                     top_vertices=prior['top_vertices']+displacement)
        tracker.surface_models[keys[1]] = moved
        frame, labels = primitive_scene(objects)
        points, valid = inventory_points(frame)
        self.assertEqual(len(depth_components(valid & inventory_color_mask(frame['rgb'], 'blue'), points)), 1)
        tracker.update(frame, {keys[1]: moved['center']})
        self.assertEqual({t['id'] for t in tracker.tracks.values() if t['visible']}, set(keys))
        for index, key in enumerate(keys):
            selected = tracker.tracks[key]['mask']
            self.assertGreater(np.count_nonzero(selected), 8)
            self.assertGreater(np.mean(labels[selected] == index), .99)
            isolated = tracker.isolated_frame(frame, key)
            self.assertEqual(np.count_nonzero(isolated['valid_depth']), np.count_nonzero(selected))
        self.assertFalse(np.any(tracker.tracks[keys[0]]['mask'] & tracker.tracks[keys[1]]['mask']))

    def test_ambiguous_overlapping_models_cannot_create_a_visible_third_object(self):
        tracker, objects, keys = learned_pair()
        # Two predictions overlap exactly. Pixels provide no evidence of which
        # identity is visible, even though a colored surface really is present.
        tracker.surface_models[keys[1]] = dict(tracker.surface_models[keys[0]])
        frame, _ = primitive_scene(objects[:1])
        tracker.update(frame)
        self.assertFalse(any(track['visible'] for track in tracker.tracks.values()))
        for key in keys:
            with self.assertRaisesRegex(ValueError, 'occluded'):
                tracker.isolated_frame(frame, key)

    def test_blank_depth_never_confirms_predicted_instances(self):
        tracker, objects, keys = learned_pair()
        frame, _ = primitive_scene(objects)
        frame['valid_depth'][:] = False
        frame['depth_m'][:] = np.nan
        tracker.update(frame, {key: tracker.surface_models[key]['center'] for key in keys})
        self.assertFalse(any(track['visible'] for track in tracker.tracks.values()))
        self.assertEqual(set(tracker.tracks), set(keys))

    def test_unexplained_same_color_object_remains_discoverable(self):
        tracker, objects, keys = learned_pair()
        objects.append({'shape': 'cube', 'center': np.array([.01, .075, .015]),
                        'dims': np.array([.03, .03]), 'height': .03})
        frame, labels = primitive_scene(objects)
        tracker.update(frame)
        unexplained = [t for t in tracker.tracks.values() if t['visible'] and t['id'] not in keys]
        self.assertEqual(len(unexplained), 1)
        self.assertGreater(np.mean(labels[unexplained[0]['mask']] == 2), .99)

    def test_box_surface_distance_respects_learned_rotation_and_axis_order(self):
        rotation = Rotation.from_euler('xyz', [.12, -.09, .6]).as_matrix()
        dimensions = np.array([.014, .032, .06])
        center = np.array([.02, -.03, .08])
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        vertices = center + (signs*dimensions/2) @ rotation.T
        model = {'shape': 'thin_prism', 'center': center, 'normal': rotation[:, 2],
                 'dims': dimensions[:2], 'height': dimensions[2],
                 'rotation': rotation[:, [2, 0, 1]], 'vertices': vertices}
        samples = center + np.array([[dimensions[0]/2, .008, .012], [0, -dimensions[1]/2, -.02]]) @ rotation.T
        np.testing.assert_allclose(column_surface_distance(samples, model), 0., atol=1e-12)
        legacy_model = {key: value for key, value in model.items() if key != 'rotation'}
        np.testing.assert_allclose(column_surface_distance(samples, legacy_model), 0., atol=1e-12)
        outside = samples[0]+.004*rotation[:, 0]
        self.assertAlmostEqual(float(column_surface_distance(outside[None], model)[0]), .004, places=10)


if __name__ == '__main__':
    unittest.main()
