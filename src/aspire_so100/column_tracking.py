"""Separate touching same-color instances using previously observed surfaces."""
import numpy as np

from aspire_so100.instance_geometry import InstanceTracker, inventory_points, inventory_color_mask


def column_surface_distance(points, model):
    """Unsigned distance to the learned closed primitive, without scene truth."""
    normal = np.asarray(model['normal'], float)
    normal = normal / np.linalg.norm(normal)
    ex = np.array([1., 0., 0.]) - normal * normal[0]
    ex /= np.linalg.norm(ex)
    rotation = np.column_stack((ex, np.cross(normal, ex), normal))
    if model['shape'] != 'cylinder':
        if 'rotation' in model:
            rotation = np.asarray(model['rotation'])
        elif 'vertices' in model:
            # Older visual fits retain corners without a full rotation matrix.
            # Three mutually perpendicular edges recover that same box basis.
            vertices = np.asarray(model['vertices'])
            edges = vertices[1:] - vertices[0]
            lengths = np.linalg.norm(edges, axis=1)
            axes = []
            for index in np.argsort(lengths):
                if lengths[index] <= 1e-9:
                    continue
                direction = edges[index] / lengths[index]
                if all(abs(np.dot(direction, axis)) < 1e-4 for axis in axes):
                    axes.append(direction)
                if len(axes) == 3:
                    rotation = np.column_stack(axes)
                    break
    local = (points - model['center']) @ rotation
    if model['shape'] == 'cylinder':
        radial = np.linalg.norm(local[..., :2], axis=-1) - min(model['dims']) / 2
        axial = abs(local[..., 2]) - model['height'] / 2
        q = np.stack((radial, axial), axis=-1)
    else:
        if 'vertices' in model:
            # Fitted axes need not retain the initial XYZ ordering.
            half = np.max(abs((np.asarray(model['vertices']) - model['center']) @ rotation), axis=0)
        else:
            half = np.r_[model['dims'], model['height']] / 2
        q = abs(local) - half
    return abs(np.linalg.norm(np.maximum(q, 0.), axis=-1) + np.minimum(np.max(q, axis=-1), 0.))


class ColumnTracker(InstanceTracker):
    """Keep geometric IDs when color regions merge during a tower placement.

    Surface models come from earlier RGB-D fits, with held poses propagated by
    robot motion. Only visible color/depth pixels can confirm a track; model
    overlap and unexplained pixels remain unassigned.
    """
    def __init__(self, plane=None):
        self.plane = plane
        self.tracks = {}
        self.next_id = 0
        self.surface_models = {}

    def update(self, frame, predictions=None):
        previous = {key: dict(track) for key, track in self.tracks.items()}
        super().update(frame, predictions)
        points, valid = inventory_points(frame)
        for color in ('red', 'yellow', 'blue', 'green'):
            keys = [key for key in self.surface_models if key in previous and previous[key]['color'] == color]
            if len(keys) < 2:
                continue
            mask = valid & inventory_color_mask(frame['rgb'], color)
            cloud = points[mask]
            if not len(cloud):
                continue
            distances = np.column_stack([column_surface_distance(cloud, self.surface_models[key]) for key in keys])
            nearest = np.argmin(distances, axis=1)
            ordered = np.sort(distances, axis=1)
            certain = (ordered[:, 0] < .006) & (ordered[:, 1] - ordered[:, 0] > .001)
            explained = np.zeros(mask.shape, bool)
            explained[mask] = ordered[:, 0] < .006
            for index, key in enumerate(keys):
                pixels = np.zeros(mask.shape, bool)
                pixels[mask] = certain & (nearest == index)
                count = int(np.count_nonzero(pixels))
                track = dict(previous[key], mask=pixels, visible=count >= 8, pixel_count=count)
                if count >= 8:
                    sample = points[pixels]
                    track['observed_center'] = np.median(sample, axis=0)
                    track['center'] = np.asarray(self.surface_models[key]['center']).copy()
                self.tracks[key] = track
            # Ambiguous overlap still belongs to modeled geometry, rather
            # than a newly discovered third object. Retain only unexplained
            # pixels on other tracks, preserving separate unknown objects.
            for key in list(self.tracks):
                if key in keys or self.tracks[key]['color'] != color:
                    continue
                track = self.tracks[key]
                pixels = track['mask'] & ~explained
                count = int(np.count_nonzero(pixels))
                if count < 8 and key not in previous:
                    del self.tracks[key]
                else:
                    track.update(mask=pixels, visible=bool(track['visible'] and count >= 8), pixel_count=count)
                    if track['visible']:
                        track['observed_center'] = np.median(points[pixels], axis=0)
        return [track for track in self.tracks.values() if track['visible']]
