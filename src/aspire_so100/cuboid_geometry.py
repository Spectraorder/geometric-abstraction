"""Visual rectangular-prism geometry for initially upright, axis-aligned objects."""
import numpy as np
from scipy import ndimage
from aspire_so100.cube_geometry import rgbd_points, color_mask as cube_color_mask


def color_mask(rgb, color):
    r, g, b = np.asarray(rgb, dtype=float).transpose(2, 0, 1)
    if color == 'red':
        # Orange remains red-dominant in shadow; its green/blue ratio differs.
        return cube_color_mask(rgb, color) & (g < 1.5*b+10)
    if color == 'purple':
        return (r > 80) & (b > 90) & (r > 1.5*g) & (b > 1.5*g)
    if color == 'orange':
        return (r > 100) & (g > 50) & (r > 1.35*g) & (g > 2.6*b)
    if color == 'cyan':
        return (g > 70) & (b > 70) & (g > 1.4*r) & (b > 1.4*r) & (g > .75*b)
    return cube_color_mask(rgb, color)


def estimate_cuboid_geometry(frame, color, dimensions_m, near=None, table_plane=None):
    """Fit an upright cuboid from calibrated depth and visually measured dimensions.

    dimensions_m describes its initial world-aligned X/Y/Z extents. The scene
    starts with zero yaw; the nearest world axes assign fitted faces to these
    extents during small rotations. Arbitrary large rotations are not supported.
    Hidden corners are inferred; no simulator dimensions or poses are used.
    """
    dimensions_m = np.asarray(dimensions_m, float)
    if dimensions_m.shape != (3,) or not np.isfinite(dimensions_m).all() or np.any(dimensions_m <= 0):
        raise ValueError('Invalid visual cuboid dimensions')
    points, valid = rgbd_points(frame)
    if table_plane is None:
        r, g, b = np.asarray(frame['rgb'], dtype=float).transpose(2, 0, 1)
        table = valid & (r > 95) & (g > 65) & (r > 1.15*g) & (r < 1.65*g) & (g > 1.35*b)
        sample = points[table][::max(1, int(np.count_nonzero(table)) // 3000)]
        if len(sample) < 50:
            raise ValueError('Insufficient tabletop depth')
        design = np.column_stack((sample[:, :2], np.ones(len(sample))))
        keep = abs(sample[:, 2] - np.median(sample[:, 2])) < .025
        for _ in range(4):
            if np.count_nonzero(keep) < 30:
                raise ValueError('Unreliable tabletop plane')
            table_plane = np.linalg.lstsq(design[keep], sample[keep, 2], rcond=None)[0]
            keep = abs(sample[:, 2] - design @ table_plane) < .003
    height = points[..., 2] - points[..., :2] @ np.asarray(table_plane)[:2] - table_plane[2]
    labels, count = ndimage.label(color_mask(frame['rgb'], color) & valid & (height > .004) & (height < .35))
    candidates = []
    for label in range(1, count + 1):
        mask = labels == label
        sample = points[mask]
        if len(sample) < 12:
            continue
        distance = 0.0 if near is None else float(np.linalg.norm(np.median(sample, axis=0) - near))
        if distance < .12:
            candidates.append((len(sample) / (1 + 60 * distance), mask))
    if not candidates:
        raise ValueError('No usable visible cuboid: ' + color)
    mask = max(candidates, key=lambda item: item[0])[1]
    cloud = points[mask]
    dx = points[1:-1, 2:] - points[1:-1, :-2]
    dy = points[2:, 1:-1] - points[:-2, 1:-1]
    normals = np.cross(dx, dy)
    length = np.linalg.norm(normals, axis=2)
    interior = (mask[1:-1, 1:-1] & mask[1:-1, 2:] & mask[1:-1, :-2]
                & mask[2:, 1:-1] & mask[:-2, 1:-1] & (length > 1e-10))
    samples = points[1:-1, 1:-1][interior]
    normals = normals[interior] / length[interior, None]
    # Bound fitting work even for a close-up image or a spurious large mask.
    stride = max(1, int(np.ceil(len(samples) / 500)))
    samples, normals = samples[::stride], normals[::stride]
    if len(samples) < 6:
        raise ValueError('Too few interior depth samples: ' + color)
    camera = np.asarray(frame['camera_to_world'])[:3, 3]
    normals *= np.where(np.sum(normals * (camera - samples), axis=1) >= 0, 1, -1)[:, None]

    # Dominant normal clusters, followed by a plane fit. Silhouette/depth-edge
    # pixels are excluded above; blended normals at cuboid edges are outliers.
    faces = []
    available = np.ones(len(samples), dtype=bool)
    for _ in range(6):
        indices = np.flatnonzero(available)
        if len(indices) < 4:
            break
        directions = normals[indices]
        votes = directions @ directions.T > np.cos(np.deg2rad(12))
        seed = int(np.argmax(np.sum(votes, axis=1)))
        members = indices[votes[seed]]
        available[members] = False
        if len(members) < 4:
            continue
        face = samples[members]
        center = np.mean(face, axis=0)
        _, singular, vh = np.linalg.svd(face - center, full_matrices=False)
        if singular[1] < .0005:
            continue
        normal = vh[-1]
        if np.dot(normal, camera - center) < 0:
            normal = -normal
        rms = float(np.sqrt(np.mean(((face - center) @ normal) ** 2)))
        if rms > .0008:
            continue
        if any(abs(np.dot(normal, f['normal'])) > .3 for f in faces):
            continue
        faces.append({'normal': normal, 'point': center, 'samples': len(face), 'rms_m': rms})
        if len(faces) == 3:
            break
    if not faces:
        raise ValueError('No reliable face planes: ' + color)

    axes = [f['normal'] for f in faces]
    if len(axes) == 1:
        seed_axis = np.eye(3)[int(np.argmin(np.abs(axes[0])))]
        axis = seed_axis - np.dot(seed_axis, axes[0]) * axes[0]
        axes.append(axis / np.linalg.norm(axis))
    if len(axes) == 2:
        axes.append(np.cross(axes[0], axes[1]))
    basis = np.column_stack(axes)
    u, _, vh = np.linalg.svd(basis)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1
    sizes = dimensions_m[np.argmax(np.abs(rotation), axis=0)]
    local = cloud @ rotation
    center_local = (np.percentile(local, 1, axis=0) + np.percentile(local, 99, axis=0)) / 2
    constrained = np.zeros(3, dtype=bool)
    plane_rms = []
    for face in faces:
        axis = int(np.argmax(np.abs(rotation.T @ face['normal'])))
        sign = float(np.sign(np.dot(rotation[:, axis], face['normal'])))
        aligned = normals @ (sign * rotation[:, axis]) > np.cos(np.deg2rad(15))
        if np.count_nonzero(aligned) < 3:
            raise ValueError('Occluded face has insufficient consistent depth samples: ' + color)
        level = float(np.median(samples[aligned] @ rotation[:, axis]))
        center_local[axis] = level - sign * sizes[axis] / 2
        constrained[axis] = True
        plane_rms.append(float(np.sqrt(np.mean((samples[aligned] @ rotation[:, axis] - level) ** 2))))
    centroid = rotation @ center_local
    signs = np.array([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
    vertices = centroid + (signs * sizes / 2) @ rotation.T
    residual = np.min(np.abs(np.abs((cloud - centroid) @ rotation) - sizes / 2), axis=1)
    inside = np.all(np.abs((cloud - centroid) @ rotation) <= sizes / 2 + .002, axis=1)
    inlier_fraction = float(np.mean((residual < .0015) & inside))
    reliable = bool(np.all(constrained) and max(plane_rms) < .0008 and inlier_fraction > .8)
    up_axis = int(np.argmax(np.abs(rotation[2])))
    up = rotation[:, up_axis] * np.sign(rotation[2, up_axis])
    top_center = centroid + up * sizes[up_axis] / 2
    return {
        'centroid': centroid, 'surface_centroid': np.mean(cloud, axis=0),
        'vertices': vertices, 'vertices_kind': 'inferred_cuboid_model',
        'rotation': rotation, 'dimensions_m': sizes,
        'top_center': top_center, 'top_normal': up,
        'top_vertices': vertices[signs[:, up_axis] == np.sign(rotation[2, up_axis])],
        'bottom_z': float(np.min(vertices[:, 2])), 'top_z': float(np.max(vertices[:, 2])),
        'observed_face_count': len(faces), 'constrained_axes': constrained,
        'plane_rms_m': float(max(plane_rms)), 'inlier_fraction': inlier_fraction,
        'reliable': reliable, 'pixel_count': len(cloud),
    }
