"""Cube geometry from calibrated RGB-D only; hidden corners are model estimates.

This module deliberately uses only numerical dependencies so it can be bundled
into the isolated policy worker without granting access to simulator state.
"""
import numpy as np
from scipy import ndimage


def rgbd_points(frame):
    depth = np.asarray(frame['depth_m'], dtype=float)
    valid = np.asarray(frame['valid_depth'], dtype=bool) & np.isfinite(depth) & (depth > 0)
    v, u = np.indices(depth.shape)
    rays = np.stack((u, v, np.ones_like(u)), axis=-1) @ np.linalg.inv(frame['intrinsics']).T
    pose = np.asarray(frame['camera_to_world'], dtype=float)
    points = (rays * np.where(valid, depth, 0)[..., None]) @ pose[:3, :3].T + pose[:3, 3]
    return points, valid


def color_mask(rgb, color):
    r, g, b = np.asarray(rgb, dtype=float).transpose(2, 0, 1)
    if color == 'red':
        return (r > 75) & (r > 1.65 * g) & (r > 1.55 * b)
    if color == 'green':
        return (g > 65) & (g > 1.45 * r) & (g > 1.3 * b)
    if color == 'blue':
        return (b > 65) & (b > 1.55 * r) & (b > 1.2 * g)
    raise ValueError('Expected red, green, or blue')


def support_margin(point, top_vertices, normal):
    """Signed distance to the nearest edge of an inferred square support face."""
    corners = np.asarray(top_vertices, dtype=float)
    center = np.mean(corners, axis=0)
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    x = corners[0] - center
    x = x / np.linalg.norm(x)
    y = np.cross(normal, x)
    uv = np.column_stack(((corners - center) @ x, (corners - center) @ y))
    order = np.argsort(np.arctan2(uv[:, 1], uv[:, 0]))
    corners = corners[order]
    edges = np.roll(corners, -1, axis=0) - corners
    inward = np.cross(normal, edges)
    inward /= np.linalg.norm(inward, axis=1)[:, None]
    return float(np.min(np.sum((np.asarray(point) - corners) * inward, axis=1)))


def estimate_cube_geometry(frame, color, side_m, near=None, table_plane=None):
    """Fit visible face planes and infer a solid cube's centroid and 8 vertices.

    side_m is a positive side length previously measured from an unobstructed
    RGB-D view. No object pose, segmentation IDs, or scene dimensions are used.
    Three perpendicular observed planes constrain all three center coordinates.
    With fewer faces, missing coordinates use visible extents and are explicitly
    marked unreliable under occlusion. Returned vertices are always inferred.
    """
    if not np.isfinite(side_m) or side_m <= 0:
        raise ValueError('A positive measured cube side is required')
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
        raise ValueError('No usable visible cube: ' + color)
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
    # pixels are excluded above; blended normals at cube edges are outliers.
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
        center_local[axis] = level - sign * side_m / 2
        constrained[axis] = True
        plane_rms.append(float(np.sqrt(np.mean((samples[aligned] @ rotation[:, axis] - level) ** 2))))
    centroid = rotation @ center_local
    signs = np.array([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
    vertices = centroid + (signs * side_m / 2) @ rotation.T
    residual = np.min(np.abs(np.abs((cloud - centroid) @ rotation) - side_m / 2), axis=1)
    inside = np.all(np.abs((cloud - centroid) @ rotation) <= side_m / 2 + .002, axis=1)
    inlier_fraction = float(np.mean((residual < .0015) & inside))
    reliable = bool(np.all(constrained) and max(plane_rms) < .0008 and inlier_fraction > .8)
    up_axis = int(np.argmax(np.abs(rotation[2])))
    up = rotation[:, up_axis] * np.sign(rotation[2, up_axis])
    top_center = centroid + up * side_m / 2
    return {
        'centroid': centroid, 'surface_centroid': np.mean(cloud, axis=0),
        'vertices': vertices, 'vertices_kind': 'inferred_cube_model',
        'rotation': rotation, 'side_m': float(side_m),
        'top_center': top_center, 'top_normal': up,
        'top_vertices': vertices[signs[:, up_axis] == np.sign(rotation[2, up_axis])],
        'bottom_z': float(np.min(vertices[:, 2])), 'top_z': float(np.max(vertices[:, 2])),
        'observed_face_count': len(faces), 'constrained_axes': constrained,
        'plane_rms_m': float(max(plane_rms)), 'inlier_fraction': inlier_fraction,
        'reliable': reliable, 'pixel_count': len(cloud),
    }
