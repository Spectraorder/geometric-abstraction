"""Visible gripper surface geometry from the side depth image.

This is a partial observation, not a complete collision model: hidden finger
surfaces cannot be certified clear. Coordinates use the calibrated camera frame
and the robot's reported tool position; no simulator geometry is required.
"""
import numpy as np


def estimate_gripper_geometry(frame, tool_position, support_z=None):
    depth = np.asarray(frame['depth_m'], dtype=float)
    valid = np.asarray(frame['valid_depth'], dtype=bool) & np.isfinite(depth) & (depth > 0)
    rows, cols = np.indices(depth.shape)
    rays = np.stack((cols, rows, np.ones_like(cols)), axis=-1) @ np.linalg.inv(frame['intrinsics']).T
    pose = np.asarray(frame['camera_to_world'])
    cloud = (rays * np.where(valid, depth, 0)[..., None]) @ pose[:3, :3].T + pose[:3, 3]
    rgb = np.asarray(frame['rgb'], dtype=float)
    neutral = (np.max(rgb, axis=2)-np.min(rgb, axis=2) < 35) & (np.mean(rgb, axis=2) > 65)
    relative = cloud-np.asarray(tool_position)
    mask = valid & neutral & (np.linalg.norm(relative[..., :2], axis=2) < .060)
    mask &= (relative[..., 2] > -.050) & (relative[..., 2] < .075)
    points = cloud[mask]
    if len(points) < 12:
        raise ValueError('Too little visible gripper depth')
    lowest = float(np.percentile(points[:, 2], 1))
    tips = points[points[:, 2] <= lowest+.012]
    opening = None
    jaw_centers = None
    if len(tips) >= 8:
        xy = tips[:, :2]
        _, _, axes = np.linalg.svd(xy-np.mean(xy, axis=0), full_matrices=False)
        coordinates = xy @ axes[0]
        order = np.argsort(coordinates)
        gaps = np.diff(coordinates[order])
        # Require at least three observed points on each jaw.
        gaps[:2] = 0
        gaps[-2:] = 0
        cut = int(np.argmax(gaps))
        if gaps[cut] > .004:
            left, right = tips[order[:cut+1]], tips[order[cut+1:]]
            jaw_centers = np.array([np.mean(left, axis=0), np.mean(right, axis=0)])
            opening = float(gaps[cut])
    return {'visible_points': points, 'tip_points': tips,
            'lowest_visible_z': lowest,
            'fingertip_drop_m': float(tool_position[2]-lowest),
            'observed_opening_m': opening, 'jaw_centers': jaw_centers,
            'visible_clearance_m': None if support_z is None else lowest-float(support_z),
            'visible_only': True, 'pixel_count': len(points)}
