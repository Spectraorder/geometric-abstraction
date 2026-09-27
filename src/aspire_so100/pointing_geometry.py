"""CPU validation of single-image Molmo 2 points against camera RGB-D.

A point selects visible evidence, not an object center, identity guarantee, or
grasp pose. This optional adapter is not bundled into the frozen policies.
"""
import re

import numpy as np

from aspire_so100.instance_geometry import (
    depth_components, inventory_color_mask, inventory_points, inventory_table,
)


def parse_molmo2_point(text, width, height):
    """Require one point in image 1 using Molmo 2's 0..1000 coordinate scale.

    Official format: <points coords="1 1 500 500">label</points>.
    Multiple points/images, legacy percentage coordinates, and prose abstain.
    Right/bottom image boundaries do not correspond to addressable pixels.
    """
    if width <= 0 or height <= 0:
        raise ValueError('Invalid image dimensions')
    if not isinstance(text, str):
        raise ValueError('Expected a text response')
    tags = re.findall(r'<points\b[^>]*\bcoords="([^"]*)"[^>]*>', text)
    if len(tags) != 1 or len(re.findall(r'<points?\b', text)) != 1:
        raise ValueError('Expected exactly one Molmo 2 points tag')
    match = re.fullmatch(r'1 +([1-9][0-9]*) +([0-9]{3,4}) +([0-9]{3,4})', tags[0])
    if match is None:
        raise ValueError('Expected exactly one point in image 1')
    x, y = int(match[2]), int(match[3])
    if not (0 <= x < 1000 and 0 <= y < 1000):
        raise ValueError('Point is outside addressable image bounds')
    return np.array([x * width / 1000., y * height / 1000.])


def locate_pointed_component(frame, point_px, color):
    """Select a color/depth component containing the pointed pixel, without snapping.

    Table-relative gates and connectivity reuse the existing CPU perception.
    Color/depth consistency cannot certify shape or distinguish merged objects;
    downstream geometry and identity checks remain necessary.
    """
    point = np.asarray(point_px, float)
    height, width = np.asarray(frame['depth_m']).shape
    if (point.shape != (2,) or not np.all(np.isfinite(point))
            or not (0 <= point[0] < width and 0 <= point[1] < height)):
        raise ValueError('Point is outside addressable image bounds')
    u, v = np.floor(point).astype(int)
    points, valid = inventory_points(frame)
    if not valid[v, u]:
        raise ValueError('Point has no valid depth')
    color_mask = inventory_color_mask(frame['rgb'], color)
    if not color_mask[v, u]:
        raise ValueError('Point does not match the requested color')
    plane = inventory_table(frame, points, valid)
    heights = points[..., 2] - points[..., :2] @ plane[:2] - plane[2]
    mask = color_mask & valid & (heights > .004) & (heights < .35)
    matches = [part for part in depth_components(mask, points) if part[v, u]]
    if len(matches) != 1:
        raise ValueError('Point has no supported object component')
    component = matches[0]
    return {
        'point_px': point,
        'pixel_index_uv': np.array([u, v]),
        'surface_world_m': points[v, u],
        'visible_surface_centroid_world_m': points[component].mean(axis=0),
        'component_pixels': int(component.sum()),
        'mask': component,
        'table_plane': plane,
        'complete_object_geometry': False,
        'identity_verified': False,
    }
