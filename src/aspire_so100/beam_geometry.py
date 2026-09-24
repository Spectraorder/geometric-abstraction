"""Visual identification and orientation checks for tabletop beam sources.

Dimensions are learned from RGB-D. Ratios describe shape hypotheses, not
metric catalog dimensions. Flat sources use measured world X/Y/Z extents.
"""
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from aspire_so100.instance_geometry import inventory_points, inventory_color_mask
from aspire_so100.column_geometry import select_column_levels, column_size_uncertainty


def flat_beam_aligned(model, tolerance_degrees=8.):
    """Check the measured thin and long physical axes, allowing face symmetry."""
    if 'rotation' not in model or 'axis_dimensions' not in model:
        return False
    dimensions = np.asarray(model['axis_dimensions'])
    rotation = np.asarray(model['rotation'])
    shortest, longest = int(np.argmin(dimensions)), int(np.argmax(dimensions))
    threshold = np.cos(np.deg2rad(tolerance_degrees))
    return bool(abs(rotation[2, shortest]) >= threshold and abs(rotation[0, longest]) >= threshold)


def select_structure_nodes(tracker, nodes, frame):
    """Resolve flat thin prisms from measured shape evidence before association."""
    for node in nodes:
        if node['orientation'] != 'flat_x':
            continue
        if node['shape'] != 'thin_prism':
            raise ValueError('Flat sources currently require a thin prism')
        matches = []
        for key, track in tracker.tracks.items():
            if not track['visible'] or track['color'] != node['color']:
                continue
            dimensions = np.asarray(track['dimensions'])
            ratios = dimensions / max(float(dimensions[0]), 1e-9)
            features = track.get('features', {})
            if (not .36 < ratios[1] < .68 or not .18 < ratios[2] < .34
                    or features.get('sloped_fraction', 1.) > .2
                    or features.get('normal_samples', 0) < 10):
                continue
            try:
                fit = estimate_flat_beam_geometry(tracker.isolated_frame(frame, key),
                                                node['color'], tracker.plane)
            except ValueError:
                continue
            # A thin slab may expose only its broad top plane. The full finite
            # silhouette and table-relative thickness still constrain its shape.
            if flat_beam_aligned(fit) and fit.get('observed_face_count', 0) >= 1:
                matches.append(key)
        if len(matches) != 1:
            raise ValueError('No unique visually supported flat '+node['color']+' thin prism')
        tracker.tracks[matches[0]]['shape'] = 'thin_prism'
    return select_column_levels(tracker, nodes, frame)


def estimate_flat_beam_geometry(frame, color, table_plane, model=None):
    """Initialize or refresh a flat slab from its complete visible top boundary.

    The thin vertical sides can be less than one pixel high. A rectangular top
    boundary supplies finite XY constraints without inventing side-plane normals.
    This initialization assumes small yaw, verified by corner coverage; later
    supported poses retain learned thickness. Held poses use the ordinary rigid
    model and measured surfaces.
    """
    points, valid = inventory_points(frame)
    selected = valid & inventory_color_mask(frame['rgb'], color)
    cloud = points[selected]
    if len(cloud) < 30:
        raise ValueError('Too little flat beam surface evidence')
    heights = cloud[:, 2]-cloud[:, :2] @ table_plane[:2]-table_plane[2]
    top_height = float(np.percentile(heights, 95))
    top = cloud[abs(heights-top_height) < .0005]
    if len(top) < 25 or not .009 < top_height < .15:
        raise ValueError('No broad planar tabletop beam face')
    mean = np.mean(top, axis=0)
    _, _, axes = np.linalg.svd(top-mean, full_matrices=False)
    normal = axes[-1]
    if normal[2] < 0:
        normal = -normal
    if normal[2] < np.cos(np.deg2rad(3)):
        raise ValueError('Beam source is not visibly flat')
    ex = np.array([1., 0., 0.])-normal*normal[0]
    ex /= np.linalg.norm(ex)
    ey = np.cross(normal, ex)
    rotation = np.column_stack((ex, ey, normal))
    uv = top @ rotation[:, :2]
    low, high = np.min(uv, axis=0), np.max(uv, axis=0)
    dims = high-low
    physical_height = top_height if model is None else model['height']
    if not .36 < dims[1]/dims[0] < .68 or not .18 < physical_height/dims[0] < .34:
        raise ValueError('Top silhouette is not a flat thin prism')
    try:
        hull = ConvexHull(uv)
    except QhullError as error:
        raise ValueError('Beam top boundary is degenerate') from error
    if hull.volume/np.prod(dims) < .9:
        raise ValueError('Beam top boundary is clipped or rotated')
    # All four corners must have pixel support. A single visible plane with
    # an occluded end cannot establish a complete physical length.
    corners = np.array([[x, y] for x in (low[0], high[0]) for y in (low[1], high[1])])
    pixel = float(np.max(dims)/np.sqrt(len(top)))
    if np.max(np.min(np.linalg.norm(corners[:, None]-uv[None], axis=2), axis=1)) > 1.5*pixel:
        raise ValueError('Beam top corners are not all observed')
    top_center = rotation[:, :2] @ ((low+high)/2)+normal*float(np.median(top @ normal))
    if model is None:
        height = float(top_center[2]-top_center[:2] @ table_plane[:2]-table_plane[2])
        model = {'height': height, 'dims': dims}
        model.update(column_size_uncertainty(frame, color, model, table_plane))
    else:
        height = float(model['height'])
        if np.max(abs(dims-np.asarray(model['dims']))) > 2*pixel:
            raise ValueError('Top boundary contradicts the learned beam dimensions')
        dims = np.asarray(model['dims'])
    center = top_center-normal*height/2
    sizes = np.r_[dims, height]
    signs = np.array([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
    vertices = center+(signs*sizes/2) @ rotation.T
    return {'shape': 'thin_prism', 'center': center, 'normal': normal, 'height': height,
            'dims': dims, 'model': model, 'rotation': rotation, 'axis_dimensions': sizes,
            'axis_dimension_indices': np.arange(3), 'vertices': vertices,
            'top_vertices': vertices[signs[:, 2] > 0], 'top': float(top_center[2]),
            'bottom': float(vertices[:, 2].min()), 'extent_z': float(np.ptp(vertices[:, 2])/2),
            'observed_face_count': 1, 'constrained_axes': np.array([False, False, True]),
            'geometry_reliable': False, 'inferred_orientation': True,
            'surface_rms_m': float(np.sqrt(np.mean(((top-mean) @ normal)**2)))}
