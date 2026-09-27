"""CPU geometry for uniquely colored convex prisms with a visible planar cap.

The first observation assumes an upright extrusion resting on the measured table.
Later observations use current boundary evidence and an initialized shape,
never a cached pose. Partial boundaries require a separately validated registration. No simulator or object catalog is used.
"""
import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull


def prism_simplify_samples(polygon, tolerance):
    polygon = polygon.copy()
    while len(polygon) > 3:
        before, after = np.roll(polygon, 1, axis=0), np.roll(polygon, -1, axis=0)
        edge = after-before
        distance = abs(edge[:, 0]*(polygon-before)[:, 1]-edge[:, 1]*(polygon-before)[:, 0])/np.linalg.norm(edge, axis=1)
        index = int(np.argmin(distance))
        if distance[index] > tolerance:
            break
        polygon = np.delete(polygon, index, axis=0)
    return polygon



def prism_simplify(polygon, tolerance):
    """Remove raster-scale chamfers before fitting physical boundary lines."""
    polygon = prism_simplify_samples(polygon, tolerance)
    while len(polygon) > 3:
        before, after = np.roll(polygon, 1, axis=0), np.roll(polygon, -1, axis=0)
        incoming, outgoing = polygon-before, after-polygon
        cosine = np.sum(incoming*outgoing, axis=1)/(np.linalg.norm(incoming, axis=1)*np.linalg.norm(outgoing, axis=1))
        index = int(np.argmax(cosine))
        if cosine[index] > np.cos(np.deg2rad(18)):
            polygon = np.delete(polygon, index, axis=0)
            continue
        length = np.linalg.norm(outgoing, axis=1)
        index = int(np.argmin(length))
        if length[index] >= 5*tolerance:
            break
        next_index = (index+1)%len(polygon)
        matrix = np.column_stack((incoming[index], -outgoing[next_index]))
        if abs(np.linalg.det(matrix)) < 1e-10:
            break
        offset = np.linalg.solve(matrix, polygon[next_index]-polygon[index])[0]
        corner = polygon[index]+offset*incoming[index]
        if np.linalg.norm(corner-polygon[index]) > 3*tolerance:
            break
        polygon[index] = corner
        polygon = np.delete(polygon, next_index, axis=0)
    return polygon


def prism_contour_points(frame, cap, normal, level):
    """Project half-pixel mask transitions onto the fitted plane, not pixel centers."""
    row_x, col_x = np.where(cap[:, 1:] != cap[:, :-1])
    row_y, col_y = np.where(cap[1:, :] != cap[:-1, :])
    pixels = np.column_stack((np.r_[col_x+.5, col_y], np.r_[row_x, row_y+.5],
                              np.ones(len(row_x)+len(row_y))))
    pose = np.asarray(frame['camera_to_world'])
    rays = pixels@np.linalg.inv(frame['intrinsics']).T@pose[:3, :3].T
    denominator = rays@normal
    if np.any(abs(denominator) < .05):
        raise ValueError('Cap boundary rays are too oblique to resolve edges')
    scale = (level-pose[:3, 3]@normal)/denominator
    if np.any(scale <= 0):
        raise ValueError('Cap boundary projects behind the camera')
    return pose[:3, 3]+scale[:, None]*rays


def prism_fit_edges(boundary, polygon, pixel_size):
    """Fit straight contour segments and intersect neighboring lines in the cap plane."""
    starts, ends = polygon, np.roll(polygon, -1, axis=0)
    edges = ends-starts
    relative = boundary[:, None]-starts[None]
    fraction = np.sum(relative*edges[None], axis=2)/np.sum(edges**2, axis=1)
    closest = starts[None]+np.clip(fraction, 0, 1)[..., None]*edges[None]
    distance = np.linalg.norm(boundary[:, None]-closest, axis=2)
    assignment = np.argmin(distance, axis=1)
    normals, levels = [], []
    for i, edge in enumerate(edges):
        selected = (assignment == i) & (fraction[:, i] > .1) & (fraction[:, i] < .9)
        samples = boundary[selected]
        if len(samples) < 4 or np.ptp(fraction[selected, i]) < .45:
            raise ValueError('Insufficient contour samples to resolve polygon edges')
        center = np.mean(samples, axis=0)
        _, _, vh = np.linalg.svd(samples-center, full_matrices=False)
        normal = vh[-1]
        outward = np.array([edge[1], -edge[0]])/np.linalg.norm(edge)
        normal *= 1. if normal@outward >= 0 else -1.
        if normal@outward < np.cos(np.deg2rad(25)):
            raise ValueError('Contour does not support a stable straight edge')
        if np.sqrt(np.mean(((samples-center)@normal)**2)) > pixel_size*.65:
            raise ValueError('Contour edge is not sufficiently straight')
        normals.append(normal)
        # Half-pixel transitions remove pixel-center shrinkage. The true
        # subpixel edge still has sampling uncertainty, reported separately.
        levels.append(float(center@normal))
    normals, levels = np.asarray(normals), np.asarray(levels)
    corners = []
    for i in range(len(polygon)):
        matrix = normals[[i-1, i]]
        if abs(np.linalg.det(matrix)) < .15:
            raise ValueError('Adjacent contour edges do not resolve a vertex')
        corners.append(np.linalg.solve(matrix, levels[[i-1, i]]))
    corners = np.asarray(corners)
    if np.max(corners@normals.T-levels) > pixel_size*.25:
        raise ValueError('Fitted contour is not convex')
    sine = abs(normals[:, 0]*np.roll(normals[:, 1], 1)-normals[:, 1]*np.roll(normals[:, 0], 1))
    # Vertex position becomes less resolved when its two incident lines approach
    # parallel; propagate this conditioning instead of using a fixed pixel radius.
    if np.any(np.linalg.norm(corners-polygon, axis=1) > 3*pixel_size/sine):
        raise ValueError('Line intersections extrapolate beyond observed corners')
    return corners


def prism_cap_boundary(frame, points, valid, cap, normal, level, tolerance):
    """Reject image cuts, missing depth and nearer occluders on the outer cap edge."""
    filled = ndimage.binary_fill_holes(cap)
    ring = ndimage.binary_dilation(filled) & ~filled
    if np.any(filled[0]) or np.any(filled[-1]) or np.any(filled[:, 0]) or np.any(filled[:, -1]):
        raise ValueError('Polygon cap boundary is clipped by the image')
    if np.any(ring & ~valid):
        raise ValueError('Polygon cap boundary has missing or excluded depth')
    camera = np.asarray(frame['camera_to_world'])[:3, 3]
    if np.any((points[ring]@normal-level) > tolerance):
        raise ValueError('Polygon cap boundary is occluded by nearer geometry')
    if np.count_nonzero(cap) < .9*np.count_nonzero(filled):
        raise ValueError('Polygon cap has substantial missing interior evidence')
    if float(camera@normal-level) <= 0:
        raise ValueError('Polygon cap is not viewed from above')


def prism_vertex_topology(count):
    """Construct extrusion connectivity from a measured boundary of arbitrary N."""
    lower = np.arange(count)
    upper = lower+count
    edges = np.concatenate((np.column_stack((upper, np.roll(upper, -1))),
                            np.column_stack((lower, np.roll(lower, -1))),
                            np.column_stack((lower, upper))))
    faces = [upper.tolist(), lower[::-1].tolist()]
    faces += [[i, (i+1)%count, (i+1)%count+count, i+count] for i in range(count)]
    return edges, faces


def prism_cap_evidence(frame, color, table_plane):
    points, valid = inventory_points(frame)
    plane = np.asarray(table_plane, dtype=float)
    if plane.shape != (3,) or not np.isfinite(plane).all():
        raise ValueError('Polygon prism requires a finite table plane')
    elevation = points[..., 2]-points[..., :2]@plane[:2]-plane[2]
    mask = valid & inventory_color_mask(frame['rgb'], color) & (elevation > .004) & (elevation < .35)
    labels, count = ndimage.label(mask)
    components = [labels == i for i in range(1, count+1) if np.count_nonzero(labels == i) >= 20]
    if len(components) != 1:
        raise ValueError('Require one uniquely colored visible polygonal prism')
    mask = components[0]
    dx = points[1:-1, 2:]-points[1:-1, :-2]
    dy = points[2:, 1:-1]-points[:-2, 1:-1]
    normals = np.cross(dx, dy)
    length = np.linalg.norm(normals, axis=2)
    interior = (mask[1:-1, 1:-1] & mask[1:-1, 2:] & mask[1:-1, :-2] &
                mask[2:, 1:-1] & mask[:-2, 1:-1] & (length > 1e-10))
    samples = points[1:-1, 1:-1][interior]
    normals = normals[interior]/length[interior, None]
    normals *= np.where(normals[:, 2:3] >= 0, 1., -1.)
    upward = normals[:, 2] > .8
    samples, normals = samples[upward], normals[upward]
    if len(samples) < 20:
        raise ValueError('Insufficient planar top evidence for polygonal prism')
    samples = samples[::max(1, int(np.ceil(len(samples)/1500)))]
    mean = np.mean(samples, axis=0)
    _, singular, vh = np.linalg.svd(samples-mean, full_matrices=False)
    normal = vh[-1]*(1. if vh[-1, 2] >= 0 else -1.)
    rms = float(np.sqrt(np.mean(((samples-mean)@normal)**2)))
    if normal[2] < .8 or singular[1] < .002 or rms > .0006:
        raise ValueError('No reliable near-upright planar cap')
    level = float(mean@normal)
    cap = mask & (abs(points@normal-level) < max(.00015, 4*rms))
    origin = mean
    xaxis = np.array([1., 0., 0.])-normal[0]*normal
    xaxis /= np.linalg.norm(xaxis)
    basis = np.column_stack((xaxis, np.cross(normal, xaxis)))
    pixel_size = float(np.median(np.linalg.norm(dx[interior], axis=1))/2)
    return {'points': points, 'valid': valid, 'mask': mask, 'dx': dx, 'dy': dy,
            'interior': interior, 'normal': normal, 'rms': rms, 'level': level,
            'cap': cap, 'origin': origin, 'xaxis': xaxis, 'basis': basis,
            'pixel_size': pixel_size, 'plane': plane}


def prism_record(frame, model, evidence, polygon, registration=None):
    """Project a fitted or registered boundary into the existing policy record."""
    points, mask, cap = evidence['points'], evidence['mask'], evidence['cap']
    dx, dy, interior = evidence['dx'], evidence['dy'], evidence['interior']
    origin, basis, xaxis = evidence['origin'], evidence['basis'], evidence['xaxis']
    normal, rms, plane = evidence['normal'], evidence['rms'], evidence['plane']
    pixel_size = evidence['pixel_size']
    tolerance = max(.00025, 1.5*pixel_size)
    area, center2, covariance = point_polygon_moments(polygon)
    cap_center = origin+basis@center2
    # Compare the observed pixel area with the hull to reject visible concavity.
    top_interior = interior & cap[1:-1, 1:-1]
    areas = np.linalg.norm(np.cross(dx, dy), axis=2)[top_interior]/4
    pixel_area = float(np.median(areas))
    coverage = min(1., np.count_nonzero(cap)*pixel_area/area)
    if registration is None and coverage < .90:
        raise ValueError('Polygon cap is concave or lacks sufficient visible area')
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    identified = bool(eigenvalues[1] > 1.12*max(eigenvalues[0], 1e-12))
    direction = basis@eigenvectors[:, 1] if identified else xaxis
    if direction@xaxis < 0:
        direction = -direction
    rotation = np.column_stack((direction, np.cross(normal, direction), normal))
    top_vertices = origin+polygon@basis.T
    local_polygon = (top_vertices-cap_center)@rotation[:, :2]
    edge = np.roll(local_polygon, -1, axis=0)-local_polygon
    direction2 = edge/np.linalg.norm(edge, axis=1)[:, None]
    sine = abs(direction2[:, 0]*np.roll(direction2[:, 1], 1)-direction2[:, 1]*np.roll(direction2[:, 0], 1))
    vertex_resolution = float(pixel_size/np.min(sine))
    perimeter = float(np.linalg.norm(np.roll(polygon, -1, axis=0)-polygon, axis=1).sum())
    if model is None:
        table_normal = np.r_[-plane[:2], 1.]
        table_normal /= np.linalg.norm(table_normal)
        if normal@table_normal < np.cos(np.deg2rad(3)):
            raise ValueError('Initial polygon prism must rest upright on the table')
        height = float((cap_center[2]-cap_center[:2]@plane[:2]-plane[2])/(normal[2]-normal[:2]@plane[:2]))
        model = {'kind': 'visual_polygon_prism_v1', 'height': height, 'cap_area_m2': area,
                 'cap_perimeter_m': perimeter, 'footprint_local_m': local_polygon.copy(),
                 'boundary_resolution_m': pixel_size, 'vertex_resolution_m': vertex_resolution,
                 'assumption': 'convex_constant_cross_section_initially_on_table'}
    else:
        if model.get('kind') != 'visual_polygon_prism_v1':
            raise ValueError('Cannot mix polygon and legacy shape models')
        height = float(model['height'])
        previous = np.asarray(model['footprint_local_m'], dtype=float)
        if previous.shape != local_polygon.shape:
            raise ValueError('Current contour has a different number of resolved edges')
        # Pairwise distances are independent of the cap's in-plane rotation and
        # vertex starting index; area and perimeter alone do not identify shape.
        rows, cols = np.triu_indices(len(polygon), k=1)
        before = np.sort(np.linalg.norm(previous[rows]-previous[cols], axis=1))
        current = np.sort(np.linalg.norm(local_polygon[rows]-local_polygon[cols], axis=1))
        shape_tolerance = 2*max(vertex_resolution, float(model.get('vertex_resolution_m', model['boundary_resolution_m'])))
        if np.max(abs(before-current)) > shape_tolerance:
            raise ValueError('Current contour differs from the initialized polygon')
        if abs(area-model['cap_area_m2']) > max(.08*model['cap_area_m2'], perimeter*tolerance):
            raise ValueError('Current cap area differs from the initialized shape')
        if abs(perimeter-model['cap_perimeter_m']) > .08*model['cap_perimeter_m']:
            raise ValueError('Current cap perimeter differs from the initialized shape')
    if not np.isfinite(height) or not .006 <= height <= .15:
        raise ValueError('Polygon prism height is outside the supported block range')
    cloud = points[mask]
    axial = (cloud-cap_center)@normal
    if np.any(axial < -height-.002):
        raise ValueError('Visible depth contradicts the prism height')
    footprint = ConvexHull(local_polygon)
    side = axial < -.002
    distances = ((cloud-cap_center)@rotation[:, :2])@footprint.equations[:, :2].T+footprint.equations[:, 2]
    # Combine contour sampling resolution with the measured plane residual.
    surface_tolerance = max(.001, 1.5*pixel_size)+3*rms
    if np.any(np.max(distances, axis=1) > surface_tolerance):
        raise ValueError('Visible surfaces extend outside the polygonal footprint')
    if np.count_nonzero(side) > 10 and np.mean(np.min(abs(distances[side]), axis=1) > surface_tolerance) > .1:
        raise ValueError('Visible side surfaces contradict a constant cross section')
    bottom_vertices = top_vertices-normal*height
    vertices = np.concatenate((bottom_vertices, top_vertices))
    edge_indices, face_indices = prism_vertex_topology(len(top_vertices))
    edges = vertices[edge_indices]
    center = cap_center-normal*height/2
    dimensions = np.r_[np.ptp(local_polygon, axis=0), height]
    result = {'shape': 'polygon_prism', 'center': center, 'rotation': rotation,
            'axis_dimensions': dimensions, 'dims': dimensions[:2], 'height': height, 'normal': normal,
            'top': float(vertices[:, 2].max()), 'bottom': float(vertices[:, 2].min()),
            'extent_z': float(center[2]-vertices[:, 2].min()), 'vertices': vertices,
            'vertices_kind': 'fitted_polygonal_cap_boundary_and_inferred_parallel_bottom',
            'top_polygon_m': top_vertices, 'bottom_polygon_m': bottom_vertices,
            'edge_vertex_indices': edge_indices, 'face_vertex_indices': face_indices,
            'vertex_sources': (['inferred_parallel_bottom_under_prism_model']*len(bottom_vertices)+
                               ['fitted_from_current_rgbd_contour']*len(top_vertices)),
            'vertex_id_scope': 'current_observation_only_not_material_point_tracking',
            'edges_m': edges, 'boundary_resolution_m': pixel_size, 'vertex_resolution_m': vertex_resolution,
            'edges_kind': 'fitted_top_edges_and_inferred_bottom_and_vertical_edges',
            'inplane_axis_identified': identified, 'model': model,
            'constrained_axes': np.ones(3, dtype=bool), 'position_fully_observed': True,
            'inferred_orientation': False, 'orientation_semantics': 'current_geometric_frame_not_material_tracking',
            'geometry_reliable': True, 'observed_face_count': 1, 'inlier_fraction': coverage,
            'plane_rms_m': rms, 'coordinate_frame': 'world', 'units': 'meters',
            'observation_time': float(frame['time']), 'source': 'current_rgbd_polygon_prism_fit',
            'observability_condition': 'complete_current_cap_boundary_and_valid_initialized_extrusion_model'}

    if registration is not None:
        result.update(registration)
        result['pose_update_mode'] = 'initialized_shape_registered_to_current_partial_edges'
        result['vertices_kind'] = 'initialized_vertices_registered_to_current_edges_and_inferred_bottom'
        result['vertex_sources'] = (['inferred_parallel_bottom_under_prism_model']*len(bottom_vertices)+
                                    ['inferred_initialized_vertex_registered_to_current_edges']*len(top_vertices))
        result['edges_kind'] = 'initialized_connectivity_registered_to_current_partial_edges'
        result['observability_condition'] = 'current_cap_plane_and_two_independent_edges_with_unique_depth_consistent_registration'
        result['source'] = 'current_rgbd_partial_polygon_registration'
    return result


def observe_polygon_prism(frame, color, model, table_plane):
    evidence = prism_cap_evidence(frame, color, table_plane)
    points, valid, cap = evidence['points'], evidence['valid'], evidence['cap']
    normal, level = evidence['normal'], evidence['level']
    origin, basis, pixel_size = evidence['origin'], evidence['basis'], evidence['pixel_size']
    try:
        prism_cap_boundary(frame, points, valid, cap, normal, level, .0015)
    except ValueError as error:
        recoverable = ('Polygon cap boundary has missing or excluded depth',
                       'Polygon cap boundary is occluded by nearer geometry',
                       'Polygon cap has substantial missing interior evidence')
        if model is None or str(error) not in recoverable:
            raise
        polygon, registration = prism_register_partial_boundary(frame, model, evidence)
        return prism_record(frame, model, evidence, polygon, registration)
    coordinates = (points[cap]-origin)@basis
    boundary = coordinates[ConvexHull(coordinates).vertices]
    tolerance = max(.00025, 1.5*pixel_size)
    polygon = prism_simplify(boundary, tolerance)
    if not 3 <= len(polygon) <= 12:
        raise ValueError('Cap is not a simple polygon with 3 to 12 sides')
    contour_points = (prism_contour_points(frame, cap, normal, level)-origin)@basis
    polygon = prism_fit_edges(contour_points, polygon, pixel_size)
    return prism_record(frame, model, evidence, polygon)


def polygon_prism_face(fit, top=True):
    value = vertex_geometry_reference(fit, 'face_0' if top else 'face_1')
    value['semantics'] = ('model_registered_polygonal_top_with_occluded_boundary' if fit.get('pose_update_mode') else
                          'observed_polygonal_top') if top else 'inferred_parallel_bottom_under_prism_model'
    return value


def polygon_prism_reference(fit, reference, role):
    if role.startswith(('vertex_', 'edge_', 'face_')):
        value = vertex_geometry_reference(fit, role)
    elif role == 'center':
        value = {'type': 'point', 'position_m': fit['center'],
                 'semantics': 'centroid_under_initialized_extrusion_model'}
    elif role in ('top_face', 'bottom_face'):
        value = polygon_prism_face(fit, role == 'top_face')
    elif role in ('axis', 'long_axis', 'short_axis'):
        index = 2
        if role != 'axis':
            dimensions = np.asarray(fit['axis_dimensions'])
            index = int(np.argmax(dimensions) if role == 'long_axis' else np.argmin(dimensions))
            if np.count_nonzero(abs(dimensions-dimensions[index]) < .001) != 1 or (index != 2 and not fit['inplane_axis_identified']):
                raise ValueError('Polygon symmetry does not identify a unique '+role)
        value = {'type': 'unoriented_axis', 'origin_m': fit['center'], 'direction': fit['rotation'][:, index],
                 'semantics': 'extrusion_axis' if role == 'axis' else 'current_geometric_axis_not_material_tracking'}
    elif role.startswith('corner_'):
        raise ValueError('Polygon prisms do not expose indexed box corners')
    else:
        raise ValueError('Unknown geometric role: '+role)
    return metric_copy(dict(value, reference=reference, coordinate_frame='world', units='meters',
                            observation_time=fit['observation_time'], source=fit['source'],
                            position_fully_observed=fit['position_fully_observed'],
                            inferred_orientation=fit['inferred_orientation'],
                            observability_condition=fit['observability_condition']))


def validate_polygon_prism(fit):
    top = np.asarray(fit['top_polygon_m'], float)
    bottom = np.asarray(fit['bottom_polygon_m'], float)
    normal = np.asarray(fit['normal'], float)
    if top.ndim != 2 or top.shape[1:] != (3,) or not 3 <= len(top) <= 12 or bottom.shape != top.shape:
        raise ValueError('Invalid polygonal cap vertices')
    if not np.isfinite(top).all() or not np.isfinite(bottom).all():
        raise ValueError('Nonfinite polygonal cap vertices')
    if not np.allclose(top-bottom, normal*fit['height'], atol=1e-6):
        raise ValueError('Polygon cap planes disagree with the extrusion model')
    if not np.allclose(fit['vertices'], np.concatenate((bottom, top)), atol=1e-6):
        raise ValueError('Polygon vertices disagree with face geometry')
    for group in fit['edge_vertex_indices']:
        point_reference_indices(group, 2*len(top), 'edge')
    for group in fit['face_vertex_indices']:
        point_reference_indices(group, 2*len(top), 'face')
    if fit['vertex_id_scope'] != 'current_observation_only_not_material_point_tracking':
        raise ValueError('Polygon vertex indices must be observation-local')
    expected_edges, expected_faces = prism_vertex_topology(len(top))
    if not np.array_equal(fit['edge_vertex_indices'], expected_edges) or fit['face_vertex_indices'] != expected_faces:
        raise ValueError('Polygon connectivity disagrees with its boundary')
    if fit.get('pose_update_mode'):
        if fit['pose_update_mode'] != 'initialized_shape_registered_to_current_partial_edges':
            raise ValueError('Unknown polygon pose update mode')
        evidence = np.asarray([fit[key] for key in ('registration_supported_edges', 'registration_edge_rms_m',
                              'registration_condition', 'registration_candidate_count', 'registration_center_spread_m')], float)
        pixel = float(fit['boundary_resolution_m'])
        if (not np.isfinite(evidence).all() or not np.isfinite(pixel) or pixel <= 0 or
                evidence[0] < 2 or not 0 <= evidence[1] <= .85*pixel or not 1 <= evidence[2] <= 1/.12 or
                evidence[3] < 1 or not 0 <= evidence[4] <= max(.002, 1.5*pixel)):
            raise ValueError('Partial polygon pose lacks bounded current registration evidence')
    top_source = ('inferred_initialized_vertex_registered_to_current_edges' if fit.get('pose_update_mode') else
                  'fitted_from_current_rgbd_contour')
    expected_sources = ['inferred_parallel_bottom_under_prism_model']*len(bottom)+[top_source]*len(top)
    if fit['vertex_sources'] != expected_sources or not np.allclose(fit['edges_m'], np.asarray(fit['vertices'])[expected_edges]):
        raise ValueError('Polygon point evidence or edge projection is inconsistent')
    center = np.asarray(fit['center'], float)
    if np.max(abs((top-center)@normal-fit['height']/2)) > .001:
        raise ValueError('Polygon top is not a plane at the declared height')
    rotation = np.asarray(fit['rotation'], float)
    if not np.allclose(normal, rotation[:, 2], atol=1e-6):
        raise ValueError('Polygon frame disagrees with its cap normal')
    polygon = (top-center)@rotation[:, :2]
    edge = np.roll(polygon, -1, axis=0)-polygon
    next_edge = np.roll(edge, -1, axis=0)
    if np.any(edge[:, 0]*next_edge[:, 1]-edge[:, 1]*next_edge[:, 0] <= 0):
        raise ValueError('Polygon vertices must form a convex counterclockwise boundary')
    _, centroid, _ = point_polygon_moments(polygon)
    if np.linalg.norm(centroid) > 1e-6:
        raise ValueError('Polygon center must be its area centroid under the prism model')
    expected = np.r_[np.ptp(polygon, axis=0), fit['height']]
    if not np.allclose(fit['axis_dimensions'], expected, atol=1e-6):
        raise ValueError('Polygon dimensions disagree with the fitted boundary')
    vertices = np.concatenate((bottom, top))
    if not np.allclose([fit['bottom'], fit['top'], fit['extent_z']],
                       [vertices[:, 2].min(), vertices[:, 2].max(), center[2]-vertices[:, 2].min()], atol=1e-6):
        raise ValueError('Polygon vertical extents disagree with the fitted vertices')

