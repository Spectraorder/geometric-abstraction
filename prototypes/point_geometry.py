"""Shape-independent references over measured points and declared connectivity.

This module does not detect surfaces or infer connectivity. The observer supplies
ordered face boundaries and a source label for each point. Indices are local to
one observation; they are not material-point correspondences across frames.
"""
import numpy as np


def point_polygon_moments(polygon):
    q = np.roll(polygon, -1, axis=0)
    cross = polygon[:, 0]*q[:, 1]-q[:, 0]*polygon[:, 1]
    area = float(np.sum(cross)/2)
    if area <= 1e-8:
        raise ValueError('Degenerate polygon')
    center = np.sum((polygon+q)*cross[:, None], axis=0)/(6*area)
    xx = np.sum((polygon[:, 0]**2+polygon[:, 0]*q[:, 0]+q[:, 0]**2)*cross)/(12*area)
    yy = np.sum((polygon[:, 1]**2+polygon[:, 1]*q[:, 1]+q[:, 1]**2)*cross)/(12*area)
    xy = np.sum((2*polygon[:, 0]*polygon[:, 1]+polygon[:, 0]*q[:, 1]+
                 q[:, 0]*polygon[:, 1]+2*q[:, 0]*q[:, 1])*cross)/(24*area)
    covariance = np.array([[xx, xy], [xy, yy]])-np.outer(center, center)
    return area, center, covariance


def point_reference_indices(group, count, kind):
    """Reject malformed connectivity before indexing or converting its dtype."""
    indices = np.asarray(group)
    if indices.ndim != 1 or indices.dtype.kind not in ('i', 'u'):
        raise ValueError('Connectivity must contain integer vertex indices')
    if (kind == 'vertex' and len(indices) != 1) or (kind == 'edge' and len(indices) != 2) or (kind == 'face' and len(indices) < 3):
        raise ValueError('Invalid vertex count for '+kind)
    if np.any(indices < 0) or np.any(indices >= count):
        raise ValueError('Connectivity index is out of range')
    if len(np.unique(indices)) != len(indices):
        raise ValueError('Connectivity repeats a vertex')
    return indices.astype(int)


def point_face_projection(selected):
    """Validate a simple planar boundary and return its oriented 2D projection."""
    differences = selected-selected[0]
    candidates = np.cross(differences[1:-1], differences[2:])
    normal = candidates[int(np.argmax(np.linalg.norm(candidates, axis=1)))]
    if np.linalg.norm(normal) < 1e-10:
        raise ValueError('Face vertices do not define a plane')
    normal /= np.linalg.norm(normal)
    if np.max(abs(differences@normal)) > 1e-6:
        raise ValueError('Face vertices are not coplanar')
    # Orient with the entire ordered boundary, including concave polygons.
    area_vector = np.sum(np.cross(differences, np.roll(differences, -1, axis=0)), axis=0)
    if np.linalg.norm(area_vector) < 1e-10:
        raise ValueError('Face boundary has zero signed area')
    normal *= 1. if normal@area_vector > 0 else -1.
    xaxis = differences[1]/np.linalg.norm(differences[1])
    basis = np.column_stack((xaxis, np.cross(normal, xaxis)))
    polygon = differences@basis
    ends = np.roll(polygon, -1, axis=0)
    for i in range(len(polygon)):
        for j in range(i+1, len(polygon)):
            if j == i+1 or (i == 0 and j == len(polygon)-1):
                continue
            a, b, c, d = polygon[i], ends[i], polygon[j], ends[j]
            if np.any(np.maximum(a, b) < np.minimum(c, d)-1e-9) or np.any(np.maximum(c, d) < np.minimum(a, b)-1e-9):
                continue
            ab, cd = b-a, d-c
            u = np.array([ab[0]*(c-a)[1]-ab[1]*(c-a)[0], ab[0]*(d-a)[1]-ab[1]*(d-a)[0]])
            v = np.array([cd[0]*(a-c)[1]-cd[1]*(a-c)[0], cd[0]*(b-c)[1]-cd[1]*(b-c)[0]])
            # Shared or crossing nonadjacent segments do not bound a simple face.
            if (np.min(u) <= 1e-12 and np.max(u) >= -1e-12 and
                    np.min(v) <= 1e-12 and np.max(v) >= -1e-12):
                raise ValueError('Face boundary intersects itself')
    return normal, basis, polygon


def vertex_geometry_reference(fit, role):
    """Resolve one point, segment or planar face without reading a shape label."""
    if not isinstance(role, str) or role.count('_') != 1:
        raise ValueError('Expected vertex_i, edge_i or face_i with a nonnegative index')
    kind, index_text = role.split('_')
    if kind not in ('vertex', 'edge', 'face') or not index_text.isascii() or not index_text.isdigit():
        raise ValueError('Expected vertex_i, edge_i or face_i with a nonnegative index')
    index = int(index_text)
    points = np.asarray(fit['vertices'], dtype=float)
    if points.ndim != 2 or points.shape[1:] != (3,) or not len(points) or not np.isfinite(points).all():
        raise ValueError('Vertices must be finite world XYZ points')
    sources = fit['vertex_sources']
    if len(sources) != len(points) or not all(isinstance(source, str) and source for source in sources):
        raise ValueError('Each vertex requires an evidence source')
    groups = ([[i] for i in range(len(points))] if kind == 'vertex' else
              fit['edge_vertex_indices' if kind == 'edge' else 'face_vertex_indices'])
    if index >= len(groups):
        raise ValueError('Geometric reference index is out of range')
    indices = point_reference_indices(groups[index], len(points), kind)
    selected = points[indices]
    if len(selected) > 1:
        rows, cols = np.triu_indices(len(selected), k=1)
        if np.any(np.linalg.norm(selected[rows]-selected[cols], axis=1) < 1e-9):
            raise ValueError('Reference vertices must be geometrically distinct')
    evidence = [sources[int(i)] for i in indices]
    inferred = any(source.startswith('inferred_') for source in evidence)
    common = {'vertex_indices': indices, 'vertex_sources': evidence,
              'id_scope': 'current_observation_only_not_material_point_tracking'}
    if kind == 'vertex':
        return dict(common, type='point', position_m=selected[0],
                    semantics='model_inferred_point' if inferred else 'point_from_current_fitted_vertices')
    if kind == 'edge':
        return dict(common, type='line_segment', endpoints_m=selected,
                    semantics='model_inferred_segment' if inferred else 'segment_from_declared_vertex_connectivity')
    # Connectivity identifies a physical/model face; coplanarity alone does not.
    normal, basis, polygon = point_face_projection(selected)
    _, centroid, _ = point_polygon_moments(polygon)
    return dict(common, type='plane_patch', vertices_m=selected, normal=normal,
                center_m=selected[0]+basis@centroid,
                semantics='model_inferred_face' if inferred else 'face_from_current_fitted_vertices')
