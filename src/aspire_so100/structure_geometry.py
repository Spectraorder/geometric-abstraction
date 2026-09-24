"""CPU convex contact patches and gravity balance for multiple supports.

Inputs are explicit geometric models. The online caller supplies visual models;
the private evaluator independently supplies simulator geometry.
"""
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull


def convex_polygon(points):
    points = np.unique(np.asarray(points, dtype=float), axis=0)
    if len(points) < 3 or not np.isfinite(points).all():
        return np.empty((0, 2))
    centered = points-points.mean(axis=0)
    if np.linalg.matrix_rank(centered, tol=1e-9) < 2:
        return np.empty((0, 2))
    return points[ConvexHull(points).vertices]


def clip_polygon(subject, boundary, margin=0.):
    """Intersect a convex polygon with inward-offset CCW boundary half-planes."""
    polygon = np.asarray(subject, dtype=float)
    boundary = np.asarray(boundary, dtype=float)
    if len(polygon) < 3 or len(boundary) < 3:
        return np.empty((0, 2))
    for index, a in enumerate(boundary):
        b = boundary[(index+1) % len(boundary)]
        edge = b-a
        normal = np.array([-edge[1], edge[0]])
        threshold = float(margin*np.linalg.norm(edge))
        result = []
        for j, start in enumerate(polygon):
            end = polygon[(j+1) % len(polygon)]
            first = float((start-a) @ normal-threshold)
            last = float((end-a) @ normal-threshold)
            if first >= -1e-12:
                result.append(start)
            if (first >= 0) != (last >= 0):
                result.append(start+(end-start)*first/(first-last))
        polygon = np.asarray(result, dtype=float).reshape(-1, 2)
        if len(polygon) < 3:
            return np.empty((0, 2))
    return convex_polygon(polygon)


def model_footprint(model, face='top'):
    """Project the requested contact face, preserving observed tilt and yaw."""
    if face not in ('top', 'bottom'):
        raise ValueError('Contact face must be top or bottom')
    center = np.asarray(model['center'], dtype=float)
    dims = np.asarray(model['dims'], dtype=float)
    normal = np.asarray(model.get('normal', [0., 0., 1.]), dtype=float)
    sign = 1. if face == 'top' else -1.
    cap = center+sign*normal*float(model.get('height', 0.))/2
    if model.get('shape') == 'cylinder':
        ex = np.array([1., 0., 0.])-normal*normal[0]
        if np.linalg.norm(ex) < 1e-8:
            ex = np.array([0., 1., 0.])-normal*normal[1]
        ex /= np.linalg.norm(ex)
        ey = np.cross(normal, ex)
        angle = np.arange(32)*2*np.pi/32
        points = cap+min(dims)/2*(np.cos(angle)[:, None]*ex+np.sin(angle)[:, None]*ey)
        return convex_polygon(points[:, :2])
    field = face+'_vertices'
    if field in model:
        return convex_polygon(np.asarray(model[field])[:, :2])
    if 'vertices' in model:
        vertices = np.asarray(model['vertices'])
        indices = np.argsort(sign*(vertices @ normal))[-4:]
        return convex_polygon(vertices[indices, :2])
    return cap[:2]+np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])*dims/2


def support_equilibrium(models, parents, *, margin=.002, minimum_load=.001):
    """Check nonnegative support reactions and XY moments for every object.

    Each vertex force acts upward on its child and downward on its parent.
    This shares bridge loads instead of assigning the entire upper mass to
    each support. Masses may be normalized volume proxies online.
    """
    names = list(models)
    if set(names) != set(parents):
        raise ValueError('Every model must have an explicit support list')
    index = {name: i for i, name in enumerate(names)}
    columns, patches = [], []
    for child in names:
        upper = model_footprint(models[child], face='bottom')
        for parent in parents[child] or [None]:
            if parent is not None and parent not in models:
                raise ValueError('Unknown supporting model')
            patch = upper if parent is None else clip_polygon(upper, model_footprint(models[parent]))
            patch = clip_polygon(patch, patch, margin=margin)
            if len(patch) < 3:
                return {'feasible': False, 'reason': 'No support patch with requested margin', 'edge': [parent, child]}
            first = len(columns)
            for point in patch:
                col = np.zeros(3*len(names))
                wrench = np.r_[1., point]
                col[3*index[child]:3*index[child]+3] += wrench
                if parent is not None:
                    col[3*index[parent]:3*index[parent]+3] -= wrench
                columns.append(col)
            patches.append((parent, child, first, len(columns)))
    masses = np.array([float(models[n]['mass']) for n in names])
    if np.any(masses <= 0) or not np.isfinite(masses).all():
        raise ValueError('Mass or volume proxies must be finite and positive')
    masses /= masses.sum()
    rhs = np.concatenate([m*np.r_[1., np.asarray(models[n]['center'])[:2]] for n, m in zip(names, masses)])
    inequalities = np.zeros((len(patches), len(columns)))
    for row, (_, _, start, end) in enumerate(patches):
        inequalities[row, start:end] = -1.
    result = linprog(np.zeros(len(columns)), A_ub=inequalities,
                     b_ub=np.repeat(-minimum_load, len(patches)),
                     A_eq=np.asarray(columns).T, b_eq=rhs, bounds=(0., None), method='highs')
    return {'feasible': bool(result.success), 'reason': None if result.success else 'No balanced nonnegative support reactions',
            'loads': [[p, c, float(result.x[a:b].sum())] for p, c, a, b in patches] if result.success else []}
