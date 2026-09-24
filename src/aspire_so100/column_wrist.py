"""CPU wrist-RGB silhouette constraints for a visually learned held-object model.

Known dimensions provide metric scale; side RGB-D supplies orientation and any
already observed translation axes. Only the remaining translation coordinates
are fitted. A complete, isolated silhouette is required: partial masks do not
silently become smaller objects or evidence of an unobserved grasp correction.
"""
import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull

from aspire_so100.instance_geometry import inventory_color_mask


def _wrist_project(frame, vertices):
    pose = np.asarray(frame['camera_to_world'], float)
    local = (vertices - pose[:3, 3]) @ pose[:3, :3]
    if np.min(local[:, 2]) <= .015:
        raise ValueError('Object model is behind or too close to the wrist camera')
    projected = local @ np.asarray(frame['intrinsics'], float).T
    return projected[:, :2] / projected[:, 2, None]


def _wrist_bounds(projected):
    return np.r_[projected.min(axis=0), projected.max(axis=0)]


def _wrist_visibility(projected, mask):
    hull = ConvexHull(projected)
    bounds = _wrist_bounds(projected)
    x0, y0 = np.maximum(np.floor(bounds[:2] - 2), 0).astype(int)
    x1, y1 = np.minimum(np.ceil(bounds[2:] + 2), [mask.shape[1] - 1, mask.shape[0] - 1]).astype(int)
    yy, xx = np.mgrid[y0:y1+1, x0:x1+1]
    pixels = np.column_stack((xx.ravel(), yy.ravel()))
    distance = pixels @ hull.equations[:, :2].T + hull.equations[:, 2]
    inner = np.all(distance < -1.2, axis=1)
    expanded = np.all(distance <= 1.2, axis=1)
    observed = mask[yy, xx].ravel()
    if np.count_nonzero(inner) < 20:
        raise ValueError('Wrist object is too small for silhouette validation')
    coverage = float(np.mean(observed[inner]))
    included = int(np.count_nonzero(observed & expanded))
    outside = float(1. - included / max(1, np.count_nonzero(mask)))
    if coverage < .88 or outside > .04:
        raise ValueError('Wrist silhouette is occluded or inconsistent with the learned model')
    return coverage, outside


def refine_column_with_wrist(frame, color, block, *, max_correction_m=.025,
                             constrained_directions=None):
    """Refine hidden translation coordinates using an RGB silhouette.

    ``block`` is the side-camera estimate in the column controller schema. Its
    vertices and dimensions must have been learned from vision, not a catalog.
    ``constrained_directions`` optionally supplies observed world-space axes as
    columns. By default ``rotation[:, constrained_axes]`` is used. A narrow
    cylinder arc needs its explicit radial direction because its two Boolean
    horizontal-axis flags cannot represent that partial radial constraint.

    Returns a shifted block plus wrist-fit diagnostics. Raises ``ValueError``
    for ambiguity, clipping, occlusion, poor conditioning or disagreement with
    the side camera. Observed side-camera axes are held fixed throughout, and
    finite side-surface center intervals remain hard constraints.
    """
    if not {'center', 'vertices', 'shape', 'top', 'bottom'} <= block.keys():
        raise ValueError('A complete learned object model is required')
    if block.get('inferred_orientation', False) or block.get('orientation_observed') is False:
        raise ValueError('Wrist translation fitting requires an independently observed orientation')
    if not .001 <= max_correction_m <= .04:
        raise ValueError('Wrist correction bound must be between 1 and 40 mm')
    center = np.asarray(block['center'], float)
    vertices = np.asarray(block['vertices'], float)
    if center.shape != (3,) or vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 8:
        raise ValueError('Expected a finite learned solid-object model')
    if not np.isfinite(center).all() or not np.isfinite(vertices).all():
        raise ValueError('Expected a finite learned solid-object model')
    if constrained_directions is None:
        if (block['shape'] == 'cylinder' and block.get('angular_information', 1.) < .025
                and not block.get('top_only_update', False)):
            raise ValueError('A narrow cylinder arc needs its observed radial constraint')
        if 'rotation' not in block or 'constrained_axes' not in block:
            raise ValueError('Side-camera translation constraints are missing')
        rotation = np.asarray(block['rotation'], float)
        constrained = np.asarray(block['constrained_axes'], bool)
        if rotation.shape != (3, 3) or constrained.shape != (3,):
            raise ValueError('Side-camera translation constraints are missing')
        constrained_directions = rotation[:, constrained]
    observed_axes = np.asarray(constrained_directions, float)
    if observed_axes.ndim != 2 or observed_axes.shape[0] != 3 or not np.isfinite(observed_axes).all():
        raise ValueError('Observed translation directions must have shape (3, n)')
    if observed_axes.shape[1]:
        _, singular, vh = np.linalg.svd(observed_axes.T, full_matrices=True)
        rank = int(np.count_nonzero(singular > 1e-6))
        free = vh[rank:].T
    else:
        free = np.eye(3)
    finite_bounds = None
    if 'finite_center_bounds_m' in block:
        # Finite side-surface intervals use absolute world points projected
        # onto the block's fitted rotation axes: local = world @ rotation.
        # They are not offsets relative to the current center or wrist frame.
        finite_bounds = np.asarray(block['finite_center_bounds_m'], float)
        finite_rotation = np.asarray(block.get('rotation'), float)
        if (finite_bounds.shape != (3, 2) or finite_rotation.shape != (3, 3)
                or not np.isfinite(finite_bounds).all() or not np.isfinite(finite_rotation).all()
                or np.any(finite_bounds[:, 0] > finite_bounds[:, 1])):
            raise ValueError('Invalid finite side-camera center bounds')
    rejected_finite_bounds = False
    relative = vertices - center
    predicted = _wrist_project(frame, vertices)
    predicted_bounds = _wrist_bounds(predicted)
    predicted_middle = (predicted_bounds[:2] + predicted_bounds[2:]) / 2
    expected_area = float(ConvexHull(predicted).volume)
    labels, count = ndimage.label(inventory_color_mask(frame['rgb'], color))
    candidates = []
    for label in range(1, count + 1):
        mask = labels == label
        rows, cols = np.nonzero(mask)
        if len(rows) < max(30, .15 * expected_area):
            continue
        bounds = np.array([cols.min()-.5, rows.min()-.5, cols.max()+.5, rows.max()+.5])
        if (bounds[0] < 1 or bounds[1] < 1 or bounds[2] > mask.shape[1]-2 or bounds[3] > mask.shape[0]-2
                or min(bounds[2:] - bounds[:2]) < 8):
            continue
        middle = (bounds[:2] + bounds[2:]) / 2
        if np.linalg.norm(middle - predicted_middle) > max(25., np.linalg.norm(predicted_bounds[2:] - predicted_bounds[:2])):
            continue

        def residual(parameters):
            shifted = center + free @ parameters
            return _wrist_bounds(_wrist_project(frame, relative + shifted)) - bounds

        if free.shape[1]:
            fit = least_squares(residual, np.zeros(free.shape[1]),
                                bounds=(-max_correction_m, max_correction_m), max_nfev=50,
                                ftol=1e-8, xtol=1e-8, gtol=1e-8)
            correction = free @ fit.x
            singular = np.linalg.svd(fit.jac, compute_uv=False)
            sigma = .75 / max(float(singular[-1]), 1e-12)
            if sigma > .003 or np.linalg.norm(correction) > max_correction_m-.0002:
                continue
            errors = fit.fun
        else:
            correction, sigma = np.zeros(3), 0.
            errors = residual(np.empty(0))
        if np.max(abs(errors)) > 2.5 or np.sqrt(np.mean(errors**2)) > 1.5:
            continue
        if finite_bounds is not None:
            local_center = (center + correction) @ finite_rotation
            if (np.any(local_center < finite_bounds[:, 0] - 1e-9)
                    or np.any(local_center > finite_bounds[:, 1] + 1e-9)):
                rejected_finite_bounds = True
                continue
        projection = _wrist_project(frame, vertices + correction)
        try:
            coverage, outside = _wrist_visibility(projection, mask)
        except ValueError:
            continue
        candidates.append((correction, errors, coverage, outside, sigma, bounds))
    if not candidates and rejected_finite_bounds:
        raise ValueError('Wrist silhouette conflicts with finite side-camera center bounds')
    if len(candidates) != 1:
        raise ValueError('No unique complete wrist silhouette agrees with the side-camera model')
    correction, errors, coverage, outside, sigma, bounds = candidates[0]
    fresh = dict(block)
    fresh['center'] = center + correction
    fresh['vertices'] = vertices + correction
    if 'top_vertices' in block:
        fresh['top_vertices'] = np.asarray(block['top_vertices']) + correction
    fresh['top'] = float(block['top'] + correction[2])
    fresh['bottom'] = float(block['bottom'] + correction[2])
    fresh['wrist_translation_observed'] = True
    fresh['wrist_fit'] = {'correction_m': correction, 'bbox_rms_px': float(np.sqrt(np.mean(errors**2))),
                          'silhouette_coverage': coverage, 'outside_fraction': outside,
                          'pixel_uncertainty_m': sigma, 'observed_bbox_px': bounds,
                          'fitted_axes': free.shape[1], 'source': 'RGB silhouette and visually learned dimensions'}
    return fresh
