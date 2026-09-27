"""Typed symbolic relations with metric residuals and three-valued visual evidence.

Truth values concern estimated geometry, not physical contact or task completion.
Intervals use heuristic translation uncertainty and omit orientation uncertainty;
therefore these are reasoning aids, not calibrated safety guarantees.
"""
import numpy as np
from aspire_so100.rigid_state import support_relation


def _geometric_truth(value, limit, radius, available):
    """Classify the estimated inequality value <= limit with an uncertainty band."""
    if not available:
        return 'unknown'
    if value + radius <= limit:
        return 'true'
    if value - radius > limit:
        return 'false'
    return 'unknown'


def _directions_observed(state, directions):
    projection = state['observed_translation_projector']
    return bool(all(np.linalg.norm((np.eye(3)-projection) @ d) < .05 for d in directions))


def geometric_predicate(memory, predicate, args, time=None, tolerance_m=.004, minimum_margin_m=.003):
    """Evaluate a geometric statement; unknown is distinct from false.

    Supported predicates: above, aligned_xy, support_candidate, contact.
    args = [upper_object_id, lower_object_id]. All numeric inputs use meters.
    contact is deliberately always unknown because the estimator has no contact
    sensor. support_candidate is a geometric relation, not an alias for contact.
    """
    if predicate not in ('above', 'aligned_xy', 'support_candidate', 'contact'):
        raise ValueError('Unknown geometric predicate: '+str(predicate))
    if not isinstance(args, (list, tuple)) or len(args) != 2 or args[0] == args[1]:
        raise ValueError('Expected two distinct object identifiers')
    if not np.isfinite([tolerance_m, minimum_margin_m]).all() or tolerance_m <= 0 or minimum_margin_m < 0:
        raise ValueError('Predicate tolerances must be finite meters')
    upper, lower = (memory.estimate(name, time) for name in args)
    delta = upper['center']-lower['center']
    covariance = upper['covariance_m2']+lower['covariance_m2']
    vertical = [np.array([0., 0., 1.])]
    horizontal = [np.array([1., 0., 0.]), np.array([0., 1., 0.])]
    z_available = all(_directions_observed(s, vertical) for s in (upper, lower))
    xy_available = all(_directions_observed(s, horizontal) for s in (upper, lower))
    z_radius = 2*float(np.sqrt(max(0., covariance[2, 2])))
    xy_radius = 2*float(np.sqrt(max(0., np.linalg.eigvalsh(covariance[:2, :2]).max())))
    truth = 'unknown'
    residuals = {}
    available = False
    if predicate == 'above':
        available = z_available
        truth = _geometric_truth(-float(delta[2]), 0., z_radius, available)
        residuals = {'center_z_difference_mm': round(float(delta[2])*1000, 2), 'band_mm': round(z_radius*1000, 2)}
    elif predicate == 'aligned_xy':
        available = xy_available
        residual = float(np.linalg.norm(delta[:2]))
        truth = _geometric_truth(residual, tolerance_m, xy_radius, available)
        residuals = {'xy_distance_mm': round(residual*1000, 2), 'tolerance_mm': tolerance_m*1000,
                     'band_mm': round(xy_radius*1000, 2)}
    elif predicate == 'support_candidate':
        relation = support_relation(upper, lower)
        available = bool(z_available and xy_available and not upper['inferred_orientation'] and not lower['inferred_orientation'])
        checks = [_geometric_truth(-float(delta[2]), 0., z_radius, available),
                  _geometric_truth(abs(relation['lowest_vertex_gap_m']), tolerance_m, z_radius, available),
                  _geometric_truth(-relation['projected_center_margin_m'], -minimum_margin_m, xy_radius, available)]
        truth = 'false' if 'false' in checks else ('true' if all(x=='true' for x in checks) else 'unknown')
        residuals = {'gap_mm': round(relation['lowest_vertex_gap_m']*1000, 2),
                     'center_margin_mm': round(relation['projected_center_margin_m']*1000, 2),
                     'gap_tolerance_mm': tolerance_m*1000, 'minimum_margin_mm': minimum_margin_m*1000}
    return {'predicate': predicate, 'args': list(args), 'truth': truth,
            'residuals': residuals, 'required_directions_observed': available,
            'evidence': 'visual_geometry_with_heuristic_translation_bands' if predicate!='contact' else 'no_contact_sensor',
            'contact_or_stability_certified': False}


def symbolic_packet(memory, time=None, details=False):
    """Serialize typed objects and derived facts, not narrative interpretations."""
    packet = memory.packet(time, details)
    facts = []
    names = list(packet['objects'])
    for upper in names:
        for lower in names:
            if upper == lower:
                continue
            for predicate in ('above', 'support_candidate'):
                facts.append(geometric_predicate(memory, predicate, [upper, lower], time))
    for name, value in packet['objects'].items():
        value['type'] = 'rigid_cube'
        value['references'] = {'center': name+'.center', 'top_face': name+'.top_face', 'bottom_face': name+'.bottom_face'}
    return dict(packet, schema='symbolic_metric_state_v1', facts=facts,
                truth_semantics='true/false/unknown about estimated geometry; not simulator truth',
                contact_state='unknown_without_a_contact_sensor')


def resolve_reference(memory, reference, time=None):
    """Resolve typed geometric roles; cube corners are not tracked material marks.

    Examples: red.center, blue.top_face, red.bottom_face, blue.corner_0.
    Top/bottom identify current world-up faces; cube symmetry means invisible
    original material-face identities cannot be recovered from uniform RGB-D.
    """
    if not isinstance(reference, str) or reference.count('.') != 1:
        raise ValueError('A reference must be object_id.landmark_or_face')
    name, role = reference.split('.')
    state = memory.estimate(name, time)
    normal = state['normal']
    if role in ('top_face', 'bottom_face'):
        direction = normal if role=='top_face' else -normal
        center = state['center']+direction*state['height']/2
        distances = state['vertices'] @ direction
        indices = np.argsort(distances)[-4:]
        geometry = {'type': 'plane_patch', 'center_m': center, 'normal': direction,
                    'vertices_m': state['vertices'][indices]}
    elif role in memory.landmarks(name, time):
        geometry = {'type': 'point', 'position_m': memory.landmarks(name, time)[role]}
    else:
        raise ValueError('Unknown geometric reference: '+reference)
    return dict(geometry, reference=reference, frame='world', source=state['source'],
                last_observed=state['last_observed'], position_fully_observed=state['position_fully_observed'],
                semantics='inferred_geometry_not_material_point_tracking')
