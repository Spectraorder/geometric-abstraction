"""Small per-action records for one active object, without scene-wide snapshots.

Input geometry is supplied by the existing RGB-D observer. This recorder performs
no sensing, fitting, planning, contact inference or pose fusion. Display coordinates
are rounded to micrometers; raw execution geometry is never modified.
"""
import numpy as np


def step_vector(value):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError('Expected a finite world XYZ vector')
    return result.copy()


def step_display(value):
    return np.round(step_vector(value), 6).tolist()


class GeometryStepState:
    """One focus object, one optional goal point, one preceding observation."""

    def __init__(self, object_id):
        if not isinstance(object_id, str) or not object_id or '.' in object_id:
            raise ValueError('Expected one object identity')
        self.object_id = object_id
        self.goal = None
        self.previous = None
        self.latest_time = None

    def set_goal(self, point_m, reference='center'):
        if not isinstance(reference, str) or not reference or '.' in reference:
            raise ValueError('Use a local reference name such as center or bottom_face')
        self.goal = {'reference': reference, 'point_m': step_vector(point_m)}

    def clear_goal(self):
        self.goal = None

    def update(self, step, fit, now, references=(), action_status=None):
        now = float(now)
        if not np.isfinite(now) or (self.latest_time is not None and now < self.latest_time):
            raise ValueError('Step time must be finite and nondecreasing')
        if not isinstance(step, str) or not step:
            raise ValueError('An explicit action-step label is required')
        output = {'object': self.object_id, 'step': step, 'time': round(now, 6),
                  'frame': 'world', 'units': 'meters', 'status': 'unavailable'}
        if action_status is not None:
            if not isinstance(action_status, str):
                raise ValueError('Action status must be a string')
            output['action_status'] = action_status
        if self.goal is not None:
            output['goal'] = {'reference': self.goal['reference'], 'point_m': step_display(self.goal['point_m'])}
            output['remaining_translation_m'] = None
        if fit is None or fit.get('status') in ('unavailable', 'stale'):
            self.latest_time = now
            self.previous = None
            return output
        if fit.get('coordinate_frame') != 'world' or fit.get('units') != 'meters':
            raise ValueError('Require a world-meter observation')
        stamp = float(fit['observation_time'])
        if not np.isfinite(stamp):
            raise ValueError('Observation timestamp must be finite')
        if abs(stamp-now) > 1e-6:
            output['status'] = 'stale'
            self.latest_time = now
            self.previous = None
            return output
        center = step_vector(fit['center'])
        observed = bool(fit['position_fully_observed'])
        geometry = {'shape': fit['shape'], 'center_m': step_display(center),
                    'position_fully_observed': observed,
                    'orientation_inferred': bool(fit['inferred_orientation'])}
        if 'axis_dimensions' in fit:
            geometry['dimensions_m'] = step_display(fit['axis_dimensions'])
        if 'normal' in fit:
            geometry['normal'] = step_display(fit['normal'])
        output.update(status='current_estimate', geometry=geometry)
        points = {'center': center}
        requested = {}
        for ref in references:
            name = ref['reference']
            if not name.startswith(self.object_id+'.') or name.count('.') != 1:
                raise ValueError('Only active-object references belong in this record')
            ref_time = float(ref['observation_time'])
            if ref['coordinate_frame'] != 'world' or ref['units'] != 'meters' or not np.isfinite(ref_time) or abs(ref_time-now) > 1e-6:
                raise ValueError('Reference must be current world-meter geometry')
            role = name.split('.')[1]
            if role in requested:
                raise ValueError('Duplicate requested reference')
            if ref['type'] not in ('point', 'plane_patch', 'disk'):
                raise ValueError('Request a point or face center')
            key = 'position_m' if ref['type'] == 'point' else 'center_m'
            point = step_vector(ref[key]); points[role] = point
            requested[role] = {'point_m': step_display(point)}
            if 'normal' in ref:
                requested[role]['normal'] = step_display(ref['normal'])
            if ref.get('semantics'):
                requested[role]['semantics'] = ref['semantics']
        if requested:
            output['requested_points'] = requested
        if self.previous is not None:
            output['estimated_change'] = {'since': round(self.previous['time'], 6),
                                          'center_delta_m': step_display(center-self.previous['center']),
                                          'both_positions_fully_observed': observed and self.previous['observed']}
        if self.goal is not None and self.goal['reference'] in points:
            output['remaining_translation_m'] = step_display(self.goal['point_m']-points[self.goal['reference']])
        self.latest_time = now
        self.previous = {'center': center.copy(), 'time': now, 'observed': observed}
        return output
