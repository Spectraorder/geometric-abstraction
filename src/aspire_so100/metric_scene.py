"""Task-free current geometry, observation history and separate desired states.

This ledger never estimates a pose from history. Missing or stale observations
remain unavailable. Predictions are separate conditional records and are disabled
by default; they never become measurements or drive an executor implicitly.
"""
import numpy as np


def metric_copy(value, plain=False):
    if isinstance(value, np.ndarray):
        return metric_copy(value.tolist(), True) if plain else value.copy()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: metric_copy(v, plain) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [metric_copy(v, plain) for v in value]
    return value


def metric_vector(value, shape, label):
    result = np.array(value, dtype=float, copy=True)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(label+' requires a finite '+str(shape)+' array')
    return result


def metric_rotation(value):
    result = metric_vector(value, (3, 3), 'rotation')
    if not np.allclose(result.T@result, np.eye(3), atol=1e-5) or np.linalg.det(result) < .999:
        raise ValueError('Require a proper rigid rotation')
    return result


class MetricScene:
    def __init__(self, predictions=False):
        self.records = {}
        self.targets = {}
        self.predictions = {}
        self.predictions_enabled = bool(predictions)
        self.latest_time = 0.

    def time(self, stamp=None):
        stamp = self.latest_time if stamp is None else float(stamp)
        if not np.isfinite(stamp) or stamp < self.latest_time-1e-6:
            raise ValueError('Scene timestamps must be finite and nondecreasing')
        return stamp

    def observe(self, name, fit):
        if not isinstance(name, str) or not name or '.' in name:
            raise ValueError('Object identifiers must be nonempty and contain no period')
        if fit.get('kind') not in (None, 'current_visual_estimate'):
            raise ValueError('Desired states and predictions cannot be ingested as observations')
        if fit.get('coordinate_frame') != 'world' or fit.get('units') != 'meters':
            raise ValueError('Current observations require world coordinates in meters')
        stamp = self.time(fit['observation_time'])
        center = metric_vector(fit['center'], (3,), 'center'); rotation = metric_rotation(fit['rotation'])
        shape = fit['shape']
        if shape not in ('cube', 'prism', 'thin_prism', 'cylinder'):
            raise ValueError('Unsupported rigid shape')
        dimensions = metric_vector(fit.get('axis_dimensions', np.r_[fit['dims'], fit['height']]), (3,), 'axis dimensions')
        if np.any(dimensions <= 0):
            raise ValueError('Measured dimensions must be positive')
        normal = metric_vector(fit['normal'], (3,), 'normal')
        vertices = np.array(fit['vertices'], dtype=float, copy=True)
        scalars = np.asarray([fit[key] for key in ('height', 'top', 'bottom', 'extent_z')], float)
        if (abs(np.linalg.norm(normal)-1.) > 1e-5 or vertices.ndim != 2 or vertices.shape[1] != 3
                or not np.isfinite(vertices).all() or not np.isfinite(scalars).all()
                or scalars[0] <= 0 or scalars[3] <= 0 or scalars[2] > scalars[1]
                or (shape != 'cylinder' and len(vertices) != 8)
                or (shape == 'cylinder' and len(vertices) < 6)):
            raise ValueError('Invalid finite rigid landmarks or extents')
        if shape == 'cylinder' and (not np.isfinite(fit['model']['radius']) or fit['model']['radius'] <= 0):
            raise ValueError('Cylinder radius must be finite and positive')
        axes = np.asarray(fit['constrained_axes'], dtype=bool)
        if axes.shape != (3,):
            raise ValueError('Three translation observability flags are required')
        value = metric_copy(fit)
        value.update(center=center, rotation=rotation, axis_dimensions=dimensions, constrained_axes=axes.copy(),
                     observed_translation_projector=rotation[:, axes]@rotation[:, axes].T,
                     observed_translation_rank=int(np.count_nonzero(axes)), observation_time=stamp,
                     position_fully_observed=bool(np.all(axes)), kind='current_visual_estimate')
        if 'finite_center_bounds_m' in value:
            value['finite_bounds_coordinate_frame'] = 'center @ rotation; fitted local basis'
        previous = self.records.get(name)
        if previous and previous.get('last_observation') is not None and previous['last_observation']['shape'] != shape:
            raise ValueError('An object cannot change its declared rigid shape')
        prior = None if previous is None else previous.get('last_observation')
        change = None if prior is None else {'from_time': prior['observation_time'], 'to_time': stamp,
                    'estimated_translation_m': center-np.asarray(prior['center']),
                    'both_positions_fully_observed': bool(np.all(axes) and prior['position_fully_observed']),
                    'semantics': 'difference_between_estimates_not_certified_physical_motion'}
        residual = None
        predicted = self.predictions.get(name)
        if predicted is not None and abs(predicted['prediction_time']-stamp) <= 1e-6:
            delta = center-predicted['center']
            residual = {'prediction_time': stamp, 'condition': predicted['condition'],
                        'translation_residual_m': delta,
                        'observable_translation_residual_m': value['observed_translation_projector']@delta,
                        'condition_verified': False}
        self.records[name] = {'status': 'available', 'time': stamp, 'last_observation': value,
                              'previous_observation': metric_copy(prior), 'change': change,
                              'prediction_residual': residual}
        self.latest_time = stamp
        return self.get(name)

    def unavailable(self, name, stamp, reason):
        stamp = self.time(stamp)
        if not isinstance(name, str) or not name or '.' in name or not isinstance(reason, str) or not reason:
            raise ValueError('Unavailable observations need an object identifier and reason')
        previous = self.records.get(name, {})
        self.records[name] = dict(previous, status='unavailable', time=stamp, reason=reason)
        self.latest_time = stamp

    def get(self, name, stamp=None):
        stamp = self.time(stamp)
        record = self.records[name]
        if record['status'] != 'available' or abs(record['time']-stamp) > 1e-6:
            raise ValueError('No current observation for '+name)
        return metric_copy(record['last_observation'])

    def history(self, name):
        record = self.records[name]
        return metric_copy({'last_observation': record.get('last_observation'),
                            'previous_observation': record.get('previous_observation'),
                            'change': record.get('change'), 'semantics': 'historical_evidence_not_current_pose'})

    def target(self, name, center, rotation=None, stamp=None):
        target = {'kind': 'desired_state', 'center': metric_vector(center, (3,), 'desired center'),
                  'rotation': None if rotation is None else metric_rotation(rotation),
                  'recorded_time': self.time(stamp), 'execution_certified': False}
        self.targets[name] = target
        return metric_copy(target)

    def goal_error(self, name, stamp=None):
        current = self.get(name, stamp); desired = self.targets[name]
        delta = desired['center']-current['center']
        return {'translation_to_goal_m': delta, 'distance_m': float(np.linalg.norm(delta)),
                'position_fully_observed': current['position_fully_observed'],
                'desired_state_recorded_time': desired['recorded_time'], 'execution_certified': False}

    def predict(self, name, transform, stamp, condition):
        if not self.predictions_enabled:
            raise ValueError('Conditional predictions are disabled for this scene')
        stamp = self.time(stamp)
        if not isinstance(condition, str) or not condition.strip():
            raise ValueError('A prediction must state its unverified condition')
        matrix = metric_vector(transform, (4, 4), 'rigid transform'); rotation = metric_rotation(matrix[:3, :3])
        if not np.allclose(matrix[3], [0., 0., 0., 1.], atol=1e-8):
            raise ValueError('Require a rigid homogeneous transform')
        source = self.get(name)
        predicted = {'kind': 'conditional_prediction', 'center': rotation@source['center']+matrix[:3, 3],
                     'rotation': rotation@source['rotation'], 'prediction_time': stamp,
                     'based_on_observation_time': source['observation_time'], 'transform': matrix.copy(), 'condition': condition,
                     'condition_verified': False, 'source': 'caller_supplied_conditional_transform'}
        self.predictions[name] = predicted
        return metric_copy(predicted)

    def packet(self, stamp=None, details=False):
        stamp = self.time(stamp); objects = {}
        for name, record in self.records.items():
            current = record['status'] == 'available' and abs(record['time']-stamp) <= 1e-6
            row = {'status': 'available' if current else ('stale' if record['status'] == 'available' else 'unavailable'),
                   'last_observed': None if record.get('last_observation') is None else record['last_observation']['observation_time']}
            if current:
                fit = record['last_observation']
                keys = ['shape', 'center', 'rotation', 'axis_dimensions', 'normal', 'top', 'bottom', 'extent_z',
                        'constrained_axes', 'inferred_orientation', 'position_fully_observed', 'source',
                        'view', 'finite_center_width_m', 'finite_center_bounds_m', 'finite_bounds_coordinate_frame', 'vertices_kind']
                row.update({k: metric_copy(fit[k]) for k in keys if k in fit})
                if details:
                    row['vertices'] = metric_copy(fit['vertices'])
                row['change'] = metric_copy(record.get('change'))
                row['prediction_residual'] = metric_copy(record.get('prediction_residual'))
                if name in self.targets:
                    row['goal_error'] = self.goal_error(name, stamp)
            elif record.get('reason'):
                row['reason'] = record['reason']
            objects[name] = row
        return metric_copy({'schema': 'mixed_metric_scene_v1', 'coordinate_frame': 'world', 'units': 'meters',
                            'time': stamp, 'objects': objects, 'desired_states': self.targets,
                            'conditional_predictions': self.predictions,
                            'temporal_pose_fusion': False, 'contact_or_stability_certified': False}, plain=True)
