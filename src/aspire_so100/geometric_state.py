"""CPU cube belief memory and conditional rigid predictions for text reasoning.

Only visual fits and caller-supplied rigid transforms enter this action-free layer.
Covariances are heuristic bookkeeping, not calibrated probabilities. A prediction,
a remembered pose, or a desired target never becomes a new visual measurement.
"""
import numpy as np
from aspire_so100.rigid_state import cube_vertices, support_relation


def _vector(value, shape, label):
    result = np.array(value, dtype=float, copy=True)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(label + ' must have finite shape ' + str(shape))
    return result


def _rotation(value):
    result = _vector(value, (3, 3), 'rotation')
    if not np.allclose(result.T @ result, np.eye(3), atol=1e-5) or np.linalg.det(result) < .999:
        raise ValueError('rotation must be a proper orthonormal matrix')
    return result


def _cube_frame(rotation, previous):
    """Keep geometric landmark labels continuous, not invisible material identities."""
    best, score = rotation, -np.inf
    for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        for x in (-1., 1.):
            for y in (-1., 1.):
                for z in (-1., 1.):
                    symmetry = np.eye(3)[:, perm] @ np.diag([x, y, z])
                    if np.linalg.det(symmetry) < 0:
                        continue
                    candidate = rotation @ symmetry
                    value = float(np.trace(previous.T @ candidate))
                    if value > score:
                        best, score = candidate, value
    return best.copy()


class GeometricState:
    """Object memory, targets, changes and conditional predictions; no robot API.

    observe(name, fit, time, view) ingests observe_cube() output. Partial fits
    update only constrained directions in the FIT'S coordinate frame. Fitted
    rotations may change axis labels between views; projection uses the original
    fit frame before landmark labels are matched to the object's previous frame.
    """
    def __init__(self, process_noise_m=.0005, measurement_floor_m=.001,
                 unknown_position_m=.03, contradiction_m=.01):
        parameters = np.array([process_noise_m, measurement_floor_m, unknown_position_m, contradiction_m])
        if not np.isfinite(parameters).all() or np.any(parameters <= 0):
            raise ValueError('Uncertainty parameters must be positive finite meters')
        self.process_noise_m = float(process_noise_m)
        self.measurement_floor_m = float(measurement_floor_m)
        self.unknown_position_m = float(unknown_position_m)
        self.contradiction_m = float(contradiction_m)
        self.objects = {}
        self.targets = {}
        self.events = []
        self.time = 0.
        self.sequence = 0

    def check_time(self, value):
        value = float(value)
        if not np.isfinite(value) or value < self.time:
            raise ValueError('State timestamps must be finite and nondecreasing')
        return value

    def record_event(self, event):
        self.sequence += 1
        self.events.append(dict(event, sequence=self.sequence))
        self.events = self.events[-24:]

    def position_covariance(self, item, time):
        return item['covariance'] + np.eye(3) * self.process_noise_m**2 * max(0., time-item['time'])

    def observe(self, name, fit, time, view='side'):
        time = self.check_time(time)
        if not isinstance(name, str) or not name:
            raise ValueError('An object identity is required')
        if fit.get('shape', 'cube') != 'cube':
            raise ValueError('This first state adapter supports cubes only')
        center = _vector(fit['center'], (3,), 'center')
        rotation = _rotation(fit['rotation'])
        side = float(fit['height'])
        if not np.isfinite(side) or side <= 0:
            raise ValueError('Measured cube size must be positive')
        axes = np.asarray(fit.get('constrained_axes', [fit.get('position_fully_observed', False)]*3), bool)
        if axes.shape != (3,) or not np.any(axes):
            raise ValueError('At least one measured translation direction is required')
        basis = rotation[:, axes]
        projection = basis @ basis.T
        remainder = np.eye(3)-projection
        rms = float(fit.get('surface_rms_m', self.measurement_floor_m))
        if not np.isfinite(rms) or rms < 0:
            raise ValueError('Surface residual must be finite and nonnegative')
        sigma = max(self.measurement_floor_m, rms)
        previous = self.objects.get(name)
        prediction = previous.get('prediction') if previous else None
        residual = None
        contradicted = False
        if previous is None:
            mean = center.copy()
            covariance = np.eye(3)*self.unknown_position_m**2
            information = np.zeros((3, 3))
            first_observed = time
            delta = np.zeros(3)
        else:
            # Rigid dimensions are learned once, not re-estimated while airborne.
            if abs(side-previous['height']) > .1*previous['height']:
                raise ValueError('Observation conflicts with stored rigid cube dimensions')
            side = previous['height']
            mean = previous['center'].copy()
            covariance = self.position_covariance(previous, time)
            information = previous['observed_information'].copy() if time == previous['last_observed'] else np.zeros((3, 3))
            first_observed = previous['first_observed']
            if prediction is not None:
                if prediction['time'] > time:
                    raise ValueError('A measurement cannot precede the pending prediction')
                residual = float(np.linalg.norm(projection @ (center-prediction['center'])))
                contradicted = residual > self.contradiction_m
                if contradicted:
                    covariance += np.eye(3)*self.unknown_position_m**2
                else:
                    mean = prediction['center'].copy()
                    covariance = prediction['covariance'].copy()
            if not fit.get('inferred_orientation', True):
                rotation = _cube_frame(rotation, previous['rotation'])
            else:
                rotation = (prediction['rotation'] if prediction is not None and not contradicted else previous['rotation']).copy()
        measured_mean = mean + projection @ (center-mean)
        delta = measured_mean-previous['center'] if previous else np.zeros(3)
        covariance = remainder @ covariance @ remainder.T + projection*sigma**2
        information += projection
        orientation_observed = not bool(fit.get('inferred_orientation', True))
        if previous and time == previous['last_observed']:
            orientation_observed |= previous['orientation_observed']
        item = {'center': measured_mean, 'rotation': rotation, 'height': side,
                'covariance': covariance, 'time': time, 'last_observed': time,
                'first_observed': first_observed, 'last_view': str(view),
                'observed_information': information, 'orientation_observed': orientation_observed,
                'measurement_center': center, 'measurement_projector': projection,
                'delta': delta, 'prediction': None, 'prediction_residual_m': residual,
                'prediction_contradicted': contradicted, 'missing': False,
                'source': 'visual_planes_and_memory' if previous else 'visual_planes_and_inferred_extents'}
        self.objects[name] = item
        self.time = time
        self.record_event({'time': time, 'object': name, 'kind': 'prediction_contradicted' if contradicted else 'observed',
                     'view': str(view), 'measured_rank': int(np.linalg.matrix_rank(projection, tol=1e-6)),
                     'delta_mm': np.round(delta*1000, 2).tolist(),
                     'prediction_residual_mm': None if residual is None else round(residual*1000, 2)})
        return self.estimate(name, time)

    def missing(self, name, time, reason='occluded_or_unobserved'):
        time = self.check_time(time)
        item = self.objects[name]
        # A second camera missing the object does not erase a same-time sighting.
        if time > item['last_observed']:
            item['missing'] = True
        self.time = time
        self.record_event({'time': time, 'object': name, 'kind': 'unobserved', 'reason': str(reason)})

    def predict(self, name, transform, time, condition):
        """Store a CONDITIONAL rigid displacement, not a successful action effect.

        transform is a world-to-world SE(3) delta. For a held-object hypothesis,
        derive it from measured robot poses after motion, not requested commands.
        A target release coordinate is not a predicted observation.
        """
        time = self.check_time(time)
        transform = _vector(transform, (4, 4), 'rigid transform')
        rotation = _rotation(transform[:3, :3])
        if not np.allclose(transform[3], [0., 0., 0., 1.]) or not isinstance(condition, str) or not condition.strip():
            raise ValueError('SE(3) transform and explicit prediction condition required')
        item = self.objects[name]
        current = item['prediction'] if item['prediction'] is not None else item
        covariance = self.position_covariance(current, time)
        item['prediction'] = {'center': rotation @ current['center']+transform[:3, 3],
                              'rotation': rotation @ current['rotation'],
                              'covariance': rotation @ covariance @ rotation.T,
                              'time': time, 'condition': condition}
        self.time = time
        self.record_event({'time': time, 'object': name, 'kind': 'conditional_prediction', 'condition': condition})
        return self.estimate(name, time)

    def estimate(self, name, time=None):
        time = self.time if time is None else self.check_time(time)
        item = self.objects[name]
        current = item['prediction'] if item['prediction'] is not None else item
        predicted = item['prediction'] is not None
        fresh = time == item['last_observed'] and not item['missing'] and not predicted
        eigenvalues, eigenvectors = np.linalg.eigh(item['observed_information'])
        basis = eigenvectors[:, eigenvalues > .15] if fresh else np.empty((3, 0))
        projector = basis @ basis.T
        rank = basis.shape[1]
        rotation, center, side = current['rotation'].copy(), current['center'].copy(), item['height']
        vertices = cube_vertices(center, rotation, side)
        axis = int(np.argmax(abs(rotation[2])))
        normal = rotation[:, axis]*np.sign(rotation[2, axis])
        status = 'conditional_prediction' if predicted else ('observed' if fresh else 'remembered')
        covariance = self.position_covariance(current, time)
        return {'shape': 'cube', 'center': center, 'rotation': rotation, 'height': side,
                'normal': normal, 'vertices': vertices, 'top': float((center+normal*side/2)[2]),
                'bottom': float(vertices[:, 2].min()), 'extent_z': float(np.sum(abs(rotation[2]))*side/2),
                'dims': np.repeat(side, 2), 'axis_dimensions': np.repeat(side, 3),
                'position_fully_observed': bool(rank == 3),
                'geometry_reliable': bool(rank == 3 and item['orientation_observed']),
                'inferred_orientation': not bool(fresh and item['orientation_observed']),
                'source': status, 'last_observed': item['last_observed'], 'age_s': time-item['last_observed'],
                'observed_translation_rank': rank, 'observed_translation_projector': projector, 'covariance_m2': covariance,
                'uncertainty_semantics': 'heuristic_not_calibrated_probability',
                'prediction_condition': current.get('condition') if predicted else None,
                'vertices_kind': 'inferred_rigid_landmarks_not_measured_individual_points'}

    def set_target(self, name, center, rotation=None):
        item = self.objects[name]
        center = _vector(center, (3,), 'target center')
        rotation = item['rotation'].copy() if rotation is None else _rotation(rotation)
        self.targets[name] = {'center': center, 'rotation': rotation,
                              'vertices': cube_vertices(center, rotation, item['height']),
                              'source': 'desired_not_observed'}

    def landmarks(self, name, time=None):
        state = self.estimate(name, time)
        result = {'center': state['center'], 'top_center': state['center']+state['normal']*state['height']/2,
                  'bottom_center': state['center']-state['normal']*state['height']/2}
        result.update({'corner_'+str(i): point.copy() for i, point in enumerate(state['vertices'])})
        return result

    def packet(self, time=None, details=False):
        """Finite, JSON-compatible text payload; concise by default, points on demand."""
        time = self.time if time is None else self.check_time(time)
        objects = {}
        estimates = {}
        for name in sorted(self.objects):
            state = self.estimate(name, time)
            estimates[name] = state
            item = self.objects[name]
            record = {'center_mm': np.round(state['center']*1000, 2).tolist(),
                      'size_mm': round(state['height']*1000, 2), 'status': state['source'],
                      'age_s': round(state['age_s'], 3), 'last_view': item['last_view'],
                      'observed_translation_rank': state['observed_translation_rank'],
                      'orientation_observed_now': not state['inferred_orientation'],
                      'position_sigma_mm': np.round(np.sqrt(np.maximum(0., np.diag(state['covariance_m2'])))*1000, 2).tolist(),
                      'last_update_delta_mm': np.round(item['delta']*1000, 2).tolist(),
                      'prediction_condition': state['prediction_condition'],
                      'prediction_contradicted': item['prediction_contradicted']}
            if name in self.targets:
                record['target_delta_mm'] = np.round((self.targets[name]['center']-state['center'])*1000, 2).tolist()
            if details:
                record['landmarks_mm'] = {key: np.round(value*1000, 2).tolist() for key, value in self.landmarks(name, time).items()}
                record['landmark_semantics'] = state['vertices_kind']
            objects[name] = record
        relations = []
        for upper, up in estimates.items():
            for lower, low in estimates.items():
                if upper == lower or up['center'][2] <= low['center'][2]:
                    continue
                rel = support_relation(up, low)
                relations.append({'upper': upper, 'lower': lower,
                                  'gap_mm': round(rel['lowest_vertex_gap_m']*1000, 2),
                                  'center_margin_mm': round(rel['projected_center_margin_m']*1000, 2),
                                  'fresh_full_poses': rel['both_poses_fully_observed'],
                                  'contact_confirmed': False})
        return {'schema': 'geometric_state_v1', 'time': time, 'frame': 'world', 'units': 'mm',
                'uncertainty': 'heuristic_not_calibrated', 'objects': objects,
                'relations': relations, 'recent_events': [dict(e) for e in self.events[-6:]],
                'task_success_certified': False}

    def text(self, time=None, details=False):
        packet = self.packet(time, details)
        if details:
            return str(packet)
        lines = ['Geometry state t='+str(packet['time'])+'; world frame; distances mm.',
                 'Uncertainty is heuristic. Remembered/predicted positions are not new observations.']
        for name, obj in packet['objects'].items():
            line = (name+': p='+str(obj['center_mm'])+' size='+str(obj['size_mm'])+
                    ' status='+obj['status']+' age_s='+str(obj['age_s'])+
                    ' measured_rank='+str(obj['observed_translation_rank'])+'/3'+
                    ' sigma~'+str(obj['position_sigma_mm'])+' delta='+str(obj['last_update_delta_mm']))
            if obj['prediction_condition']:
                line += ' IF '+obj['prediction_condition']
            if obj['prediction_contradicted']:
                line += ' PREDICTION_CONTRADICTED'
            if 'target_delta_mm' in obj:
                line += ' to_goal='+str(obj['target_delta_mm'])
            lines.append(line)
        for relation in packet['relations']:
            lines.append(relation['upper']+' above '+relation['lower']+
                         ': gap='+str(relation['gap_mm'])+' center_margin='+str(relation['center_margin_mm'])+
                         ' fresh_full_poses='+str(relation['fresh_full_poses']))
        lines.append('Relations are geometry estimates; contact and task success are not certified.')
        return '\n'.join(lines)
