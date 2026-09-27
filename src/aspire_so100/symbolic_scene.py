"""Typed geometric references and desired constraints over the same numeric scene.

No sensing, task selection, action execution or independent pose estimation occurs
here. Relations concern estimated geometry, never certified contact or stability.
"""
import numpy as np
from aspire_so100.metric_scene import metric_copy, metric_vector
from aspire_so100.rigid_goals import point_axis_goal
from aspire_so100.structure_geometry import convex_polygon, clip_polygon


def scene_face(fit, top=True):
    center, rotation = np.asarray(fit['center']), np.asarray(fit['rotation'])
    sign = 1. if top else -1.
    if fit['shape'] == 'cylinder':
        axis = rotation[:, 2]
        up = axis*(1. if axis[2] >= 0 else -1.)
        normal = sign*up; height = float(fit['axis_dimensions'][2])
        radius = float(fit['model']['radius'])
        face_center = center+normal*height/2
        angle = np.arange(32)*2*np.pi/32
        rim = face_center+radius*(np.cos(angle)[:, None]*rotation[:, 0]+np.sin(angle)[:, None]*rotation[:, 1])
        return {'type': 'disk', 'center_m': face_center, 'normal': normal, 'radius_m': radius,
                'rim_samples_m': rim, 'sample_semantics': 'model_rim_samples_not_physical_vertices',
                'semantics': 'upward_facing_end_cap_not_topmost_curved_surface'}
    axis = int(np.argmax(abs(rotation[2]))); normal = sign*np.sign(rotation[2, axis])*rotation[:, axis]
    dimensions = np.asarray(fit['axis_dimensions'])
    face_center = center+normal*dimensions[axis]/2
    other = [i for i in range(3) if i != axis]
    signs = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    corners = face_center+(signs*dimensions[other]/2)@rotation[:, other].T
    return {'type': 'plane_patch', 'center_m': face_center, 'normal': normal, 'vertices_m': corners,
            'semantics': 'current_geometric_face_not_identified_material_face'}


def scene_direction_observed(fit, directions):
    projection = np.asarray(fit['observed_translation_projector'])
    return bool(all(np.linalg.norm((np.eye(3)-projection)@direction) < .05 for direction in directions))


def scene_polygon(face):
    points = face['rim_samples_m'] if face['type'] == 'disk' else face['vertices_m']
    return convex_polygon(np.asarray(points)[:, :2])


class SymbolicScene:
    def __init__(self, metric):
        self.metric = metric

    def reference(self, reference, stamp=None):
        if not isinstance(reference, str) or reference.count('.') != 1:
            raise ValueError('References use object_id.geometric_role')
        name, role = reference.split('.')
        fit = self.metric.get(name, stamp)
        if role == 'center':
            value = {'type': 'point', 'position_m': fit['center']}
        elif role in ('top_face', 'bottom_face'):
            value = scene_face(fit, role == 'top_face')
        elif role in ('axis', 'long_axis', 'short_axis'):
            if fit['shape'] == 'cylinder':
                if role != 'axis':
                    raise ValueError('Cylinder references use axis, not artificial box axes')
                direction = fit['rotation'][:, 2]
            else:
                if role == 'axis':
                    raise ValueError('A box requires a uniquely longest or shortest axis')
                dims = np.asarray(fit['axis_dimensions']); axis = int(np.argmax(dims) if role == 'long_axis' else np.argmin(dims))
                if np.count_nonzero(abs(dims-dims[axis]) < 1e-6) != 1:
                    raise ValueError('Equal lengths do not identify a unique '+role)
                direction = fit['rotation'][:, axis]
            value = {'type': 'unoriented_axis', 'origin_m': fit['center'], 'direction': direction,
                     'semantics': 'positive_and_negative_directions_are_equivalent'}
        elif role.startswith('corner_'):
            if fit['shape'] == 'cylinder':
                raise ValueError('Cylinders do not have physical corners')
            index = int(role[7:]); vertices = np.asarray(fit['vertices'])
            if not 0 <= index < 8:
                raise ValueError('Box corner index must be 0..7')
            value = {'type': 'point', 'position_m': vertices[index],
                     'semantics': 'inferred_current_corner_not_material_point_tracking'}
        else:
            raise ValueError('Unknown geometric role: '+role)
        return metric_copy(dict(value, reference=reference, coordinate_frame='world', units='meters',
                                observation_time=fit['observation_time'], source=fit['source'],
                                position_fully_observed=fit['position_fully_observed'],
                                inferred_orientation=fit['inferred_orientation']))

    def relation(self, predicate, first, second, stamp=None, tolerance_m=.004):
        if predicate not in ('above', 'aligned_xy', 'support_candidate', 'contact'):
            raise ValueError('Unsupported geometric predicate')
        if first == second or not np.isfinite(tolerance_m) or tolerance_m <= 0:
            raise ValueError('Relations need distinct objects and a positive meter tolerance')
        result = {'predicate': predicate, 'arguments': [first, second], 'truth': 'unknown', 'residuals': {},
                  'semantics': 'thresholds_on_current_estimated_geometry_not_physical_truth',
                  'contact_or_stability_certified': False}
        if predicate == 'contact':
            return dict(result, reason='No contact sensor')
        try:
            upper, lower = self.metric.get(first, stamp), self.metric.get(second, stamp)
        except (ValueError, KeyError):
            return dict(result, reason='Current object observation unavailable')
        delta = upper['center']-lower['center']; directions = np.eye(3)
        if predicate == 'above':
            available = all(scene_direction_observed(fit, directions[2:]) for fit in (upper, lower))
            satisfies = bool(delta[2] > 0); residuals = {'center_z_difference_m': float(delta[2])}
        elif predicate == 'aligned_xy':
            available = all(scene_direction_observed(fit, directions[:2]) for fit in (upper, lower))
            distance = float(np.linalg.norm(delta[:2])); satisfies = distance <= tolerance_m
            residuals = {'center_xy_distance_m': distance, 'tolerance_m': tolerance_m}
        else:
            top, bottom = scene_face(lower), scene_face(upper, False)
            normal = top['normal']; origin = top['center_m']
            if upper['shape'] == 'cylinder':
                alignment = float(np.clip(abs(upper['rotation'][:, 2]@normal), 0., 1.))
                extent = alignment*upper['axis_dimensions'][2]/2+upper['model']['radius']*np.sqrt(1-alignment**2)
                gap = float((upper['center']-origin)@normal-extent)
            else:
                gap = float(np.min((upper['vertices']-origin)@normal))
            patch = clip_polygon(scene_polygon(bottom), scene_polygon(top))
            area = 0. if len(patch) < 3 else float(abs(np.sum(patch[:, 0]*np.roll(patch[:, 1], -1)-patch[:, 1]*np.roll(patch[:, 0], -1)))/2)
            orientation = all(not fit['inferred_orientation'] for fit in (upper, lower))
            horizontal_faces = bool(top['normal'][2] > np.cos(np.deg2rad(8)) and -bottom['normal'][2] > np.cos(np.deg2rad(8)))
            above_known = all(scene_direction_observed(fit, directions[2:]) for fit in (upper, lower))
            gap_known = orientation and all(scene_direction_observed(fit, [normal]) for fit in (upper, lower))
            overlap_known = orientation and all(scene_direction_observed(fit, directions[:2]) for fit in (upper, lower))
            known = [above_known, gap_known, overlap_known, orientation]
            satisfied = [bool(delta[2] > 0), abs(gap) <= tolerance_m, area > 0, horizontal_faces]
            checks = [('true' if yes else 'false') if seen else 'unknown' for seen, yes in zip(known, satisfied)]
            truth = 'false' if 'false' in checks else ('true' if all(value == 'true' for value in checks) else 'unknown')
            available = all(known); satisfies = all(satisfied)
            residuals = {'signed_surface_gap_m': gap, 'projected_overlap_area_m2': area, 'gap_tolerance_m': tolerance_m,
                         'overlap_approximation': 'convex_XY_projection_with_sampled_cylinder_rims'}
            result.update(residuals=residuals, estimate_satisfies=satisfies, required_geometry_observed=available,
                          truth=truth, component_truth=dict(zip(('above', 'gap', 'overlap', 'horizontal_faces'), checks)))
            return result
        result.update(residuals=residuals, estimate_satisfies=satisfies, required_geometry_observed=available,
                      truth=('true' if satisfies else 'false') if available else 'unknown')
        return result

    def align(self, source_reference, target_point, target_normal, stamp=None):
        """Construct a desired rigid pose from a current face and caller's target.

        target_normal is the desired OUTWARD normal of the source face. To mate a
        bottom face to a support top face, the policy chooses the opposite normal.
        This operator chooses no object order, support or world target location.
        """
        source = self.reference(source_reference, stamp)
        if source['type'] not in ('plane_patch', 'disk'):
            raise ValueError('Alignment requires a face reference')
        name = source_reference.split('.')[0]; fit = self.metric.get(name, stamp)
        goal = point_axis_goal(source['center_m'], source['normal'], target_point, target_normal)
        goal.update(object_id=name, source_reference=source_reference, source_observation_time=fit['observation_time'],
                    desired_center_m=goal['rotation']@fit['center']+goal['translation'],
                    desired_rotation=goal['rotation']@fit['rotation'],
                    source_position_fully_observed=fit['position_fully_observed'],
                    source_orientation_inferred=fit['inferred_orientation'])
        return goal

    def packet(self, stamp=None, details=False):
        packet = self.metric.packet(stamp, details)
        names = [name for name, row in packet['objects'].items() if row['status'] == 'available']
        facts = []
        for first in names:
            packet['objects'][first]['references'] = {role: first+'.'+role for role in ('center', 'top_face', 'bottom_face')}
            for second in names:
                if first != second:
                    facts.extend(self.relation(pred, first, second, stamp) for pred in ('above', 'support_candidate'))
        return metric_copy(dict(packet, schema='mixed_symbolic_scene_v1', facts=facts,
                                contact_state='unknown_without_a_contact_sensor'), plain=True)

    def text(self, stamp=None):
        packet = self.packet(stamp); lines = ['World geometry in meters at t='+str(packet['time'])]
        for name, value in packet['objects'].items():
            if value['status'] != 'available':
                lines.append(name+': '+value['status']+'; last observation='+str(value['last_observed']))
            else:
                center = [round(v, 5) for v in value['center']]
                lines.append(name+' ('+value['shape']+'): center='+str(center)+'; observed axes='+str(value['constrained_axes']))
        for fact in packet['facts']:
            if fact['predicate'] == 'support_candidate':
                lines.append(str(fact['arguments'])+': support_candidate='+fact['truth']+'; '+str(fact['residuals']))
        return '\n'.join(lines)
