"""Sensor-only mixed rigid geometry and support-graph goals for unique colors."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from aspire_so100.ablation_geometry import observe_cube
from aspire_so100.beam_geometry import estimate_flat_beam_geometry
from aspire_so100.column_geometry import estimate_column_geometry
from aspire_so100.instance_geometry import inventory_points, inventory_color_mask
from aspire_so100.rigid_state import cube_rotation_error
from aspire_so100.structure_geometry import model_footprint, clip_polygon


def observe_mixed(frame, color, shape, model, table_plane, orientation='upright'):
    """Associate by unique color; shape is a public task label, sizes are visual.

    New models require an initial on-table view. Later calls retain dimensions.
    No stale pose prior is inserted when a current view cannot constrain a pose.
    """
    points, valid = inventory_points(frame)
    plane = np.asarray(table_plane)
    elevation = points[..., 2] - points[..., :2] @ plane[:2] - plane[2]
    mask = valid & inventory_color_mask(frame['rgb'], color) & (elevation > .004) & (elevation < .35)
    selected = dict(frame, valid_depth=mask)
    if shape == 'cube':
        measured = observe_cube(selected, color, model, plane)
    elif shape == 'thin_prism' and orientation == 'flat_x':
        try:
            measured = estimate_flat_beam_geometry(selected, color, plane, model)
        except ValueError:
            if model is None:
                raise
            measured = estimate_column_geometry(selected, color, shape, model, plane)
    elif shape in ('cylinder', 'prism', 'thin_prism'):
        measured = estimate_column_geometry(selected, color, shape, model, plane)
    else:
        raise ValueError('Unsupported declared rigid shape')
    constrained = np.asarray(measured['constrained_axes'], bool)
    measured['position_fully_observed'] = bool(np.all(constrained))
    measured['inferred_orientation'] = bool(measured.get('inferred_orientation', not measured.get('orientation_observed', False)))
    measured['observed_face_count'] = int(measured.get('observed_face_count', int(measured.get('observed_top', False))))
    measured['inlier_fraction'] = float(measured.get('inlier_fraction', 0.))
    if shape == 'cylinder':
        measured['axis'] = np.asarray(measured['normal'])
        measured['radius'] = float(measured['model']['radius'])
        measured['end_centers'] = np.asarray(measured['center']) + np.array([[-1.], [1.]])*measured['axis']*measured['height']/2
        measured['vertices_kind'] = 'sampled_model_rims_not_physical_vertices'
    else:
        measured['vertices_kind'] = 'inferred_rigid_box_corners'
    return measured


def shape_target(shape, center, dimensions):
    """Construct world-axis-aligned boxes or an upright cylinder from measured sizes."""
    dimensions, center = np.asarray(dimensions, float), np.asarray(center, float)
    if dimensions.shape != (3,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
        raise ValueError('Finite positive measured dimensions required')
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError('Finite goal center required')
    rotation = np.eye(3)
    if shape == 'cylinder':
        radius = float(min(dimensions[:2])/2)
        angle = np.arange(32)*2*np.pi/32
        ring = radius*np.column_stack((np.cos(angle), np.sin(angle), np.zeros(32)))
        vertices = np.concatenate([center+ring+[0., 0., sign*dimensions[2]/2] for sign in (1., -1.)])
    else:
        signs = np.array([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
        vertices = center+signs*dimensions/2
    result = {'shape': shape, 'center': center, 'rotation': rotation, 'normal': rotation[:, 2],
              'dims': dimensions[:2], 'height': float(dimensions[2]), 'axis_dimensions': dimensions,
              'vertices': vertices, 'geometry_reliable': False,
              'source': 'constructed_task_goal_from_visual_dimensions_and_public_graph'}
    if shape == 'cylinder':
        result.update(radius=radius, axis=rotation[:, 2],
                      end_centers=center+np.array([[0., 0., -dimensions[2]/2], [0., 0., dimensions[2]/2]]),
                      vertices_kind='sampled_model_rims_not_physical_vertices')
    else:
        result['vertices_kind'] = 'constructed_rigid_box_corners'
    return result


def graph_targets(nodes, models, table_plane, origin, unit):
    """Compile a public support DAG into rigid goals using measured dimensions."""
    colors = [node['color'] for node in nodes]
    if len(colors) != len(set(colors)):
        raise ValueError('This experiment requires one object per color')
    if not np.isfinite(unit) or unit <= 0:
        raise ValueError('A positive visually measured length unit is required')
    origin, plane = np.asarray(origin, float), np.asarray(table_plane, float)
    if origin.shape != (2,) or plane.shape != (3,) or not np.isfinite(np.r_[origin, plane]).all():
        raise ValueError('Finite world XY origin and tabletop plane required')
    targets, by_key = {}, {}
    for node in nodes:
        model = models[node['color']]
        dimensions = np.r_[model['dims'], model['height']]
        if node['orientation'] == 'flat_x':
            dimensions = np.sort(dimensions)[::-1]
        xy = origin + np.asarray(node['grid'])*unit
        supports = node['supports']
        if any(key not in by_key for key in supports):
            raise ValueError('Support nodes must precede their children')
        bottom = max((by_key[key]['center'][2]+by_key[key]['height']/2 for key in supports),
                     default=float(xy @ plane[:2]+plane[2]))
        target = shape_target(node['shape'], np.r_[xy, bottom+dimensions[2]/2], dimensions)
        target.update(support=[by_key[key]['color'] for key in supports] or ['table'],
                      color=node['color'], key=node['key'])
        by_key[node['key']] = target
        targets[node['color']] = target
    return targets


def mixed_state_error(observed, target):
    delta = np.asarray(target['center'])-observed['center']
    result = {'translation_to_goal_m': delta, 'center_error_m': float(np.linalg.norm(delta)),
              'xy_error_m': float(np.linalg.norm(delta[:2])),
              'pose_fully_observed': bool(observed['geometry_reliable'] and not observed['inferred_orientation']),
              'interpretation': 'visual_estimate_not_contact_or_stability_truth'}
    if observed['shape'] == 'cylinder':
        angle = float(np.arccos(np.clip(abs(np.asarray(observed['normal']) @ target['normal']), 0., 1.)))
        result.update(rotation_error_rad=angle, rotation_metric='unoriented_cylinder_axis; axial_yaw_ignored',
                      radius_error_m=float(observed['model']['radius']-target['radius']))
    else:
        distances = np.linalg.norm(np.asarray(observed['vertices'])[:, None]-target['vertices'][None, :], axis=2)
        rows, cols = linear_sum_assignment(distances**2)
        result['vertex_rms_m'] = float(np.sqrt(np.mean(distances[rows, cols]**2)))
        if observed['shape'] == 'cube':
            angle = cube_rotation_error(observed['rotation'], target['rotation'])
        else:
            axes = np.argsort(observed['axis_dimensions'])[::-1]
            angle = max(float(np.arccos(np.clip(abs(observed['rotation'][:, axis] @ target['rotation'][:, index]), 0., 1.)))
                        for index, axis in enumerate(axes))
        result.update(rotation_error_rad=angle, rotation_metric='cube_symmetry' if observed['shape']=='cube' else 'maximum_dimension_axis_error')
    return result


def mixed_support_relation(upper, lower):
    normal = np.asarray(lower['normal'])
    points = np.asarray(lower['vertices'])
    top_level = float(np.max(points @ normal))
    if upper['shape'] == 'cylinder':
        axis = np.asarray(upper['normal'])
        projection = float(np.clip(abs(axis @ normal), 0., 1.))
        extent = projection*upper['height']/2 + upper['model']['radius']*np.sqrt(1-projection**2)
        gap = float(np.asarray(upper['center']) @ normal-extent-top_level)
    else:
        gap = float(np.min(np.asarray(upper['vertices']) @ normal)-top_level)
    patch = clip_polygon(model_footprint(upper, 'bottom'), model_footprint(lower))
    area = 0. if len(patch)<3 else float(abs(np.sum(patch[:, 0]*np.roll(patch[:, 1], -1)-patch[:, 1]*np.roll(patch[:, 0], -1)))/2)
    return {'signed_surface_gap_m': gap, 'projected_overlap_area_m2': area,
            'both_poses_fully_observed': bool(upper['geometry_reliable'] and lower['geometry_reliable']),
            'contact_or_stability_certified': False}
