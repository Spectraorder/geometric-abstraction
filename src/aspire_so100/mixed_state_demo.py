"""Unique-color support-graph geometry diagnostics around the existing controller."""
import numpy as np
from aspire_so100.rigid_state_demo import RigidStateObserver
from aspire_so100.rigid_state import geometry_json
from aspire_so100.instance_geometry import inventory_table
from aspire_so100.mixed_rigid_state import observe_mixed, graph_targets, mixed_state_error, mixed_support_relation


def select_structure_nodes(tracker, nodes, frame):
    """Use declared task shapes and unique color association, without classifying shape."""
    colors = [node['color'] for node in nodes]
    if len(colors) != len(set(colors)):
        raise ValueError('Unique colors are required for this geometry experiment')
    chosen = []
    for node in nodes:
        matches = [key for key, track in tracker.tracks.items() if track['visible'] and track['color']==node['color']]
        if len(matches) != 1:
            raise ValueError('Expected exactly one visible instance for '+node['color'])
        tracker.tracks[matches[0]]['shape'] = node['shape']
        chosen.append(matches[0])
    return chosen


class MixedStateObserver(RigidStateObserver):
    def __init__(self, api):
        self.api = api
        goal = api.task()
        self.nodes = goal['nodes']
        self.order = [node['color'] for node in self.nodes]
        if len(self.order) != len(set(self.order)):
            raise ValueError('Unique colors required')
        self.unit_color = next(node['color'] for node in self.nodes if node['key']==goal['length_unit']['observed_cube'])
        self.models = {}
        self.plane = None
        self.sequence = 0
        self.origin = None
        self.unit = None
        self.anchor_label = 'initial_provisional'

    def geometry_plan(self, origin, unit):
        self.origin = np.asarray(origin).copy()
        self.unit = float(unit)
        self.anchor_label = 'controller_visual_plan_revision_'+str(self.sequence)
        self.snapshot('target_plan')

    def snapshot(self, label, reanchor=False):
        side, wrist = self.api.camera('side'), self.api.camera('wrist')
        if self.plane is None:
            self.plane = inventory_table(side)
        views = {'side': side}
        if 'depth_m' in wrist:
            views['wrist'] = wrist
        objects, diagnostics = {}, {}
        for node in self.nodes:
            color = node['color']
            observations = {}
            for view, frame in views.items():
                try:
                    if color not in self.models and self.sequence != 0:
                        raise ValueError('No initial on-table dimension model; do not infer size after moving')
                    value = observe_mixed(frame, color, node['shape'], self.models.get(color), self.plane, node['orientation'])
                    if color not in self.models:
                        self.models[color] = value['model']
                    observations[view] = value
                except ValueError as error:
                    observations[view] = {'unavailable': str(error)}
            available = [(view, value) for view, value in observations.items() if 'unavailable' not in value]
            if available:
                view, value = max(available, key=lambda pair: (pair[1]['geometry_reliable'], np.count_nonzero(pair[1]['constrained_axes']), pair[1]['inlier_fraction']))
                objects[color] = dict(value, selected_view=view)
            diagnostics[color] = observations
        if self.unit is None and self.unit_color in self.models:
            self.unit = float(self.models[self.unit_color]['height'])
        if self.origin is None and self.unit_color in objects:
            reference = next(node for node in self.nodes if node['color']==self.unit_color)
            self.origin = objects[self.unit_color]['center'][:2]-np.asarray(reference['grid'])*self.unit
        targets = {}
        if self.origin is not None and all(color in self.models for color in self.order):
            targets = graph_targets(self.nodes, self.models, self.plane, self.origin, self.unit)
        errors = {color: mixed_state_error(value, targets[color]) for color, value in objects.items() if color in targets}
        by_key = {node['key']: node['color'] for node in self.nodes}
        relations = {}
        for node in self.nodes:
            for support in node['supports']:
                upper, lower = node['color'], by_key[support]
                if upper in objects and lower in objects:
                    relations[upper+'_on_'+lower] = mixed_support_relation(objects[upper], objects[lower])
        record = {'schema_version': 2, 'task': 'thin_beam_bridge', 'sequence': self.sequence, 'label': label,
                  'time': float(side['time']), 'coordinate_frame': 'world', 'units': 'meters_radians',
                  'table_plane': self.plane, 'objects': objects, 'per_view': diagnostics, 'targets': targets,
                  'target_anchor_label': self.anchor_label, 'errors': errors, 'relations': relations,
                  'goal': self.nodes, 'gripper': self.api.gripper_geometry(),
                  'selection': 'best_supported_single_view_not_fused', 'wrist_depth_enabled': 'depth_m' in wrist,
                  'control_role': 'diagnostic_observer_existing_structure_controller_executes_actions'}
        self.api.record_geometry(geometry_json(record))
        self.sequence += 1

    def note(self, text):
        self.api.note(text)
        if any(marker in text for marker in ('payload tilt=', 'release offset=', 'after withdrawal center=', 'NODE confirmed')):
            self.snapshot(text)


def run(api):
    observer = MixedStateObserver(api)
    observer.snapshot('initial')
    execute_column(observer)
    observer.snapshot('final')
