"""Action-free CPU scene geometry for policy reasoning over supplied RGB-D frames.

The interface never captures sensors or calls a robot. Dimensions are learned
from the initial on-table frame. Metric geometry, symbolic references, desired
states and conditional predictions have separate provenance.
"""
import numpy as np
from aspire_so100.ablation_geometry import observe_cube
from aspire_so100.instance_geometry import inventory_table
from aspire_so100.geometric_state import GeometricState
from aspire_so100.geometric_fusion import fuse_cube_observations
from aspire_so100.robot_visual_geometry import filter_robot_visual_geometry
from aspire_so100.symbolic_geometry import symbolic_packet, geometric_predicate, resolve_reference


class SceneGeometry:
    """Rigid-cube scene state with optional explicitly enabled temporal memory.

    With memory=False, a new timestamp discards old pose estimates, while frames
    at the same timestamp may contribute complementary visible directions.
    Rigid dimensions, the static table plane and desired targets are retained.
    Missing observations are never silently filled from simulator state.
    """
    def __init__(self, initial_frame, colors, memory=False, robot_geometry=None):
        self.identities = tuple(colors)
        if not self.identities or len(set(self.identities)) != len(self.identities):
            raise ValueError('Require distinct public color identities')
        if any(color not in ('red','green','blue','yellow') for color in self.identities):
            raise ValueError('Unsupported color identity')
        self.memory_enabled = bool(memory)
        prepared,_ = filter_robot_visual_geometry(initial_frame,robot_geometry)
        self.table_coefficients = inventory_table(prepared)
        self.robot_exclusions = {}
        self.dimension_models = {}
        self.belief = GeometricState()
        self.latest_time = float(initial_frame['time'])
        self.errors = {}
        self.current_fits = {color:{} for color in self.identities}
        self.rejected_views = []
        for color in self.identities:
            try:
                fit = observe_cube(prepared,color,table_plane=self.table_coefficients)
                self.dimension_models[color] = fit['model']
            except ValueError as error:
                self.errors[color] = 'Initial dimension measurement failed: '+str(error)
        self.refresh(initial_frame,robot_geometry=robot_geometry)

    def refresh(self, frame, view='side', robot_geometry=None):
        """Ingest one calibrated RGB-D frame; return finite symbolic/metric JSON."""
        stamp = float(frame['time'])
        if not np.isfinite(stamp) or stamp < self.latest_time:
            raise ValueError('Frame timestamps must be finite and nondecreasing')
        prepared,exclusion = filter_robot_visual_geometry(frame,robot_geometry)
        if stamp > self.latest_time:
            self.robot_exclusions = {}
        self.robot_exclusions[str(view)] = exclusion
        if not self.memory_enabled and stamp > self.latest_time:
            self.current_fits = {color:{} for color in self.identities}
        self.latest_time = stamp
        for color in self.identities:
            if color not in self.dimension_models:
                # Relearning dimensions after an unknown action could confuse
                # support height with object height, so fail explicitly instead.
                continue
            previous = self.belief.estimate(color,stamp) if self.memory_enabled and color in self.belief.objects else None
            try:
                fit = observe_cube(prepared,color,model=self.dimension_models[color],
                                   table_plane=self.table_coefficients,prior=previous)
                if self.memory_enabled:
                    self.belief.observe(color,fit,stamp,view)
                else:
                    self.current_fits[color][str(view)] = fit
                self.errors.pop(color,None)
            except ValueError as error:
                self.errors[color] = str(error)
                if self.memory_enabled and color in self.belief.objects:
                    self.belief.missing(color,stamp,str(error))
                elif not self.memory_enabled:
                    self.current_fits[color].pop(str(view),None)
        if not self.memory_enabled:
            self.refresh_instantaneous(stamp)
        return self.packet()

    def refresh_instantaneous(self, stamp):
        """Rebuild same-time estimates so camera arrival order cannot dominate."""
        updated = GeometricState()
        updated.targets = self.belief.targets
        updated.time = stamp
        self.rejected_views = []
        for color in self.identities:
            observations = [{'view':view,'fit':fit} for view,fit in self.current_fits[color].items()]
            if not observations:
                continue
            fused = fuse_cube_observations(color,observations,stamp)
            self.rejected_views.extend(dict(item,object=color) for item in fused['rejected'])
            if fused['state'] is None:
                self.errors[color] = fused['reason']
                continue
            updated.objects[color] = fused['state'].objects[color]
            for event in fused['state'].events:
                updated.record_event(event)
            if fused['rejected']:
                self.errors[color] = 'Rejected inconsistent views: '+', '.join(item['view'] for item in fused['rejected'])
            else:
                self.errors.pop(color,None)
        self.belief = updated

    def get(self, color):
        """Metric geometry in meters; arrays are copies, and partiality is explicit."""
        if color not in self.belief.objects:
            raise ValueError('No current geometry for '+str(color)+': '+self.errors.get(color,'unobserved identity'))
        return self.belief.estimate(color,self.latest_time)

    def table_height(self, xy):
        xy = np.asarray(xy,dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError('Table query requires finite world XY')
        return float(np.dot(self.table_coefficients[:2],xy)+self.table_coefficients[2])

    def reference(self, name):
        """Resolve object.center/top_face/bottom_face/corner_N without actions."""
        return resolve_reference(self.belief,name,self.latest_time)

    def relation(self, predicate, upper, lower):
        """Return true/false/unknown and metric residuals; never contact truth."""
        return geometric_predicate(self.belief,predicate,[upper,lower],self.latest_time)

    def target(self, color, center, rotation=None):
        """Record a desired metric state separately from current geometry."""
        self.belief.set_target(color,center,rotation)
        return self.packet()

    def predict(self, color, transform, time, condition):
        """Optional memory-only conditional SE(3) prediction; no action execution."""
        if not self.memory_enabled:
            raise ValueError('Conditional prediction requires explicitly enabled memory')
        result = self.belief.predict(color,transform,time,condition)
        self.latest_time = float(time)
        return result

    def packet(self, details=False):
        return dict(symbolic_packet(self.belief,self.latest_time,details),
                    memory_enabled=self.memory_enabled,
                    robot_geometry_exclusion={view:dict(value) for view,value in self.robot_exclusions.items()},
                    unavailable={name:self.errors.get(name,'unobserved') for name in self.identities if name not in self.belief.objects},
                    observation_warnings=dict(self.errors),
                    fusion='experimental_temporal_projection_v1' if self.memory_enabled else 'same_time_consistency_gated_v2',
                    rejected_views=[dict(item) for item in self.rejected_views])

    def text(self, details=False):
        text = self.belief.text(self.latest_time,details)
        if self.rejected_views:
            text += '\nRejected inconsistent views: '+', '.join(item['object']+'/'+item['view'] for item in self.rejected_views)
        return text


def instantaneous_scene(initial_frame, colors, robot_geometry=None):
    """Experiment interface that cannot silently enable temporal predictions."""
    return SceneGeometry(initial_frame,colors,memory=False,robot_geometry=robot_geometry)
