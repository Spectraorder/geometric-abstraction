"""Evidence-qualified retry observations over the unchanged shared motion skills."""
from online_event_skills import EventSkills
from online_reasoning_state import observed_center
import numpy as np


class ReasoningSkills(EventSkills):
    def initialize_workspace(self):
        if not getattr(self.api, 'reserve_pick_space', False):
            return super().initialize_workspace()
        from online_pick_clearance import select_workspace, conditional_center_box
        from scipy.ndimage import binary_erosion
        b,api,session=self.bridge,self.api,self.session
        fits={key:b.observe(session,api,self.nodes[key]['color'],attempts=2) for key in self.order}
        if any(fit is None for fit in fits.values()):
            raise ValueError('Pickup clearance needs current initial shape fits')
        self.scale=b.reference_side(self.nodes[self.task['length_unit']['observed_cube']],
                                    fits[self.task['length_unit']['observed_cube']])
        region=b.table_region(api)
        if self.scale is None or region is None:raise ValueError('No current table/scale estimate')
        rgb,xyz,valid=session.pnp.capture()
        color_mask=type(session.backend).__init__.__globals__['inventory_color_mask']
        envelopes={};bounds={}
        for key,fit in fits.items():
            radius=np.zeros(3)
            if observed_center(fit) is None:
                raw=session.get(key)
                mask=binary_erosion(color_mask(rgb,self.nodes[key]['color']),iterations=1)&valid
                points=xyz[mask]
                center,radius,bounds[key]=conditional_center_box(points,raw['rotation'],raw['axis_dimensions'],
                    raw['center'],raw['constrained_axes'])
                fit=dict(fit,center=center);fits[key]=fit
            envelope=api.pickup_envelopes(fit,session.pnp.grasp_fraction)
            envelope[:,0]-=radius;envelope[:,1]+=radius
            envelopes[key]=envelope
        self.origin,_,counts=select_workspace(b,api,session,self.nodes,self.order,fits,region,self.scale,envelopes)
        api.record_geometry({'kind':'pickup_clearance_workspace','origin':self.origin,'scale':self.scale,
                             'sampled_jaw_envelopes':envelopes,'candidate_counts':counts,
                             'conditional_planning_bounds':bounds,'planning_fits':fits,
                             'semantics':'Sampled robot-only jaw AABBs; not a complete swept-volume certificate.'})
        if self.origin is None:raise ValueError('No reachable worksite preserves sampled pickup clearance')

    def evidence(self):
        fits = {key: observed_center(self.bridge.observe(self.session, self.api,
                                                       node['color'], attempts=1))
                for key, node in self.nodes.items()}
        held = self.holding()
        fits['@holding'] = (None if held is None else
                            [list(self.nodes).index(held['color'])+1, int(held['lift_confirmed'])])
        return fits
