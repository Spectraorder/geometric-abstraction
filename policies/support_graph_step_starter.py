"""Common starter with active-object checkpoints after pick, place and settle."""
import numpy as np


def run(api):
    goal = api.task(); nodes = goal['nodes']; by_key = {node['key']: node for node in nodes}
    session = Session(api, nodes)
    initial = {node['key']: session.observe(node['color']) for node in nodes}
    unit_key = goal['length_unit']['observed_cube']; unit = initial[unit_key]['height']
    anchor = initial[unit_key]['center'][:2]-unit*np.asarray(by_key[unit_key]['grid'])
    for node in nodes:
        if node['key'] == unit_key:
            continue
        if not node['supports']:
            xy = anchor+unit*np.asarray(node['grid']); point = np.r_[xy, session.table_z(xy)]
        else:
            supports = [session.observe(by_key[key]['color']) for key in node['supports']]
            offset = np.asarray(node['grid'])-np.mean([by_key[key]['grid'] for key in node['supports']], axis=0)
            xy = np.mean([fit['center'][:2] for fit in supports], axis=0)+unit*offset
            point = np.r_[xy, max(fit['top'] for fit in supports)]
        session.checkpoint(node['color'], 'before_pick')
        picked = session.pick(node['color'], transit_z=float(point[2]+.085))
        session.checkpoint(node['color'], 'pick', action_status=picked['status'])
        if not picked['held']:
            api.note(node['key']+': '+picked['status'])
            return
        result = session.place(point)
        session.checkpoint(node['color'], 'place', action_status=result['status'])
        api.note(node['key']+': '+result['status'])
        if result['status'] != 'released':
            return
        api.wait(.5)
        session.checkpoint(node['color'], 'settled', refresh=True)
