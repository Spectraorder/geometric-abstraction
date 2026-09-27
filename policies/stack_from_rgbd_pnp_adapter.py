"""Thin single-column task adapter, separate from the reusable PnP mechanism.

Bundle with pick_place_rgbd_base.py. The public goal supplies colors/order;
all dimensions and placement coordinates come from camera observations.
"""
import numpy as np


def run(api):
    goal = api.task()
    if goal.get('kind') != 'single_column':
        raise ValueError('This adapter expects a public single-column task')
    levels = goal['levels']
    colors = [level['color'] for level in levels]
    if len(colors) < 2 or len(set(colors)) != len(colors):
        raise ValueError('At least two uniquely colored objects are required')
    pnp = RGBDPickPlace(api)
    for color in colors:
        pnp.observe(color)
    for lower, upper in zip(colors[:-1], colors[1:]):
        support = pnp.observe(lower)
        result = pnp.pick_and_place(upper, np.r_[support['center'][:2], support['top']])
        api.note(upper+' PnP status='+result['status'])
        if result['status'] != 'released':
            return
    api.note('All requested pick-and-place operations returned released; task success is not certified.')
