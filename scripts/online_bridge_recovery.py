"""A shared sensor-grounded clearing primitive for obstructed construction sites."""
import time
import numpy as np

from online_bridge_skills import append


def clearing_candidates(bridge, region, nodes, fits, origin, scale, key):
    """Screen table points against observed objects and the planned worksite."""
    source = fits[key]
    bounds = np.asarray(region['bounds'], float)
    xs = np.linspace(bounds[0, 0], bounds[1, 0], 15)
    ys = np.linspace(bounds[0, 1], bounds[1, 1], 15)
    footprint = source['footprint']
    margin = .35*scale
    candidates = []
    queries = [source['center'][:2].copy()]+[np.array([x, y]) for x in xs for y in ys]
    for xy in queries:
        if not bridge.inside_table(region, xy, footprint/2, margin):
            continue
        occupied = False
        for other, fit in fits.items():
            widths = (footprint+fit['footprint'])/2+margin
            if other != key and np.all(np.abs(xy-fit['center'][:2]) < widths):
                occupied = True
                break
            goal_xy = origin+scale*np.asarray(nodes[other]['grid'], float)
            if np.all(np.abs(xy-goal_xy) < widths):
                occupied = True
                break
        if not occupied:
            candidates.append(xy)
    return sorted(candidates, key=lambda point: float(np.linalg.norm(point-source['center'][:2])))


def clear_object(skills, key):
    """Move one obstruction aside; no task-progress oracle or environment reset."""
    if key not in skills.nodes:
        raise ValueError('Clear requires a public object key')
    api, session, b = skills.api, skills.session, skills.bridge
    started = time.monotonic()
    node = skills.nodes[key]
    result = None
    try:
        held = skills.holding()
        if held is not None and held['color'] != node['color']:
            result = {'status': 'another_object_held'}
        elif held is not None and not held['lift_confirmed']:
            result = {'status': 'holding_uncertain'}
        else:
            fits = {name: b.observe(session, api, item['color'], attempts=2)
                    for name, item in skills.nodes.items()}
            region = b.table_region(api)
            if region is None or any(fit is None for fit in fits.values()):
                result = {'status': 'clearance_geometry_unavailable'}
            else:
                points = clearing_candidates(b, region, skills.nodes, fits, skills.origin, skills.scale, key)
                point = None
                for xy in points[:12]:
                    surface = np.r_[xy, session.table_z(xy)]
                    if b.reachable(api, surface+np.array([0., 0., fits[key]['height']*1.2+.003])):
                        point = surface
                        break
                if point is None:
                    result = {'status': 'no_observed_reachable_clearance'}
                else:
                    already_clear = (held is None and np.linalg.norm(fits[key]['center'][:2]-point[:2]) < .003
                                     and abs(fits[key]['bottom']-point[2]) < .004)
                    if already_clear:
                        result = {'status': 'cleared_visually'}
                    elif held is None:
                        picked = session.pick(node['color'], transit_z=max(
                            float(point[2])+fits[key]['height']+.085, fits[key]['top']+.065))
                        if not picked['held'] or picked['status'] != 'lift_visually_confirmed':
                            result = {'status': picked['status']}
                    if result is None:
                        placed = session.place(point.tolist())
                        result = {'status': placed['status']}
                        if not placed['held']:
                            api.wait(.4)
                            fit = b.observe(session, api, node['color'], attempts=2)
                            consistent = (fit is not None and np.linalg.norm(fit['center'][:2]-point[:2]) <= .005
                                          and abs(fit['bottom']-point[2]) <= .005)
                            result = {'status': 'cleared_visually' if consistent else 'clearance_unconfirmed'}
    except (RuntimeError, ValueError) as error:
        result = {'status': 'recoverable_error', 'error': str(error)}
    result.update(action='clear', node=key, holding=skills.holding(), wall_seconds=time.monotonic()-started)
    skills.last_result = result
    append(api.output/'skill_results.jsonl', result)
    return result
