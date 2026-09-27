"""Development-only cooperative motion boundaries over the shared RGB-D controller.

The original controller's numerical motions and durations are retained. Generator
boundaries run on the simulator owner's thread, without concurrent scene access.
"""
import ast
import copy
from pathlib import Path
from types import MethodType
import time
import numpy as np
from online_bridge_skills import SharedSkills, append


def motion_generator(name):
    source = Path(__file__).resolve().parents[1]/'policies/pick_place_rgbd_shared.py'
    tree = ast.parse(source.read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RGBDPickPlace')
    function = copy.deepcopy(next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name))
    function.name = name+'_steps'
    class Boundaries(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) in ('self.move', 'self.api.wait'):
                event = ast.parse(f"yield {{'phase': {name!r}, 'boundary': {ast.unparse(node.value)!r}}}").body[0]
                return [node, event]
            return node
    function = Boundaries().visit(function)
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    scope = {'np': np}
    exec(compile(module, str(source)+'::cooperative', 'exec'), scope)
    return scope[function.name]


class EventSkills(SharedSkills):
    def __init__(self, api):
        super().__init__(api)
        for name in ('pick', 'place'):
            setattr(self.session.pnp, name+'_steps', MethodType(motion_generator(name), self.session.pnp))
        def pick_steps(session, color, transit_z=None):
            if session.pnp.held is None:
                session.placement_intent = None
            return (yield from session.pnp.pick_steps(color, transit_z))
        def place_steps(session, point):
            if session.pnp.held is not None and session.pnp.held['confirmed']:
                session.placement_intent = {'object': session.pnp.held['color'], 'point': np.asarray(point, float)}
            return (yield from session.pnp.place_steps(point))
        self.session.pick_steps = MethodType(pick_steps, self.session)
        self.session.place_steps = MethodType(place_steps, self.session)

    def evidence(self):
        """Private skill observations for retry gating; never supplied to Base."""
        fits = {}
        for key, node in self.nodes.items():
            fit = self.bridge.observe(self.session, self.api, node['color'], attempts=1)
            fits[key] = None if fit is None else np.asarray(fit['center'], float).tolist()
        held = self.holding()
        # A release changes execution preconditions even if an object's center
        # does not move enough to pass the geometric retry threshold.
        fits['@holding'] = (None if held is None else
                            [list(self.nodes).index(held['color'])+1, int(held['lift_confirmed'])])
        return fits

    def run_steps(self, skill, key):
        if key not in self.nodes or skill not in ('bridge', 'stack'):
            raise ValueError('Unknown skill/node')
        node = self.nodes[key]
        if (len(node['supports']) == 1) != (skill == 'stack'):
            raise ValueError('Skill does not match the public support graph')
        start = time.monotonic()
        try:
            result = yield from self._attempt(node)
        except (RuntimeError, ValueError) as error:
            result = {'status': 'recoverable_error', 'error': str(error)}
        result.update(skill=skill, node=key, holding=self.holding(), wall_seconds=time.monotonic()-start)
        self.last_result = result
        append(self.api.output/'skill_results.jsonl', result)
        return result

    def _attempt(self, node):
        b, api, session = self.bridge, self.api, self.session
        target, _, fresh = b.current_target(api, session, node, self.nodes, self.origin, self.scale)
        if not fresh:
            return {'status': 'support_unavailable'}
        held = self.holding()
        if held is None:
            good, _, _ = b.settled_check(api, session, node, target, self.scale)
            if good:
                return {'status': 'visually_consistent'}
            current = b.observe(session, api, node['color'], attempts=1)
            if current is None:
                return {'status': 'object_unavailable'}
            if b.orientation_evidence(node, b.fit_geometry(current)) is False:
                return {'status': 'orientation_incompatible'}
            picked = yield from session.pick_steps(node['color'],
                transit_z=float(target[2]+current['height']+.085))
            if not picked['held'] or picked['status'] != 'lift_visually_confirmed':
                return {'status': picked['status']}
            yield {'phase': 'lift_confirmed', 'boundary': 'refresh_supports_before_transfer'}
        elif held['color'] != node['color'] or not held['lift_confirmed']:
            return {'status': 'holding_uncertain' if held['color'] == node['color'] else 'another_object_held'}
        refreshed, _, fresh = b.current_target(api, session, node, self.nodes, self.origin, self.scale)
        if fresh:
            target = refreshed
        # A retained destination is a plan, not a transported measurement. The
        # unchanged local controller still observes the held object before release.
        placed = yield from session.place_steps(target.tolist())
        if placed['held']:
            return {'status': placed['status']}
        yield {'phase': 'released', 'boundary': 'settle_and_reobserve'}
        api.wait(.3)
        target, _, fresh = b.current_target(api, session, node, self.nodes, self.origin, self.scale)
        good = False
        if fresh:
            good, _, _ = b.settled_check(api, session, node, target, self.scale)
        return {'status': 'visually_consistent' if good and fresh else 'unconfirmed'}


def clear_steps(skills, key):
    """Expose identical clearing motions through the cooperative controller."""
    import online_bridge_recovery as recovery
    source = Path(recovery.__file__)
    tree = ast.parse(source.read_text())
    function = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'clear_object'))
    class Calls(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if ast.unparse(node.func) in ('session.pick', 'session.place'):
                node.func.attr += '_steps'
                return ast.copy_location(ast.YieldFrom(value=node), node)
            return node
    function = Calls().visit(function)
    namespace = dict(vars(recovery))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 str(source)+'::cooperative', 'exec'), namespace)
    return (yield from namespace['clear_object'](skills, key))


def drive(skills, operation, on_boundary=None):
    """Advance one finite command at a time; cancellation discards stale targets."""
    generator = (clear_steps(skills, operation['node']) if operation['action'] == 'clear'
                 else skills.run_steps(operation['skill'], operation['node']))
    try:
        while True:
            start = time.monotonic()
            event = next(generator)
            event.update(wall_seconds=time.monotonic()-start, holding=skills.holding(),
                         node=operation['node'], time=skills.session.time())
            append(skills.api.output/'motion_boundaries.jsonl', event)
            if on_boundary is not None and on_boundary(event):
                result = {'status': 'support_changed', 'node': operation['node'], 'holding': skills.holding()}
                skills.last_result = result
                append(skills.api.output/'skill_results.jsonl', result)
                return result
    except StopIteration as done:
        return done.value
    finally:
        generator.close()
