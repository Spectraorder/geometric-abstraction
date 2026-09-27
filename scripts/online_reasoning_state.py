"""Compact geometric constraint evidence and dependency-local retry gates.

This development representation does not choose actions or certify physical success.
"""
from copy import deepcopy
import math


def truth(value):
    if value is True or value == 'true':
        return 'true'
    if value is False or value == 'false':
        return 'false'
    return 'unknown'


def conjunction(values):
    values = list(values)
    return 'false' if 'false' in values else 'unknown' if 'unknown' in values else 'true'


def ancestors(nodes, key):
    found = set()
    visiting = set()
    def walk(node):
        if node in visiting:
            raise ValueError('Cyclic support graph')
        visiting.add(node)
        for parent in nodes[node]['supports']:
            if parent not in nodes:
                raise ValueError('Unknown support')
            if parent not in found:
                walk(parent)
                found.add(parent)
        visiting.remove(node)
    walk(key)
    return found


def geometric_constraints(packet, nodes):
    """Separate local alignment from transitive goal-geometry consistency."""
    source = {row['node']: row for row in packet['objects']}
    local = {}
    checks = {}
    for key, node in nodes.items():
        row = source.get(key, {})
        checks[key] = {'translation': truth(row.get('placement', {}).get('translation_within_tolerance')),
                       'orientation': truth(row.get('orientation_compatible'))}
        # A residual must refer to a fully observed current position. Missing
        # evidence cannot silently become an accepted metric pose.
        if row.get('status') != 'current_estimate' or not row.get('position_fully_observed'):
            checks[key]['translation'] = 'unknown'
        checks[key].update({f'support:{parent}': truth(row.get('supports', {}).get(parent))
                           for parent in node['supports']})
        local[key] = conjunction(checks[key].values())
    objects = []
    for key in nodes:
        row = source.get(key, {})
        parents = ancestors(nodes, key)
        item = {'id': key, 'local_goal_geometry': local[key],
                'support_chain_geometry': conjunction(local[parent] for parent in parents),
                'violated': [name for name, value in checks[key].items() if value == 'false'],
                'unknown': [name for name, value in checks[key].items() if value == 'unknown']}
        unmet = sorted(parent for parent in parents if local[parent] == 'false')
        uncertain = sorted(parent for parent in parents if local[parent] == 'unknown')
        if unmet:
            item['violated_ancestors'] = unmet
        if uncertain:
            item['unknown_ancestors'] = uncertain
        residual = row.get('placement', {}).get('remaining_translation_m')
        if residual is not None and checks[key]['translation'] != 'unknown':
            item['translation_error_m'] = residual
        objects.append(item)
    return {'time': packet['time'], 'objects': objects,
            'translation_reference': packet['placement_reference'],
            'tolerance_m': packet['placement_tolerance_m']}


def project_feedback(packet, nodes):
    result = deepcopy(packet)
    if 'geometry' in result:
        result['geometry'] = geometric_constraints(result['geometry'], nodes)
    held = result.get('holding')
    if held is not None:
        result['holding_constraint'] = (
            'Only resume or clear the held object; observe, release, wait and replan are also allowed.'
            if held['lift_confirmed'] else
            'No placement or new pick is available until holding is resolved; observe or release.')
    return result


class RelevantRetryGate:
    """Only useful changes in an operation's dependencies unlock its retry."""
    def __init__(self, nodes):
        self.nodes = nodes
        self.failed = {}

    def keys(self, operation):
        action, _, key = operation
        if action == 'clear':
            # Free-space selection depends on every visible object.
            return set(self.nodes) | {'@holding'}
        return ancestors(self.nodes, key) | {key, '@holding'}

    def project(self, operation, evidence):
        return {key: deepcopy(evidence.get(key)) for key in self.keys(operation)}

    def blocked(self, operation, evidence):
        if operation not in self.failed:
            return False
        old = self.failed[operation]
        current = self.project(operation, evidence)
        for key, before in old.items():
            after = current[key]
            if key == '@holding':
                if before != after:
                    return False
            elif after is not None and (before is None or math.dist(before, after) > .01):
                return False
        return True

    def record(self, operation, evidence, success):
        if success:
            self.failed.pop(operation, None)
        else:
            self.failed[operation] = self.project(operation, evidence)


def observed_center(fit):
    """Return a point only when every translation direction is observed."""
    if fit is None or fit.get('fully_observed') is not True:
        return None
    center = list(map(float, fit['center']))
    return center if len(center) == 3 and all(math.isfinite(v) for v in center) else None
