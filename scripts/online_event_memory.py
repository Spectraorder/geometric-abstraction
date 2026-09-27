"""Bounded public memory, structured plan parsing and evidence-based retry gates."""
from collections import deque
import math


def changed(before, after, threshold=.01):
    if set(before) != set(after):
        return True
    for key in before:
        a, b = before[key], after[key]
        if (a is None) != (b is None):
            return True
        if a is not None and math.dist(a, b) > threshold:
            return True
    return False


class RetryGate:
    def __init__(self):
        self.failed = {}

    def blocked(self, operation, evidence):
        return operation in self.failed and not changed(self.failed[operation], evidence)

    def record(self, operation, evidence, success):
        if success:
            self.failed.pop(operation, None)
        else:
            self.failed[operation] = evidence


def parse_plan(entries, nodes):
    if len(entries) > 5:
        raise ValueError('A plan has at most five remaining operations')
    result = []
    for entry in entries:
        parts = entry.split(':')
        if len(parts) != 2 or parts[0] not in ('bridge', 'stack') or parts[1] not in nodes:
            raise ValueError('Plan entries must be skill:public_color')
        skill, key = parts
        if (len(nodes[key]['supports']) == 1) != (skill == 'stack'):
            raise ValueError('Plan skill is incompatible with its node')
        result.append({'action': 'execute', 'skill': skill, 'node': key})
    return result


class PublicMemory:
    def __init__(self):
        self.events = deque(maxlen=4)
        self.last_attempts = {}

    def record(self, decision, outcome):
        event = {k: decision[k] for k in ('action', 'skill', 'node')}
        event['status'] = outcome['status']
        self.events.append(event)
        if decision['node']:
            self.last_attempts.pop(decision['node'], None)
            self.last_attempts[decision['node']] = dict(event)
            while len(self.last_attempts) > 5:
                self.last_attempts.pop(next(iter(self.last_attempts)))

    def packet(self):
        return {'recent_events': list(self.events), 'last_attempt_per_object': self.last_attempts,
                'semantics': 'Historical action results only; not current object poses or certified success.'}


def validate_operation(decision, nodes):
    action, skill, node = (decision[k] for k in ('action', 'skill', 'node'))
    if action == 'execute':
        parse_plan([skill+':'+node], nodes)
    elif action == 'clear':
        if skill != 'none' or node not in nodes:
            raise ValueError('Clear requires a known public node and no skill')
    elif action in ('observe', 'release', 'replan', 'wait'):
        if skill != 'none' or node:
            raise ValueError('This operation requires no skill or object key')
    else:
        raise ValueError('Unknown operation')
