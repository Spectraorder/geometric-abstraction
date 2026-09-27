"""Bounded, episode-scoped reasoning checkpoints and separate conditional effects.

Prototype only: excluded from the frozen mixed-scene policy libraries. It does
not fuse poses, infer contacts, choose actions, or turn expectations into evidence.
"""
from collections import OrderedDict
import copy
import numpy as np
from checkpoint_geometry_prototype import compare_checkpoints


def common_translation_constraint(first, second):
    """Intersect reported world-space subspaces; not a calibrated error model."""
    # A vector belongs to both projector ranges exactly when their sum has
    # eigenvalue two. The tolerance is numerical, not a sensing confidence bound.
    values, vectors = np.linalg.eigh(np.asarray(first)+np.asarray(second))
    basis = vectors[:, values >= 2.-1e-8]
    return basis@basis.T, basis.shape[1]


class ReasoningCheckpoints:
    def __init__(self, episode, capacity=4):
        if not isinstance(episode, str) or not episode or type(capacity) is not int or capacity < 1:
            raise ValueError('Require an explicit episode ID and positive integer capacity')
        self.episode = episode
        self.capacity = capacity
        self.checkpoints = OrderedDict()
        self.effects = {}
        self.latest_time = -np.inf

    def _packet(self, packet, episode):
        if episode != self.episode:
            raise ValueError('Checkpoint episodes must match')
        if packet.get('coordinate_frame') != 'world' or packet.get('units') != 'meters':
            raise ValueError('Require world meters')
        stamp = float(packet['time'])
        if not np.isfinite(stamp) or stamp < self.latest_time:
            raise ValueError('Require finite nondecreasing observation time')
        objects = {}
        for name, row in packet['objects'].items():
            if row.get('status') != 'available':
                objects[name] = {'status': 'unknown', 'reason': 'No current visual estimate'}
                continue
            observed = row.get('last_observed')
            if observed is None or not np.isfinite(observed) or abs(float(observed)-stamp) > 1e-6:
                objects[name] = {'status': 'unknown', 'reason': 'Missing or stale observation timestamp'}
                continue
            center = np.asarray(row['center'], float)
            if center.shape != (3,) or not np.isfinite(center).all():
                raise ValueError('Require a finite estimated center')
            objects[name] = {'status': 'available', 'center': center.tolist(),
                             'position_fully_observed': bool(row.get('position_fully_observed', False))}
            if 'rotation' in row and 'constrained_axes' in row:
                rotation = np.asarray(row['rotation'], float)
                axes = np.asarray(row['constrained_axes'])
                if (rotation.shape != (3,3) or not np.isfinite(rotation).all()
                        or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-5)
                        or np.linalg.det(rotation) < .999 or axes.shape != (3,)
                        or axes.dtype != np.dtype(bool)):
                    raise ValueError('Require a proper rotation and three boolean observability flags')
                objects[name]['translation_projector'] = (rotation[:,axes]@rotation[:,axes].T).tolist()
                objects[name]['observable_rank'] = int(axes.sum())
        return {'coordinate_frame': 'world', 'units': 'meters', 'time': stamp, 'objects': objects}

    def remember(self, label, packet, *, episode):
        if not isinstance(label, str) or not label or label in self.checkpoints:
            raise ValueError('Require a new nonempty checkpoint label')
        current = self._packet(packet, episode)
        self.checkpoints[label] = current
        self.latest_time = current['time']
        while len(self.checkpoints) > self.capacity:
            removed, _ = self.checkpoints.popitem(last=False)
            self.effects.pop(removed, None)
        return copy.deepcopy(current)

    def expect_transform(self, label, name, transform, condition, *, episode):
        """Record an unverified rigid attachment/effect assumption before reobservation."""
        if episode != self.episode:
            raise ValueError('Checkpoint episodes must match')
        checkpoint = self.checkpoints[label]
        if checkpoint['time'] != self.latest_time:
            raise ValueError('Cannot create an expectation after a later observation')
        source = checkpoint['objects'][name]
        if source['status'] != 'available':
            raise ValueError('An expectation requires a current visual source')
        if not isinstance(condition, str) or not condition.strip():
            raise ValueError('State the unverified effect condition')
        matrix = np.asarray(transform, float)
        if (matrix.shape != (4,4) or not np.isfinite(matrix).all()
                or not np.allclose(matrix[3], [0,0,0,1], atol=1e-8)
                or not np.allclose(matrix[:3,:3].T@matrix[:3,:3], np.eye(3), atol=1e-5)
                or np.linalg.det(matrix[:3,:3]) < .999):
            raise ValueError('Require a finite proper rigid transform')
        if name in self.effects.get(label, {}):
            raise ValueError('An existing expectation is immutable')
        predicted = matrix[:3,:3]@np.asarray(source['center'])+matrix[:3,3]
        effect = {'kind': 'conditional_center_expectation', 'center_m': predicted.tolist(),
                  'condition': condition, 'condition_verified': False,
                  'based_on_checkpoint': label, 'based_on_time': checkpoint['time'],
                  'source_position_fully_observed': source['position_fully_observed'],
                  'source_translation_projector': copy.deepcopy(source.get('translation_projector')),
                  'source': 'caller_supplied_transform_of_visual_estimate',
                  'transform': matrix.tolist()}
        self.effects.setdefault(label, {})[name] = effect
        return copy.deepcopy(effect)

    def compare(self, label, packet, *, episode):
        current = self._packet(packet, episode)
        before = self.checkpoints[label]
        changes = compare_checkpoints(before, current)
        for name, change in changes['objects'].items():
            if change['status'] != 'estimated_change':
                continue
            first = before['objects'][name].get('translation_projector')
            second = current['objects'][name].get('translation_projector')
            change['jointly_constrained_translation_m'] = None
            change['joint_constraint_rank'] = None
            if first is not None and second is not None:
                projector, rank = common_translation_constraint(first, second)
                change['joint_constraint_rank'] = rank
                if rank:
                    change['jointly_constrained_translation_m'] = (projector@np.asarray(change['estimated_translation_m'])).tolist()
        expectations = {}
        for name, effect in self.effects.get(label, {}).items():
            item = copy.deepcopy(effect)
            observed = current['objects'].get(name, {'status': 'unknown'})
            item['comparison_status'] = 'unknown'
            if observed['status'] == 'available':
                residual = np.asarray(observed['center'])-np.asarray(effect['center_m'])
                item.update(comparison_status='estimated_residual', estimated_residual_m=residual.tolist())
                if 'translation_projector' in observed:
                    item['observable_rank'] = observed['observable_rank']
                    item['observable_residual_m'] = ((np.asarray(observed['translation_projector'])@residual).tolist()
                                                       if observed['observable_rank'] else None)
                    source_projector = effect['source_translation_projector']
                    item['jointly_constrained_residual_m'] = None
                    item['joint_constraint_rank'] = None
                    if source_projector is not None:
                        rotation = np.asarray(effect['transform'])[:3,:3]
                        transported = rotation@np.asarray(source_projector)@rotation.T
                        projector, rank = common_translation_constraint(transported, observed['translation_projector'])
                        item['joint_constraint_rank'] = rank
                        if rank:
                            item['jointly_constrained_residual_m'] = (projector@residual).tolist()
                item['current_position_fully_observed'] = observed['position_fully_observed']
            expectations[name] = item
        self.latest_time = current['time']
        return {'schema': 'reasoning_checkpoint_prototype_v1', 'episode': self.episode,
                'checkpoint': label, 'current_estimates': current, 'checkpoint_changes': changes,
                'conditional_effects': expectations, 'temporal_pose_fusion': False,
                'contact_or_stability_certified': False,
                'semantics': 'Current estimates, checkpoint differences and unverified expectations remain separate.'}
