"""Explicit initial-scene assumptions, independent of desired task poses.

Development prototype only. It does not change the frozen study or infer any
initial assumption from a task node's desired orientation.
"""


class ObservationContext:
    """Bind a complete public inventory and its declared initial assumptions."""

    def __init__(self, packet):
        if set(packet) != {'schema', 'source', 'objects'}:
            raise ValueError('Expected only schema, source and initial objects')
        if packet['schema'] != 'initial_observation_context_v1':
            raise ValueError('Unsupported initial observation context')
        if packet['source'] != 'public_initial_scene_description':
            raise ValueError('Initial assumptions must be explicitly public')
        items = packet['objects']
        if not isinstance(items, list) or not items:
            raise ValueError('A nonempty public inventory is required')
        canonical = []
        for item in items:
            if set(item) != {'color', 'shape', 'initial_orientation_assumption'}:
                raise ValueError('Separate initial assumptions from goals and metric state')
            color, shape = item['color'], item['shape']
            orientation = item['initial_orientation_assumption']
            if color not in ('red', 'yellow', 'blue', 'green'):
                raise ValueError('Unsupported color identity')
            if shape not in ('cube', 'prism', 'thin_prism', 'cylinder'):
                raise ValueError('Unsupported observer shape')
            if orientation not in ('upright', 'flat_x') or (orientation == 'flat_x' and shape != 'thin_prism'):
                raise ValueError('Unsupported explicit initial orientation assumption')
            canonical.append((color, shape, orientation))
        if len({item[0] for item in canonical}) != len(canonical):
            raise ValueError('Initial inventory requires unique colors')
        self.items = tuple(canonical)

    def packet(self):
        return {'schema': 'initial_observation_context_v1',
                'source': 'public_initial_scene_description',
                'objects': [{'color': color, 'shape': shape,
                             'initial_orientation_assumption': orientation}
                            for color, shape, orientation in self.items]}

    def descriptors(self, requested):
        """Validate identities; task-only fields cannot change perception setup.

        The complete inventory is required by the shared session's initial model
        acquisition contract. Input ordering and desired orientation are ignored.
        The context order is stable across equivalent actor descriptor lists.
        """
        identities = []
        for item in requested:
            if not isinstance(item, dict) or 'color' not in item or 'shape' not in item:
                raise ValueError('Each requested object needs shape and color')
            identities.append((item['color'], item['shape']))
        expected = [(color, shape) for color, shape, _ in self.items]
        if len(identities) != len(expected) or set(identities) != set(expected):
            raise ValueError('Requested inventory differs from the public context')
        return [{'color': color, 'shape': shape, 'orientation': orientation}
                for color, shape, orientation in self.items]


def bind_session(factory, packet):
    """Bind trusted initial context before exposing Session(api, objects)."""
    context = ObservationContext(packet)

    def session(api, objects):
        return factory(api, context.descriptors(objects))

    return session
