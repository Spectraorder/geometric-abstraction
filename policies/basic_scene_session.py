"""Surface-only reference interface over the unchanged historical RGB-D PnP."""


class BasicSession:
    def __init__(self, api, objects):
        self.api = api
        self.colors = [item['color'] for item in objects]
        if len(self.colors) != len(set(self.colors)):
            raise ValueError('Unique colors required')
        self.pnp = RGBDPickPlace(api)
        self.records = {}
        for color in self.colors:
            self.observe(color)

    def observe(self, color):
        try:
            result = self.pnp.observe(color)
        except (ValueError, RuntimeError) as error:
            self.records[color] = {'status': 'unavailable', 'reason': str(error),
                                   'observation_time': float(self.api.proprioception()['time'])}
            raise
        self.records[color] = {'status': 'available', 'observation_time': float(self.api.proprioception()['time']),
                               'observation': metric_copy(result)}
        return metric_copy(result)

    def snapshot(self):
        for color in self.colors:
            try:
                self.observe(color)
            except (ValueError, RuntimeError) as error:
                self.records[color] = {'status': 'unavailable', 'reason': str(error),
                                       'observation_time': float(self.api.proprioception()['time'])}
        return self.state()

    def state(self):
        stamp = float(self.api.proprioception()['time']); objects = {}
        for color, record in self.records.items():
            current = record['status'] == 'available' and abs(record['observation_time']-stamp) <= 1e-6
            row = {'status': 'available' if current else ('stale' if record['status'] == 'available' else 'unavailable'),
                   'last_observed': record['observation_time']}
            if current:
                row.update({k: record['observation'][k] for k in ('center', 'normal', 'height', 'dims', 'top', 'bottom', 'extent_z', 'source')})
                row['hidden_extent_inferred'] = True
            elif 'reason' in record:
                row['reason'] = record['reason']
            objects[color] = row
        return metric_copy({'schema': 'basic_surface_scene_v1', 'time': stamp, 'coordinate_frame': 'world',
                            'units': 'meters', 'objects': objects, 'contact_or_stability_certified': False}, plain=True)

    def get(self, color):
        row = self.state()['objects'][color]
        if row['status'] != 'available':
            raise ValueError('No current observation for '+color)
        return metric_copy(self.records[color]['observation'])

    def table_z(self, xy):
        return self.pnp.table_z(xy)

    def tool(self):
        return self.pnp.tool()

    def pick(self, color, transit_z=None):
        return metric_copy(self.pnp.pick(color, transit_z))

    def place(self, surface_point):
        return metric_copy(self.pnp.place(surface_point))

    def pick_and_place(self, color, surface_point):
        return metric_copy(self.pnp.pick_and_place(color, surface_point))
