"""Optional pixel-guided reassociation; geometry remains the controller's gate."""
import numpy as np
from aspire_so100.instance_geometry import inventory_points, instance_evidence
from aspire_so100.pointing_geometry import locate_pointed_component


class PointingRecovery:
    def __init__(self, api, tracker):
        self.api = api
        self.tracker = tracker
        self.attempts = set()
        self.overrides = {}
        self.protected = set()

    def apply_frame(self, frame):
        for key, saved in list(self.overrides.items()):
            if (float(frame['time']) != saved['time']
                    or not np.array_equal(frame['rgb'], saved['rgb'])):
                del self.overrides[key]
                continue
            self.tracker.tracks[key] = dict(saved['track'])
            for other in saved['duplicates']:
                if other in self.tracker.tracks:
                    self.tracker.tracks[other]['visible'] = False

    def reassociate(self, frame, node, name=None):
        token = (float(frame['time']), node['color'], node['shape'], name)
        if token in self.attempts:
            return False
        self.attempts.add(token)
        self.api.note('POINTING request color=' + node['color'] + ' shape=' + node['shape'])
        reply = self.api.point_target(node['color'], node['shape'], node['orientation'], float(frame['time']))
        if reply['status'] != 'point':
            self.api.note('POINTING abstained')
            return False
        try:
            selected = locate_pointed_component(frame, reply['point_px'], node['color'])
        except ValueError as error:
            self.api.note('POINTING rejected: ' + str(error))
            return False
        mask = selected['mask']
        overlaps = []
        for key, track in self.tracker.tracks.items():
            if track['color'] == node['color'] and track['visible']:
                overlap = np.count_nonzero(mask & track['mask']) / max(1, np.count_nonzero(mask))
                if overlap > .8:
                    overlaps.append(key)
        if name is None:
            if len(overlaps) > 1:
                self.api.note('POINTING rejected: multiple existing identities overlap')
                return False
            if overlaps:
                name = overlaps[0]
            else:
                name = node['color'] + '_' + str(self.tracker.next_id)
                self.tracker.next_id += 1
        if any(key != name and key in self.protected for key in overlaps):
            self.api.note('POINTING rejected: proposed region belongs to another task instance')
            return False
        points, valid = inventory_points(frame)
        evidence = instance_evidence(points, mask, self.tracker.plane)
        # A semantic proposal only labels the candidate. The unmodified
        # downstream geometry and placement checks must still accept it.
        track = dict(evidence, color=node['color'], shape=node['shape'], id=name,
                     mask=mask, visible=True, initial_eligible=True)
        self.overrides[name] = {'time': float(frame['time']), 'rgb': frame['rgb'].copy(),
                                'track': track, 'duplicates': [key for key in overlaps if key != name]}
        self.apply_frame(frame)
        self.api.note('POINTING component selected id=' + name + ' pixel=' + str(reply['point_px'])
                      + ' pixels=' + str(selected['component_pixels']) + '; geometry still required')
        return True
