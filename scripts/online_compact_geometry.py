"""Lossless, self-contained factoring of repeated geometric reference metadata."""
from copy import deepcopy
import json

ENCODING = 'shared_reference_metadata_v1'
SHARED_FIELDS = ('coordinate_frame', 'units', 'observation_time', 'source',
                 'position_fully_observed', 'inferred_orientation', 'semantics')
CONTRACT = '''Compact geometry uses shared_reference_metadata_v1. In its geometry.objects
references, $shared:true inherits reference_defaults; explicit fields override them.
References without $shared inherit nothing, including unavailable references.
Coordinates are unchanged. Observation times are evidence timestamps, not guarantees
of freshness at execution. No metadata is inherited from earlier messages.
'''


def wire(value):
    """Canonical JSON also distinguishes booleans, integers and floating values."""
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def references(geometry):
    for obj in geometry['objects']:
        yield from obj['references'].values()


def compact_geometry(geometry):
    """Factor identical evidence fields; retain raw geometry if factoring is larger."""
    wire(geometry)
    if 'encoding' in geometry:
        raise ValueError('Geometry already uses an encoding envelope')
    original_refs = list(references(geometry))
    if any('$shared' in ref for ref in original_refs):
        raise ValueError('Reserved reference metadata key')
    eligible = [ref for ref in original_refs
                if ref.get('position_fully_observed') is True and ref.get('status') != 'unavailable']
    if not eligible:
        return deepcopy(geometry)
    shared = {key: deepcopy(eligible[0][key]) for key in SHARED_FIELDS
              if key in eligible[0] and all(key in ref and wire(ref[key]) == wire(eligible[0][key])
                                            for ref in eligible)}
    compact = deepcopy(geometry)
    for ref in references(compact):
        if ref.get('position_fully_observed') is True and ref.get('status') != 'unavailable':
            for key in shared:
                del ref[key]
            ref['$shared'] = True
    encoded = dict(encoding=ENCODING, reference_defaults=shared, geometry=compact)
    return encoded if len(wire(encoded)) < len(wire(geometry)) else deepcopy(geometry)


def expand_geometry(value):
    """Restore the exact reference records; missing fields never acquire defaults."""
    wire(value)
    if 'encoding' not in value:
        return deepcopy(value)
    if value['encoding'] != ENCODING or set(value) != {'encoding', 'reference_defaults', 'geometry'}:
        raise ValueError('Unknown geometry encoding')
    defaults = value['reference_defaults']
    if not isinstance(defaults, dict) or not set(defaults) <= set(SHARED_FIELDS):
        raise ValueError('Unknown inherited metadata')
    result = deepcopy(value['geometry'])
    for ref in references(result):
        if '$shared' not in ref:
            continue
        if ref['$shared'] is not True or ref.get('status') == 'unavailable':
            raise ValueError('Invalid inheritance marker')
        explicit = {k: v for k, v in ref.items() if k != '$shared'}
        ref.clear()
        ref.update(deepcopy(defaults))
        ref.update(explicit)
    return result


def compact_prompt(prompt, *, explain):
    """Transform only Full geometry; leave other feedback and Base text unchanged."""
    prefix, marker, body = prompt.partition('Current feedback:\n')
    if not marker:
        raise ValueError('Missing feedback boundary')
    packet = json.loads(body)
    if prefix.endswith('Interface: basic\n'):
        if 'geometry' in packet:
            raise ValueError('Base must not receive geometric references')
        return prompt, False
    if not prefix.endswith('Interface: symbolic\n') or 'geometry' not in packet:
        raise ValueError('Expected Full geometric feedback')
    compact = compact_geometry(packet['geometry'])
    if compact.get('encoding') != ENCODING:
        return prompt, False
    if wire(expand_geometry(compact)) != wire(packet['geometry']):
        raise ValueError('Compact geometry did not preserve all fields')
    packet['geometry'] = compact
    explanation = CONTRACT + '\n' if explain else ''
    return prefix + explanation + marker + json.dumps(packet, separators=(',', ':'), allow_nan=False), True
