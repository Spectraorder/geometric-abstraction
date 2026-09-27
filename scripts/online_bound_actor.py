"""An independent decision schema over the frozen persistent actor transport."""
from types import FunctionType
import json
from online_bridge_actor import OnlineActor
from online_bound_target_skills import surface_point

SCHEMA = {
    'type':'object', 'additionalProperties':False,
    'required':['action','skill','node','surface_point_world_m','frame_id','pixels','reason'],
    'properties':{
        'action':{'type':'string','enum':['execute','sample_depth','observe','release','wait']},
        'skill':{'type':'string','enum':['bridge','stack','none']},
        'node':{'type':'string'},
        'surface_point_world_m':{'type':'array','items':{'type':'number'},'maxItems':3},
        'frame_id':{'type':'string'},
        'pixels':{'type':'array','maxItems':16,'items':{'type':'array','items':{'type':'integer'},'minItems':2,'maxItems':2}},
        'reason':{'type':'string'},
    },
}


def validate_response(response):
    if not isinstance(response,dict) or set(response)!=set(SCHEMA['required']):
        raise ValueError('Unexpected decision fields')
    if len(json.dumps(response,allow_nan=False))>6000:
        raise ValueError('Decision exceeded response size')
    for key in ('action','skill'):
        if response[key] not in SCHEMA['properties'][key]['enum']:
            raise ValueError('Invalid '+key)
    if not all(isinstance(response[key],str) for key in ('node','frame_id','reason')):
        raise ValueError('Invalid decision text fields')
    if not isinstance(response['surface_point_world_m'],list) or not isinstance(response['pixels'],list):
        raise ValueError('Target and pixels must be arrays')
    action=response['action']
    if action=='execute':
        surface_point(response['surface_point_world_m'])
        if response['skill']=='none' or not response['node'] or response['frame_id'] or response['pixels']:
            raise ValueError('Execute requires a node, skill and target only')
    else:
        if response['skill']!='none' or response['node'] or response['surface_point_world_m']:
            raise ValueError('Non-execution actions cannot bind a skill or target')
        if action=='sample_depth':
            if (not response['frame_id'] or not 1<=len(response['pixels'])<=16
                    or any(not isinstance(p,list) or len(p)!=2 or any(type(x) is not int for x in p)
                           for p in response['pixels'])):
                raise ValueError('Depth query requires a frame and 1–16 integer pixels')
        elif response['frame_id'] or response['pixels']:
            raise ValueError('Only sample_depth accepts pixel arguments')
    return response


def bound_request():
    # Reuse the audited transport bytecode with independent schema globals.
    # No mutation of the frozen module or cross-actor global patching occurs.
    original=OnlineActor.request
    namespace=dict(original.__globals__,SCHEMA=SCHEMA,validate_response=validate_response)
    function=FunctionType(original.__code__,namespace,'request',original.__defaults__,original.__closure__)
    function.__kwdefaults__=original.__kwdefaults__
    return function


class BoundActor(OnlineActor):
    request=bound_request()
