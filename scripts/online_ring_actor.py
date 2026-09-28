"""Persistent online Ring decisions using the audited actor and uniform HTTP."""
import ast
import inspect
import json
import math
import textwrap
from online_bridge_actor import OnlineActor
from bridge_actor_transport import HTTP_ARGS

SCHEMA = {'type':'object','additionalProperties':False,
    'required':['action','point_m','roll','jaw','seconds','frame_id','pixels','reason'],
    'properties':{
        'action':{'type':'string','enum':['grasp_lift','move_pad','release','observe','sample_depth','wait']},
        'point_m':{'type':'array','items':{'type':'number'},'maxItems':3},
        'roll':{'type':'number'},'jaw':{'type':'number'},'seconds':{'type':'number'},
        'frame_id':{'type':'string'},
        'pixels':{'type':'array','maxItems':16,'items':{'type':'array','items':{'type':'integer'},'minItems':2,'maxItems':2}},
        'reason':{'type':'string'}}}


def validate_response(value):
    if not isinstance(value,dict) or set(value) != set(SCHEMA['required']):
        raise ValueError('Unexpected decision fields')
    if len(json.dumps(value,allow_nan=False)) > 6000:
        raise ValueError('Decision too large')
    if value['action'] not in SCHEMA['properties']['action']['enum']:
        raise ValueError('Unknown action')
    if not isinstance(value['reason'],str) or not isinstance(value['frame_id'],str):
        raise ValueError('Invalid decision text')
    for key in ('roll','jaw','seconds'):
        if type(value[key]) not in (int,float) or not math.isfinite(value[key]):
            raise ValueError('Invalid numeric decision')
    if not -2.79 <= value['roll'] <= 2.79 or not -.174 <= value['jaw'] <= 1.75:
        raise ValueError('Invalid joint target')
    if not isinstance(value['point_m'],list) or not isinstance(value['pixels'],list):
        raise ValueError('Invalid array fields')
    if value['action'] == 'move_pad':
        p=value['point_m']
        if len(p)!=3 or any(type(x) not in (int,float) or not math.isfinite(x) or abs(x)>2 for x in p):
            raise ValueError('Expected finite world XYZ target')
        if not .1 <= value['seconds'] <= 10:
            raise ValueError('Invalid motion duration')
    elif value['point_m']:
        raise ValueError('Only move_pad accepts a target')
    if value['action'] == 'wait' and not .1 <= value['seconds'] <= 10:
        raise ValueError('Invalid wait duration')
    if value['action'] == 'sample_depth':
        if not value['frame_id'] or not 1 <= len(value['pixels']) <= 16 or any(
            not isinstance(p,list) or len(p)!=2 or any(type(x) is not int for x in p) for p in value['pixels']):
            raise ValueError('Invalid depth query')
    elif value['frame_id'] or value['pixels']:
        raise ValueError('Only sample_depth accepts pixels')
    return value


def request_function():
    source=textwrap.dedent(inspect.getsource(OnlineActor.request))
    marker="    command.append('-')"
    if source.count(marker)!=1:
        raise ValueError('Frozen online transport changed')
    source=source.replace(marker,'    command += HTTP_ARGS\n'+marker)
    namespace=dict(OnlineActor.request.__globals__,SCHEMA=SCHEMA,
                   validate_response=validate_response,HTTP_ARGS=HTTP_ARGS)
    exec(compile(ast.parse(source),__file__,'exec'),namespace)
    return namespace['request']


class RingActor(OnlineActor):
    request=request_function()
