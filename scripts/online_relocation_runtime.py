"""Sensor-only size transfer and existing explicit-target skills for relocation."""
import ast
import inspect
import textwrap
from types import FunctionType
from unittest.mock import patch
import online_bridge_skills as shared_skills
from build_online_bridge_geometry import online_library
from online_bound_actor import SCHEMA, validate_response
from online_bound_decisions import BoundDecisionLoop, ExplicitSpatialSkills, CONTRACT as BASE_CONTRACT
from online_compact_geometry import compact_prompt
from online_ring_actor import RingActor

CONTRACT = BASE_CONTRACT.replace(
    'Build the public support graph in this continuing scene using shared bridge and stack skills.',
    'Achieve the public relocation goal in this continuing scene using shared pick/place skills.') + '''
The initial scene contains a three-cube column and a separate destination cube.
All cubes have equal dimensions; the local controller measures size from destination.
Keep both foundations at their initial locations in the final scene.
Accidental foundation displacement can be repaired by an explicit bridge placement.
The task graph describes the final supports, not the current supports or an action order.
The stack skill for a movable cube accepts any explicit surface point, including a
free temporary table location. It does not require that point to be the final support.
Observe actual outcomes before continuing. The shared skill performs one pick/place,
not a relocation plan. No initial cube positions or hidden success state are supplied.
'''


def relocation_library():
    tree = ast.parse(online_library())
    owner = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MixedSession')
    init = next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    if not isinstance(init.body[-1],ast.For):
        raise ValueError('Shared initialization changed')
    init.body[-1:] = ast.parse('''
reference = api.task()['roles']['destination']
self.pnp.observe(reference)
measured = self.pnp.models[reference]
for color in self.colors:
    if color != reference:
        self.pnp.models[color] = {'height': float(measured['height']),
            'dims': np.array(measured['dims'], copy=True),
            'source': 'equal_cube_size_from_destination_rgbd', 'reference_color': reference}
for color in self.colors:
    try:
        self.pnp.observe(color)
    except (ValueError, RuntimeError) as error:
        self.ledger.unavailable(color, self.time(), str(error))
''').body
    return ast.unparse(ast.fix_missing_locations(tree))+'\n'


class RelocationSkills(ExplicitSpatialSkills):
    def __init__(self, api):
        source = textwrap.dedent(inspect.getsource(shared_skills.SharedSkills.__init__))
        original = "self.order = self.bridge.construction_order(self.task['nodes'])"
        if source.count(original) != 1:
            raise ValueError('Shared initialization order changed')
        source = source.replace(original, 'self.order = list(self.nodes)')
        scope = dict(shared_skills.SharedSkills.__init__.__globals__, online_library=relocation_library)
        exec(compile(source, __file__, 'exec'), scope)
        with patch.object(shared_skills.SharedSkills,'__init__',scope['__init__']):
            super().__init__(api)
        self.session.pnp.grasp_fraction = .70
        self.provenance['grasp_fraction'] = .70
        self.provenance['dimension_assumption'] = 'Equal cubes, measured destination size; independent current poses'



def clone(function, **updates):
    copied = FunctionType(function.__code__,dict(function.__globals__,**updates),
                          function.__name__,function.__defaults__,function.__closure__)
    copied.__kwdefaults__ = function.__kwdefaults__
    return copied


class RelocationActor(RingActor):
    _bound_request = clone(RingActor.request,SCHEMA=SCHEMA,validate_response=validate_response)

    def request(self,prompt,images,*,slow,advance,deadline):
        transformed,encoded = compact_prompt(prompt,explain=not getattr(self,'_compact_contract_sent',False))
        result = self._bound_request(transformed,images,slow=slow,advance=advance,deadline=deadline)
        if encoded:self._compact_contract_sent=True
        return result


class RelocationLoop(BoundDecisionLoop):
    step = clone(BoundDecisionLoop.step,CONTRACT=CONTRACT)

    def __init__(self,env,api,skills,arm,**kwargs):
        super().__init__(env,api,skills,arm,actor_factory=RelocationActor)
