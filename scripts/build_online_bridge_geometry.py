"""Build a five-color online bundle without modifying registered observer sources."""
import ast
from build_polygon_prism_session import library, definition, insert_dispatch
from run_vision_trial import validate_source


def online_library():
    tree=ast.parse(library('symbolic',grasp_fraction=.4))
    mask=definition(tree.body,'inventory_color_mask')
    insert_dispatch(mask,"""
if color == 'magenta':
    r, g, b = np.asarray(rgb, float).transpose(2, 0, 1)
    return (r > 65) & (b > 65) & (r > 1.8*g) & (b > 1.8*g) & (r < 1.4*b) & (b < 1.4*r)
""")
    for owner,method in (('MixedPnPObservation','__init__'),('ObservedRGBDPickPlace','observe'),('ObservationContext','__init__')):
        node=definition(definition(tree.body,owner).body,method)
        matches=[v for v in ast.walk(node) if isinstance(v,ast.Tuple) and
                 all(isinstance(x,ast.Constant) for x in v.elts) and
                 tuple(x.value for x in v.elts)==('red','yellow','blue','green')]
        if len(matches)!=1:
            raise ValueError('Expected one color contract in '+owner)
        matches[0].elts.append(ast.Constant('magenta'))
    source=ast.unparse(tree)+'\n'
    validate_source(source)
    return source
