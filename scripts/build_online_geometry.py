"""Export the action-free online state interface into the isolated policy worker."""
import ast
from pathlib import Path
from run_geometry_ablation import build_geometry_library, ROOT
from run_vision_trial import validate_source


def build():
    tree=ast.parse(build_geometry_library(include_state=True))
    excluded={'stack_targets','upright_cube_rotation','state_error','cube_rotation_error','geometry_json',
              'GeometryAPI','StateGeometryAPI','MemoryGeometryAPI'}
    nodes=[node for node in tree.body if getattr(node,'name',None) not in excluded
           and not (isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='GEOMETRY' for target in node.targets))]
    source=ast.unparse(ast.Module(body=nodes,type_ignores=[]))
    tree=ast.parse((ROOT/'src/aspire_so100/geometric_fusion.py').read_text())
    nodes=[n for n in tree.body if not isinstance(n,ast.ImportFrom) or not (n.module or '').startswith('aspire_so100')]
    source+='\n'+ast.unparse(ast.Module(body=nodes,type_ignores=[]))
    tree=ast.parse((ROOT/'src/aspire_so100/robot_visual_geometry.py').read_text())
    source+='\n'+ast.unparse(tree)
    tree=ast.parse((ROOT/'src/aspire_so100/online_geometry.py').read_text())
    nodes=[n for n in tree.body if not isinstance(n,ast.ImportFrom) or not (n.module or '').startswith('aspire_so100')]
    source+='\n'+ast.unparse(ast.Module(body=nodes,type_ignores=[]))
    source+='''
class OnlineGeometryAPI:
    __slots__ = ()
    scene = staticmethod(instantaneous_scene)
GEOMETRY = OnlineGeometryAPI()
'''
    validate_source(source)
    return source


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(build())
