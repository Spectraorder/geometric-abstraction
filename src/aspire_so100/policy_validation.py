"""Static numerical-policy validation, independent of simulator initialization."""
import ast


def validate_source(source):
    tree = ast.parse(source)
    forbidden = {'open','exec','eval','compile','getattr','setattr','delattr','globals','locals','vars',
                 '__import__','input','breakpoint','print','load','save','savez','savez_compressed','fromfile',
                 'tofile','memmap','ctypeslib','ctypes','sys','os','subprocess'}
    for node in ast.walk(tree):
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or '']
            if any(name.split('.')[0] not in {'numpy','scipy','math'} for name in names):
                raise ValueError('Import outside numerical libraries')
        if isinstance(node,ast.Attribute) and (node.attr.startswith('_') or node.attr in forbidden):
            raise ValueError('Forbidden attribute')
        if isinstance(node,ast.Name) and node.id in forbidden:
            raise ValueError('Forbidden name')
    compile(tree,'<vision policy>','exec')
