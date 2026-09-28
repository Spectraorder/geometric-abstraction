"""Use one HTTP-only transport for all actors without editing the frozen client."""
import ast
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import geometry_actor_session as actor
from summarize_mixed_scene_learning import require

HTTP_ARGS = ['-c', 'model_provider="openai-http"', '-c',
             'model_providers.openai-http={name="OpenAI HTTP",wire_api="responses",requires_openai_auth=true,supports_websockets=false}']


def actor_source():
    source = Path(actor.__file__).read_text()
    marker = "    command += ['-']\n"
    require(source.count(marker) == 1, 'Actor command builder changed')
    return source.replace(marker, '    command += '+repr(HTTP_ARGS)+'\n'+marker)


def http_actor():
    functions = [node for node in ast.parse(actor_source()).body
                 if isinstance(node, ast.FunctionDef) and node.name == '_propose_locked']
    require(len(functions) == 1, 'Missing actor request function')
    namespace = dict(vars(actor))
    exec(compile(ast.Module(body=functions, type_ignores=[]), __file__, 'exec'), namespace)
    return namespace['_propose_locked']


@contextmanager
def installed():
    with patch.object(actor, '_propose_locked', http_actor()):
        yield
