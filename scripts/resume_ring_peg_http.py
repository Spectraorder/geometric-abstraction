"""Registered HTTP transport continuation; preserve both interrupted attempts."""
import argparse
import ast
from pathlib import Path
from unittest.mock import patch

import resume_ring_peg_comparison as previous

HTTP_ARGS = ['-c', 'model_provider="openai-http"', '-c',
             'model_providers.openai-http={name="OpenAI HTTP",wire_api="responses",requires_openai_auth=true,supports_websockets=false}']


SIDECAR_CHANGE = {
    'path': 'logs_2.sqlite-shm',
    'registered_sha256': 'e057b88b061eef0a943347a1e8d6e34bd5cae9040fa7b77a8a233ec5c0bfb1ba',
    'current_sha256': '3910490e7a66aa5b805682d54fb6c0e543f1302ad3bab4d6d73e920f920005a7',
    'reason': 'The earlier read-only SQLite diagnosis changed a WAL shared-memory index. Database, WAL, transcript and all experiment evidence hashes are unchanged.'}


def verify_previous(root):
    source = Path(previous.__file__).read_text()
    old = "require(digest(archived_home/relative) == sha, 'Archived actor home changed')"
    new = "require(digest(archived_home/relative) == (SIDECAR_CHANGE['current_sha256'] if relative == SIDECAR_CHANGE['path'] and sha == SIDECAR_CHANGE['registered_sha256'] else sha), 'Archived actor home changed')"
    previous.require(source.count(old) == 1, 'Prior archive check changed')
    tree = ast.parse(source.replace(old, new))
    function = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'check']
    namespace = dict(vars(previous), SIDECAR_CHANGE=SIDECAR_CHANGE)
    exec(compile(ast.Module(body=function, type_ignores=[]), __file__, 'exec'), namespace)
    return namespace['check'](root)


def http_actor():
    source = Path(previous.study.actor.__file__).read_text()
    old = "    command += ['-']\n"
    previous.require(source.count(old) == 1, 'Actor command builder changed')
    source = source.replace(old, "    command += "+repr(HTTP_ARGS)+"\n"+old)
    functions = [node for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef) and node.name == '_propose_locked']
    previous.require(len(functions) == 1, 'Missing actor request function')
    namespace = dict(vars(previous.study.actor))
    exec(compile(ast.Module(body=functions, type_ignores=[]), __file__, 'exec'), namespace)
    return namespace['_propose_locked'], source


def build_adapter():
    """Specialize the frozen recovery implementation without editing its source."""
    source = Path(previous.__file__).read_text()
    changes = {
        "REPAIR = 'resume_01'": "REPAIR = 'resume_02_http'",
        "    protocol = study.check(root)\n    require(read(root/'lifecycle.json')['status'] == 'failed'":
            "    verify_previous(root)\n    protocol = study.check(root)\n    require(read(root/'lifecycle.json')['status'] == 'failed'",
        "require(len(failed_leases) == 1, 'Expected exactly one interrupted request')":
            "require(len(failed_leases) == 2, 'Expected exactly two interrupted requests')",
        "== 22, 'Unexpected generation inventory'": "== 28, 'Unexpected generation inventory'",
        "== 140, 'Unexpected training inventory'": "== 185, 'Unexpected training inventory'",
        "'completed_generations': 22, 'remaining_generations': 8":
            "'completed_generations': 28, 'remaining_generations': 2",
        "'model_attempt_budget': 31, 'incomplete_attempts': 1":
            "'model_attempt_budget': 32, 'incomplete_attempts': 2",
        "'incomplete_model_attempts': 1, 'model_attempt_budget': 31":
            "'incomplete_model_attempts': 2, 'model_attempt_budget': 32",
        "'remaining_training_slots': 60": "'remaining_training_slots': 15",
        "read(root/REPAIR/'lifecycle.json')['elapsed_seconds']":
            "sum(read(root/REPAIR/'lifecycle.json').get(key, 0) for key in ('elapsed_seconds', 'prior_elapsed_seconds'))",
        "'incomplete_usage': usage, 'future_failure_rule'":
            "'incomplete_usage': [read(root/'resume_01/registration.json')['incomplete_usage'], usage],\n        'transport': 'OpenAI HTTP-only provider alias; same login, model and reasoning; no custom base URL',\n        'transport_args': HTTP_ARGS, 'future_failure_rule'",
        "No runtime, prompts, seeds, selection, timeouts or model settings change.":
            "Only remaining model requests use the HTTP-only OpenAI provider alias. No runtime, prompts, seeds, selection, timeouts, model or reasoning changes.",
        "    shutil.copyfile(Path(__file__), destination/Path(__file__).name)\n":
            "    shutil.copyfile(Path(__file__), destination/Path(__file__).name)\n    shutil.copyfile(Path(prior_adapter.__file__), destination/'base_adapter.py')\n    write(destination/'diagnostic_sidecar_change.json', SIDECAR_CHANGE)\n    (destination/'http_actor.py').write_text(http_actor()[1])\n",
        "    study.check(root)\n    destination = root/REPAIR\n":
            "    study.check(root)\n    destination = root/REPAIR\n    verify_previous(root)\n    require(digest(Path(prior_adapter.__file__)) == digest(destination/'base_adapter.py'), 'Recovery base changed')\n    require(http_actor()[1] == (destination/'http_actor.py').read_text(), 'HTTP actor changed')\n",
        "[dict(receipt['incomplete_usage'], arm='basic', revision=2)]":
            "[dict(usage, arm='basic', revision=2) for usage in receipt['incomplete_usage']]",
        "result['incomplete_model_attempts'] = [receipt['incomplete_usage']]":
            "result['incomplete_model_attempts'] = receipt['incomplete_usage']",
        "One interrupted Base R2 request has unknown final token usage":
            "Two interrupted Base R2 requests have unknown final token usage",
    }
    for old, new in changes.items():
        previous.require(source.count(old) == 1, 'Recovery specialization mismatch: '+old)
        source = source.replace(old, new)
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if not isinstance(node, ast.If)]
    namespace = dict(__name__='ring_http_recovery', __file__=__file__, prior_adapter=previous,
                     HTTP_ARGS=HTTP_ARGS, http_actor=http_actor, verify_previous=verify_previous, SIDECAR_CHANGE=SIDECAR_CHANGE)
    exec(compile(tree, __file__, 'exec'), namespace)
    return namespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase', choices=('register', 'check', 'run', 'chain', 'summarize'), required=True)
    parser.add_argument('--replica')
    parser.add_argument('--arm', choices=previous.study.ARMS)
    args = parser.parse_args()
    root = args.root.resolve()
    adapter = build_adapter()
    with patch.object(previous.study.actor, '_propose_locked', http_actor()[0]):
        if args.phase == 'register':
            with adapter['exclusive'](root/'coordinator.lock'):
                receipt = adapter['register'](root)
                print('Registered HTTP continuation:', receipt['remaining_generations'], 'generations')
        elif args.phase == 'check':
            print('HTTP continuation verified:', adapter['check'](root)['model_attempt_budget'], 'maximum attempts')
        elif args.phase == 'chain':
            try:
                adapter['check'](root)
                chain, _, _ = adapter['corrected_functions']()
                chain(root, args.replica, args.arm)
            except BaseException as error:
                previous.study.stop(root, error)
                raise
        elif args.phase == 'summarize':
            adapter['summarize'](root)
        else:
            adapter['run'](root)


if __name__ == '__main__':
    main()
