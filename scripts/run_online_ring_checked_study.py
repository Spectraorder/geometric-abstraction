"""Run online Ring with nested source paths supported by the legacy resource audit.

Development episodes snapshot repository-relative paths, including assets. The
legacy audit fallback used Path.with_name(), which accepts basenames only. This
adapter changes source lookup, not episode runtime, model inputs or outcomes.
"""
import argparse
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
import audit_online_reasoning_development as resource
import run_online_ring_study as study


def archived_source(root,episode,name):
    relative=Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Source identifier must be repository relative')
    candidates=(episode/'sources'/relative,root/'sources'/relative,
                root/'sources/scripts'/relative,study.ROOT/relative,
                Path(resource.__file__).parent/relative)
    return next((path for path in candidates if path.exists()),candidates[-1])


@contextmanager
def installed():
    with patch.object(resource,'archived_source',archived_source):
        yield


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--phase',choices=('prepare','check','run','summarize'),required=True)
    parser.add_argument('--development',type=Path,default=study.ROOT/'outputs/online_ring_dev/live_01')
    args=parser.parse_args();root=args.root.resolve()
    with installed():
        if args.phase=='prepare':study.prepare(root,args.development.resolve())
        elif args.phase=='check':study.check(root);print('Registered sources match')
        elif args.phase=='run':study.execute(root)
        else:print(study.summarize(root)['arms'])
