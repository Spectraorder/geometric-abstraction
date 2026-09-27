"""Register or run the fixed five-layout, 200-second compact Base/Full comparison."""
import argparse
from pathlib import Path
import run_online_bound_study as shared

SEEDS = tuple(range(2028105001, 2028105006))


def prepare(root):
    if shared.DEFAULT_BUDGET_S != 200.:
        raise ValueError('This study requires the authorized 200-second budget')
    shared.prepare(root, seeds=SEEDS, compact_text=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase', choices=('prepare', 'check', 'run', 'summarize'), required=True)
    args = parser.parse_args(); root = args.root.resolve()
    if args.phase == 'prepare':
        prepare(root)
    else:
        protocol = shared.verify_sources(root)
        if (protocol['name'] != 'online_compact_explicit_target_200' or protocol['seeds'] != list(SEEDS)
                or protocol['budget_s'] != 200. or not protocol['compact_text']):
            raise ValueError('Registered settings do not match this study')
        if args.phase == 'run':
            shared.execute(root)
        elif args.phase == 'summarize':
            shared.summarize(root)
        else:
            print('Registered sources and settings match')
