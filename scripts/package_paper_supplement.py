#!/usr/bin/env python3
"""Package self-contained result guides, exported data, and sparse task videos."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'docs/evaluations'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def portable_guide(name):
    """Retain data definitions while replacing repository-only navigation."""
    original = (DATA/name/'README.md').read_text()
    if name == 'single_policy':
        body = original[original.index('## Contents'):original.index('## Reproduction')]
        title = '# Single-policy results and data\n\n'
    else:
        body = original[original.index('| File |'):original.index('## Reproduction and verification')]
        title = '# Online results and data\n\n'
    ending = ('## Source archives\n\n'
              'The CSV/JSON data and figures are included here. Repository-relative\n'
              '`source` paths and source hashes identify original study archives; those\n'
              'raw archives and the repository exporters are outside this submission\n'
              'bundle. See the [bundle guide](../README.md) for scope and video details.\n')
    return (title+'[Bundle guide](../README.md)\n\n'+body+ending).encode()


def build(output):
    contents = {'README.md': (DATA/'SUPPLEMENTARY_MATERIALS.md').read_bytes()}
    origins = {'README.md': 'docs/evaluations/SUPPLEMENTARY_MATERIALS.md'}
    for name in ('single_policy', 'online'):
        provenance = json.loads((DATA/name/'provenance.json').read_text())
        for filename, expected in provenance['exported_files'].items():
            assert digest((DATA/name/filename).read_bytes()) == expected, filename
        for path in sorted((DATA/name).iterdir()):
            if path.suffix not in {'.md', '.csv', '.json', '.pdf', '.png', '.svg'}:
                raise ValueError(f'Unexpected result file: {path}')
            key = f'{name}/{path.name}'
            contents[key] = portable_guide(name) if path.name == 'README.md' else path.read_bytes()
            origins[key] = str(path.relative_to(ROOT))
    videos = json.loads((DATA/'videos/manifest.json').read_text())
    assert len(videos['videos']) == 6
    for video in videos['videos']:
        assert digest((DATA/'videos'/video['file']).read_bytes()) == video['video_sha256']
    for path in sorted((DATA/'videos').iterdir()):
        if path.suffix not in {'.mp4', '.png', '.md', '.json', '.csv'}:
            raise ValueError(f'Unexpected video file: {path}')
        key = f'videos/{path.name}'
        contents[key] = path.read_bytes()
        origins[key] = str(path.relative_to(ROOT))
    # All navigable Markdown links must resolve within the zip, with no source checkout.
    for name, data in contents.items():
        if not name.endswith('.md'):
            continue
        for target in re.findall(r'\]\(([^)]+)\)', data.decode()):
            if '://' in target or target.startswith('#'):
                continue
            parts = []
            for part in (PurePosixPath(name).parent/target.split('#')[0]).parts:
                if part == '..':
                    if not parts:
                        raise ValueError(f'Link escapes bundle: {name}: {target}')
                    parts.pop()
                elif part != '.':
                    parts.append(part)
            assert '/'.join(parts) in contents, (name, target)
    manifest = dict(kind='paper_supplementary_results_and_sparse_replays',
        builder='scripts/package_paper_supplement.py',
        builder_sha256=digest(Path(__file__).read_bytes()),
        adapted_guides=['single_policy/README.md', 'online/README.md'],
        files={name:dict(bytes=len(data), sha256=digest(data), repository_source=origins[name])
               for name, data in sorted(contents.items())})
    contents['manifest.json'] = (json.dumps(manifest, indent=2)+'\n').encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, data in sorted(contents.items()):
            info = zipfile.ZipInfo('supplementary_material/'+name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        for name, data in contents.items():
            assert archive.read('supplementary_material/'+name) == data
    checksum = output.with_suffix(output.suffix+'.sha256')
    checksum.write_text(f'{digest(output.read_bytes())}  {output.name}\n')
    print(json.dumps(dict(path=str(output), files=len(contents), videos=6,
                         size_mib=round(output.stat().st_size/1024**2, 2),
                         checksum=str(checksum), markdown_links='all internal links resolve'), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'dist/paper_supplement.zip')
    build(parser.parse_args().output)
