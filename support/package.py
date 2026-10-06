#!/usr/bin/env python3
"""Build or verify a self-contained submission archive, without Git or network access."""
import argparse
import gzip
import hashlib
import io
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_ROOTS = {'results', 'build', 'dist', '.venv', '.git', '.agents', '.codex'}
EXCLUDED_PARTS = {'__pycache__', '.pytest_cache', '.mplconfig'}


def files(root):
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if relative.parts[0] in EXCLUDED_ROOTS or EXCLUDED_PARTS.intersection(relative.parts):
            continue
        if relative.parts[:2] == ('reproduction', 'build'):
            continue
        if path.name in {'.DS_Store', 'SHA256SUMS'} or path.suffix in {'.pyc', '.part'}:
            continue
        if 'roms' in relative.parts and path.suffix == '.bin':
            continue
        if path.is_symlink():
            raise ValueError(f'Package must not depend on a symlink: {relative}')
        if path.is_file():
            yield path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checksums(root):
    return {p.relative_to(root).as_posix(): digest(p) for p in files(root)}


def verify(root):
    expected = {}
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        sha, name = line.split('  ', 1)
        expected[name] = sha
    actual = checksums(root)
    if actual != expected:
        different = sorted(name for name in actual.keys() | expected.keys()
                           if actual.get(name) != expected.get(name))
        raise ValueError('Package files differ: ' + ', '.join(different[:20]))
    print(f'Verified all {len(expected)} packaged files.')


def build(output):
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if output.is_relative_to(ROOT) and not output.is_relative_to(ROOT / 'dist'):
        raise ValueError('Put archives in dist/ or outside this package.')
    contents = list(files(ROOT))
    sums = ''.join(f'{digest(p)}  {p.relative_to(ROOT).as_posix()}\n' for p in contents).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode='w') as archive:
            for path in contents:
                name = 'stackelberg-pomdp/' + path.relative_to(ROOT).as_posix()
                data = path.read_bytes()
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime = len(data), 0o644, 0
                archive.addfile(info, io.BytesIO(data))
            info = tarfile.TarInfo('stackelberg-pomdp/SHA256SUMS')
            info.size, info.mode, info.mtime = len(sums), 0o644, 0
            archive.addfile(info, io.BytesIO(sums))
    output.with_suffix(output.suffix + '.sha256').write_text(f'{digest(output)}  {output.name}\n')
    print(f'Packaged {len(contents)} files into {output} ({output.stat().st_size / 1e6:.1f} MB).')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--output', type=Path, help='Create a new .tar.gz and its SHA-256.')
    selection.add_argument('--verify', action='store_true', help='Verify an extracted archive.')
    args = parser.parse_args()
    if args.verify:
        verify(ROOT)
    else:
        build(args.output)


if __name__ == '__main__':
    main()
