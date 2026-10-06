#!/usr/bin/env python3
"""Verify figure inputs and regenerated outputs against the paper's fingerprints."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
REFERENCE = Path(__file__).with_name('paper_reference.json')



def split_platform_numbers(source):
    return split_generated_block(source, 'PLATFORM NUMBERS')


def split_generated_block(source, name):
    begin = f'% BEGIN GENERATED {name}\n'
    end = f'% END GENERATED {name}'
    if source.count(begin) != 1 or source.count(end) != 1:
        raise ValueError(f'Expected one marked {name} block in the manuscript')
    start = source.index(begin) + len(begin)
    stop = source.index(end, start)
    return source[:start], source[start:stop], source[stop:]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rendered_digest(pdf):
    renderer = shutil.which('pdftoppm')
    if renderer is None:
        raise RuntimeError('--compare-rendered requires pdftoppm (Poppler)')
    with tempfile.TemporaryDirectory(prefix='stackpomdp-figure-qa-') as temporary:
        prefix = Path(temporary) / 'rendered'
        subprocess.run([renderer, '-png', '-singlefile', '-r', '150', str(pdf), str(prefix)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return digest(prefix.with_suffix('.png'))


def tex_figure_paths(tex_source):
    references = set()
    pending, seen = [tex_source], set()
    while pending:
        source = pending.pop()
        if source in seen:
            continue
        seen.add(source)
        for raw_line in source.read_text(encoding='utf-8').splitlines():
            line = raw_line.split('%', 1)[0]
            for included in re.findall(r'\\(?:input|include)\{([^}]+)\}', line):
                path = tex_source.parent / included
                pending.append(path if path.suffix else path.with_suffix('.tex'))
            for match in re.findall(r'figures/([^}\s]+)', line):
                path = Path('figures') / match
                references.add((path if path.suffix else path.with_suffix('.pdf')).as_posix())
    return references


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, help='Check regenerated figures and numerical tables.')
    parser.add_argument('--compare-rendered', action='store_true',
                        help='Require pixel-identical figures using the recorded Poppler rendering (needs --build-dir).')
    parser.add_argument('--paper-dir', type=Path,
                        help='Also check the live manuscript and its figure/table inventory.')
    args = parser.parse_args()
    if args.compare_rendered and args.build_dir is None:
        parser.error('--compare-rendered requires --build-dir')
    reference = json.loads(REFERENCE.read_text())
    if reference.get('schema_version') != 2:
        raise ValueError('Expected the final-rerun paper reference (schema 2)')
    rows = reference['figures']
    if not rows or set(rows[0]) != {'figure', 'kind', 'script', 'data'}:
        raise ValueError('Unexpected figure manifest schema')
    declared_figures = {row['figure'] for row in rows}
    if len(declared_figures) != len(rows):
        raise ValueError('Duplicate figures in manifest')
    if declared_figures != {name for name in reference['files'] if name.startswith('figures/')}:
        raise ValueError('Figure manifest does not match the recorded paper inventory')
    for relative, expected in {**reference['inputs'], **reference['scripts']}.items():
        if digest(BUNDLE_ROOT / relative) != expected:
            raise ValueError(f'Input differs from the recorded rerun: {relative}')
    if __package__:
        from .records import check_records
    else:
        from records import check_records
    check_records(BUNDLE_ROOT)
    for row in rows:
        if row['kind'] == 'data-driven':
            if not (BUNDLE_ROOT / row['script']).is_file():
                raise FileNotFoundError(row['script'])
            if row['figure'] not in reference['rendered_sha256']:
                raise ValueError(f"Missing rendering fingerprint: {row['figure']}")
        elif row['kind'] != 'source-native':
            raise ValueError(f"Unknown figure kind: {row['kind']}")
        for relative in filter(None, row['data'].split(';')):
            path = BUNDLE_ROOT / relative
            if not path.exists():
                raise FileNotFoundError(path)
    if args.paper_dir:
        tex_source = args.paper_dir / reference['manuscript']
        if digest(tex_source) != reference['manuscript_sha256']:
            raise ValueError('The manuscript differs from the recorded paper version')
        if tex_figure_paths(tex_source) != declared_figures:
            raise ValueError('Figure manifest does not match LaTeX references')
        actual_figures = {p.relative_to(args.paper_dir).as_posix()
                          for p in (args.paper_dir / 'figures').rglob('*') if p.is_file()}
        if actual_figures != declared_figures:
            raise ValueError('Manuscript figure directory contains missing or unused files')
        for block, filename in (('PLATFORM NUMBERS', 'hidden_cost_numbers.tex'),
                                ('STATISTICAL COMPARISONS', 'statistical_comparisons.tex')):
            _, numbers, _ = split_generated_block(tex_source.read_text(encoding='utf-8'), block)
            if hashlib.sha256(numbers.encode()).hexdigest() != reference['generated_files'][filename]:
                raise ValueError(f'Inline {block} differ from the recorded paper')
        for relative, expected in reference['files'].items():
            if digest(args.paper_dir / relative) != expected:
                raise ValueError(f'Manuscript input differs from recorded version: {relative}')
    if args.build_dir:
        build = args.build_dir.resolve()
        for name, expected in reference['generated_files'].items():
            if digest(build / name) != expected:
                raise ValueError(f'Generated numerical table differs from the paper: {name}')
        if args.paper_dir:
            source_text = re.sub(r'\s+', '', tex_source.read_text(encoding='utf-8'))
            with (build / 'tables/mspm_welfare.csv').open(newline='') as handle:
                for row in csv.DictReader(handle):
                    expected = (f"{row['types']}&${float(row['ex_post_welfare']):.4f}$&"
                                f"${float(row['optimal_spm_welfare']):.4f}$&"
                                f"${float(row['best_welfare_mean']):.4f}\\;"
                                f"({float(row['best_welfare_se']):.4f})$")
                    if expected not in source_text:
                        raise ValueError(f"MSPM manuscript table differs for {row['types']} types")
        for row in rows:
            if row['kind'] != 'data-driven':
                continue
            output = build / row['figure']
            if not output.is_file() or output.stat().st_size == 0:
                raise FileNotFoundError(output)
            if args.compare_rendered and rendered_digest(output) != reference['rendered_sha256'][row['figure']]:
                raise ValueError(f"Figure rendering differs from the paper: {row['figure']}; "
                                 f"reference used {reference['rendering']['version']}")
    print(f"Verified {len(rows)} paper figure mappings and {len(reference['inputs'])} figure/table inputs."
          + (' Regenerated figures and numerical tables match the recorded paper.' if args.build_dir else ''))


if __name__ == '__main__':
    main()
