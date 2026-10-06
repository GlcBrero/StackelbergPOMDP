"""Source identity and output handling shared by paper runners."""
import hashlib
import json
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
PACKAGE = CODE


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def source_manifest():
    import stackelberg_pomdp
    actual = Path(stackelberg_pomdp.__file__).resolve().parent
    if actual != CODE / 'stackelberg_pomdp':
        raise RuntimeError(f'Wrong library imported: {actual}')
    files = {}
    for path in sorted(CODE.rglob('*')):
        if path.is_file() and path.suffix in {'.py', '.cpp', '.yml', '.json'}:
            if not any(part in {'__pycache__', '.pytest_cache', 'results', 'build', 'reproduction'} for part in path.relative_to(CODE).parts):
                files[path.relative_to(CODE).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {'schema': 'stackpomdp.unified-source.v1', 'files': files}


def fresh_output(path):
    path = Path(path).resolve()
    for protected in ('stackelberg_pomdp', 'experiments', 'support', 'pretrained', 'checkpoints', 'configs', 'tests', 'scripts', 'docs', 'reproduction'):
        if path.is_relative_to(PACKAGE / protected):
            raise ValueError(f'Use a new results directory outside {protected}/')
    path.mkdir(parents=True, exist_ok=False)
    return path


def record_run(output, experiment, protocol):
    write_json(output / 'source_manifest.json', source_manifest())
    write_json(output / 'protocol.json', protocol)
    write_json(output / 'run_identity.json', {
        'experiment': experiment,
        'cohort': 'unified-code rerun; separate from retained paper results',
        'protocol_sha256': hashlib.sha256((output / 'protocol.json').read_bytes()).hexdigest(),
    })


def verify_run(output):
    output = Path(output)
    recorded = json.loads((output / 'source_manifest.json').read_text())
    if recorded != source_manifest():
        raise ValueError('Code differs from this run; retain the original code before continuing it.')
    identity = json.loads((output / 'run_identity.json').read_text())
    digest = hashlib.sha256((output / 'protocol.json').read_bytes()).hexdigest()
    if digest != identity['protocol_sha256']:
        raise ValueError('Run protocol changed after initialization')
    return json.loads((output / 'protocol.json').read_text())
