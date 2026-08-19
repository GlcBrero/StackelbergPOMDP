from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def load(name):
    """Load a legacy environment module by filename.

    New code should import environment modules normally. This compatibility
    helper remains for older callers without relying on the removed ``imp``
    module.
    """
    path = Path(__file__).resolve().parent / name
    module_name = f"{__name__}._dynamic_{path.stem}"
    spec = spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load environment module from {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
