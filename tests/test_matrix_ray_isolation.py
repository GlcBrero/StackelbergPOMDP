import ast
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
RLLIB_MODULE = Path("stackelberg_pomdp/matrix_ablations/rllib_es.py")


def _ray_imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    parents = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    for node in ast.walk(tree):
        imported = []
        if isinstance(node, ast.Import):
            imported = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported = [node.module]
        if any(name == "ray" or name.startswith("ray.") for name in imported):
            functions = []
            parent = parents.get(node)
            while parent is not None:
                if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions.append(parent.name)
                parent = parents.get(parent)
            yield node.lineno, functions


def test_ray_imports_exist_only_in_the_lazy_matrix_es_boundary():
    found = []
    for path in REPO_ROOT.rglob("*.py"):
        for line, functions in _ray_imports(path):
            found.append((path.relative_to(REPO_ROOT), line, functions))
    assert found
    assert all(path == RLLIB_MODULE for path, _, _ in found)
    assert all("require_rllib_es" in functions for _, _, functions in found)


def test_importing_main_experiment_does_not_import_ray():
    code = (
        "import sys; "
        "import stackelberg_pomdp.experiments.matrix_ablations; "
        "assert not any(n == 'ray' or n.startswith('ray.') for n in sys.modules)"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        check=True,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )


def test_ray_is_absent_from_main_environment_and_pinned_in_dedicated_env():
    main = (REPO_ROOT / "environment.yml").read_text(encoding="utf-8").lower()
    dedicated = (REPO_ROOT / "environment-ray-es.yml").read_text(
        encoding="utf-8"
    )
    assert "ray[rllib]" not in main
    assert "name: stackelberg-pomdp-ray-es" in dedicated
    assert dedicated.count("ray[rllib]==2.0.1") == 1


def test_obsolete_active_es_modules_are_absent():
    package = REPO_ROOT / "stackelberg_pomdp/matrix_ablations"
    for filename in ("ars.py", "es.py", "ray_style_es.py"):
        assert not (package / filename).exists()
