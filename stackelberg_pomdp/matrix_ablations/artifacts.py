"""Run directories, immutable manifests, checksums, and shared logging."""

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import numpy as np
import gym
import stable_baselines3
import torch

from pathlib import Path


SCHEMA_VERSION = 3
DEFAULT_RESULTS_ROOT = Path("replication/matrix_ablations/results/single_runs")
REPO_ROOT = Path(__file__).resolve().parents[2]


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError("cannot serialize {!r}".format(type(value)))


def canonical_json(payload):
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=_json_default
    )


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
    temporary.replace(path)


def append_jsonl(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(payload))
        handle.write("\n")


def file_sha256(path):
    path = Path(path)
    if path.is_dir():
        digest = hashlib.sha256(b"tree-sha256-v1\0")
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            relative = child.relative_to(path).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(bytes.fromhex(file_sha256(child)))
        return digest.hexdigest()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)) if len(values) else None,
        "median": float(np.median(values)) if len(values) else None,
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "sem": (
            float(np.std(values, ddof=1) / np.sqrt(len(values)))
            if len(values) > 1 else 0.0
        ),
        "minimum": float(np.min(values)) if len(values) else None,
        "maximum": float(np.max(values)) if len(values) else None,
    }


def _git_value(*args):
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def provenance():
    source_paths = [
        REPO_ROOT / "stackelberg_pomdp/experiments/matrix_ablations.py",
        REPO_ROOT / "stackelberg_pomdp/envs/matrix.py",
        *sorted(Path(__file__).parent.glob("*.py")),
        REPO_ROOT / "stackelberg_pomdp/algorithms/on_policy.py",
        REPO_ROOT / "stackelberg_pomdp/policies/generic.py",
        REPO_ROOT / "stackelberg_pomdp/policies/cache.py",
        REPO_ROOT / "stackelberg_pomdp/callbacks.py",
        REPO_ROOT / "stackelberg_pomdp/rl_trainer_setup.py",
        REPO_ROOT / "environment.yml",
    ]
    status = _git_value("status", "--porcelain")
    return {
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
        "hostname": platform.node(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "gym_version": gym.__version__,
        "stable_baselines3_version": stable_baselines3.__version__,
        "torch_version": torch.__version__,
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "source_sha256": {
            str(path.relative_to(REPO_ROOT)): file_sha256(path)
            for path in source_paths if path.exists()
        },
    }


def start_manifest(run_dir, config):
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "started_unix": time.time(),
        "config_sha256": hashlib.sha256(
            canonical_json(config).encode("utf-8")
        ).hexdigest(),
        "provenance": provenance(),
    }
    write_json(run_dir / "run_manifest.json", payload)
    return payload


def finish_manifest(
        run_dir, manifest, status, artifacts=None, error=None, metadata=None
):
    payload = dict(manifest)
    payload.update({"status": status, "finished_unix": time.time()})
    if metadata:
        payload.update(metadata)
    if artifacts:
        payload["artifacts"] = {
            name: {
                "path": str(path),
                "sha256": file_sha256(path),
            }
            for name, path in artifacts.items() if Path(path).exists()
        }
    if error is not None:
        payload["error"] = str(error)
        payload["traceback"] = traceback.format_exc()
    write_json(run_dir / "run_manifest.json", payload)


def _checkpoint_path(path):
    path = Path(path)
    if path.exists():
        return path.resolve()
    zipped = Path(str(path) + ".zip")
    if zipped.exists():
        return zipped.resolve()
    raise FileNotFoundError("checkpoint does not exist: {}".format(path))


def _run_id(config):
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()[:12]


def _run_dir(output_root, config):
    if config["stage"] == "meta_follower":
        parts = (
            "meta_follower",
            config["memory_mode"],
            config["algorithm"].lower(),
            "seed{}-{}".format(config["seed"], _run_id(config)),
        )
    else:
        parts = (
            config["experiment"],
            config["matrix"],
            config["algorithm"].lower(),
            config["condition"],
            "seed{}-{}".format(config["seed"], _run_id(config)),
        )
    path = Path(output_root).joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    protected = (
        "config.json", "run_manifest.json", "progress.jsonl",
        "evaluation.json", "model.zip",
        "response_lookup.json",
    )
    existing = [name for name in protected if (path / name).exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite run artifacts {} in {}".format(
                ", ".join(existing), path
            )
        )
    return path


def _maybe_wandb(args, config, run_name):
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("--wandb requested but wandb is unavailable") from exc
    return wandb.init(
        project=args.wandb_project,
        group=args.wandb_group,
        name=args.wandb_name or run_name,
        config=config,
    )


def validate_logging_args(args):
    if args.attempt < 0:
        raise ValueError("attempt must be nonnegative")
    if bool(args.sweep_id) != bool(args.record_key):
        raise ValueError("sweep_id and record_key must be supplied together")
    if args.sweep_plan:
        sweep_plan = Path(args.sweep_plan)
        if not sweep_plan.is_file():
            raise FileNotFoundError("sweep plan does not exist: {}".format(sweep_plan))
        if not args.sweep_id:
            raise ValueError("sweep_plan requires sweep_id and record_key")
