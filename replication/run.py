"""Run named paper replication targets from a JSON manifest."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path(__file__).with_name("targets.json")


def load_targets(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def format_value(value, seed):
    if isinstance(value, str):
        return value.format(seed=seed)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def build_command(target, seed, extra_args=None):
    if "module" not in target:
        return None
    command = [sys.executable, "-m", target["module"]]
    for key, value in target.get("args", {}).items():
        if value is None:
            continue
        command.extend([f"--{key}", format_value(value, seed)])
    command.extend(extra_args or [])
    return command


def main():
    parser = argparse.ArgumentParser(description="Run a named replication target.")
    parser.add_argument("target", nargs="?", help="Target name from the manifest.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--list", action="store_true", help="List available targets.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running it.")
    args, extra_args = parser.parse_known_args()

    targets = load_targets(args.manifest)
    if args.list:
        for name, target in targets.items():
            status = target.get("status", "owned_here")
            output = target.get("figure_or_table", "")
            suffix = f" [{output}]" if output else ""
            print(f"{name} ({status}){suffix}: {target.get('description', '')}")
        return

    if not args.target:
        parser.error("target is required unless --list is used")
    if args.target not in targets:
        parser.error(f"unknown target {args.target!r}; use --list")

    target = targets[args.target]
    command = build_command(target, args.seed, extra_args=extra_args)
    if command is None:
        print(f"{args.target}: {target.get('description', '')}")
        print(f"status: {target.get('status', 'todo')}")
        if target.get("todo"):
            print(f"todo: {target['todo']}")
        if target.get("owner"):
            print(f"owner: {target['owner']}")
        return

    print(" ".join(command))
    if not args.dry_run:
        env = dict(os.environ)
        env["PYTHONNOUSERSITE"] = "1"
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        mpl_config_dir = Path(tempfile.gettempdir()) / "stackelberg-pomdp-matplotlib"
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        env.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
        xdg_cache_dir = Path(tempfile.gettempdir()) / "stackelberg-pomdp-cache"
        xdg_cache_dir.mkdir(parents=True, exist_ok=True)
        env.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))
        subprocess.run(command, cwd=ROOT, check=True, env=env)


if __name__ == "__main__":
    main()
