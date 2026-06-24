"""Run named replication targets from a JSON manifest."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path(__file__).with_name("bertrand_targets.json")


def load_targets(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def format_value(value, seed):
    if isinstance(value, str):
        return value.format(seed=seed)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def build_command(target, seed):
    command = [sys.executable, "-m", target["module"]]
    for key, value in target.get("args", {}).items():
        if value is None:
            continue
        command.extend([f"--{key}", format_value(value, seed)])
    return command


def main():
    parser = argparse.ArgumentParser(description="Run a named replication target.")
    parser.add_argument("target", nargs="?", help="Target name from the manifest.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--list", action="store_true", help="List available targets.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running it.")
    args = parser.parse_args()

    targets = load_targets(args.manifest)
    if args.list:
        for name, target in targets.items():
            print(f"{name}: {target.get('description', '')}")
        return

    if not args.target:
        parser.error("target is required unless --list is used")
    if args.target not in targets:
        parser.error(f"unknown target {args.target!r}; use --list")

    command = build_command(targets[args.target], args.seed)
    print(" ".join(command))
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
