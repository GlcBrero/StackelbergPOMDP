"""Run and validate named paper-replication targets from a JSON manifest."""

import argparse
import importlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path(__file__).with_name("targets.json")

# ``python replication/run.py`` places ``replication/`` rather than the
# repository root on sys.path.  Validation imports the same modules that the
# spawned ``python -m`` command will load from ROOT, so make that location
# explicit instead of relying on an editable installation.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class ManifestError(ValueError):
    """Raised when a replication target cannot be expanded unambiguously."""


def load_targets(path):
    with open(path, "r", encoding="utf-8") as handle:
        targets = json.load(handle)
    if not isinstance(targets, dict):
        raise ManifestError("the manifest root must be a JSON object")
    return targets


def format_value(value, seed):
    """Render the one supported manifest placeholder without generic formatting.

    Using ``str.format`` here previously made descriptive values such as
    ``{observed|hidden}`` fail at runtime with a ``KeyError``.  Variants are now
    represented structurally, and ``{seed}`` is the only legal placeholder.
    """
    if isinstance(value, str):
        rendered = value.replace("{seed}", str(seed))
        if "{" in rendered or "}" in rendered:
            raise ManifestError(
                f"unsupported placeholder in argument value {value!r}; "
                "use a named manifest variant instead"
            )
        return rendered
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _as_tokens(value, field):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    raise ManifestError(f"{field} must be a string or a list of strings")


def _variant_specs(target, selected_variant=None):
    variants = target.get("variants")
    if variants is None:
        if selected_variant is not None:
            raise ManifestError("this target has no named variants")
        return [(None, target)]
    if not isinstance(variants, list) or not variants:
        raise ManifestError("variants must be a non-empty list")

    expanded = []
    seen = set()
    for variant in variants:
        if not isinstance(variant, dict) or not variant.get("name"):
            raise ManifestError("every variant must be an object with a non-empty name")
        name = variant["name"]
        if name in seen:
            raise ManifestError(f"duplicate variant name {name!r}")
        seen.add(name)
        if selected_variant is not None and name != selected_variant:
            continue
        merged = dict(target)
        merged["args"] = {**target.get("args", {}), **variant.get("args", {})}
        merged["parameter_exceptions"] = {
            **target.get("parameter_exceptions", {}),
            **variant.get("parameter_exceptions", {}),
        }
        merged["flags"] = [*target.get("flags", []), *variant.get("flags", [])]
        merged["command"] = [
            *_as_tokens(target.get("command"), "command"),
            *_as_tokens(variant.get("command"), "variant command"),
        ]
        expanded.append((name, merged))

    if selected_variant is not None and not expanded:
        available = ", ".join(sorted(seen))
        raise ManifestError(
            f"unknown variant {selected_variant!r}; available variants: {available}"
        )
    return expanded


def _validate_optimizer_overrides(spec):
    """Require a value-specific reason for nondefault mechanism target knobs.

    Explicit user CLI overrides remain available for diagnostics. Atari and
    the appendix PG/SimpleQ recipes have separate scientific protocols.
    """
    module = spec["module"]
    if module not in {
        "stackelberg_pomdp.experiments." + name for name in
        ("simple_allocation", "matrix_design", "mspm", "spm_baseline",
         "price_collusion", "normal_form")
    }:
        return
    from stackelberg_pomdp.training_defaults import algorithm_defaults

    args = spec.get("args", {})
    algorithm = args.get("algorithm", "A2C" if module.endswith("price_collusion") else "PPO")
    defaults = algorithm_defaults(algorithm)
    expected = {"learning_rate": defaults["learning_rate"], "ent_coef": defaults["ent_coef"]}
    if algorithm == "PPO":
        expected.update(ppo_batch_size=defaults["batch_size"],
                        ppo_n_epochs=defaults["n_epochs"],
                        ppo_episodes_per_batch=None,
                        ppo_rollout_geometry="complete_episodes")
    for key, default in expected.items():
        if key not in args or args[key] is None or args[key] == default:
            continue
        exception = spec.get("parameter_exceptions", {}).get(key, {})
        if exception.get("value") != args[key] or not exception.get("reason", "").strip():
            raise ManifestError(
                f"{module}: nondefault {key}={args[key]!r} requires a matching "
                "parameter_exceptions value and reason; see replication/PARAMETERS.md"
            )


def build_commands(target, seed, variant=None, extra_args=None):
    """Expand one target into ``(variant_name, argv)`` command pairs."""
    if "module" not in target or target["module"] is None:
        return []
    if not isinstance(target["module"], str):
        raise ManifestError("module must be a string or null")

    commands = []
    extra_options = {token.split("=", 1)[0] for token in (extra_args or [])}
    for variant_name, spec in _variant_specs(target, variant):
        _validate_optimizer_overrides(spec)
        command = [sys.executable, "-m", spec["module"]]
        command.extend(_as_tokens(spec.get("command"), "command"))
        args = spec.get("args", {})
        if not isinstance(args, dict):
            raise ManifestError("args must be a JSON object")
        for key, value in args.items():
            if value is None:
                continue
            # A CLI override may select the other response-budget unit.
            if ((key == "mw_response_cycles" and "--tot_num_response_episodes" in extra_options)
                    or (key == "tot_num_response_episodes" and "--mw_response_cycles" in extra_options)):
                continue
            if not isinstance(key, str) or not key:
                raise ManifestError("argument names must be non-empty strings")
            command.extend([f"--{key}", format_value(value, seed)])
        for flag in spec.get("flags", []):
            if not isinstance(flag, str) or not flag:
                raise ManifestError("flags must contain non-empty strings")
            command.append(flag if flag.startswith("--") else f"--{flag}")
        command.extend(extra_args or [])
        commands.append((variant_name, command))
    return commands


def build_command(target, seed, extra_args=None):
    """Backward-compatible helper for targets that expand to one command."""
    commands = build_commands(target, seed, extra_args=extra_args)
    if not commands:
        return None
    if len(commands) != 1:
        raise ManifestError("target has multiple variants; use build_commands")
    return commands[0][1]


def validate_command(command):
    """Parse a generated command with the target module's public parser."""
    try:
        module_index = command.index("-m") + 1
        module_name = command[module_index]
    except (ValueError, IndexError) as exc:
        raise ManifestError(f"not a Python module command: {command!r}") from exc
    module = importlib.import_module(module_name)
    build_parser = getattr(module, "build_parser", None)
    if build_parser is None:
        raise ManifestError(f"{module_name} does not expose build_parser()")
    parser = build_parser()
    try:
        parser.parse_args(command[module_index + 1 :])
    except SystemExit as exc:
        raise ManifestError(
            f"generated arguments do not parse for {module_name}: "
            f"{' '.join(command[module_index + 1:])}"
        ) from exc


def validate_targets(targets, seed=1, target_name=None, variant=None, extra_args=None):
    """Validate all runnable expansions, returning their identifying labels."""
    selected = (
        [(target_name, targets[target_name])]
        if target_name is not None
        else list(targets.items())
    )
    labels = []
    for name, target in selected:
        for variant_name, command in build_commands(
            target, seed, variant=variant, extra_args=extra_args
        ):
            validate_command(command)
            labels.append(name if variant_name is None else f"{name}:{variant_name}")
    return labels


def _runtime_environment():
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
    return env


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a named replication target.")
    parser.add_argument("target", nargs="?", help="Target name from the manifest.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--variant", help="Run one named variant of an expanded target.")
    parser.add_argument("--list", action="store_true", help="List available targets.")
    parser.add_argument(
        "--validate", action="store_true",
        help="Parser-validate the selected target, or the whole manifest.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print without running.")
    args, extra_args = parser.parse_known_args(argv)

    targets = load_targets(args.manifest)
    if args.list:
        for name, target in targets.items():
            status = target.get("status", "owned_here")
            output = target.get("figure_or_table", "")
            suffix = f" [{output}]" if output else ""
            variants = target.get("variants", [])
            variant_suffix = ""
            if variants:
                variant_suffix = " {" + ", ".join(v["name"] for v in variants) + "}"
            print(
                f"{name}{variant_suffix} ({status}){suffix}: "
                f"{target.get('description', '')}"
            )
        return 0

    if args.target is not None and args.target not in targets:
        parser.error(f"unknown target {args.target!r}; use --list")
    if args.variant is not None and args.target is None:
        parser.error("--variant requires a target")

    if args.validate:
        labels = validate_targets(
            targets,
            seed=args.seed,
            target_name=args.target,
            variant=args.variant,
            extra_args=extra_args,
        )
        for label in labels:
            print(f"validated: {label}")
        return 0

    if not args.target:
        parser.error("target is required unless --list or --validate is used")

    target = targets[args.target]
    commands = build_commands(
        target, args.seed, variant=args.variant, extra_args=extra_args
    )
    if not commands:
        print(f"{args.target}: {target.get('description', '')}")
        print(f"status: {target.get('status', 'todo')}")
        if target.get("todo"):
            print(f"todo: {target['todo']}")
        if target.get("owner"):
            print(f"owner: {target['owner']}")
        if target.get("entrypoints"):
            print("entrypoints:")
            for entrypoint in target["entrypoints"]:
                print(f"  - {entrypoint}")
        return 0

    env = _runtime_environment()
    for variant_name, command in commands:
        prefix = f"[{variant_name}] " if variant_name is not None else ""
        print(prefix + " ".join(command))
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
