#!/usr/bin/env python3
"""Plan, explicitly execute, or inspect the two appendix diagnostics."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stackelberg_pomdp.matrix_ablations.presets import (
    APPENDIX_CURVES, FIGURE_EXPERIMENTS, LEADER_DEFAULTS,
)

DEFAULT_ROOT = Path("replication/matrix_ablations/results")


def parse_seeds(value):
    seeds = []
    for token in value.split(","):
        token = token.strip()
        if "-" in token:
            start, end = (int(part) for part in token.split("-", 1))
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(token))
    seeds = sorted(set(seeds))
    if not seeds or any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("seeds must be nonnegative")
    return seeds


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plan_path(args):
    return Path(args.results_root).resolve() / args.sweep_id / "plan.json"


def common_output(
        root, plan_file, sweep_id, record_key, attempt,
        wandb=False, wandb_project="StackPOMDP",
):
    output = [
        "--output-root", str(root / "runs"),
        "--sweep-id", sweep_id,
        "--record-key", record_key,
        "--sweep-plan", str(plan_file),
        "--attempt", str(attempt),
    ]
    if wandb:
        output.extend([
            "--wandb", "--wandb-project", wandb_project,
            "--wandb-group", sweep_id,
            "--wandb-name", "{}.attempt{}".format(record_key, attempt),
        ])
    else:
        output.append("--no-wandb")
    return output


def make_plan(args):
    root = Path(args.results_root).resolve() / args.sweep_id
    path = root / "plan.json"
    if path.exists():
        raise FileExistsError("refusing to overwrite existing plan: {}".format(path))
    figure = getattr(args, "figure", "all")
    experiments = list(APPENDIX_CURVES) if figure == "all" else [FIGURE_EXPERIMENTS[figure]]
    seeds_by_experiment = {
        experiment: args.seeds or list(range(1, LEADER_DEFAULTS[experiment]["seeds"] + 1))
        for experiment in experiments
    }
    records = []
    for seed in seeds_by_experiment.get("phase_observability", []):
        key = "e1.seed{}".format(seed)
        records.append({
            "key": key,
            "stage": "meta-follower",
            "seed": seed,
            "match": {
                "sweep_id": args.sweep_id,
                "record_key": key,
                "stage": "meta_follower",
                "profile_id": "paper_joint_v1",
                "matrix": "prisoners_dilemma",
                "memory_mode": "joint",
                "algorithm": "REINFORCE",
                "seed": seed,
                "timesteps": args.e1_timesteps,
                "learning_rate": 0.02,
            },
            "argv": [
                "-m",
                "stackelberg_pomdp.experiments.matrix_ablations",
                "meta-follower", "--matrix", "prisoners_dilemma",
                "--memory-mode", "joint", "--algorithm", "REINFORCE",
                "--seed", str(seed), "--timesteps", str(args.e1_timesteps),
            ],
        })

    def leader_record(seed, experiment, condition, matrix, learning_rate):
        defaults = LEADER_DEFAULTS[experiment]
        algorithm = defaults["algorithm"]
        timesteps = args.leader_timesteps if args.leader_timesteps is not None else defaults["timesteps"]
        key = ".".join((experiment, matrix, algorithm.lower(), condition,
                        "lr" + str(learning_rate).replace(".", "p"), "seed{}".format(seed)))
        argv = [
            "-m",
            "stackelberg_pomdp.experiments.matrix_ablations",
            "leader", "--experiment", experiment,
            "--condition", condition, "--algorithm", algorithm,
            "--matrix", matrix, "--seed", str(seed),
            "--timesteps", str(timesteps),
            "--eval-freq", str(args.eval_freq),
        ]
        argv.extend(["--learning-rate", str(learning_rate)])
        if experiment == "phase_observability":
            argv.extend(["--response-algorithm", "REINFORCE"])
        profile_id = (
            "paper_joint_v1" if experiment == "phase_observability" else "one_shot_{}_v1".format(matrix)
        )
        match = {
            "sweep_id": args.sweep_id, "record_key": key,
            "stage": "leader", "seed": seed,
            "profile_id": profile_id,
            "experiment": experiment, "condition": condition,
            "algorithm": algorithm, "matrix": matrix,
            "memory_mode": (
                "joint" if experiment == "phase_observability" else "none"
            ),
            "timesteps": timesteps,
            "learning_rate": learning_rate,
            "eval_freq": args.eval_freq,
        }
        if experiment == "response_reward":
            match["q_protocol"] = {
                "alpha": 0.2,
                "epsilon": 0.1,
                "exploration": "parameter_noise",
                "initialization": "zero",
                "initialization_std": 0.01,
            }
        return {
            "key": key,
            "stage": "leader",
            "seed": seed,
            "response_seed": (
                seed if experiment == "phase_observability"
                else None
            ),
            "match": match,
            "argv": argv,
        }

    for experiment in experiments:
        for seed in seeds_by_experiment[experiment]:
            for condition, matrix, learning_rate in APPENDIX_CURVES[experiment]:
                records.append(leader_record(
                    seed, experiment, condition, matrix, learning_rate,
                ))

    payload = {
        "schema_version": 1,
        "sweep_id": args.sweep_id,
        "seeds": sorted({seed for seeds in seeds_by_experiment.values() for seed in seeds}),
        "seeds_by_experiment": seeds_by_experiment,
        "memory_mode": "joint",
        "profile_id": "paper_joint_v1",
        "planner_python": sys.executable,
        "uncertainty": "sample SEM across independent training seeds",
        "e1_runs": sum(row["stage"] == "meta-follower" for row in records),
        "leader_runs": sum(row["stage"] == "leader" for row in records),
        "records": records,
    }
    write_json(path, payload)
    print(json.dumps({
        "planned": len(records),
        "e1_runs": payload["e1_runs"],
        "leader_runs": payload["leader_runs"],
        "plan": str(path),
    }, sort_keys=True))


def completed_configs(root):
    result = []
    for manifest_path in (root / "runs").glob("**/run_manifest.json"):
        manifest = read_json(manifest_path)
        config_path = manifest_path.with_name("config.json")
        if config_path.exists():
            result.append((read_json(config_path), manifest, manifest_path.parent))
    return result


def matches(config, expected):
    return all(config.get(key) == value for key, value in expected.items())


def matching_attempts(record, configs):
    return [item for item in configs if matches(item[0], record["match"])]


def attempt_number(item):
    return int(item[0].get("attempt", 0))


def record_status(record, configs):
    found = matching_attempts(record, configs)
    if not found:
        return "missing", None
    completed = [item for item in found if item[1].get("status") == "completed"]
    if len(completed) > 1:
        return "duplicate", None
    if completed:
        return "completed", completed[0][2]
    running = [item for item in found if item[1].get("status") == "running"]
    if running:
        return "running", max(running, key=attempt_number)[2]
    failed = [item for item in found if item[1].get("status") == "failed"]
    if failed:
        return "failed", max(failed, key=attempt_number)[2]
    latest = max(found, key=attempt_number)
    return latest[1].get("status", "unknown"), latest[2]


def response_checkpoint(seed, configs, plan):
    expected_records = [
        record for record in plan["records"]
        if record["stage"] == "meta-follower" and record["seed"] == seed
    ]
    if len(expected_records) != 1:
        raise RuntimeError("plan has no unique E1 dependency for seed {}".format(seed))
    found = [
        item for item in configs
        if matches(item[0], expected_records[0]["match"])
    ]
    completed = [item for item in found if item[1].get("status") == "completed"]
    if len(completed) != 1:
        raise RuntimeError(
            "expected one completed E1 checkpoint for seed {}; found {}".format(
                seed, len(completed)
            )
        )
    checkpoint = completed[0][2] / "model.zip"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def select_records(payload, stage, record_group="all"):
    records = [row for row in payload["records"] if row["stage"] == stage]
    if stage != "leader" and record_group != "all":
        raise ValueError("record-group applies only to the leader stage")
    if record_group == "independent":
        records = [row for row in records if row.get("response_seed") is None]
    elif record_group == "meta-dependent":
        records = [row for row in records if row.get("response_seed") is not None]
    return records


def run_stage(args):
    payload = read_json(plan_path(args))
    root = Path(args.results_root).resolve() / args.sweep_id
    plan_file = plan_path(args)
    records = select_records(payload, args.stage, args.record_group)
    if args.record_index is not None:
        if args.record_index < 0 or args.record_index >= len(records):
            raise IndexError(
                "record-index {} is outside 0..{} for stage {}".format(
                    args.record_index, len(records) - 1, args.stage
                )
            )
        records = [records[args.record_index]]
    failures = []
    for index, record in enumerate(records, start=1):
        configs = completed_configs(root)
        status, directory = record_status(record, configs)
        if status == "completed":
            try:
                if record.get("response_seed") is not None:
                    checkpoint = response_checkpoint(
                        record["response_seed"], configs, payload
                    )
                    existing = read_json(directory / "config.json")
                    if existing.get("response_checkpoint_sha256") != file_sha256(
                        checkpoint
                    ):
                        raise RuntimeError(
                            "completed run uses the wrong E1 checkpoint"
                        )
            except Exception as exc:
                failures.append((record["key"], str(exc)))
                print("invalid completed {}: {}".format(record["key"], exc))
                continue
            print("skip completed {} ({})".format(record["key"], directory), flush=True)
            continue
        if status == "duplicate":
            failures.append((record["key"], "multiple completed attempts"))
            print("skip duplicate {}".format(record["key"]), flush=True)
            continue
        if status == "running":
            failures.append((record["key"], "an attempt is already running"))
            print("skip running {} ({})".format(record["key"], directory), flush=True)
            continue
        if status == "failed" and not args.retry_failed:
            failures.append((record["key"], "failed; pass --retry-failed"))
            print("skip failed {} ({})".format(record["key"], directory), flush=True)
            continue
        if status not in ("missing", "failed"):
            failures.append((record["key"], "non-rerunnable status {}".format(status)))
            print("skip {} status {}".format(record["key"], status), flush=True)
            continue

        attempts = matching_attempts(record, configs)
        attempt = 0 if not attempts else max(map(attempt_number, attempts)) + 1
        argv = [
            sys.executable,
            *record["argv"],
            *common_output(
                root, plan_file, args.sweep_id, record["key"], attempt,
                wandb=args.wandb, wandb_project=args.wandb_project,
            ),
        ]
        try:
            if record.get("response_seed") is not None:
                checkpoint = response_checkpoint(
                    record["response_seed"], configs, payload
                )
                argv.extend(["--response-checkpoint", str(checkpoint)])
        except Exception as exc:
            failures.append((record["key"], "blocked by E1: {}".format(exc)))
            print("blocked {}: {}".format(record["key"], exc), flush=True)
            continue
        print("[{}/{}] {}".format(index, len(records), record["key"]), flush=True)
        try:
            subprocess.run(argv, check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            failures.append((record["key"], str(exc)))
            print("failed {}: {}".format(record["key"], exc), flush=True)

    print(json.dumps({
        "stage": args.stage,
        "records": len(records),
        "failures": len(failures),
    }, sort_keys=True))
    if failures:
        details = "; ".join("{}: {}".format(*item) for item in failures)
        raise RuntimeError("{} sweep records unresolved: {}".format(
            len(failures), details
        ))


def show_status(args):
    payload = read_json(plan_path(args))
    root = Path(args.results_root).resolve() / args.sweep_id
    configs = completed_configs(root)
    counts = {}
    for record in payload["records"]:
        status, _ = record_status(record, configs)
        counts[status] = counts.get(status, 0) + 1
    print(json.dumps({
        "sweep_id": args.sweep_id,
        "total": len(payload["records"]),
        "status": counts,
    }, indent=2, sort_keys=True))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="Write commands; execute nothing.")
    plan.add_argument("--sweep-id", required=True)
    plan.add_argument("--results-root", default=str(DEFAULT_ROOT))
    plan.add_argument("--figure", choices=("all", *FIGURE_EXPERIMENTS), default="all")
    plan.add_argument("--seeds", type=parse_seeds,
                      help="Override paper counts: phase/reward 10, reset 3.")
    plan.add_argument("--e1-timesteps", type=int, default=25_000)
    plan.add_argument("--leader-timesteps", type=int)
    plan.add_argument("--eval-freq", type=int, default=2_000)

    run = subparsers.add_parser("run", help="Explicitly execute one stage.")
    run.add_argument("--sweep-id", required=True)
    run.add_argument("--results-root", default=str(DEFAULT_ROOT))
    run.add_argument("--stage", choices=("meta-follower", "leader"), required=True)
    run.add_argument(
        "--retry-failed", action="store_true",
        help="Create a new immutable attempt for failed logical records.",
    )
    run.add_argument(
        "--record-index", type=int,
        help="Run one zero-based record within the selected stage (for arrays).",
    )
    run.add_argument(
        "--record-group",
        choices=("all", "independent", "meta-dependent"),
        default="all",
        help="Restrict leader records by whether they require an E1 checkpoint.",
    )
    run.add_argument(
        "--wandb", action=argparse.BooleanOptionalAction, default=False,
        help="Enable per-record W&B evaluation logging.",
    )
    run.add_argument("--wandb-project", default="StackPOMDP")

    status = subparsers.add_parser("status", help="Inspect manifests only.")
    status.add_argument("--sweep-id", required=True)
    status.add_argument("--results-root", default=str(DEFAULT_ROOT))
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "plan":
        make_plan(args)
    elif args.command == "run":
        run_stage(args)
    else:
        show_status(args)


if __name__ == "__main__":
    main()
