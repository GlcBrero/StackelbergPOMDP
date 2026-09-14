"""Plan and run the 210 mechanism-design reruns from the paper manifest.

The two normal-form diagnostics use matrix_ablations/sweep.py. This runner
adds immutable launch records around the existing mechanism training commands;
it does not change their training or evaluation protocols.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from replication.run import build_commands, load_targets, validate_command


COHORTS = {
    "simple_allocation": [
        ("fig_simple_allocation_stackpomdp_mappo_m1", None, 25),
        ("fig_simple_allocation_stackpomdp_mappo_m2", None, 25),
        ("fig_simple_allocation_stackpomdp_mappo", None, 25),
        ("fig_simple_allocation_hidden_queries_mappo", None, 25),
        ("fig_simple_allocation_stackpomdp_ppo", None, 25),
        ("fig_simple_allocation_hidden_queries_ppo", None, 25),
    ],
    "matrix_design": [
        ("fig_matrix_design_ablation", "basic_mappo", 25),
        ("fig_matrix_design_ablation", "basic_ppo", 25),
    ],
    "spm": [("fig_pi_spm_2types", None, 10)],
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def make_records(targets):
    from stackelberg_pomdp.experiments.common import finalized_config
    from stackelberg_pomdp.run_setups import _experiment_name

    records = []
    for group, specifications in COHORTS.items():
        for target, variant, count in specifications:
            for seed in range(1, count + 1):
                expansions = build_commands(targets[target], seed, variant)
                if len(expansions) != 1:
                    raise ValueError("each cohort record must expand to one command")
                argv = expansions[0][1]
                # Print less training chatter; paper evaluation cadence is unchanged.
                argv.extend(["--reward_print_freq", "1000"])
                validate_command(argv)
                module = importlib.import_module(argv[2])
                args = module.build_parser().parse_args(argv[3:])
                experiment = {
                    "simple_allocation": "simple_allocation:{}".format(
                        getattr(args, "num_messages", 3)),
                    "matrix_design": "matrix_design",
                    "spm": "spm:PI:2",
                }[group]
                config = finalized_config(args, experiment)
                key = ".".join(filter(None, (target, variant, "seed{}".format(seed))))
                records.append({
                    "key": key, "group": group, "target": target,
                    "variant": variant, "seed": seed,
                    "argv": argv[1:], "resolved_config": config,
                    "log_directory": "stackelberg_pomdp/logs/" + _experiment_name(config),
                })
    if len({row["log_directory"] for row in records}) != len(records):
        raise ValueError("cohort records would overwrite one another's logs")
    return records


def prepare(batch_root):
    batch_root = Path(batch_root).resolve()
    batch_root.mkdir(parents=True, exist_ok=True)
    path = batch_root / "mechanism_plan.json"
    if path.exists():
        raise FileExistsError("refusing to overwrite an existing plan")
    manifest = ROOT / "replication/targets.json"
    records = make_records(load_targets(manifest))
    write_json(path, {
        "schema_version": 1, "created_at": now(), "code_root": str(ROOT),
        "targets_sha256": sha256(manifest), "records": records,
        "counts": dict(Counter(row["group"] for row in records)),
        "scope": "User-approved reruns excluding the critic flag as a rerun trigger; current fix retained.",
    })
    print(json.dumps({"plan": str(path), "runs": len(records)}), flush=True)


def run_record(batch_root, group, index):
    batch_root = Path(batch_root).resolve()
    plan_path = batch_root / "mechanism_plan.json"
    plan = json.loads(plan_path.read_text())
    if plan["code_root"] != str(ROOT):
        raise ValueError("plan belongs to a different source snapshot")
    if plan["targets_sha256"] != sha256(ROOT / "replication/targets.json"):
        raise ValueError("target manifest changed after planning")
    records = [row for row in plan["records"] if row["group"] == group]
    if not 0 <= index < len(records):
        raise IndexError("record index is outside the planned cohort")
    record = records[index]
    log_directory = ROOT / record["log_directory"]
    if log_directory.exists():
        raise FileExistsError("refusing to overwrite existing training artifacts")
    record_dir = batch_root / "mechanism_runs" / record["key"]
    record_dir.mkdir(parents=True, exist_ok=False)
    status_path = record_dir / "run.json"
    state = {
        **record, "status": "running", "started_at": now(),
        "plan_sha256": sha256(plan_path), "python": sys.executable,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    source_manifest = batch_root / "source_manifest.json"
    if source_manifest.exists():
        state["source_manifest_sha256"] = sha256(source_manifest)
    write_json(status_path, state)
    print("Starting " + record["key"], flush=True)
    try:
        with (record_dir / "training.log").open("x") as output:
            subprocess.run(
                [sys.executable, *record["argv"]], cwd=ROOT,
                stdout=output, stderr=subprocess.STDOUT, check=True,
            )
        errors = [line.strip() for line in (record_dir / "training.log").read_text().splitlines()
                  if line.startswith("Eval error:")]
        if errors:
            raise RuntimeError("evaluation failed: " + "; ".join(errors[:3]))
        checkpoints = list(log_directory.glob("rl_model_*_steps.zip"))
        if not checkpoints or not (log_directory / "progress.csv").is_file():
            raise RuntimeError("training ended without its checkpoint or progress CSV")
        state["artifacts"] = {
            str(path.relative_to(ROOT)): sha256(path)
            for path in sorted(log_directory.rglob("*")) if path.is_file()
        }
        state["status"] = "completed"
    except BaseException as exc:
        state.update(status="failed", error=str(exc))
        raise
    finally:
        state["finished_at"] = now()
        write_json(status_path, state)
    print("Completed " + record["key"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run"))
    parser.add_argument("--batch-root", required=True)
    parser.add_argument("--group", choices=tuple(COHORTS))
    parser.add_argument("--record-index", type=int)
    args = parser.parse_args()
    if args.action == "plan":
        prepare(args.batch_root)
    elif args.group is None or args.record_index is None:
        parser.error("run requires --group and --record-index")
    else:
        run_record(args.batch_root, args.group, args.record_index)


if __name__ == "__main__":
    main()
