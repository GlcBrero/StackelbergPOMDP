#!/usr/bin/env python3
"""Summarize qualitative trends from a plan-bound matrix sweep.

The monitor is deliberately read-only with respect to run artifacts.  It
validates manifests and final evaluations, collapses immutable attempts to
logical plan records, and writes only its derived JSON report.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time


DEFAULT_ROOT = Path("replication/matrix_ablations/results")
STATUSES = ("completed", "failed", "running", "missing")
DELTA_ORDER = (
    "phase_observability",
    "q_reset",
    "response_reward",
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reject_nonfinite(value, location):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite value at {}".format(location))
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_nonfinite(child, "{}.{}".format(location, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_nonfinite(child, "{}[{}]".format(location, index))


def read_json(path):
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read valid JSON from {}: {}".format(path, exc))
    _reject_nonfinite(payload, str(path))
    return payload


def write_json_atomic(path, payload):
    """Replace ``path`` atomically after writing a complete finite JSON file."""
    path = Path(path)
    _reject_nonfinite(payload, "report")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("{}.{}.tmp".format(path.name, os.getpid()))
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _matches(config, expected):
    return all(config.get(key) == value for key, value in expected.items())


def _validate_plan(plan, plan_path):
    if not isinstance(plan, dict) or not isinstance(plan.get("records"), list):
        raise ValueError("invalid sweep plan structure: {}".format(plan_path))
    sweep_id = plan.get("sweep_id")
    if not isinstance(sweep_id, str) or not sweep_id:
        raise ValueError("sweep plan has no valid sweep_id")
    records = {}
    for index, record in enumerate(plan["records"]):
        if not isinstance(record, dict):
            raise ValueError("plan record {} is not an object".format(index))
        key = record.get("key")
        if not isinstance(key, str) or not key:
            raise ValueError("plan record {} has no valid key".format(index))
        if key in records:
            raise ValueError("duplicate plan record key: {}".format(key))
        if record.get("stage") not in ("meta-follower", "leader"):
            raise ValueError("unknown plan stage for {}".format(key))
        expected = record.get("match")
        if not isinstance(expected, dict):
            raise ValueError("plan record {} has no match contract".format(key))
        if expected.get("sweep_id") != sweep_id:
            raise ValueError("plan record {} has the wrong sweep_id".format(key))
        if expected.get("record_key") != key:
            raise ValueError("plan record {} has the wrong record_key".format(key))
        records[key] = record
    return records


def _artifact_path(run_dir, manifest, name):
    artifact = (manifest.get("artifacts") or {}).get(name)
    if not isinstance(artifact, dict) or not artifact.get("sha256"):
        raise ValueError(
            "completed manifest has no {} checksum: {}".format(name, run_dir)
        )
    path_value = artifact.get("path")
    if not isinstance(path_value, str) or Path(path_value).name != "{}.json".format(
            name
    ):
        raise ValueError("invalid {} artifact path in {}".format(name, run_dir))
    return run_dir / "{}.json".format(name), artifact["sha256"]


def _load_completed_leader_evaluation(run_dir, config, manifest):
    evaluation_path, expected_hash = _artifact_path(
        run_dir, manifest, "evaluation"
    )
    if not evaluation_path.is_file():
        raise ValueError("missing completed evaluation: {}".format(evaluation_path))
    if file_sha256(evaluation_path) != expected_hash:
        raise ValueError("evaluation hash mismatch: {}".format(evaluation_path))
    evaluation = read_json(evaluation_path)
    if evaluation.get("config") != config:
        raise ValueError("evaluation config mismatch: {}".format(evaluation_path))
    per_stage = evaluation.get("per_stage_summary")
    if not isinstance(per_stage, dict):
        raise ValueError("evaluation has no per_stage_summary: {}".format(
            evaluation_path
        ))
    value = per_stage.get("mean")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("evaluation has no numeric per-stage mean: {}".format(
            evaluation_path
        ))
    if not math.isfinite(float(value)):
        raise ValueError("non-finite per-stage mean: {}".format(evaluation_path))
    n = per_stage.get("n")
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("evaluation has invalid per-stage n: {}".format(
            evaluation_path
        ))
    for field in ("std", "sem"):
        statistic = per_stage.get(field)
        if statistic is not None and (
                isinstance(statistic, bool)
                or not isinstance(statistic, (int, float))
                or not math.isfinite(float(statistic))
                or statistic < 0
        ):
            raise ValueError("evaluation has invalid {}: {}".format(
                field, evaluation_path
            ))
    return float(value)


def _load_attempts(sweep_root, plan, records, plan_hash):
    attempts = {key: [] for key in records}
    runs_root = sweep_root / "runs"
    for manifest_path in sorted(runs_root.glob("**/run_manifest.json")):
        run_dir = manifest_path.parent
        config_path = run_dir / "config.json"
        if not config_path.is_file():
            raise ValueError("manifest has no config: {}".format(manifest_path))
        config = read_json(config_path)
        manifest = read_json(manifest_path)
        key = config.get("record_key")
        if config.get("sweep_id") != plan["sweep_id"]:
            raise ValueError("foreign sweep config under {}: {}".format(
                runs_root, config_path
            ))
        if key not in records:
            raise ValueError("config references unknown plan record: {}".format(
                config_path
            ))
        if config.get("sweep_plan_sha256") != plan_hash:
            raise ValueError("sweep plan hash mismatch: {}".format(config_path))
        if not _matches(config, records[key]["match"]):
            raise ValueError("config violates plan match contract: {}".format(
                config_path
            ))
        if manifest.get("config_sha256") != canonical_sha256(config):
            raise ValueError("config hash mismatch: {}".format(config_path))
        status = manifest.get("status")
        if status not in ("running", "failed", "completed"):
            raise ValueError("invalid manifest status in {}".format(manifest_path))
        attempt = config.get("attempt", 0)
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
            raise ValueError("invalid attempt number in {}".format(config_path))
        value = None
        if status == "completed" and records[key]["stage"] == "leader":
            value = _load_completed_leader_evaluation(
                run_dir, config, manifest
            )
        attempts[key].append({
            "attempt": attempt,
            "status": status,
            "run_dir": str(run_dir),
            "config": config,
            "per_stage_mean": value,
        })
    for key, found in attempts.items():
        numbers = [item["attempt"] for item in found]
        if len(numbers) != len(set(numbers)):
            raise ValueError("duplicate attempt number for record {}".format(key))
    return attempts


def collapse_attempts(key, attempts):
    completed = [item for item in attempts if item["status"] == "completed"]
    if len(completed) > 1:
        raise ValueError("multiple completed attempts for record {}".format(key))
    if completed:
        return "completed", completed[0]
    running = [item for item in attempts if item["status"] == "running"]
    if running:
        return "running", max(running, key=lambda item: item["attempt"])
    failed = [item for item in attempts if item["status"] == "failed"]
    if failed:
        return "failed", max(failed, key=lambda item: item["attempt"])
    return "missing", None


def sample_summary(values):
    values = [float(value) for value in values]
    if not values:
        return {"n": 0, "mean": None, "std": None, "sem": None}
    mean = math.fsum(values) / len(values)
    if len(values) == 1:
        return {"n": 1, "mean": mean, "std": None, "sem": None}
    variance = math.fsum((value - mean) ** 2 for value in values) / (
        len(values) - 1
    )
    std = math.sqrt(max(0.0, variance))
    return {
        "n": len(values),
        "mean": mean,
        "std": std,
        "sem": std / math.sqrt(len(values)),
    }


def _status_counts(rows):
    result = {"planned": len(rows)}
    result.update({status: 0 for status in STATUSES})
    for row in rows:
        result[row["status"]] += 1
    return result


def _cell_summaries(rows):
    groups = {}
    for row in rows:
        if row["stage"] != "leader" or row["status"] != "completed":
            continue
        key = (
            row["experiment"], row["matrix"], row["algorithm"],
            row["condition"], row.get("learning_rate"),
        )
        by_seed = groups.setdefault(key, {})
        if row["seed"] in by_seed:
            raise ValueError("duplicate completed training seed for cell {}".format(
                key
            ))
        by_seed[row["seed"]] = row["per_stage_mean"]
    summaries = []
    for key, by_seed in sorted(groups.items()):
        seeds = sorted(by_seed)
        summaries.append({
            "experiment": key[0],
            "matrix": key[1],
            "algorithm": key[2],
            "condition": key[3],
            "learning_rate": key[4],
            "seeds": seeds,
            **sample_summary([by_seed[seed] for seed in seeds]),
        })
    return summaries


def _completed_seed_map(rows, experiment, matrix, algorithm, condition, learning_rate):
    result = {}
    for row in rows:
        if (
                row["stage"] == "leader"
                and row["status"] == "completed"
                and row["experiment"] == experiment
                and row["matrix"] == matrix
                and row["algorithm"] == algorithm
                and row["condition"] == condition
                and row.get("learning_rate") == learning_rate
        ):
            if row["seed"] in result:
                raise ValueError("duplicate seed in condition comparison")
            result[row["seed"]] = row["per_stage_mean"]
    return result


def _comparison(
        rows, comparison_id, experiment, matrix, algorithm,
        positive_condition, negative_condition, kind="signed_delta",
        learning_rate=None,
):
    positive = _completed_seed_map(
        rows, experiment, matrix, algorithm, positive_condition, learning_rate
    )
    negative = _completed_seed_map(
        rows, experiment, matrix, algorithm, negative_condition, learning_rate
    )
    seeds = sorted(set(positive).intersection(negative))
    deltas = [positive[seed] - negative[seed] for seed in seeds]
    result = {
        "id": comparison_id if learning_rate is None else comparison_id + "_lr{:g}".format(learning_rate),
        "experiment": experiment,
        "matrix": matrix,
        "algorithm": algorithm,
        "learning_rate": learning_rate,
        "kind": kind,
        "definition": "{} - {}".format(
            positive_condition, negative_condition
        ),
        "positive_condition": positive_condition,
        "negative_condition": negative_condition,
        "paired_seeds": seeds,
        "positive_mean_on_paired_seeds": (
            sample_summary([positive[seed] for seed in seeds])["mean"]
        ),
        "negative_mean_on_paired_seeds": (
            sample_summary([negative[seed] for seed in seeds])["mean"]
        ),
        **sample_summary(deltas),
    }
    if kind == "gap":
        result["absolute_mean_gap"] = (
            None if result["mean"] is None else abs(result["mean"])
        )
    return result


def _planned_axes(rows, experiment, algorithms=None):
    axes = set()
    for row in rows:
        if row["stage"] != "leader" or row["experiment"] != experiment:
            continue
        if algorithms is not None and row["algorithm"] not in algorithms:
            continue
        axes.add((row["matrix"], row["algorithm"], row.get("learning_rate")))
    return sorted(axes)


def condition_deltas(rows):
    result = {name: [] for name in DELTA_ORDER}
    for matrix, algorithm, learning_rate in _planned_axes(rows, "phase_observability"):
        result["phase_observability"].append(_comparison(
            rows,
            "phase_{}_visible_minus_hidden".format(algorithm.lower()),
            "phase_observability",
            matrix,
            algorithm,
            "visible",
            "hidden",
            learning_rate=learning_rate,
        ))
    for matrix, algorithm, learning_rate in _planned_axes(rows, "q_reset"):
        result["q_reset"].append(_comparison(
            rows,
            "q_reset_{}_reset_minus_ongoing".format(algorithm.lower()),
            "q_reset",
            matrix,
            algorithm,
            "reset",
            "ongoing",
            learning_rate=learning_rate,
        ))
    for matrix, algorithm, learning_rate in _planned_axes(rows, "response_reward"):
        result["response_reward"].append(_comparison(
            rows,
            "response_reward_{}_{}_excluded_minus_included".format(
                matrix, algorithm.lower()
            ),
            "response_reward",
            matrix,
            algorithm,
            "excluded",
            "included",
            learning_rate=learning_rate,
        ))
    return result


def build_summary(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = read_json(plan_path)
    records = _validate_plan(plan, plan_path)
    plan_hash = file_sha256(plan_path)
    sweep_root = plan_path.parent
    attempts = _load_attempts(sweep_root, plan, records, plan_hash)
    rows = []
    for key, record in records.items():
        status, selected = collapse_attempts(key, attempts[key])
        row = {
            "record_key": key,
            "stage": record["stage"],
            "seed": int(record["seed"]),
            "status": status,
            "attempts": len(attempts[key]),
            "selected_attempt": None if selected is None else selected["attempt"],
            "run_dir": None if selected is None else selected["run_dir"],
        }
        if record["stage"] == "leader":
            match = record["match"]
            row.update({
                "experiment": match["experiment"],
                "matrix": match["matrix"],
                "algorithm": match["algorithm"],
                "condition": match["condition"],
                "learning_rate": match.get("learning_rate"),
                "per_stage_mean": (
                    None if selected is None else selected["per_stage_mean"]
                ),
            })
        rows.append(row)
    stage_rows = {
        stage: [row for row in rows if row["stage"] == stage]
        for stage in ("meta-follower", "leader")
    }
    return {
        "schema_version": 1,
        "generated_unix": time.time(),
        "sweep_id": plan["sweep_id"],
        "plan": {"path": str(plan_path), "sha256": plan_hash},
        "uncertainty": (
            "sample standard error across independent training seeds; "
            "condition deltas are paired by seed"
        ),
        "status_counts": _status_counts(rows),
        "status_counts_by_stage": {
            stage: _status_counts(found) for stage, found in stage_rows.items()
        },
        "records": rows,
        "final_per_stage": _cell_summaries(rows),
        "condition_deltas": condition_deltas(rows),
    }


def _format_statistic(row):
    if row["mean"] is None:
        return "unavailable (n=0)"
    sem = "n/a" if row["sem"] is None else "{:.3f}".format(row["sem"])
    text = "{:.3f} +/- {} (n={})".format(row["mean"], sem, row["n"])
    if row.get("kind") == "gap":
        text += "; absolute gap {:.3f}".format(row["absolute_mean_gap"])
    return text


def print_summary(summary, output_path):
    counts = summary["status_counts_by_stage"]["leader"]
    print(
        "{}: leaders {}/{} completed; {} failed, {} running, {} missing".format(
            summary["sweep_id"], counts["completed"], counts["planned"],
            counts["failed"], counts["running"], counts["missing"],
        )
    )
    for experiment in DELTA_ORDER:
        for row in summary["condition_deltas"][experiment]:
            print("{} [{}]: {}".format(
                row["id"], row["definition"], _format_statistic(row)
            ))
    print("report: {}".format(output_path))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-id", required=True)
    parser.add_argument("--results-root", default=str(DEFAULT_ROOT))
    parser.add_argument(
        "--output",
        help="Derived JSON path (default: <sweep root>/trend_summary.json).",
    )
    return parser


def main():
    args = build_parser().parse_args()
    sweep_root = Path(args.results_root).resolve() / args.sweep_id
    plan_path = sweep_root / "plan.json"
    output_path = (
        Path(args.output).resolve()
        if args.output else sweep_root / "trend_summary.json"
    )
    summary = build_summary(plan_path)
    write_json_atomic(output_path, summary)
    print_summary(summary, output_path)


if __name__ == "__main__":
    main()
