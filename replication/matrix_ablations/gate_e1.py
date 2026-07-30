#!/usr/bin/env python3
"""Certify the matrix meta-follower checkpoints required by an E2 sweep."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time


SCHEMA_VERSION = 1
DEFAULT_MAX_REGRET = 0.25


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def canonical_json(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_sha256(config):
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def _matches(config, expected):
    return all(config.get(key) == value for key, value in expected.items())


def _artifact_sha(manifest, name, path, errors):
    artifacts = manifest.get("artifacts", {})
    artifact = artifacts.get(name) if isinstance(artifacts, dict) else None
    if not isinstance(artifact, dict):
        errors.append("manifest is missing the {!r} artifact".format(name))
        return None
    expected = artifact.get("sha256")
    if not isinstance(expected, str):
        errors.append("manifest artifact {!r} has no SHA256".format(name))
        return None
    if not path.is_file():
        errors.append("artifact {!r} is missing: {}".format(name, path))
        return None
    actual = sha256(path)
    if actual != expected:
        errors.append("artifact {!r} SHA256 does not match manifest".format(name))
    return actual


def _numeric_max_regret(value, source, errors):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append("{} max_regret is not numeric".format(source))
        return None
    value = float(value)
    if not math.isfinite(value):
        errors.append("{} max_regret is not finite".format(source))
        return None
    return value


def _centered_follower_payoffs(config, errors):
    try:
        payoffs = config["payoffs"]
        offset = float(config["reward_offset"])
        return [
            [float(joint_payoffs[1]) + offset for joint_payoffs in leader_row]
            for leader_row in payoffs
        ]
    except (KeyError, TypeError, ValueError, IndexError):
        errors.append("config does not define valid follower payoffs and reward_offset")
        return None


def _validate_contract_alignment(contract, config, errors):
    if contract.get("schema_version") != 2:
        errors.append("unsupported response contract schema")
    if contract.get("kind") != "matrix_meta_follower":
        errors.append("response contract kind is not matrix_meta_follower")
    if contract.get("algorithm") != config.get("algorithm"):
        errors.append("response contract algorithm does not match config")

    game = contract.get("response_game")
    if not isinstance(game, dict):
        errors.append("response contract has no response_game object")
        return
    comparisons = {
        "profile_id": config.get("profile_id"),
        "memory_mode": config.get("memory_mode"),
        "episode_length": config.get("episode_length"),
        "query_states": config.get("query_states"),
        "training_reward_offset": config.get("reward_offset"),
    }
    for key, expected in comparisons.items():
        if game.get(key) != expected:
            errors.append("response_game {} does not match config".format(key))
    if game.get("profile") != config.get("profile"):
        errors.append("response_game profile does not match config")
    centered = _centered_follower_payoffs(config, errors)
    if centered is not None and game.get("centered_follower_payoffs") != centered:
        errors.append("response_game follower payoffs do not match config")


def _validate_completed_run(run_dir, config, manifest, expected, plan_hash, threshold):
    errors = []
    if not isinstance(expected, dict):
        errors.append("planned E1 match is not a JSON object")
        expected = {}
    if not _matches(config, expected):
        errors.append("config does not match its planned E1 record")
    if config.get("sweep_plan_sha256") != plan_hash:
        errors.append("config is not bound to the current plan SHA256")
    if manifest.get("status") != "completed":
        errors.append("manifest status is not completed")
    if manifest.get("config_sha256") != config_sha256(config):
        errors.append("manifest config_sha256 does not match config")

    model_path = run_dir / "model.zip"
    contract_path = run_dir / "response_contract.json"
    evaluation_path = run_dir / "evaluation.json"
    model_hash = _artifact_sha(manifest, "model", model_path, errors)
    contract_hash = _artifact_sha(
        manifest, "response_contract", contract_path, errors
    )
    evaluation_hash = _artifact_sha(
        manifest, "evaluation", evaluation_path, errors
    )

    contract = None
    evaluation = None
    if contract_path.is_file():
        try:
            contract = read_json(contract_path)
        except (OSError, ValueError) as exc:
            errors.append("cannot read response contract: {}".format(exc))
    if evaluation_path.is_file():
        try:
            evaluation = read_json(evaluation_path)
        except (OSError, ValueError) as exc:
            errors.append("cannot read evaluation: {}".format(exc))

    contract_regret = None
    evaluation_regret = None
    if isinstance(contract, dict):
        _validate_contract_alignment(contract, config, errors)
        if contract.get("checkpoint_filename") != model_path.name:
            errors.append("response contract names the wrong checkpoint")
        if model_hash is not None and contract.get("checkpoint_sha256") != model_hash:
            errors.append("checkpoint SHA256 does not match response contract")
        metadata = contract.get("metadata", {})
        if not isinstance(metadata, dict):
            errors.append("response contract metadata is not an object")
            metadata = {}
        if metadata.get("seed") != config.get("seed"):
            errors.append("response contract seed does not match config")
        if metadata.get("training_config_sha256") != config_sha256(config):
            errors.append("response contract training config hash does not match")
        contract_evaluation = metadata.get("evaluation", {})
        if not isinstance(contract_evaluation, dict):
            errors.append("response contract evaluation metadata is not an object")
            contract_evaluation = {}
        contract_regret = _numeric_max_regret(
            contract_evaluation.get("max_regret"),
            "response contract",
            errors,
        )
    elif contract is not None:
        errors.append("response contract is not a JSON object")
    if isinstance(evaluation, dict):
        if evaluation.get("config") != config:
            errors.append("evaluation is not bound to the run config")
        summary = evaluation.get("summary", {})
        if not isinstance(summary, dict):
            errors.append("evaluation summary is not an object")
            summary = {}
        evaluation_regret = _numeric_max_regret(
            summary.get("max_regret"),
            "evaluation",
            errors,
        )
    elif evaluation is not None:
        errors.append("evaluation is not a JSON object")
    if contract_regret is not None and evaluation_regret is not None:
        if not math.isclose(contract_regret, evaluation_regret, rel_tol=0.0, abs_tol=1e-12):
            errors.append("contract and evaluation max_regret disagree")
    max_regret = contract_regret
    if max_regret is not None and max_regret > threshold:
        errors.append(
            "max_regret {:.12g} exceeds threshold {:.12g}".format(
                max_regret, threshold
            )
        )

    return {
        "passed": not errors,
        "run_dir": str(run_dir.resolve()),
        "max_regret": max_regret,
        "checkpoint": str(model_path.resolve()),
        "checkpoint_sha256": model_hash,
        "response_contract_sha256": contract_hash,
        "evaluation_sha256": evaluation_hash,
        "errors": errors,
    }


def _planned_e1_records(plan):
    errors = []
    if not isinstance(plan, dict):
        return [], [], ["plan is not a JSON object"]
    seeds = plan.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        return [], [], ["plan seeds must be a nonempty list"]
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        errors.append("plan seeds must be nonnegative integers")
    valid_seeds = [
        seed for seed in seeds
        if not isinstance(seed, bool) and isinstance(seed, int) and seed >= 0
    ]
    if len(set(valid_seeds)) != len(valid_seeds):
        errors.append("plan seeds contain duplicates")
    records_payload = plan.get("records")
    if not isinstance(records_payload, list):
        return seeds, [], errors + ["plan records must be a list"]
    records = [
        record for record in records_payload
        if isinstance(record, dict) and record.get("stage") == "meta-follower"
    ]
    if any(not isinstance(record, dict) for record in records_payload):
        errors.append("plan records must be JSON objects")
    by_seed = {}
    for record in records:
        seed = record.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            errors.append("E1 record has an invalid seed: {!r}".format(seed))
            continue
        by_seed.setdefault(seed, []).append(record)
    for seed in valid_seeds:
        if len(by_seed.get(seed, [])) != 1:
            errors.append(
                "plan requires exactly one E1 record for seed {}; found {}".format(
                    seed, len(by_seed.get(seed, []))
                )
            )
    extras = sorted(set(by_seed) - set(valid_seeds), key=lambda value: str(value))
    if extras:
        errors.append("plan has E1 records for unplanned seeds: {}".format(extras))
    ordered = [
        by_seed[seed][0] for seed in valid_seeds
        if len(by_seed.get(seed, [])) == 1
    ]
    return seeds, ordered, errors


def _discover_attempts(sweep_root, sweep_id):
    attempts = []
    errors = []
    meta_root = sweep_root / "runs" / "meta_follower"
    if not meta_root.exists():
        return attempts, errors
    for config_path in sorted(meta_root.glob("**/config.json")):
        run_dir = config_path.parent
        manifest_path = run_dir / "run_manifest.json"
        try:
            config = read_json(config_path)
        except (OSError, ValueError) as exc:
            errors.append("cannot read {}: {}".format(config_path, exc))
            continue
        if not isinstance(config, dict):
            errors.append("{} is not a JSON object".format(config_path))
            continue
        if config.get("sweep_id") != sweep_id:
            continue
        if not manifest_path.is_file():
            continue
        try:
            manifest = read_json(manifest_path)
        except (OSError, ValueError) as exc:
            errors.append("cannot read {}: {}".format(manifest_path, exc))
            continue
        if not isinstance(manifest, dict):
            errors.append("{} is not a JSON object".format(manifest_path))
            continue
        attempts.append((config, manifest, run_dir))
    for manifest_path in sorted(meta_root.glob("**/run_manifest.json")):
        if not manifest_path.with_name("config.json").is_file():
            errors.append("completed-state manifest has no config: {}".format(
                manifest_path
            ))
    return attempts, errors


def run_gate(sweep_root, plan_path=None, max_regret=DEFAULT_MAX_REGRET, output=None):
    sweep_root = Path(sweep_root).resolve()
    plan_path = Path(plan_path).resolve() if plan_path else sweep_root / "plan.json"
    output = Path(output).resolve() if output else sweep_root / "e1_gate.json"
    try:
        threshold = float(max_regret)
    except (TypeError, ValueError):
        threshold = None
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "matrix_e1_gate",
        "generated_unix": time.time(),
        "sweep_root": str(sweep_root),
        "plan": str(plan_path),
        "max_regret_threshold": threshold,
        "passed": False,
        "status": "failed",
        "errors": [],
        "e1": [],
    }
    if threshold is None or not math.isfinite(threshold) or threshold < 0:
        report["errors"].append("max_regret threshold must be finite and nonnegative")
        write_json(output, report)
        return report
    try:
        plan = read_json(plan_path)
        plan_hash = sha256(plan_path)
    except (OSError, ValueError) as exc:
        report["errors"].append("cannot read sweep plan: {}".format(exc))
        write_json(output, report)
        return report
    report["plan_sha256"] = plan_hash
    if not isinstance(plan, dict):
        report["errors"].append("plan is not a JSON object")
        write_json(output, report)
        return report
    report["sweep_id"] = plan.get("sweep_id")
    seeds, records, plan_errors = _planned_e1_records(plan)
    report["planned_seeds"] = seeds
    report["errors"].extend(plan_errors)
    if not isinstance(plan.get("sweep_id"), str) or not plan["sweep_id"]:
        report["errors"].append("plan has no valid sweep_id")

    attempts, discovery_errors = _discover_attempts(
        sweep_root, plan.get("sweep_id")
    )
    report["errors"].extend(discovery_errors)
    for record in records:
        seed = record["seed"]
        row = {
            "seed": seed,
            "record_key": record.get("key"),
            "passed": False,
            "errors": [],
        }
        candidates = [
            item for item in attempts
            if item[0].get("seed") == seed
            and item[1].get("status") == "completed"
        ]
        if len(candidates) != 1:
            row["errors"].append(
                "expected exactly one completed E1; found {}".format(len(candidates))
            )
            row["completed_run_dirs"] = [
                str(item[2].resolve()) for item in candidates
            ]
        else:
            config, manifest, run_dir = candidates[0]
            validated = _validate_completed_run(
                run_dir,
                config,
                manifest,
                record.get("match", {}),
                plan_hash,
                threshold,
            )
            row.update(validated)
        report["e1"].append(row)

    report["passed"] = (
        not report["errors"]
        and len(report["e1"]) == len(seeds)
        and all(row["passed"] for row in report["e1"])
    )
    report["status"] = "passed" if report["passed"] else "failed"
    write_json(output, report)
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", required=True)
    parser.add_argument(
        "--plan", help="Plan path (default: SWEEP_ROOT/plan.json)."
    )
    parser.add_argument(
        "--max-regret", type=float, default=DEFAULT_MAX_REGRET
    )
    parser.add_argument(
        "--output", help="Gate report path (default: SWEEP_ROOT/e1_gate.json)."
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    report = run_gate(
        args.sweep_root,
        plan_path=args.plan,
        max_regret=args.max_regret,
        output=args.output,
    )
    print(json.dumps({
        "status": report["status"],
        "e1_runs": len(report["e1"]),
        "output": str(Path(args.output).resolve()) if args.output else str(
            Path(args.sweep_root).resolve() / "e1_gate.json"
        ),
    }, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
