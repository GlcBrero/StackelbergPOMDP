#!/usr/bin/env python3
"""Audit immutable E1 response checkpoints without loading trained models.

The audit scans completed matrix meta-follower runs below ``--root``.  It
checks the config/manifest/artifact hash chain, then validates every row of the
REINFORCE checkpoint ledger against its checkpoint, response contract,
training profile, evaluation metrics, and rollout counters.  Run artifacts are
never modified.  The compact report is written atomically to a separate JSON
file and also printed on stdout.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time


AUDIT_SCHEMA_VERSION = 1
RUN_SCHEMA_VERSION = 2
DEFAULT_REGRET_TOLERANCE = 1e-12
DEFAULT_REPORT_NAME = "e1_checkpoint_audit.json"
REQUIRED_ARTIFACTS = (
    "checkpoint_evaluations",
    "evaluation",
    "model",
    "response_contract",
)


def canonical_json(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload):
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path, payload):
    """Atomically replace ``path`` with compact, fsynced JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=".{}.".format(path.name),
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(canonical_json(payload))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _error(errors, message):
    if message not in errors:
        errors.append(message)


def _is_int(value, minimum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        return False
    return minimum is None or value >= minimum


def _finite_number(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _same_number(left, right, tolerance=1e-12):
    return (
        _finite_number(left)
        and _finite_number(right)
        and math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=tolerance
        )
    )


def _safe_leaf(value, field, errors):
    if not isinstance(value, str) or not value:
        _error(errors, "{} is not a nonempty filename".format(field))
        return None
    path = Path(value)
    if path.name != value or value in (".", ".."):
        _error(errors, "{} must name a file in the run directory".format(field))
        return None
    return value


def _load_object(path, label, errors):
    try:
        payload = read_json(path)
    except (OSError, ValueError) as exc:
        _error(errors, "cannot read {}: {}".format(label, exc))
        return None
    if not isinstance(payload, dict):
        _error(errors, "{} is not a JSON object".format(label))
        return None
    return payload


def _validate_profile(config, errors):
    profile_id = config.get("profile_id")
    profile = config.get("profile")
    if not isinstance(profile_id, str) or not profile_id:
        _error(errors, "config has no immutable profile_id")
    if not isinstance(profile, dict):
        _error(errors, "config profile is not an object")
        return None
    if profile.get("profile_id") != profile_id:
        _error(errors, "config profile_id does not match profile snapshot")
    comparisons = {
        "memory_mode": config.get("memory_mode"),
        "query_state_order": config.get("query_states"),
        "training_reward_offset": config.get("reward_offset"),
    }
    for key, expected in comparisons.items():
        if profile.get(key) != expected:
            _error(
                errors,
                "config profile {} does not match config".format(key),
            )
    return sha256_json(profile)


def _artifact_path(run_dir, artifact, name, errors):
    if not isinstance(artifact, dict):
        _error(errors, "manifest artifact {!r} is not an object".format(name))
        return None
    stored_path = artifact.get("path")
    if not isinstance(stored_path, str) or not stored_path:
        _error(errors, "manifest artifact {!r} has no path".format(name))
        return None
    filename = Path(stored_path).name
    if filename in ("", ".", ".."):
        _error(errors, "manifest artifact {!r} has an invalid path".format(name))
        return None
    # Manifests preserve the submission-time path.  Audits deliberately bind
    # artifacts to the downloaded run directory instead of following a stale
    # absolute path from the cluster.
    return run_dir / filename


def _validate_artifacts(run_dir, manifest, errors):
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        _error(errors, "completed manifest has no artifacts object")
        return {}, {}

    by_name = {}
    by_filename = {}
    for name in sorted(artifacts):
        artifact = artifacts[name]
        path = _artifact_path(run_dir, artifact, name, errors)
        expected_hash = artifact.get("sha256") if isinstance(artifact, dict) else None
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            _error(
                errors,
                "manifest artifact {!r} has no valid SHA256".format(name),
            )
        actual_hash = None
        if path is not None:
            if not path.is_file():
                _error(errors, "manifest artifact {!r} is missing".format(name))
            else:
                actual_hash = sha256_file(path)
                if actual_hash != expected_hash:
                    _error(
                        errors,
                        "manifest artifact {!r} SHA256 mismatch".format(name),
                    )
            filename = path.name
            existing = by_filename.get(filename)
            if existing is not None and (
                existing["expected_sha256"] != expected_hash
            ):
                _error(
                    errors,
                    "manifest aliases for {!r} disagree on SHA256".format(
                        filename
                    ),
                )
            # The final checkpoint intentionally has both checkpoint_final and
            # model aliases (likewise for its contract).  Preserve one binding
            # after verifying that aliases agree.
            by_filename[filename] = {
                "name": name,
                "path": path,
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
            }
        by_name[name] = {
            "path": path,
            "expected_sha256": expected_hash,
            "actual_sha256": actual_hash,
        }

    for name in REQUIRED_ARTIFACTS:
        if name not in artifacts:
            _error(errors, "completed manifest is missing {!r}".format(name))
    return by_name, by_filename


def _validate_contract_game(contract, config, profile_hash, errors):
    if contract.get("schema_version") != RUN_SCHEMA_VERSION:
        _error(errors, "response contract schema_version is not 2")
    if contract.get("kind") != "matrix_meta_follower":
        _error(errors, "response contract kind is not matrix_meta_follower")
    if contract.get("algorithm") != config.get("algorithm"):
        _error(errors, "response contract algorithm does not match config")

    game = contract.get("response_game")
    if not isinstance(game, dict):
        _error(errors, "response contract has no response_game object")
        return
    comparisons = {
        "profile_id": config.get("profile_id"),
        "profile": config.get("profile"),
        "memory_mode": config.get("memory_mode"),
        "episode_length": config.get("episode_length"),
        "query_states": config.get("query_states"),
        "training_reward_offset": config.get("reward_offset"),
    }
    for key, expected in comparisons.items():
        if game.get(key) != expected:
            _error(
                errors,
                "response_game {} does not match config".format(key),
            )
    if profile_hash is not None and game.get("profile_sha256") != profile_hash:
        _error(errors, "response_game profile_sha256 does not match profile")


def _validate_metric_alignment(row, summary, errors):
    if not isinstance(summary, dict):
        _error(errors, "response contract evaluation is not an object")
        return
    for key in (
        "mean_regret",
        "max_regret",
        "optimal_commitments",
        "commitments",
    ):
        if key not in summary:
            _error(errors, "response contract evaluation is missing {}".format(key))
            continue
        if key in ("optimal_commitments", "commitments"):
            if summary[key] != row.get(key):
                _error(errors, "checkpoint and contract {} disagree".format(key))
        elif not _same_number(summary[key], row.get(key)):
            _error(errors, "checkpoint and contract {} disagree".format(key))


def _rollout_geometry(config, errors):
    if config.get("algorithm") != "REINFORCE":
        _error(errors, "checkpoint ledger audit requires algorithm REINFORCE")
    protocol = config.get("reinforce_protocol")
    if not isinstance(protocol, dict):
        _error(errors, "config has no reinforce_protocol object")
        return None, None
    rollout = protocol.get("rollout_geometry")
    cadence = protocol.get("checkpoint_geometry")
    if not isinstance(rollout, dict):
        _error(errors, "reinforce_protocol has no rollout_geometry object")
        rollout = {}
    if not isinstance(cadence, dict):
        _error(errors, "reinforce_protocol has no checkpoint_geometry object")
        cadence = {}

    follower_per_update = rollout.get("follower_gradient_samples_per_update")
    executed_per_update = rollout.get("executed_env_steps_per_update")
    update_interval = cadence.get("update_interval")
    if not _is_int(follower_per_update, 1):
        _error(errors, "invalid follower samples per REINFORCE update")
        follower_per_update = None
    if not _is_int(executed_per_update, 1):
        _error(errors, "invalid executed steps per REINFORCE update")
        executed_per_update = None
    if not _is_int(update_interval, 1):
        _error(errors, "invalid checkpoint update interval")
        update_interval = None

    if follower_per_update is not None:
        if config.get("n_steps") != follower_per_update:
            _error(errors, "config n_steps does not match rollout geometry")
        expected = (
            update_interval * follower_per_update
            if update_interval is not None else None
        )
        if expected is not None and cadence.get(
            "follower_gradient_samples_interval"
        ) != expected:
            _error(errors, "checkpoint follower-step cadence is inconsistent")
    if executed_per_update is not None and update_interval is not None:
        if cadence.get("executed_env_steps_interval") != (
            update_interval * executed_per_update
        ):
            _error(errors, "checkpoint executed-step cadence is inconsistent")
    return (
        {
            "follower_per_update": follower_per_update,
            "executed_per_update": executed_per_update,
        },
        update_interval,
    )


def _validate_checkpoint_row(
        run_dir,
        row,
        line_number,
        config,
        config_hash,
        profile_hash,
        by_filename,
        geometry,
        regret_tolerance,
):
    errors = []
    result = {
        "line": line_number,
        "valid": False,
        "strict_pass": False,
        "errors": errors,
    }
    if not isinstance(row, dict):
        _error(errors, "checkpoint ledger row is not an object")
        return result

    if row.get("schema_version") != RUN_SCHEMA_VERSION:
        _error(errors, "checkpoint row schema_version is not 2")
    if row.get("profile_id") != config.get("profile_id"):
        _error(errors, "checkpoint row profile_id does not match config")
    if row.get("seed") != config.get("seed"):
        _error(errors, "checkpoint row seed does not match config")
    if not _same_number(row.get("learning_rate"), config.get("learning_rate")):
        _error(errors, "checkpoint row learning_rate does not match config")
    if row.get("training_config_sha256") != config_hash:
        _error(errors, "checkpoint row training config SHA256 mismatch")
    if row.get("checkpoint_semantics") != "post_optimizer_update":
        _error(errors, "checkpoint row is not post_optimizer_update")

    updates = row.get("completed_updates")
    follower_steps = row.get("follower_gradient_samples")
    executed_steps = row.get("executed_equivalent_steps")
    for key, value in (
        ("completed_updates", updates),
        ("follower_gradient_samples", follower_steps),
        ("executed_equivalent_steps", executed_steps),
    ):
        if not _is_int(value, 1):
            _error(errors, "checkpoint row {} is not a positive integer".format(key))
    if _is_int(updates, 1):
        follower_per_update = geometry.get("follower_per_update")
        executed_per_update = geometry.get("executed_per_update")
        if follower_per_update is not None and follower_steps != (
            updates * follower_per_update
        ):
            _error(errors, "checkpoint follower-step counter is misaligned")
        if executed_per_update is not None and executed_steps != (
            updates * executed_per_update
        ):
            _error(errors, "checkpoint executed-step counter is misaligned")

    max_regret = row.get("max_regret")
    mean_regret = row.get("mean_regret")
    optimal = row.get("optimal_commitments")
    commitments = row.get("commitments")
    if not _finite_number(max_regret) or float(max_regret) < 0.0:
        _error(errors, "checkpoint max_regret is not finite and nonnegative")
    if not _finite_number(mean_regret) or float(mean_regret) < 0.0:
        _error(errors, "checkpoint mean_regret is not finite and nonnegative")
    if not _is_int(commitments, 1):
        _error(errors, "checkpoint commitments is not a positive integer")
    if not _is_int(optimal, 0) or (
        _is_int(commitments, 1) and optimal > commitments
    ):
        _error(errors, "checkpoint optimal_commitments is invalid")

    checkpoint_name = _safe_leaf(
        row.get("checkpoint_filename"), "checkpoint_filename", errors
    )
    contract_name = _safe_leaf(
        row.get("contract_filename"), "contract_filename", errors
    )
    checkpoint_hash = row.get("checkpoint_sha256")
    contract_hash = row.get("contract_sha256")
    contract = None
    if checkpoint_name is not None:
        checkpoint_path = run_dir / checkpoint_name
        declared = by_filename.get(checkpoint_name)
        if declared is None:
            _error(errors, "checkpoint is not declared in the manifest")
        if not checkpoint_path.is_file():
            _error(errors, "checkpoint file is missing")
        else:
            actual = sha256_file(checkpoint_path)
            if checkpoint_hash != actual:
                _error(errors, "checkpoint row SHA256 mismatch")
            if declared is not None and declared["expected_sha256"] != actual:
                _error(errors, "checkpoint manifest SHA256 mismatch")
    if contract_name is not None:
        contract_path = run_dir / contract_name
        declared = by_filename.get(contract_name)
        if declared is None:
            _error(errors, "checkpoint contract is not declared in the manifest")
        if not contract_path.is_file():
            _error(errors, "checkpoint contract file is missing")
        else:
            actual = sha256_file(contract_path)
            if contract_hash != actual:
                _error(errors, "checkpoint contract row SHA256 mismatch")
            if declared is not None and declared["expected_sha256"] != actual:
                _error(errors, "checkpoint contract manifest SHA256 mismatch")
            contract = _load_object(
                contract_path,
                "checkpoint response contract at line {}".format(line_number),
                errors,
            )

    if contract is not None:
        _validate_contract_game(contract, config, profile_hash, errors)
        if contract.get("checkpoint_filename") != checkpoint_name:
            _error(errors, "response contract names the wrong checkpoint")
        if contract.get("checkpoint_sha256") != checkpoint_hash:
            _error(errors, "response contract checkpoint SHA256 mismatch")
        metadata = contract.get("metadata")
        if not isinstance(metadata, dict):
            _error(errors, "response contract metadata is not an object")
            metadata = {}
        if metadata.get("seed") != config.get("seed"):
            _error(errors, "response contract seed does not match config")
        if metadata.get("training_config_sha256") != config_hash:
            _error(errors, "response contract training config SHA256 mismatch")
        if metadata.get("response_training") != config.get("reinforce_protocol"):
            _error(errors, "response contract training protocol does not match config")
        for key, expected in (
            ("completed_updates", updates),
            ("follower_gradient_samples", follower_steps),
            ("executed_equivalent_steps", executed_steps),
        ):
            if metadata.get(key) != expected:
                _error(errors, "response contract {} counter disagrees".format(key))
        _validate_metric_alignment(row, metadata.get("evaluation"), errors)

    result.update({
        "completed_updates": updates,
        "follower_gradient_samples": follower_steps,
        "executed_equivalent_steps": executed_steps,
        "checkpoint_filename": checkpoint_name,
        "checkpoint_sha256": checkpoint_hash,
        "contract_filename": contract_name,
        "mean_regret": mean_regret,
        "max_regret": max_regret,
        "optimal_commitments": optimal,
        "commitments": commitments,
    })
    result["valid"] = not errors
    result["strict_pass"] = bool(
        result["valid"]
        and _finite_number(max_regret)
        and float(max_regret) <= regret_tolerance
        and _is_int(commitments, 1)
        and optimal == commitments
    )
    return result


def _read_checkpoint_ledger(
        path,
        run_dir,
        config,
        config_hash,
        profile_hash,
        by_filename,
        geometry,
        regret_tolerance,
        errors,
):
    rows = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    _error(errors, "checkpoint ledger has a blank line {}".format(
                        line_number
                    ))
                    continue
                try:
                    payload = json.loads(raw_line)
                except ValueError as exc:
                    row = {
                        "line": line_number,
                        "valid": False,
                        "strict_pass": False,
                        "errors": ["invalid JSON: {}".format(exc)],
                    }
                else:
                    row = _validate_checkpoint_row(
                        run_dir,
                        payload,
                        line_number,
                        config,
                        config_hash,
                        profile_hash,
                        by_filename,
                        geometry,
                        regret_tolerance,
                    )
                rows.append(row)
    except OSError as exc:
        _error(errors, "cannot read checkpoint ledger: {}".format(exc))
    if not rows:
        _error(errors, "checkpoint ledger has no rows")
    return rows


def _validate_sequence(rows, config, update_interval, errors):
    valid_counters = []
    seen_files = set()
    for row in rows:
        counters = tuple(row.get(key) for key in (
            "completed_updates",
            "follower_gradient_samples",
            "executed_equivalent_steps",
        ))
        if all(_is_int(value, 1) for value in counters):
            valid_counters.append((row["line"], counters))
        name = row.get("checkpoint_filename")
        if name is not None:
            if name in seen_files:
                _error(errors, "checkpoint ledger repeats filename {!r}".format(name))
            seen_files.add(name)

    previous = None
    for line, counters in valid_counters:
        if previous is not None and any(
            current <= old for current, old in zip(counters, previous)
        ):
            _error(errors, "checkpoint counters are not strictly increasing at line {}".format(
                line
            ))
        previous = counters

    if not rows:
        return
    last = rows[-1]
    if last.get("follower_gradient_samples") != config.get("timesteps"):
        _error(errors, "final checkpoint does not match configured timesteps")
    if last.get("checkpoint_filename") != "model.zip":
        _error(errors, "final checkpoint is not model.zip")
    if last.get("contract_filename") != "response_contract.json":
        _error(errors, "final checkpoint contract is not response_contract.json")
    if update_interval is not None:
        for row in rows[:-1]:
            updates = row.get("completed_updates")
            if _is_int(updates, 1) and updates % update_interval:
                _error(
                    errors,
                    "non-final checkpoint at line {} violates update cadence".format(
                        row["line"]
                    ),
                )


def _validate_final_evaluation(
        evaluation_path, config, final_row, errors
):
    evaluation = _load_object(evaluation_path, "final evaluation", errors)
    if evaluation is None:
        return
    if evaluation.get("schema_version") != RUN_SCHEMA_VERSION:
        _error(errors, "final evaluation schema_version is not 2")
    if evaluation.get("config") != config:
        _error(errors, "final evaluation config does not match run config")
    summary = evaluation.get("summary")
    if not isinstance(summary, dict):
        _error(errors, "final evaluation summary is not an object")
        return
    for key in (
        "mean_regret",
        "max_regret",
        "optimal_commitments",
        "commitments",
    ):
        expected = final_row.get(key)
        actual = summary.get(key)
        if key in ("optimal_commitments", "commitments"):
            aligned = actual == expected
        else:
            aligned = _same_number(actual, expected)
        if not aligned:
            _error(errors, "final evaluation {} disagrees with checkpoint".format(key))


def audit_run(run_dir, config, manifest, regret_tolerance):
    errors = []
    run_dir = Path(run_dir).resolve()
    result = {
        "run_dir": str(run_dir),
        "valid": False,
        "has_strict_pass": False,
        "strict_passes": [],
        "errors": errors,
        "checkpoints": [],
    }
    if config.get("schema_version") != RUN_SCHEMA_VERSION:
        _error(errors, "config schema_version is not 2")
    if config.get("stage") != "meta_follower":
        _error(errors, "config stage is not meta_follower")
    if manifest.get("schema_version") != RUN_SCHEMA_VERSION:
        _error(errors, "manifest schema_version is not 2")
    if manifest.get("status") != "completed":
        _error(errors, "manifest status is not completed")

    config_hash = sha256_json(config)
    if manifest.get("config_sha256") != config_hash:
        _error(errors, "manifest config_sha256 does not match config")
    profile_hash = _validate_profile(config, errors)
    artifacts, by_filename = _validate_artifacts(run_dir, manifest, errors)
    geometry, update_interval = _rollout_geometry(config, errors)
    if geometry is None:
        geometry = {}

    ledger_entry = artifacts.get("checkpoint_evaluations")
    ledger_path = ledger_entry.get("path") if ledger_entry else None
    if ledger_path is not None and ledger_path.is_file():
        rows = _read_checkpoint_ledger(
            ledger_path,
            run_dir,
            config,
            config_hash,
            profile_hash,
            by_filename,
            geometry,
            regret_tolerance,
            errors,
        )
    else:
        rows = []
        _error(errors, "checkpoint ledger is missing")
    result["checkpoints"] = rows
    _validate_sequence(rows, config, update_interval, errors)
    if rows:
        evaluation_entry = artifacts.get("evaluation")
        evaluation_path = evaluation_entry.get("path") if evaluation_entry else None
        if evaluation_path is not None and evaluation_path.is_file():
            _validate_final_evaluation(evaluation_path, config, rows[-1], errors)

    for row in rows:
        if row.get("errors"):
            _error(errors, "checkpoint ledger line {} is invalid".format(row["line"]))

    result.update({
        "profile_id": config.get("profile_id"),
        "algorithm": config.get("algorithm"),
        "seed": config.get("seed"),
        "learning_rate": config.get("learning_rate"),
        "config_sha256": config_hash,
        "checkpoint_count": len(rows),
    })
    result["strict_passes"] = [
        {
            "line": row["line"],
            "completed_updates": row.get("completed_updates"),
            "follower_gradient_samples": row.get("follower_gradient_samples"),
            "executed_equivalent_steps": row.get("executed_equivalent_steps"),
            "checkpoint_filename": row.get("checkpoint_filename"),
            "checkpoint_sha256": row.get("checkpoint_sha256"),
            "contract_filename": row.get("contract_filename"),
            "max_regret": row.get("max_regret"),
            "optimal_commitments": row.get("optimal_commitments"),
            "commitments": row.get("commitments"),
        }
        for row in rows if row.get("strict_pass")
    ]
    result["valid"] = not errors
    # A scientifically usable strict pass must belong to an entirely valid run.
    if not result["valid"]:
        result["strict_passes"] = []
        for row in rows:
            row["strict_pass"] = False
    result["has_strict_pass"] = bool(result["strict_passes"])
    return result


def _discover_completed_runs(root):
    runs = []
    errors = []
    for manifest_path in sorted(root.glob("**/run_manifest.json")):
        manifest = _load_object(manifest_path, str(manifest_path), errors)
        if manifest is None or manifest.get("status") != "completed":
            continue
        run_dir = manifest_path.parent
        config_path = run_dir / "config.json"
        config = _load_object(config_path, str(config_path), errors)
        if config is None:
            # A completed manifest without a readable config cannot be safely
            # classified, so retain it as an invalid candidate rather than
            # silently omitting evidence.
            runs.append((run_dir, {}, manifest))
            continue
        if config.get("stage") == "meta_follower":
            runs.append((run_dir, config, manifest))
    return runs, errors


def audit_root(root, regret_tolerance=DEFAULT_REGRET_TOLERANCE):
    root = Path(root).resolve()
    report = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "kind": "matrix_e1_checkpoint_audit",
        "generated_unix": time.time(),
        "root": str(root),
        "regret_tolerance": regret_tolerance,
        "status": "invalid",
        "errors": [],
        "runs": [],
    }
    if not _finite_number(regret_tolerance) or float(regret_tolerance) < 0.0:
        report["errors"].append(
            "regret tolerance must be finite and nonnegative"
        )
    elif not root.is_dir():
        report["errors"].append("root is not a directory")
    else:
        candidates, discovery_errors = _discover_completed_runs(root)
        report["errors"].extend(discovery_errors)
        for run_dir, config, manifest in candidates:
            report["runs"].append(
                audit_run(run_dir, config, manifest, float(regret_tolerance))
            )

    report["completed_run_count"] = len(report["runs"])
    report["valid_run_count"] = sum(
        int(run["valid"]) for run in report["runs"]
    )
    report["invalid_run_count"] = (
        report["completed_run_count"] - report["valid_run_count"]
    )
    report["checkpoint_count"] = sum(
        run.get("checkpoint_count", 0) for run in report["runs"]
    )
    report["strict_pass_count"] = sum(
        len(run.get("strict_passes", [])) for run in report["runs"]
    )
    report["strict_run_count"] = sum(
        int(run.get("has_strict_pass", False)) for run in report["runs"]
    )
    report["has_strict_pass"] = report["strict_pass_count"] > 0
    if report["errors"] or report["invalid_run_count"]:
        report["status"] = "invalid"
    elif not report["runs"]:
        report["status"] = "empty"
    else:
        report["status"] = "valid"
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        required=True,
        help="Root below which completed meta-follower manifests are scanned.",
    )
    parser.add_argument(
        "--regret-tolerance",
        "--tol",
        dest="regret_tolerance",
        type=float,
        default=DEFAULT_REGRET_TOLERANCE,
        help="Maximum regret for a strict pass (default: %(default)g).",
    )
    parser.add_argument(
        "--output",
        help="Report path (default: ROOT/{}).".format(DEFAULT_REPORT_NAME),
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    output = (
        Path(args.output).resolve()
        if args.output else root / DEFAULT_REPORT_NAME
    )
    report = audit_root(root, regret_tolerance=args.regret_tolerance)
    report["output"] = str(output)
    write_json_atomic(output, report)
    print(canonical_json(report))
    return 0 if report["status"] == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
