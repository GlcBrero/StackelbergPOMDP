"""Assemble evaluation tables and publish complete reports atomically."""

import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from stackelberg_pomdp.atari.training import write_csv, write_json


def _slug(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")


def default_run_name(report):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    role = report["environment_config"]["leader_role"]
    seed_start = report["screen"]["seed_start"]
    seed_end = report["screen"]["seed_end"]
    return f"e2_{role}_selection_seed{seed_start}-{seed_end}_{stamp}"


def _csv_value(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _csv_row(row):
    return {key: _csv_value(value) for key, value in row.items()}


def _event_rows(results):
    rows = []
    for result in results:
        for episode in result["episode_rows"]:
            for event in episode.get("events", ()):
                rows.append({
                    "phase": result["phase"],
                    "checkpoint_path": result["checkpoint_path"],
                    "checkpoint_sha256": result["checkpoint_sha256"],
                    "economic_intervention_id": result.get(
                        "economic_intervention", {}
                    ).get("intervention_id"),
                    "economic_intervention_sha256": result.get(
                        "economic_intervention", {}
                    ).get("manifest_sha256"),
                    "economic_commitment_override": result.get(
                        "economic_intervention", {}
                    ).get("economic_commitment"),
                    "response_sha256": episode["response_sha256"],
                    "environment_config_sha256": episode[
                        "environment_config_sha256"
                    ],
                    "e2_provenance_fingerprint": episode.get(
                        "e2_provenance_fingerprint"
                    ),
                    "evaluation_episode": episode["evaluation_episode"],
                    "evaluation_seed": episode["evaluation_seed"],
                    **event,
                })
    return rows


def _counterfactual_results(results):
    return [
        control
        for result in results
        for control in result.get("counterfactual_controls", ())
    ]


def _condition_rows(results):
    rows = []
    for factual in results:
        conditions = [factual, *factual.get("counterfactual_controls", ())]
        for result in conditions:
            intervention = result.get("economic_intervention", {})
            rows.append({
                "phase": result.get("phase"),
                "checkpoint_path": result.get("checkpoint_path"),
                "checkpoint_sha256": result.get("checkpoint_sha256"),
                "response_sha256": result.get(
                    "response_checkpoint_sha256"
                ),
                "environment_config_sha256": result.get(
                    "environment_config_sha256"
                ),
                "e2_provenance_fingerprint": result.get(
                    "e2_provenance_fingerprint"
                ),
                "intervention_id": intervention.get("intervention_id"),
                "intervention_sha256": intervention.get("manifest_sha256"),
                "economic_commitment": intervention.get(
                    "economic_commitment"
                ),
                "protocol_passed": result.get("protocol", {}).get("passed"),
                **(result.get("summary") or {}),
            })
    return rows


def _paired_gate_rows(results):
    return [
        row
        for result in results
        for row in result.get("economic_gate", {}).get("paired_rows", ())
    ]


def _artifact_paths(output_dir, stem, *, has_confirmation):
    paths = {
        "report_json": output_dir / f"{stem}.json",
        "ranking_csv": output_dir / f"{stem}.ranking.csv",
        "screen_episodes_csv": output_dir / f"{stem}.screen.episodes.csv",
        "screen_transitions_csv": output_dir / f"{stem}.screen.transitions.csv",
        "screen_decisions_csv": output_dir / f"{stem}.screen.decisions.csv",
        "screen_events_csv": output_dir / f"{stem}.screen.events.csv",
        "screen_counterfactual_conditions_csv": (
            output_dir / f"{stem}.screen.counterfactual.conditions.csv"
        ),
        "screen_counterfactual_paired_csv": (
            output_dir / f"{stem}.screen.counterfactual.paired.csv"
        ),
        "screen_counterfactual_episodes_csv": (
            output_dir / f"{stem}.screen.counterfactual.episodes.csv"
        ),
        "screen_counterfactual_transitions_csv": (
            output_dir / f"{stem}.screen.counterfactual.transitions.csv"
        ),
        "screen_counterfactual_decisions_csv": (
            output_dir / f"{stem}.screen.counterfactual.decisions.csv"
        ),
        "screen_counterfactual_events_csv": (
            output_dir / f"{stem}.screen.counterfactual.events.csv"
        ),
    }
    if has_confirmation:
        paths.update({
            "confirmation_episodes_csv": (
                output_dir / f"{stem}.confirmation.episodes.csv"
            ),
            "confirmation_transitions_csv": (
                output_dir / f"{stem}.confirmation.transitions.csv"
            ),
            "confirmation_decisions_csv": (
                output_dir / f"{stem}.confirmation.decisions.csv"
            ),
            "confirmation_events_csv": (
                output_dir / f"{stem}.confirmation.events.csv"
            ),
            "confirmation_counterfactual_conditions_csv": (
                output_dir
                / f"{stem}.confirmation.counterfactual.conditions.csv"
            ),
            "confirmation_counterfactual_paired_csv": (
                output_dir / f"{stem}.confirmation.counterfactual.paired.csv"
            ),
            "confirmation_counterfactual_episodes_csv": (
                output_dir / f"{stem}.confirmation.counterfactual.episodes.csv"
            ),
            "confirmation_counterfactual_transitions_csv": (
                output_dir
                / f"{stem}.confirmation.counterfactual.transitions.csv"
            ),
            "confirmation_counterfactual_decisions_csv": (
                output_dir
                / f"{stem}.confirmation.counterfactual.decisions.csv"
            ),
            "confirmation_counterfactual_events_csv": (
                output_dir / f"{stem}.confirmation.counterfactual.events.csv"
            ),
        })
    return paths


def _selection_artifact_tables(report):
    """Return nonempty CSV tables; absent counterfactuals stay absent."""

    screen_results = report["screen"]["checkpoint_results"]
    screen_controls = _counterfactual_results(screen_results)
    tables = {
        "ranking_csv": report["selection"]["ranking_rows"],
        "screen_episodes_csv": [
            row for result in screen_results for row in result["episode_rows"]
        ],
        "screen_transitions_csv": [
            row
            for result in screen_results
            for row in result["transition_rows"]
        ],
        "screen_decisions_csv": [
            row for result in screen_results for row in result["decision_rows"]
        ],
        "screen_events_csv": _event_rows(screen_results),
    }
    if screen_controls:
        tables.update({
            "screen_counterfactual_conditions_csv": _condition_rows(
                screen_results
            ),
            "screen_counterfactual_paired_csv": _paired_gate_rows(
                screen_results
            ),
            "screen_counterfactual_episodes_csv": [
                row
                for result in screen_controls
                for row in result["episode_rows"]
            ],
            "screen_counterfactual_transitions_csv": [
                row
                for result in screen_controls
                for row in result["transition_rows"]
            ],
            "screen_counterfactual_decisions_csv": [
                row
                for result in screen_controls
                for row in result["decision_rows"]
            ],
            "screen_counterfactual_events_csv": _event_rows(screen_controls),
        })

    confirmation_attempts = report.get("confirmation_attempts") or ()
    if confirmation_attempts:
        confirmation_results = [
            attempt["result"] for attempt in confirmation_attempts
        ]
        confirmation_controls = _counterfactual_results(
            confirmation_results
        )
        tables.update({
            "confirmation_episodes_csv": [
                row
                for result in confirmation_results
                for row in result["episode_rows"]
            ],
            "confirmation_transitions_csv": [
                row
                for result in confirmation_results
                for row in result["transition_rows"]
            ],
            "confirmation_decisions_csv": [
                row
                for result in confirmation_results
                for row in result["decision_rows"]
            ],
            "confirmation_events_csv": _event_rows(confirmation_results),
        })
        if confirmation_controls:
            tables.update({
                "confirmation_counterfactual_conditions_csv": _condition_rows(
                    confirmation_results
                ),
                "confirmation_counterfactual_paired_csv": _paired_gate_rows(
                    confirmation_results
                ),
                "confirmation_counterfactual_episodes_csv": [
                    row
                    for result in confirmation_controls
                    for row in result["episode_rows"]
                ],
                "confirmation_counterfactual_transitions_csv": [
                    row
                    for result in confirmation_controls
                    for row in result["transition_rows"]
                ],
                "confirmation_counterfactual_decisions_csv": [
                    row
                    for result in confirmation_controls
                    for row in result["decision_rows"]
                ],
                "confirmation_counterfactual_events_csv": _event_rows(
                    confirmation_controls
                ),
            })
    return {
        key: [_csv_row(row) for row in rows]
        for key, rows in tables.items()
        if rows
    }


def _artifact_file_identity(path):
    try:
        status = os.lstat(path)
    except (FileNotFoundError, OSError):
        return None
    return (
        status.st_dev,
        status.st_ino,
        status.st_size,
        status.st_mtime_ns,
    )


def _same_artifact_file(path, identity):
    return _artifact_file_identity(path) == identity


def _publish_artifact_set(staged_paths, final_paths, *, lock_path):
    """Hard-link a staged set without overwriting, rolling back on failure.

    The per-stem lock serializes cooperating writers.  Hard links make each
    file publication atomic and inherently no-overwrite even if an external
    writer races after the collision check.  The report JSON is linked last and
    therefore serves as the completion marker for the whole artifact set.
    """

    lock_acquired = False
    attempted = []
    try:
        try:
            os.mkdir(lock_path)
        except FileExistsError as error:
            raise FileExistsError(
                "E2 artifact publication is already in progress for "
                f"{lock_path.name}"
            ) from error
        lock_acquired = True

        collisions = [
            str(path)
            for path in final_paths.values()
            if os.path.lexists(path)
        ]
        if collisions:
            raise FileExistsError(
                "refusing to overwrite existing E2 selection artifacts: "
                + ", ".join(collisions)
            )

        publication_order = [
            key for key in final_paths if key != "report_json"
        ] + ["report_json"]
        for key in publication_order:
            staged = staged_paths[key]
            identity = _artifact_file_identity(staged)
            if identity is None:
                raise RuntimeError(f"staged E2 artifact disappeared: {staged}")
            final = final_paths[key]
            os.link(staged, final, follow_symlinks=False)
            attempted.append((final, identity))

        mismatches = [
            str(path)
            for path, identity in attempted
            if not _same_artifact_file(path, identity)
        ]
        if mismatches:
            raise RuntimeError(
                "published E2 artifacts changed during publication: "
                + ", ".join(mismatches)
            )
    except BaseException:
        for path, identity in reversed(attempted):
            if _same_artifact_file(path, identity):
                try:
                    path.unlink()
                except (FileNotFoundError, OSError):
                    pass
        raise
    finally:
        if lock_acquired:
            try:
                Path(lock_path).rmdir()
            except (FileNotFoundError, OSError):
                pass


def write_selection_artifacts(report, *, output_dir, run_name=None):
    """Transactionally stage and publish one collision-safe artifact set."""

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _slug(run_name or default_run_name(report))
    if not stem:
        raise ValueError("selection run name cannot be empty")
    all_paths = _artifact_paths(
        output_dir,
        stem,
        has_confirmation=bool(report.get("confirmation_attempts")),
    )
    tables = _selection_artifact_tables(report)
    final_paths = {
        key: all_paths[key]
        for key in tables
    }
    final_paths["report_json"] = all_paths["report_json"]
    artifacts = {key: str(path) for key, path in final_paths.items()}
    result = {**report, "artifacts": artifacts}

    stage_dir = Path(tempfile.mkdtemp(
        prefix=f".{stem}.stage-", dir=output_dir
    ))
    try:
        staged_paths = {
            key: stage_dir / path.name for key, path in final_paths.items()
        }
        for key, rows in tables.items():
            write_csv(staged_paths[key], rows)
        write_json(staged_paths["report_json"], result)
        missing = [
            str(path) for path in staged_paths.values() if not path.is_file()
        ]
        if missing:
            raise RuntimeError(
                "E2 artifact staging did not create every declared file: "
                + ", ".join(missing)
            )
        _publish_artifact_set(
            staged_paths,
            final_paths,
            lock_path=output_dir / f".{stem}.publish.lock",
        )
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)
    return result
