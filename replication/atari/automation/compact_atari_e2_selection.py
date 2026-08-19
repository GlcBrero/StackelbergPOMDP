"""Create a small, portable view of a full Atari E2 selection report.

The selector records transition-level diagnostics and can therefore emit a
large JSON file.  Keep that full report as an archive and use this module to
extract the policy-level quantities needed for multi-seed analysis.  The
source and optional archive are content-addressed so compaction does not
weaken provenance.  Absolute operational paths are reduced to basenames; the
immutable checkpoint, ROM, provenance, source-report, and archive hashes are
retained.  Because ``rom_path`` is part of the evaluator's environment hash,
the portable environment hash and every copy of it are recomputed together.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


SCHEMA = "stackelberg_pomdp.atari.e2_multiseed_compact.v2"
LEGACY_SCHEMAS = {"stackelberg_pomdp.atari.e2_multiseed_compact.v1"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value):
    """Match the evaluator's canonical JSON hash exactly."""

    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _portable_value(value):
    """Copy a JSON value while replacing absolute paths with basenames.

    Compact reports are publication artifacts, not resumable run manifests.
    Their cryptographic hashes identify the immutable inputs, while basenames
    are sufficient to identify the corresponding files without leaking a
    workstation or cluster account path.
    """

    if isinstance(value, dict):
        return {key: _portable_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_value(item) for item in value]
    if isinstance(value, tuple):
        return [_portable_value(item) for item in value]
    if isinstance(value, str) and Path(value).is_absolute():
        return Path(value).name
    return value


def _replace_environment_hash(value, *, old_hash, new_hash):
    """Update copies of the E2 environment hash without touching other hashes."""

    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key == "environment_config_sha256" and item == old_hash:
                result[key] = new_hash
            else:
                result[key] = _replace_environment_hash(
                    item, old_hash=old_hash, new_hash=new_hash
                )
        return result
    if isinstance(value, list):
        return [
            _replace_environment_hash(
                item, old_hash=old_hash, new_hash=new_hash
            )
            for item in value
        ]
    return value


def make_portable(compact):
    """Return a path-sanitized compact report with coherent config hashes."""

    environment = compact.get("environment_config")
    if not isinstance(environment, dict):
        raise ValueError("selection report is missing environment_config")
    old_hash = compact.get("environment_config_sha256")
    if old_hash != canonical_sha256(environment):
        raise ValueError(
            "selection report environment_config_sha256 does not match its "
            "environment_config"
        )
    portable = _portable_value(compact)
    new_hash = canonical_sha256(portable["environment_config"])
    return _replace_environment_hash(
        portable, old_hash=old_hash, new_hash=new_hash
    )


def normalize_compact_report(compact):
    """Upgrade an existing v1/v2 compact report to portable schema v2.

    This supports sanitizing retained reports without the large full-report
    archives and, importantly, without rerunning either training or Atari
    evaluation.  Content hashes for those unavailable source artifacts remain
    unchanged.
    """

    if not isinstance(compact, dict):
        raise TypeError("compact report must be a JSON object")
    schema = compact.get("schema")
    if schema not in LEGACY_SCHEMAS | {SCHEMA}:
        raise ValueError(f"unsupported compact-report schema: {schema!r}")
    upgraded = dict(compact)
    upgraded["schema"] = SCHEMA
    return make_portable(upgraded)


def compact_gate(gate):
    gate = gate or {}
    return {
        "passed": gate.get("passed"),
        "checks": gate.get("checks"),
        "errors": gate.get("errors"),
        "condition_summaries": gate.get("condition_summaries"),
    }


def compact_result(result):
    result = result or {}
    keys = (
        "checkpoint_id",
        "checkpoint_path",
        "checkpoint_sha256",
        "response_checkpoint_sha256",
        "environment_config_sha256",
        "e2_provenance_fingerprint",
        "e2_provenance_manifest",
        "training_total_timesteps",
        "economic_role",
        "economic_input_mode",
        "actor_loss_mode",
        "target_kl",
        "economic_head_initialization",
        "phase",
        "seed_start",
        "seed_end",
        "summary",
        "protocol",
    )
    compact = {key: result.get(key) for key in keys}
    compact["economic_gate"] = compact_gate(result.get("economic_gate"))
    return compact


def compact_report(
        report,
        *,
        report_path,
        report_sha256,
        archive_path=None,
        archive_sha256=None,
):
    screen = report.get("screen") or {}
    confirmation = report.get("confirmation")
    compact_confirmation = None
    if confirmation is not None:
        compact_confirmation = {
            "screen_rank": confirmation.get("screen_rank"),
            "episodes": confirmation.get("episodes"),
            "seed_start": confirmation.get("seed_start"),
            "seed_end": confirmation.get("seed_end"),
            "disjoint_from_screen": confirmation.get("disjoint_from_screen"),
            "checks": confirmation.get("checks"),
            "result": compact_result(confirmation.get("result")),
        }
    selection = report.get("selection") or {}
    compact = {
        "schema": SCHEMA,
        "full_report": {
            "path_at_selection": str(Path(report_path).resolve()),
            "sha256": report_sha256,
            "archive_path": (
                None
                if archive_path is None
                else str(Path(archive_path).resolve())
            ),
            "archive_sha256": archive_sha256,
        },
        "passed": report.get("passed"),
        "evaluator": report.get("evaluator"),
        "created_at_utc": report.get("created_at_utc"),
        "environment_config": report.get("environment_config"),
        "environment_config_sha256": report.get(
            "environment_config_sha256"
        ),
        "response_checkpoint_sha256": report.get(
            "response_checkpoint_sha256"
        ),
        "response_metadata": report.get("response_metadata"),
        "screen": {
            "episodes_per_checkpoint": screen.get("episodes_per_checkpoint"),
            "seed_start": screen.get("seed_start"),
            "seed_end": screen.get("seed_end"),
            "common_seed_schedule_check": screen.get(
                "common_seed_schedule_check"
            ),
            "checkpoint_results": [
                compact_result(result)
                for result in screen.get("checkpoint_results", ())
            ],
        },
        "selection": {
            "selection_rule": selection.get("selection_rule"),
            "ranking_rows": selection.get("ranking_rows"),
            "screen_selected_checkpoint_sha256": selection.get(
                "screen_selected_checkpoint_sha256"
            ),
            "selected_checkpoint_sha256": selection.get(
                "selected_checkpoint_sha256"
            ),
        },
        "selected_alias": report.get("selected_alias"),
        "confirmation": compact_confirmation,
    }
    return make_portable(compact)


def write_json_atomic(path, value):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_sha256_sidecar(path, digest=None):
    """Write a standard SHA-256 sidecar containing only the file basename."""

    path = Path(path).expanduser().resolve()
    digest = sha256_file(path) if digest is None else str(digest)
    sidecar = Path(f"{path}.sha256")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{sidecar.name}.", suffix=".tmp", dir=sidecar.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(f"{digest}  {path.name}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, sidecar)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return sidecar


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--report", help="full evaluator report to compact"
    )
    source.add_argument(
        "--compact-input",
        help="existing v1/v2 compact report to sanitize or upgrade",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--archive")
    parser.add_argument(
        "--sha256-sidecars",
        action="store_true",
        help="write basename-only .sha256 sidecars for output and archive",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    archive_hash = None
    if args.compact_input is not None:
        if args.archive is not None:
            raise ValueError(
                "--archive is valid only when compacting a full --report"
            )
        compact_path = Path(args.compact_input).expanduser().resolve()
        with compact_path.open("r", encoding="utf-8") as handle:
            compact = normalize_compact_report(json.load(handle))
    else:
        report_path = Path(args.report).expanduser().resolve()
        report_hash = sha256_file(report_path)
        with report_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        if args.archive is not None:
            archive = Path(args.archive).expanduser().resolve()
            if not archive.is_file():
                raise FileNotFoundError(
                    f"selection archive does not exist: {archive}"
                )
            archive_hash = sha256_file(archive)
        compact = compact_report(
            report,
            report_path=report_path,
            report_sha256=report_hash,
            archive_path=args.archive,
            archive_sha256=archive_hash,
        )
    write_json_atomic(args.output, compact)
    if args.sha256_sidecars:
        write_sha256_sidecar(args.output)
        if args.archive is not None:
            write_sha256_sidecar(args.archive, archive_hash)
    print({
        "output": str(Path(args.output).expanduser().resolve()),
        "passed": compact["passed"],
        "selected_checkpoint_sha256": compact["selection"][
            "selected_checkpoint_sha256"
        ],
    }, flush=True)


if __name__ == "__main__":
    main()
