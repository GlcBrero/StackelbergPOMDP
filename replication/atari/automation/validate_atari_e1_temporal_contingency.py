"""Validate and pin the activation-gated Atari E1 temporal contingency."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace

from replication.atari import evaluate_atari_meta_response_sb3 as evaluator
from replication.atari import train_atari_meta_response_sb3 as trainer
from replication.atari.sb3_common import PHASE_BALANCED_ACTOR_LOSS_MODE
from stackelberg_pomdp.atari.e1_sampling import (
    TEMPORAL_MIX_E1_SAMPLER,
    UNIFORM_E1_SAMPLER,
    e1_sampler_provenance,
)
from stackelberg_pomdp.atari.stackpomdp_env import BUYER


SCHEMA_VERSION = 1
KIND = "atari_e1_buyer_temporal_contingency_activation"
FAMILY_KIND = "atari_e1_buyer_temporal_contingency_family"
SELECTION_GATE_KIND = "atari_e1_buyer_temporal_contingency_gate"
CANONICAL_ROM_SHA256 = (
    "7224b17462b992d67f4e06a3c85f269c9822b06df6015bf038b55f384ced0301"
)
ADDITIONAL_TIMESTEPS = 2_000_800
CHECKPOINT_INTERVAL = 400_160
NUM_CANDIDATES = 6
SCREEN_SEED_START = 6_000_001
CONFIRMATION_SEED_START = 6_100_001
FIXED_SEED_START = 6_200_001
TIMING_SEED_START = 6_300_001


class ContingencyInactive(RuntimeError):
    """The preregistered uniform-family failure condition is not satisfied."""


def sha256_file(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"expected a regular immutable file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _same_file(first, second):
    try:
        return os.path.samefile(first, second)
    except (FileNotFoundError, OSError):
        return Path(first).expanduser().resolve() == Path(second).expanduser().resolve()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _git_revision(code_root):
    root = Path(code_root).expanduser().resolve()
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _require(len(result) == 40, "code revision is not a full Git SHA")
    return result


def atomic_write_json(path, value):
    """Publish a new JSON artifact atomically without overwrite semantics."""

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary_raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_raw)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite immutable artifact: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _uniform_sampler_contract(report):
    family = report["training_family"]
    current = family.get("common_sampler_provenance")
    history = family.get("common_sampler_history")
    if current is None and history is None:
        return {"mode": UNIFORM_E1_SAMPLER, "legacy_inferred": True}
    expected = e1_sampler_provenance(
        UNIFORM_E1_SAMPLER, gameplay_horizon=200, event_tail_steps=0
    )
    _require(current == expected, "uniform report has a nonuniform sampler")
    _require(isinstance(history, list) and len(history) == 1, "uniform report has mixed sampler history")
    _require(history[0].get("start_total_timesteps") == 0, "uniform sampler history must start at zero")
    _require(history[0].get("sampler") == expected, "uniform sampler history is not canonical")
    return {"mode": UNIFORM_E1_SAMPLER, "legacy_inferred": False}


def validate_failed_uniform_report(path, *, actor_loss_mode, rom_sha256):
    """Validate failure of the final uniform family's screened rank one.

    Legacy selectors could continue down the ranking after rank one failed.
    Such a report may say ``passed=true`` because a lower-ranked checkpoint
    passed confirmation.  It is still a failure under the preregistered
    screen-winner-only rule used by the contingency and all downstream gates.
    """

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ContingencyInactive(f"final uniform report is not available: {path}")
    report = load_json(path)
    _require(type(report.get("passed")) is bool, "uniform report has no Boolean outcome")
    _require(report.get("evaluator") == evaluator.EVALUATOR_NAME, "unknown E1 evaluator")
    _require(report.get("role") == BUYER, "uniform report is not a buyer report")
    protocol = report.get("protocol", {})
    _require(protocol.get("screen_episodes") == 20, "uniform screen is not common-20")
    _require(protocol.get("confirmation_episodes") == 100, "uniform confirmation is not fresh-100")
    _require(protocol.get("outer_transitions") == 205, "uniform report has a noncanonical horizon")
    environment = report.get("environment", {})
    _require(environment.get("gameplay_horizon") == 200, "uniform report has wrong gameplay horizon")
    _require(environment.get("event_tail_steps") == 0, "uniform report has an event tail")
    _require(environment.get("rom_sha256") == rom_sha256, "uniform report used different ROM bytes")
    family = report.get("training_family", {})
    config = family.get("common_training_config", {})
    _require(config.get("actor_loss_mode") == actor_loss_mode, "uniform report actor-loss mode mismatch")
    _require(config.get("n_steps") == 205 and config.get("batch_size") == 820, "uniform report has wrong rollout geometry")
    _require(config.get("gamma") == 1.0 and config.get("gae_lambda") == 1.0, "uniform report discounts rewards")
    sampler = _uniform_sampler_contract(report)

    screen = report.get("screen", {})
    common = screen.get("common_pairing", {})
    results = screen.get("results", [])
    ranking = report.get("ranking", [])
    _require(common.get("passed") is True, "uniform report lacks a valid common screen")
    _require(common.get("candidates_checked") == NUM_CANDIDATES, "uniform screen is not all-six")
    _require(len(common.get("seed_context_pairs", [])) == 20, "uniform common screen lacks 20 seed/context pairs")
    _require(len(results) == NUM_CANDIDATES and len(ranking) == NUM_CANDIDATES, "uniform report is not an all-six family")
    screen_hashes = []
    for result in results:
        metadata = result.get("metadata", {})
        digest = metadata.get("sha256")
        _require(isinstance(digest, str) and len(digest) == 64, "screen candidate lacks SHA-256")
        _require(result.get("protocol", {}).get("passed") is True, "uniform screen candidate failed mechanics")
        _require(len(result.get("episode_rows", [])) == 20, "uniform candidate was not screened for 20 episodes")
        candidate = Path(metadata.get("path", "")).expanduser().resolve()
        _require(sha256_file(candidate) == digest, f"uniform candidate bytes changed: {candidate}")
        screen_hashes.append(digest)
    _require(len(set(screen_hashes)) == NUM_CANDIDATES, "uniform candidates are not byte-distinct")
    immutable = report.get("immutable_evaluation", {})
    _require(immutable.get("candidate_sha256") == screen_hashes, "uniform immutable candidate order changed")
    ranked_hashes = [row.get("checkpoint_sha256") for row in ranking]
    _require(set(ranked_hashes) == set(screen_hashes), "uniform ranking does not cover the common screen")
    _require([row.get("rank") for row in ranking] == list(range(1, 7)), "uniform ranking is malformed")
    rank_one = ranking[0]
    _require(rank_one.get("mechanically_valid") is True, "uniform rank-1 checkpoint is mechanically invalid")
    attempts = report.get("confirmation_attempts", [])
    _require(bool(attempts), "failed uniform report has no fresh-seed confirmation")
    attempt_hashes = [row.get("metadata", {}).get("sha256") for row in attempts]
    _require(
        attempt_hashes == ranked_hashes[:len(attempt_hashes)],
        "uniform confirmations are not a prefix of the screen ranking",
    )
    rank_one_passed = attempts[0].get("behavioral_gate", {}).get("passed") is True
    if rank_one_passed:
        raise ContingencyInactive(
            f"contingency is inactive because the uniform screen winner passed: {path}"
        )
    selected_alias = report.get("selected_alias")
    if report["passed"]:
        _require(isinstance(selected_alias, dict), "passing legacy uniform report has no alias")
        selected_hash = selected_alias.get("sha256")
        _require(
            selected_hash in ranked_hashes[1:],
            "legacy fallback did not select a lower-ranked candidate",
        )
        _require(
            attempts[-1].get("behavioral_gate", {}).get("passed") is True
            and attempt_hashes[-1] == selected_hash,
            "legacy uniform fallback alias is not its passing confirmation",
        )
        alias_path = Path(
            selected_alias.get("pinned_path", "")
        ).expanduser().resolve()
        _require(
            sha256_file(alias_path) == selected_hash,
            "legacy uniform fallback alias bytes changed",
        )
    else:
        _require(selected_alias is None, "failed uniform report retained an alias")
        _require(
            all(
                row.get("behavioral_gate", {}).get("passed") is False
                for row in attempts
            ),
            "failed uniform report contains a passing confirmation",
        )
    artifacts = report.get("artifacts", {})
    _require(_same_file(artifacts.get("json", ""), path), "uniform report does not identify itself")
    e0b = report.get("e0b_source", {})
    e0b_path = Path(e0b.get("path", "")).expanduser().resolve()
    _require(sha256_file(e0b_path) == e0b.get("sha256"), "uniform report E0b bytes changed")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "actor_loss_mode": actor_loss_mode,
        "candidate_count": NUM_CANDIDATES,
        "sampler": sampler,
        "reported_passed": bool(report["passed"]),
        "strict_screen_winner_passed": False,
        "legacy_fallback_selected": bool(report["passed"]),
        "e0b": {"path": str(e0b_path), "sha256": e0b["sha256"]},
        "rank_one": {
            "path": str(Path(rank_one["checkpoint_path"]).expanduser().resolve()),
            "sha256": rank_one["checkpoint_sha256"],
            "training_total_timesteps": int(rank_one["training_timesteps"]),
            "rank": 1,
            "mechanically_valid": True,
        },
    }


def expected_candidates(base, resume_total_timesteps):
    base = Path(base).expanduser().resolve()
    stem = base.with_suffix("")
    steps = [
        int(resume_total_timesteps) + CHECKPOINT_INTERVAL * index
        for index in range(1, 6)
    ]
    return [str(stem.with_name(f"{stem.name}_step{step}.zip")) for step in steps] + [str(base)]


def build_activation(args):
    rom = Path(args.rom).expanduser().resolve()
    rom_sha256 = sha256_file(rom)
    _require(rom_sha256 == CANONICAL_ROM_SHA256, "active ROM is not canonical Space Invaders")
    standard = validate_failed_uniform_report(
        args.standard_report, actor_loss_mode="standard", rom_sha256=rom_sha256
    )
    balanced = validate_failed_uniform_report(
        args.balanced_report,
        actor_loss_mode=PHASE_BALANCED_ACTOR_LOSS_MODE,
        rom_sha256=rom_sha256,
    )
    _require(standard["e0b"]["sha256"] == balanced["e0b"]["sha256"], "uniform families use different E0b bytes")
    source = balanced["rank_one"]
    _require(source["training_total_timesteps"] % CHECKPOINT_INTERVAL == 0, "balanced rank-1 timestep is not aligned to 400160")
    _require(sha256_file(source["path"]) == source["sha256"], "balanced rank-1 bytes changed")
    base = Path(args.base).expanduser().resolve()
    candidates = expected_candidates(base, source["training_total_timesteps"])
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "activation_condition": {
            "standard_uniform_all_six_failed": True,
            "balanced_uniform_all_six_failed": True,
        },
        "uniform_failure_reports": {"standard": standard, "balanced": balanced},
        "resume_source": source,
        "e0b_source": balanced["e0b"],
        "rom": {"path": str(rom), "sha256": rom_sha256},
        "code_revision": _git_revision(args.code_root),
        "protocol": {
            "role": BUYER,
            "actor_loss_mode": PHASE_BALANCED_ACTOR_LOSS_MODE,
            "training_sampler": e1_sampler_provenance(
                TEMPORAL_MIX_E1_SAMPLER,
                gameplay_horizon=200,
                event_tail_steps=0,
            ),
            "evaluation_sampler": e1_sampler_provenance(
                UNIFORM_E1_SAMPLER,
                gameplay_horizon=200,
                event_tail_steps=0,
            ),
            "additional_timesteps": ADDITIONAL_TIMESTEPS,
            "checkpoint_interval": CHECKPOINT_INTERVAL,
            "expected_total_timesteps": source["training_total_timesteps"] + ADDITIONAL_TIMESTEPS,
            "candidate_paths": candidates,
            "candidate_count": NUM_CANDIDATES,
            "screen": {"episodes": 20, "seed_start": SCREEN_SEED_START},
            "confirmation": {"episodes": 100, "seed_start": CONFIRMATION_SEED_START, "fallback_allowed": False},
            "fixed_seed_start": FIXED_SEED_START,
            "timing_seed_start": TIMING_SEED_START,
        },
    }


def validate_activation(path, *, rom=None):
    path = Path(path).expanduser().resolve()
    value = load_json(path)
    _require(value.get("schema_version") == SCHEMA_VERSION and value.get("kind") == KIND, "unknown contingency activation manifest")
    _require(value.get("activation_condition") == {
        "standard_uniform_all_six_failed": True,
        "balanced_uniform_all_six_failed": True,
    }, "contingency activation condition is incomplete")
    for item in value["uniform_failure_reports"].values():
        _require(sha256_file(item["path"]) == item["sha256"], "uniform failure report changed after activation")
    _require(sha256_file(value["resume_source"]["path"]) == value["resume_source"]["sha256"], "resume source changed after activation")
    _require(sha256_file(value["e0b_source"]["path"]) == value["e0b_source"]["sha256"], "E0b changed after activation")
    rom_path = value["rom"]["path"] if rom is None else rom
    _require(sha256_file(rom_path) == value["rom"]["sha256"] == CANONICAL_ROM_SHA256, "ROM changed after activation")
    protocol = value["protocol"]
    _require(protocol["additional_timesteps"] == ADDITIONAL_TIMESTEPS, "contingency budget changed")
    _require(protocol["checkpoint_interval"] == CHECKPOINT_INTERVAL, "contingency checkpoint interval changed")
    _require(protocol["candidate_paths"] == expected_candidates(protocol["candidate_paths"][-1], value["resume_source"]["training_total_timesteps"]), "contingency candidate schedule changed")
    _require(protocol["training_sampler"]["mode"] == TEMPORAL_MIX_E1_SAMPLER, "contingency training sampler changed")
    _require(protocol["evaluation_sampler"]["mode"] == UNIFORM_E1_SAMPLER, "contingency evaluation sampler changed")
    return value


def write_or_validate_activation(args):
    expected = build_activation(args)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        existing = validate_activation(output, rom=args.rom)
        for key in (
            "activation_condition", "uniform_failure_reports", "resume_source",
            "e0b_source", "rom", "code_revision", "protocol",
        ):
            _require(existing.get(key) == expected.get(key), f"existing activation manifest differs in {key}")
        return existing
    atomic_write_json(output, expected)
    return validate_activation(output, rom=args.rom)


def _validate_temporal_history(metadata, activation):
    history = metadata["atari_e1_sampler_history"]
    _require(len(history) == 2, "temporal checkpoint must have exactly uniform and temporal stages")
    _require(history[0]["sampler"]["mode"] == UNIFORM_E1_SAMPLER, "temporal lineage does not start uniform")
    temporal = history[1]
    _require(temporal["sampler"] == activation["protocol"]["training_sampler"], "temporal lineage uses another sampler")
    _require(temporal["start_total_timesteps"] == activation["resume_source"]["training_total_timesteps"], "temporal stage starts at wrong timestep")
    sources = temporal.get("resume_sources", [])
    _require(len(sources) == 1, "temporal stage must have one byte-bound resume source")
    source = sources[0]
    for key in ("path", "sha256"):
        _require(source.get(key) == activation["resume_source"][key], f"temporal resume source {key} mismatch")
    _require(source.get("resume_total_timesteps") == activation["resume_source"]["training_total_timesteps"], "temporal resume timestep mismatch")


def build_training_family(args):
    activation_path = Path(args.activation).expanduser().resolve()
    activation = validate_activation(activation_path)
    preflight = load_json(args.preflight)
    _require(preflight.get("passed") is True, "temporal preflight did not pass")
    _require(preflight.get("activation_sha256") == sha256_file(activation_path), "preflight belongs to another activation")
    candidates = [Path(path).expanduser().resolve() for path in args.candidate]
    expected = [Path(path).resolve() for path in activation["protocol"]["candidate_paths"]]
    _require(candidates == expected, "temporal family paths differ from preregistration")
    expected_steps = [
        activation["resume_source"]["training_total_timesteps"] + CHECKPOINT_INTERVAL * index
        for index in range(1, 6)
    ] + [activation["protocol"]["expected_total_timesteps"]]
    metadata = []
    hashes = []
    for path, expected_step in zip(candidates, expected_steps):
        model, item = evaluator.load_candidate(
            path,
            role=BUYER,
            e0b_sha256=activation["e0b_source"]["sha256"],
            device=args.device,
        )
        del model
        _require(item["training_timesteps"] == expected_step, "temporal candidate timestep mismatch")
        _require(item["training_config"]["actor_loss_mode"] == PHASE_BALANCED_ACTOR_LOSS_MODE, "temporal candidate is not phase-balanced")
        _require(item["atari_e1_sampler_provenance"] == activation["protocol"]["training_sampler"], "temporal candidate sampler mismatch")
        _validate_temporal_history(item, activation)
        metadata.append(item)
        hashes.append(item["sha256"])
    _require(len(set(hashes)) == NUM_CANDIDATES, "temporal candidates are not byte-distinct")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": FAMILY_KIND,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "activation": {"path": str(activation_path), "sha256": sha256_file(activation_path)},
        "preflight": {"path": str(Path(args.preflight).resolve()), "sha256": sha256_file(args.preflight)},
        "candidate_metadata": metadata,
        "candidate_sha256": hashes,
        "passed": True,
    }


def validate_training_family(path):
    value = load_json(path)
    _require(value.get("schema_version") == SCHEMA_VERSION and value.get("kind") == FAMILY_KIND, "unknown temporal family manifest")
    _require(value.get("passed") is True, "temporal family validation failed")
    _require(len(value.get("candidate_metadata", [])) == NUM_CANDIDATES, "temporal family is not all-six")
    _require(len(set(value.get("candidate_sha256", []))) == NUM_CANDIDATES, "temporal family hashes are not distinct")
    _require(sha256_file(value["activation"]["path"]) == value["activation"]["sha256"], "temporal family activation changed")
    _require(sha256_file(value["preflight"]["path"]) == value["preflight"]["sha256"], "temporal family preflight changed")
    for item in value["candidate_metadata"]:
        _require(sha256_file(item["path"]) == item["sha256"], "temporal family checkpoint changed")
    return value


def validate_selection(args):
    family = validate_training_family(args.family)
    report = load_json(args.report)
    _require(report.get("evaluator") == evaluator.EVALUATOR_NAME, "unknown temporal selector")
    _require(report.get("role") == BUYER, "temporal selector role mismatch")
    _require(report.get("protocol", {}).get("confirmation_policy") == "screen_winner_only_no_fallback", "temporal selector allows confirmation fallback")
    _require(report.get("selection", {}).get("fallback_allowed") is False, "temporal selection fallback flag is wrong")
    screen = report.get("screen", {})
    _require(len(screen.get("results", [])) == NUM_CANDIDATES, "temporal selector is not all-six")
    _require(report.get("protocol", {}).get("screen_episodes") == 20, "temporal selector screen is not common-20")
    _require(report.get("protocol", {}).get("screen_seed_start") == SCREEN_SEED_START, "temporal selector reports another screen schedule")
    _require(report.get("protocol", {}).get("confirmation_episodes") == 100, "temporal selector confirmation is not fresh-100")
    _require(report.get("protocol", {}).get("confirmation_seed_start") == CONFIRMATION_SEED_START, "temporal selector reports another confirmation schedule")
    _require(report.get("protocol", {}).get("fixed_context_seed_start") == FIXED_SEED_START, "temporal selector reports another fixed-grid schedule")
    _require(report.get("protocol", {}).get("paired_timing_seed_start") == TIMING_SEED_START, "temporal selector reports another timing schedule")
    common = screen.get("common_pairing", {})
    _require(common.get("passed") is True and common.get("candidates_checked") == NUM_CANDIDATES, "temporal common screen failed")
    screen_seeds = [
        row.get("evaluation_seed")
        for row in common.get("seed_context_pairs", [])
    ]
    _require(screen_seeds == list(range(SCREEN_SEED_START, SCREEN_SEED_START + 20)), "temporal common screen uses another seed schedule")
    attempts = report.get("confirmation_attempts", [])
    _require(len(attempts) == 1, "temporal selector must confirm only the screen winner")
    confirmation_seeds = [
        row.get("evaluation_seed")
        for row in attempts[0].get("random", {}).get("episode_rows", [])
    ]
    _require(confirmation_seeds == list(range(CONFIRMATION_SEED_START, CONFIRMATION_SEED_START + 100)), "temporal confirmation is not the preregistered fresh-100 schedule")
    screen_hashes = [row["metadata"]["sha256"] for row in screen["results"]]
    _require(screen_hashes == family["candidate_sha256"], "temporal selector screened another family")
    _require(report.get("immutable_evaluation", {}).get("candidate_sha256") == screen_hashes, "temporal immutable screen hashes changed")
    expected_sampler = family["candidate_metadata"][0]["atari_e1_sampler_provenance"]
    expected_history = family["candidate_metadata"][0]["atari_e1_sampler_history"]
    training_family = report.get("training_family", {})
    _require(training_family.get("common_sampler_provenance") == expected_sampler, "temporal selector reports another training sampler")
    _require(training_family.get("common_sampler_history") == expected_history, "temporal selector reports another sampler lineage")
    winner = report.get("selection", {}).get("screen_selected_checkpoint_sha256")
    _require(winner == report["ranking"][0]["checkpoint_sha256"], "temporal screen winner does not match rank one")
    _require(attempts[0]["metadata"]["sha256"] == winner, "temporal selector confirmed a nonwinner")
    selected = Path(args.selected).expanduser().resolve()
    if report.get("passed"):
        _require(report.get("selection", {}).get("selected_checkpoint_sha256") == winner, "temporal selected hash is not the screen winner")
        _require(selected.is_file() and sha256_file(selected) == winner, "temporal selected alias is missing or changed")
        _require(
            _same_file(
                report.get("selected_alias", {}).get("pinned_path", ""),
                selected,
            ),
            "temporal report identifies another selected alias",
        )
    else:
        _require(report.get("selected_alias") is None and not selected.exists(), "failed temporal selector retained an alias")
        _require(attempts[0].get("behavioral_gate", {}).get("passed") is False, "failed temporal selector contains a passing gate")
    _require(_same_file(report.get("artifacts", {}).get("json", ""), args.report), "temporal selector report does not identify itself")
    return report


def build_selection_gate(args, report):
    """Bind a passing temporal selection to all preregistered source bytes."""

    _require(report.get("passed") is True, "only a passing selection is a gate")
    family_path = Path(args.family).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    selected = Path(args.selected).expanduser().resolve()
    family = validate_training_family(family_path)
    activation_path = Path(family["activation"]["path"]).expanduser().resolve()
    activation = validate_activation(activation_path)
    winner = report["selection"]["screen_selected_checkpoint_sha256"]
    attempt = report["confirmation_attempts"][0]
    _require(sha256_file(selected) == winner, "selected bytes are not the screen winner")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": SELECTION_GATE_KIND,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "role": BUYER,
        "actor_loss_mode": PHASE_BALANCED_ACTOR_LOSS_MODE,
        "report": {
            "path": str(report_path),
            "sha256": sha256_file(report_path),
            "evaluator": evaluator.EVALUATOR_NAME,
        },
        "selected_checkpoint": {
            "path": str(selected),
            "sha256": winner,
        },
        "training_family": {
            "path": str(family_path),
            "sha256": sha256_file(family_path),
            "candidate_sha256": list(family["candidate_sha256"]),
        },
        "activation": {
            "path": str(activation_path),
            "sha256": sha256_file(activation_path),
            "code_revision": activation["code_revision"],
        },
        "sampler": {
            "current": report["training_family"]["common_sampler_provenance"],
            "history": report["training_family"]["common_sampler_history"],
        },
        "selection": {
            "screen_selected_checkpoint_sha256": winner,
            "confirmed_checkpoint_sha256": attempt["metadata"]["sha256"],
            "selected_checkpoint_sha256": report["selection"][
                "selected_checkpoint_sha256"
            ],
            "confirmation_attempts": len(report["confirmation_attempts"]),
            "confirmation_policy": report["protocol"]["confirmation_policy"],
            "fallback_allowed": report["selection"]["fallback_allowed"],
        },
    }


def validate_selection_gate(path, *, family=None, report=None, selected=None):
    """Revalidate a published temporal gate and every byte-bound dependency."""

    path = Path(path).expanduser().resolve()
    value = load_json(path)
    _require(
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("kind") == SELECTION_GATE_KIND,
        "unknown temporal selection-gate manifest",
    )
    _require(value.get("passed") is True, "temporal selection gate did not pass")
    _require(value.get("role") == BUYER, "temporal selection gate has wrong role")
    _require(
        value.get("actor_loss_mode") == PHASE_BALANCED_ACTOR_LOSS_MODE,
        "temporal selection gate is not phase-balanced",
    )
    family_path = Path(value["training_family"]["path"]).expanduser().resolve()
    report_path = Path(value["report"]["path"]).expanduser().resolve()
    selected_path = Path(value["selected_checkpoint"]["path"]).expanduser().resolve()
    if family is not None:
        _require(_same_file(family_path, family), "temporal gate names another family")
    if report is not None:
        _require(_same_file(report_path, report), "temporal gate names another report")
    if selected is not None:
        _require(_same_file(selected_path, selected), "temporal gate names another alias")
    _require(
        sha256_file(family_path) == value["training_family"]["sha256"],
        "temporal family changed after gate publication",
    )
    _require(
        sha256_file(report_path) == value["report"]["sha256"],
        "temporal report changed after gate publication",
    )
    _require(
        sha256_file(selected_path) == value["selected_checkpoint"]["sha256"],
        "temporal selected checkpoint changed after gate publication",
    )
    family_value = validate_training_family(family_path)
    activation_path = Path(value["activation"]["path"]).expanduser().resolve()
    _require(
        _same_file(family_value["activation"]["path"], activation_path),
        "temporal gate names another activation",
    )
    _require(
        sha256_file(activation_path) == value["activation"]["sha256"],
        "temporal activation changed after gate publication",
    )
    activation = validate_activation(activation_path)
    _require(
        activation["code_revision"] == value["activation"]["code_revision"],
        "temporal gate code revision differs from activation",
    )
    report_value = validate_selection(SimpleNamespace(
        family=str(family_path),
        report=str(report_path),
        selected=str(selected_path),
    ))
    expected = build_selection_gate(
        SimpleNamespace(
            family=str(family_path),
            report=str(report_path),
            selected=str(selected_path),
        ),
        report_value,
    )
    for key in (
        "passed", "role", "actor_loss_mode", "report", "selected_checkpoint",
        "training_family", "activation", "sampler", "selection",
    ):
        _require(value.get(key) == expected.get(key), f"temporal gate differs in {key}")
    return value


def write_or_validate_selection_gate(args, report):
    """Publish a gate only after the single confirmed screen winner passes."""

    output = Path(args.gate_output).expanduser().resolve()
    if report.get("passed") is not True:
        _require(not output.exists(), "failed temporal selection retained a gate")
        return None
    expected = build_selection_gate(args, report)
    if output.exists():
        existing = validate_selection_gate(
            output,
            family=args.family,
            report=args.report,
            selected=args.selected,
        )
        for key in (
            "passed", "role", "actor_loss_mode", "report", "selected_checkpoint",
            "training_family", "activation", "sampler", "selection",
        ):
            _require(existing.get(key) == expected.get(key), f"existing gate differs in {key}")
        return existing
    atomic_write_json(output, expected)
    return validate_selection_gate(
        output,
        family=args.family,
        report=args.report,
        selected=args.selected,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    activation = subparsers.add_parser("activation")
    activation.add_argument("--standard-report", required=True)
    activation.add_argument("--balanced-report", required=True)
    activation.add_argument("--rom", required=True)
    activation.add_argument("--base", required=True)
    activation.add_argument("--code-root", required=True)
    activation.add_argument("--output", required=True)
    validate = subparsers.add_parser("validate-activation")
    validate.add_argument("--activation", required=True)
    validate.add_argument("--rom")
    family = subparsers.add_parser("training-family")
    family.add_argument("--activation", required=True)
    family.add_argument("--preflight", required=True)
    family.add_argument("--candidate", action="append", required=True)
    family.add_argument("--output", required=True)
    family.add_argument("--device", default="cpu")
    validate_family = subparsers.add_parser("validate-training-family")
    validate_family.add_argument("--family", required=True)
    selection = subparsers.add_parser("selection")
    selection.add_argument("--family", required=True)
    selection.add_argument("--report", required=True)
    selection.add_argument("--selected", required=True)
    selection.add_argument("--gate-output")
    gate = subparsers.add_parser("validate-selection-gate")
    gate.add_argument("--gate", required=True)
    gate.add_argument("--family")
    gate.add_argument("--report")
    gate.add_argument("--selected")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.command == "activation":
            result = write_or_validate_activation(args)
        elif args.command == "validate-activation":
            result = validate_activation(args.activation, rom=args.rom)
        elif args.command == "training-family":
            result = build_training_family(args)
            atomic_write_json(args.output, result)
            result = validate_training_family(args.output)
        elif args.command == "validate-training-family":
            result = validate_training_family(args.family)
        elif args.command == "selection":
            result = validate_selection(args)
            if args.gate_output:
                gate = write_or_validate_selection_gate(args, result)
                if gate is not None:
                    result = gate
        else:
            result = validate_selection_gate(
                args.gate,
                family=args.family,
                report=args.report,
                selected=args.selected,
            )
    except ContingencyInactive as error:
        print({"active": False, "reason": str(error)}, flush=True)
        raise SystemExit(3) from error
    print({"command": args.command, "passed": True, "kind": result.get("kind")}, flush=True)


if __name__ == "__main__":
    main()
