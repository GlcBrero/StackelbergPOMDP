#!/Users/gbrero/miniconda3/envs/stackelbergPOMDP/bin/python
"""Fail-closed manifests for the Atari E1 seller-conditioning recovery.

The recovery is deliberately a two-stage *single optimizer lineage*:

1. start from canonical E0b and train 400,160 transitions with one shared
   Uniform(0,1) opponent threshold in all five events (``all-equal-v1``);
2. resume the complete model, optimizer, and timestep clock and train another
   2,000,800 transitions with five independent Uniform(0,1) thresholds.

Only five target-stage step checkpoints and the byte-distinct final target
checkpoint may be screened.  The warm-up is never a candidate.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from types import SimpleNamespace

import numpy as np


SCHEMA_VERSION = 1
ACTIVATION_KIND = "stackpomdp.atari.e1_seller_conditioning_recovery_activation.v1"
FAMILY_KIND = "stackpomdp.atari.e1_seller_conditioning_recovery_family.v1"
GATE_KIND = "stackpomdp.atari.e1_seller_conditioning_recovery_gate.v1"
SOURCE_KIND = "seller_conditioning_recovery_v1"
EVALUATOR = "clean_atari_e1_selector_v2"
ROLE = "seller"
ACTOR_LOSS_MODE = "balanced"
ALL_EQUAL_MODE = "all-equal-v1"
UNIFORM_MODE = "uniform"
WARMUP_TIMESTEPS = 400_160
TARGET_ADDITIONAL_TIMESTEPS = 2_000_800
FINAL_TIMESTEPS = WARMUP_TIMESTEPS + TARGET_ADDITIONAL_TIMESTEPS
CHECKPOINT_INTERVAL = 400_160
TARGET_STEP_TIMESTEPS = (800_320, 1_200_480, 1_600_640, 2_000_800, 2_400_960)
TARGET_CANDIDATE_TIMESTEPS = TARGET_STEP_TIMESTEPS + (FINAL_TIMESTEPS,)
SCREEN_SEED_START = 9_000_001
CONFIRMATION_SEED_START = 9_100_001
FIXED_SEED_START = 9_200_001
TIMING_SEED_START = 9_300_001
FAILED_REPORT_NAME = "e1_seller_balanced_all6_selector_v2.json"
FAILED_REPORT_SHA256 = (
    "2c47ed3cee727bd2ac2b1284dab7a6d4ea199918b6acc3b40a11022e533fb534"
)
REPORT_NAME = "e1_seller_conditioning_recovery_all6_selector_v2.json"
GATE_NAME = "e1_seller_conditioning_recovery_all6_selector_v2.gate.json"
ACTIVATION_NAME = "e1_seller_conditioning_recovery_activation_v1.json"
FAMILY_NAME = "e1_seller_conditioning_recovery_family_v1.json"
PROBE_NAME = "clean_atari_e1_seller_conditioning_probe_v1"
PROBE_GATE_NAME = "clean_e1_seller_conditioning_warmup_gate_v1"
MINIMUM_ENDPOINT_RESPONSE = 0.25
MAXIMUM_ADJACENT_REVERSAL = 0.15
CANONICAL_E0B_SHA256 = (
    "3a9ded5c53e10bf0b2215f1223f7bde15ba23d197dd0c651590bcd980c375ca3"
)
CANONICAL_ROM_SHA256 = (
    "7224b17462b992d67f4e06a3c85f269c9822b06df6015bf038b55f384ced0301"
)
PINNED_E2_CODE_HEAD = "7a193ba14b91f6ab116da29ff288e3e577d73b88"
CANONICAL_SELLER_RELEASE_SHA256 = (
    "a1f4e02ce0a99246e1bcb24dd9f9680c8ff934531518bbc069836740bca3ebb7"
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical_json(value):
    return json.loads(json.dumps(value, sort_keys=True))


def sha256_file(path):
    path = Path(path).expanduser()
    _require(not path.is_symlink(), f"expected a regular non-symlink file: {path}")
    path = path.resolve()
    _require(path.is_file(), f"expected a regular non-symlink file: {path}")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    _require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        f"file changed while hashing: {path}",
    )
    return digest.hexdigest()


def load_json(path):
    path = Path(path).expanduser()
    _require(not path.is_symlink(), f"expected a regular non-symlink JSON: {path}")
    path = path.resolve()
    _require(path.is_file(), f"expected a regular non-symlink JSON: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def atomic_write_json(path, value):
    path = Path(path).expanduser().resolve()
    _require(not os.path.lexists(path), f"refusing to overwrite immutable artifact: {path}")
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
            raise FileExistsError(
                f"refusing to overwrite immutable artifact: {path}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _same_path(actual, expected, *, label):
    _require(isinstance(actual, str) and actual, f"{label} is not a path string")
    _require(
        Path(actual).expanduser().resolve() == Path(expected).expanduser().resolve(),
        f"{label} names another path: {actual}",
    )


def canonical_uniform_sampler():
    return {
        "mode": UNIFORM_MODE,
        "gameplay_horizon": 200,
        "event_tail_steps": 0,
        "schedule": "ExactFiveEventSchedule.sample",
        "context": "five independent Uniform(0,1) prices",
        "schedule_context_rngs_independent": False,
        "legacy_sampling_path": True,
    }


def canonical_all_equal_sampler():
    return {
        "mode": ALL_EQUAL_MODE,
        "gameplay_horizon": 200,
        "event_tail_steps": 0,
        "schedule": "ExactFiveEventSchedule.sample",
        "context": "one Uniform(0,1) scalar replicated across five events",
        "context_scalar_distribution": "Uniform(0,1)",
        "context_replication_count": 5,
        "commitment_entries_all_equal": True,
        "per_event_marginal": "Uniform(0,1)",
        "schedule_context_rngs_independent": False,
        "legacy_sampling_path": False,
    }


def canonical_training_config():
    return {
        "algorithm": "PPO",
        "actor_loss_mode": ACTOR_LOSS_MODE,
        "economic_head_initialization": {
            "mean": 0.5,
            "concentration": 2.0,
        },
        "target_kl": None,
        "seed": 1,
        "learning_rate": 1.0e-4,
        "n_steps": 205,
        "batch_size": 820,
        "n_epochs": 4,
        "gamma": 1.0,
        "gae_lambda": 1.0,
        "clip_range_at_start": 0.1,
        "entropy_coefficient": 0.01,
        "value_coefficient": 0.5,
        "max_grad_norm": 0.5,
        "policy_class": (
            "stackelberg_pomdp.atari.stackpomdp_policy."
            "StackPOMDPAtariPolicy"
        ),
        "visual_features": 512,
        "state_features": 64,
        "economic_hidden": 64,
        "critic_hidden": 256,
        "pretrained_lr_scale": 0.1,
    }


def expected_target_candidates(base):
    base = Path(base).expanduser().resolve()
    stem = base.with_suffix("")
    return [
        str(stem.with_name(f"{stem.name}_step{step}.zip"))
        for step in TARGET_STEP_TIMESTEPS
    ] + [str(base)]


def _git_revision(code_root):
    root = Path(code_root).expanduser().resolve()
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _require(re.fullmatch(r"[0-9a-f]{40}", revision), "code revision is not a full SHA")
    return revision


def _git_scoped_clean(code_root):
    root = Path(code_root).expanduser().resolve()
    dirty = subprocess.run(
        [
            "git", "-C", str(root), "status", "--porcelain", "--",
            "replication/atari/train_atari_meta_response_sb3.py",
            "replication/atari/evaluate_atari_meta_response_sb3.py",
            "replication/atari/probe_atari_e1_seller_conditioning.py",
            "replication/atari/sb3_common.py",
            "replication/atari/automation",
            "stackelberg_pomdp/atari",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _require(not dirty, f"recovery activation requires clean Atari code: {dirty}")


def validate_seller_release(path):
    path = Path(path).expanduser().resolve()
    _require(
        sha256_file(path) == CANONICAL_SELLER_RELEASE_SHA256,
        "seller recovery requires the exact canonical seller-release bytes",
    )
    value = load_json(path)
    _require(value.get("schema") == "stackpomdp.atari.e1_seller_release.v2", "unknown seller release")
    _require(value.get("code_head") == PINNED_E2_CODE_HEAD, "seller release uses another E2 code head")
    _require(value.get("seller_training_actor_loss_mode") == ACTOR_LOSS_MODE, "seller release is not balanced")
    buyer = value.get("buyer_gate")
    _require(isinstance(buyer, dict), "seller release has no buyer gate")
    _require(buyer.get("source_kind") == "primary_economic_v1", "seller release is not primary-economic")
    _require(buyer.get("actor_loss_mode") == ACTOR_LOSS_MODE, "released buyer is not balanced")
    for field, hash_field in (("report", "report_sha256"), ("checkpoint", "checkpoint_sha256")):
        _require(
            sha256_file(buyer.get(field, "")) == buyer.get(hash_field),
            f"seller release buyer {field} bytes changed",
        )
    support = buyer.get("support_artifacts")
    _require(isinstance(support, dict) and support, "seller release lacks buyer support artifacts")
    for label, record in support.items():
        _require(isinstance(record, dict), f"seller release {label} record is malformed")
        _require(
            sha256_file(record.get("path", "")) == record.get("sha256"),
            f"seller release {label} bytes changed",
        )
    return value


def _canonical_failed_uniform_history(report):
    family = report.get("training_family", {})
    current = family.get("common_sampler_provenance")
    history = family.get("common_sampler_history")
    expected = canonical_uniform_sampler()
    _require(current == expected, "failed seller report is not canonical uniform")
    _require(isinstance(history, list) and len(history) == 1, "failed seller report has mixed sampler history")
    stage = history[0]
    _require(stage.get("start_total_timesteps") == 0, "failed seller history does not start at zero")
    _require(stage.get("sampler") == expected, "failed seller history sampler changed")
    _require(stage.get("resume_sources") == [], "failed seller history has resume sources")
    return {"current": current, "history": history}


def validate_failed_seller_report(path):
    path = Path(path).expanduser().resolve()
    _require(path.name == FAILED_REPORT_NAME, "recovery prerequisite has the wrong report name")
    _require(
        sha256_file(path) == FAILED_REPORT_SHA256,
        "recovery prerequisite is not the exact certified failed report",
    )
    report = load_json(path)
    _require(report.get("passed") is False, "seller recovery requires a formal failed report")
    _require(report.get("role") == ROLE and report.get("evaluator") == EVALUATOR, "failed report has the wrong role/evaluator")
    _require(report.get("selected_alias") is None, "failed seller report retained a selected alias")
    protocol = report.get("protocol", {})
    expected_protocol = {
        "screen_episodes": 20,
        "screen_seed_start": 3_500_001,
        "confirmation_episodes": 100,
        "confirmation_seed_start": 3_600_001,
        "fixed_context_episodes": 20,
        "fixed_context_seed_start": 3_700_001,
        "confirmation_policy": "screen_winner_only_no_fallback",
        "outer_transitions": 205,
    }
    for key, expected in expected_protocol.items():
        _require(protocol.get(key) == expected, f"failed seller protocol changed: {key}")
    environment = report.get("environment", {})
    _require(environment.get("gameplay_horizon") == 200, "failed seller gameplay horizon changed")
    _require(environment.get("event_tail_steps") == 0, "failed seller event tail changed")
    _require(environment.get("rom_sha256") == CANONICAL_ROM_SHA256, "failed seller ROM changed")
    _canonical_failed_uniform_history(report)
    config = report.get("training_family", {}).get("common_training_config", {})
    for key, expected in {
        "algorithm": "PPO", "actor_loss_mode": ACTOR_LOSS_MODE, "seed": 1,
        "n_steps": 205, "batch_size": 820, "n_epochs": 4,
        "learning_rate": 1.0e-4, "pretrained_lr_scale": 0.1,
        "gamma": 1.0, "gae_lambda": 1.0,
    }.items():
        _require(config.get(key) == expected, f"failed seller config changed: {key}")
    screen = report.get("screen", {})
    common = screen.get("common_pairing", {})
    _require(common.get("passed") is True and common.get("candidates_checked") == 6, "failed seller common screen is invalid")
    _require(
        [row.get("evaluation_seed") for row in common.get("seed_context_pairs", [])]
        == list(range(3_500_001, 3_500_021)),
        "failed seller screen seed schedule changed",
    )
    results = screen.get("results", [])
    _require(len(results) == 6, "failed seller report is not all-six")
    _require(
        [row.get("metadata", {}).get("training_timesteps") for row in results]
        == [400_160, 800_320, 1_200_480, 1_600_640, 2_000_800, 2_000_800],
        "failed seller candidate schedule changed",
    )
    hashes = []
    for row in results:
        metadata = row.get("metadata", {})
        digest = metadata.get("sha256")
        _require(sha256_file(metadata.get("path", "")) == digest, "failed seller candidate bytes changed")
        _require(row.get("protocol", {}).get("passed") is True, "failed seller screen mechanics failed")
        hashes.append(digest)
    _require(len(set(hashes)) == 6, "failed seller family is not byte-distinct")
    _require(report.get("immutable_evaluation", {}).get("candidate_sha256") == hashes, "failed seller candidate hashes changed")
    _require(report.get("immutable_evaluation", {}).get("e0b_sha256") == CANONICAL_E0B_SHA256, "failed seller E0b changed")
    attempts = report.get("confirmation_attempts", [])
    _require(len(attempts) == 1, "failed seller report did not confirm only its winner")
    _require(attempts[0].get("behavioral_gate", {}).get("passed") is False, "failed seller winner unexpectedly passed")
    _require(
        [row.get("evaluation_seed") for row in attempts[0].get("random", {}).get("episode_rows", [])]
        == list(range(3_600_001, 3_600_101)),
        "failed seller confirmation seed schedule changed",
    )
    fixed = attempts[0].get("fixed_contexts", [])
    _require(len(fixed) == 11, "failed seller fixed grid is incomplete")
    _require([round(float(row.get("opponent_value")), 6) for row in fixed] == [round(i / 10, 6) for i in range(11)], "failed seller fixed grid changed")
    for row in fixed:
        _require(
            [episode.get("evaluation_seed") for episode in row.get("episode_rows", [])]
            == list(range(3_700_001, 3_700_021)),
            "failed seller fixed-grid seeds changed",
        )
    _same_path(report.get("artifacts", {}).get("json"), path, label="failed seller report artifact")
    e0b = report.get("e0b_source", {})
    _require(sha256_file(e0b.get("path", "")) == e0b.get("sha256") == CANONICAL_E0B_SHA256, "failed seller E0b bytes changed")
    _require(not path.with_name(f"{path.stem}.gate.json").exists(), "failed seller report has a gate sidecar")
    return {
        "path": str(path), "sha256": FAILED_REPORT_SHA256, "passed": False,
        "evaluator": EVALUATOR, "candidate_sha256": hashes,
        "e0b_source": {"path": str(Path(e0b["path"]).resolve()), "sha256": e0b["sha256"]},
    }


def build_activation(args):
    _git_scoped_clean(args.code_root)
    failure = validate_failed_seller_report(args.failed_report)
    release_path = Path(args.seller_release).expanduser().resolve()
    validate_seller_release(release_path)
    e0b = Path(args.e0b).expanduser().resolve()
    _require(sha256_file(e0b) == CANONICAL_E0B_SHA256, "recovery E0b is not canonical")
    _require(failure["e0b_source"]["sha256"] == CANONICAL_E0B_SHA256, "failure and recovery use different E0b")
    rom = Path(args.rom).expanduser().resolve()
    _require(sha256_file(rom) == CANONICAL_ROM_SHA256, "recovery ROM is not canonical")
    warmup = Path(args.warmup_checkpoint).expanduser().resolve()
    warmup_probe = Path(args.warmup_probe).expanduser().resolve()
    target = Path(args.target_checkpoint).expanduser().resolve()
    candidates = expected_target_candidates(target)
    _require(str(warmup) not in candidates, "warm-up checkpoint entered the target family")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": ACTIVATION_KIND,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_revision": _git_revision(args.code_root),
        "activation_condition": {"formal_prior_seller_gate_failed": True},
        "prerequisite_failure": failure,
        "seller_release": {"path": str(release_path), "sha256": sha256_file(release_path)},
        "e0b_source": {"path": str(e0b), "sha256": CANONICAL_E0B_SHA256},
        "rom": {"path": str(rom), "sha256": CANONICAL_ROM_SHA256},
        "protocol": {
            "role": ROLE,
            "actor_loss_mode": ACTOR_LOSS_MODE,
            "checkpoint_interval": CHECKPOINT_INTERVAL,
            "stage_order": ["all_equal_warmup", "independent_uniform_target"],
            "warmup": {
                "fresh_e0b_initialization": True,
                "resume": False,
                "sampler": canonical_all_equal_sampler(),
                "start_total_timesteps": 0,
                "additional_timesteps": WARMUP_TIMESTEPS,
                "expected_total_timesteps": WARMUP_TIMESTEPS,
                "checkpoint": str(warmup),
                "selectable": False,
                "conditioning_probe": {
                    "path": str(warmup_probe),
                    "probe": PROBE_NAME,
                    "no_ale": True,
                    "environment_steps": 0,
                    "deterministic_statistic": "Beta mean",
                    "require_all_outputs_finite_and_unit": True,
                    "minimum_endpoint_response": MINIMUM_ENDPOINT_RESPONSE,
                    "maximum_adjacent_reversal": MAXIMUM_ADJACENT_REVERSAL,
                    "require_pass": True,
                },
            },
            "target": {
                "resume_complete_model_optimizer_clock": True,
                "resume_checkpoint": str(warmup),
                "sampler": canonical_uniform_sampler(),
                "start_total_timesteps": WARMUP_TIMESTEPS,
                "additional_timesteps": TARGET_ADDITIONAL_TIMESTEPS,
                "expected_total_timesteps": FINAL_TIMESTEPS,
                "candidate_paths": candidates,
                "candidate_timesteps": list(TARGET_CANDIDATE_TIMESTEPS),
                "candidate_count": 6,
                "warmup_selectable": False,
            },
            "evaluation": {
                "screen": {"episodes": 20, "seed_start": SCREEN_SEED_START},
                "confirmation": {
                    "episodes": 100,
                    "seed_start": CONFIRMATION_SEED_START,
                    "policy": "screen_winner_only_no_fallback",
                    "fallback_allowed": False,
                },
                "fixed_grid": {
                    "episodes_per_value": 20,
                    "seed_start": FIXED_SEED_START,
                    "values": [i / 10 for i in range(11)],
                    "event_steps": [20, 50, 80, 110, 140],
                },
                "timing_seed_start_reserved": TIMING_SEED_START,
            },
        },
    }


def _validate_activation_contract(value):
    _require(value.get("schema_version") == SCHEMA_VERSION and value.get("kind") == ACTIVATION_KIND, "unknown seller-recovery activation")
    _require(value.get("activation_condition") == {"formal_prior_seller_gate_failed": True}, "seller-recovery activation condition changed")
    _require(re.fullmatch(r"[0-9a-f]{40}", str(value.get("code_revision", ""))), "seller-recovery code revision is invalid")
    failure = value.get("prerequisite_failure", {})
    _require(
        failure.get("sha256") == FAILED_REPORT_SHA256,
        "activation is not bound to the certified failed-report SHA-256",
    )
    certified_failure = validate_failed_seller_report(
        failure.get("path", "")
    )
    _require(
        failure == certified_failure,
        "activation prerequisite differs from the fully revalidated failure",
    )
    release = value.get("seller_release", {})
    _require(sha256_file(release.get("path", "")) == release.get("sha256"), "seller release changed after activation")
    validate_seller_release(release.get("path", ""))
    e0b = value.get("e0b_source", {})
    _require(sha256_file(e0b.get("path", "")) == e0b.get("sha256") == CANONICAL_E0B_SHA256, "activation E0b changed")
    rom = value.get("rom", {})
    _require(sha256_file(rom.get("path", "")) == rom.get("sha256") == CANONICAL_ROM_SHA256, "activation ROM changed")
    protocol = value.get("protocol", {})
    _require(protocol.get("role") == ROLE and protocol.get("actor_loss_mode") == ACTOR_LOSS_MODE, "activation role/mode changed")
    _require(protocol.get("checkpoint_interval") == CHECKPOINT_INTERVAL, "activation checkpoint interval changed")
    _require(protocol.get("stage_order") == ["all_equal_warmup", "independent_uniform_target"], "activation stage order changed")
    warmup = protocol.get("warmup", {})
    target = protocol.get("target", {})
    expected_warmup = {
        "fresh_e0b_initialization": True, "resume": False,
        "sampler": canonical_all_equal_sampler(), "start_total_timesteps": 0,
        "additional_timesteps": WARMUP_TIMESTEPS,
        "expected_total_timesteps": WARMUP_TIMESTEPS,
        "checkpoint": warmup.get("checkpoint"), "selectable": False,
        "conditioning_probe": {
            "path": warmup.get("conditioning_probe", {}).get("path"),
            "probe": PROBE_NAME,
            "no_ale": True,
            "environment_steps": 0,
            "deterministic_statistic": "Beta mean",
            "require_all_outputs_finite_and_unit": True,
            "minimum_endpoint_response": MINIMUM_ENDPOINT_RESPONSE,
            "maximum_adjacent_reversal": MAXIMUM_ADJACENT_REVERSAL,
            "require_pass": True,
        },
    }
    _require(warmup == expected_warmup, "activation warm-up stage changed")
    _require(target.get("resume_complete_model_optimizer_clock") is True, "target does not resume full state")
    _same_path(target.get("resume_checkpoint"), warmup.get("checkpoint"), label="target resume source")
    _require(target.get("sampler") == canonical_uniform_sampler(), "target sampler is not independent uniform")
    _require(target.get("start_total_timesteps") == WARMUP_TIMESTEPS, "target boundary changed")
    _require(target.get("additional_timesteps") == TARGET_ADDITIONAL_TIMESTEPS, "target budget changed")
    _require(target.get("expected_total_timesteps") == FINAL_TIMESTEPS, "target final clock changed")
    _require(target.get("candidate_timesteps") == list(TARGET_CANDIDATE_TIMESTEPS), "target candidate clocks changed")
    _require(target.get("candidate_paths") == expected_target_candidates(target.get("candidate_paths", [""])[-1]), "target candidate paths changed")
    _require(target.get("candidate_count") == 6 and target.get("warmup_selectable") is False, "warm-up selection guard changed")
    _require(str(Path(warmup["checkpoint"]).resolve()) not in target["candidate_paths"], "warm-up is selectable")
    evaluation = protocol.get("evaluation", {})
    _require(evaluation.get("screen") == {"episodes": 20, "seed_start": SCREEN_SEED_START}, "recovery screen changed")
    _require(evaluation.get("confirmation") == {"episodes": 100, "seed_start": CONFIRMATION_SEED_START, "policy": "screen_winner_only_no_fallback", "fallback_allowed": False}, "recovery confirmation changed")
    _require(evaluation.get("fixed_grid") == {"episodes_per_value": 20, "seed_start": FIXED_SEED_START, "values": [i / 10 for i in range(11)], "event_steps": [20, 50, 80, 110, 140]}, "recovery fixed grid changed")
    return value


def validate_warmup_probe(*, activation, probe, checkpoint=None):
    """Verify the immutable no-ALE gate that must release target training."""

    activation = (
        _validate_activation_contract(activation)
        if isinstance(activation, dict)
        else validate_activation(activation)
    )
    contract = activation["protocol"]["warmup"]["conditioning_probe"]
    probe_path = Path(probe).expanduser().resolve()
    _same_path(str(probe_path), contract["path"], label="warm-up probe")
    value = load_json(probe_path)
    _require(value.get("probe") == PROBE_NAME, "unknown warm-up conditioning probe")
    execution = value.get("execution", {})
    _require(execution.get("read_only_checkpoint") is True, "warm-up probe was not read-only")
    _require(execution.get("ale_instantiated") is False, "warm-up probe instantiated ALE")
    _require(execution.get("environment_steps") == 0, "warm-up probe stepped an environment")
    _require(execution.get("deterministic_statistic") == "Beta mean", "warm-up probe used another statistic")
    _require(execution.get("device") == "cpu", "warm-up probe did not use the predeclared CPU device")
    checkpoint_record = value.get("checkpoint", {})
    checkpoint_path = Path(
        checkpoint
        if checkpoint is not None
        else activation["protocol"]["warmup"]["checkpoint"]
    ).expanduser().resolve()
    _same_path(checkpoint_record.get("path"), checkpoint_path, label="probe checkpoint")
    checkpoint_digest = sha256_file(checkpoint_path)
    _require(checkpoint_record.get("sha256") == checkpoint_digest, "probe checkpoint SHA-256 changed")
    _require(checkpoint_record.get("training_timesteps") == WARMUP_TIMESTEPS, "probe checkpoint is not the warm-up boundary")
    _require(checkpoint_record.get("role") == ROLE, "probe checkpoint is not a seller")
    _require(checkpoint_record.get("atari_e1_sampler_provenance") == canonical_all_equal_sampler(), "probe checkpoint sampler is not all-equal-v1")
    _require(
        checkpoint_record.get("atari_e1_sampler_history") == [{
            "start_total_timesteps": 0,
            "sampler": canonical_all_equal_sampler(),
            "inferred_for_legacy_checkpoint": False,
            "resume_sources": [],
        }],
        "probe checkpoint does not have exactly one fresh all-equal stage",
    )
    e0b = value.get("e0b_source", {})
    _require(e0b.get("sha256") == CANONICAL_E0B_SHA256, "probe used another E0b")
    all_equal_rows = value.get("all_equal_thresholds", [])
    _require(
        [row.get("threshold") for row in all_equal_rows]
        == [0.0, 0.25, 0.5, 0.75, 1.0],
        "warm-up probe all-equal threshold grid changed",
    )
    all_equal_prices = [
        row.get("event_beta_mean_prices") for row in all_equal_rows
    ]
    _require(
        all(isinstance(row, list) and len(row) == 5 for row in all_equal_prices),
        "warm-up probe all-equal measurements are incomplete",
    )
    coordinate_rows = value.get(
        "current_coordinate_only_sensitivity", {}
    ).get("rows", [])
    _require(
        [row.get("event_index") for row in coordinate_rows]
        == list(range(5)),
        "warm-up probe coordinate measurements are incomplete",
    )
    coordinate_low = [row.get("low_beta_mean_price") for row in coordinate_rows]
    coordinate_high = [row.get("high_beta_mean_price") for row in coordinate_rows]
    from replication.atari import probe_atari_e1_seller_conditioning as probe_module

    recomputed_gate = probe_module.warmup_diagnostic_gate(
        all_equal_prices, coordinate_low, coordinate_high
    )
    gate = value.get("warmup_gate", {})
    _require(
        gate == recomputed_gate,
        "warm-up probe gate does not recompute from its raw measurements",
    )
    _require(gate.get("name") == PROBE_GATE_NAME and gate.get("predeclared") is True, "warm-up probe gate is not predeclared v1")
    _require(gate.get("passed") is True, "warm-up conditioning probe did not pass")
    checks = gate.get("checks", {})
    finite = checks.get("all_outputs_finite_and_in_unit_interval", {})
    endpoint = checks.get("minimum_all_one_minus_all_zero_beta_mean_price", {})
    reversal = checks.get("largest_adjacent_threshold_price_reversal", {})
    _require(finite.get("required") is True and finite.get("actual") is True and finite.get("passed") is True, "warm-up probe has nonfinite/out-of-unit output")
    _require(endpoint.get("operator") == ">=" and endpoint.get("required") == MINIMUM_ENDPOINT_RESPONSE and endpoint.get("passed") is True, "warm-up endpoint gate changed or failed")
    _require(isinstance(endpoint.get("actual"), (int, float)) and MINIMUM_ENDPOINT_RESPONSE <= float(endpoint["actual"]) <= 1.0, "warm-up endpoint response is invalid")
    _require(reversal.get("operator") == "<=" and reversal.get("required") == MAXIMUM_ADJACENT_REVERSAL and reversal.get("passed") is True, "warm-up reversal gate changed or failed")
    _require(isinstance(reversal.get("actual"), (int, float)) and 0.0 <= float(reversal["actual"]) <= MAXIMUM_ADJACENT_REVERSAL, "warm-up reversal is invalid")
    # The JSON is evidence, not an oracle.  Re-run the deterministic no-ALE
    # inference against the bound checkpoint and E0b bytes, then compare every
    # report field except the wall-clock timestamp.  This prevents a
    # hand-authored passing measurement grid from releasing target training.
    inferred = probe_module.run_probe_from_checkpoints(
        checkpoint=checkpoint_path,
        e0b_checkpoint=activation["e0b_source"]["path"],
        device="cpu",
    )
    reported_for_comparison = _canonical_json(value)
    inferred_for_comparison = _canonical_json(inferred)
    reported_for_comparison.pop("created_utc", None)
    inferred_for_comparison.pop("created_utc", None)
    _require(
        reported_for_comparison == inferred_for_comparison,
        "warm-up probe JSON does not match fresh checkpoint inference",
    )
    return {
        "path": str(probe_path),
        "sha256": sha256_file(probe_path),
        "probe": PROBE_NAME,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_digest,
        "passed": True,
        "no_ale": True,
        "environment_steps": 0,
        "minimum_endpoint_response": float(endpoint["actual"]),
        "largest_adjacent_reversal": float(reversal["actual"]),
    }


def validate_warmup_checkpoint(*, activation, checkpoint, device="cpu"):
    """Load and validate the exact fresh 400160-step warm-up checkpoint."""

    activation_path = Path(activation).expanduser().resolve()
    activation_value = validate_activation(activation_path)
    metadata = _load_candidate_metadata(
        checkpoint, activation=activation_value, device=device
    )
    metadata["resume_source"] = metadata.get("resume_source")
    _validate_warmup_metadata(metadata, activation_value)
    return {
        "passed": True,
        "checkpoint_metadata": metadata,
    }


def validate_warmup_stage(
        *, activation, probe, checkpoint, training_log, evaluation,
        device="cpu",
):
    """Load the probed ZIP and prove that the probe releases that exact stage."""

    validated = validate_warmup_checkpoint(
        activation=activation, checkpoint=checkpoint, device=device
    )
    activation_value = validate_activation(
        Path(activation).expanduser().resolve()
    )
    metadata = validated["checkpoint_metadata"]
    probe_record = validate_warmup_probe(
        activation=activation_value,
        probe=probe,
        checkpoint=checkpoint,
    )
    _require(
        probe_record["checkpoint_sha256"] == metadata["sha256"],
        "warm-up probe/checkpoint metadata SHA mismatch",
    )
    trace_record = _validate_training_trace(
        training_log, mode=ALL_EQUAL_MODE, first_step=820,
        last_step=WARMUP_TIMESTEPS, expected_episodes=1_952,
        expected_optimizers=488, checkpoint=checkpoint,
    )
    evaluation_record = _validate_evaluation(
        evaluation,
        current=canonical_all_equal_sampler(),
        history=metadata["atari_e1_sampler_history"],
        expected_step=WARMUP_TIMESTEPS,
        checkpoint_metadata=metadata,
        random_episodes=20,
        fixed_episodes=5,
    )
    return {
        "passed": True,
        "checkpoint_metadata": metadata,
        "conditioning_probe": probe_record,
        "training_trace": trace_record,
        "evaluation": evaluation_record,
    }


def validate_activation(path, *, code_root=None):
    value = _validate_activation_contract(load_json(path))
    if code_root is not None:
        _require(_git_revision(code_root) == value["code_revision"], "runtime revision differs from activation")
        _git_scoped_clean(code_root)
    return value


def write_or_validate_activation(args):
    expected = build_activation(args)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        existing = validate_activation(output)
        for key in (
            "code_revision", "activation_condition", "prerequisite_failure",
            "seller_release", "e0b_source", "rom", "protocol",
        ):
            _require(existing.get(key) == expected.get(key), f"existing activation differs in {key}")
        return existing
    atomic_write_json(output, expected)
    return validate_activation(output)


def _optimizer_step(model):
    steps = set()
    for state in model.policy.optimizer.state.values():
        raw = state.get("step")
        if raw is None:
            continue
        steps.add(int(raw.item() if hasattr(raw, "item") else raw))
    _require(len(steps) == 1, f"checkpoint optimizer has inconsistent or absent Adam steps: {steps}")
    return next(iter(steps))


def _load_candidate_metadata(path, *, activation, device):
    from replication.atari import evaluate_atari_meta_response_sb3 as evaluator

    model, metadata = evaluator.load_candidate(
        path,
        role=ROLE,
        e0b_sha256=activation["e0b_source"]["sha256"],
        device=device,
    )
    metadata = _canonical_json(metadata)
    metadata["optimizer_adam_step"] = _optimizer_step(model)
    metadata["resume_source"] = _canonical_json(
        getattr(model, "atari_e1_resume_source_provenance", None)
    )
    del model
    return metadata


def _expected_history(activation, warmup_digest):
    warmup_path = str(Path(activation["protocol"]["warmup"]["checkpoint"]).resolve())
    return [
        {
            "start_total_timesteps": 0,
            "sampler": canonical_all_equal_sampler(),
            "inferred_for_legacy_checkpoint": False,
            "resume_sources": [],
        },
        {
            "start_total_timesteps": WARMUP_TIMESTEPS,
            "sampler": canonical_uniform_sampler(),
            "inferred_for_legacy_checkpoint": False,
            "resume_sources": [{
                "path": warmup_path,
                "sha256": warmup_digest,
                "training_total_timesteps": WARMUP_TIMESTEPS,
                "resume_total_timesteps": WARMUP_TIMESTEPS,
            }],
        },
    ]


def _validate_warmup_metadata(metadata, activation):
    _same_path(metadata.get("path"), activation["protocol"]["warmup"]["checkpoint"], label="warm-up checkpoint")
    _require(metadata.get("training_timesteps") == WARMUP_TIMESTEPS, "warm-up clock is not 400160")
    _require(metadata.get("atari_e1_sampler_provenance") == canonical_all_equal_sampler(), "warm-up sampler is not all-equal-v1")
    _require(metadata.get("atari_e1_sampler_history") == [{
        "start_total_timesteps": 0,
        "sampler": canonical_all_equal_sampler(),
        "inferred_for_legacy_checkpoint": False,
        "resume_sources": [],
    }], "warm-up has a mixed or resumed history")
    _require(metadata.get("resume_source") is None, "warm-up was not initialized fresh from E0b")
    _require(metadata.get("optimizer_adam_step") == 1_952, "warm-up optimizer clock is wrong")
    _require(
        metadata.get("training_config") == canonical_training_config(),
        "warm-up PPO configuration/seed changed",
    )


def _validate_target_metadata(metadata, *, expected_path, expected_step, activation, warmup_digest, final):
    _same_path(metadata.get("path"), expected_path, label="target candidate")
    _require(metadata.get("training_timesteps") == expected_step, "target candidate clock mismatch")
    _require(metadata.get("atari_e1_sampler_provenance") == canonical_uniform_sampler(), "target current sampler is not uniform")
    _require(metadata.get("atari_e1_sampler_history") == _expected_history(activation, warmup_digest), "target mixed sampler history is invalid")
    resume = metadata.get("resume_source")
    _require(isinstance(resume, dict), "target lacks direct resume provenance")
    _same_path(resume.get("path"), activation["protocol"]["warmup"]["checkpoint"], label="target direct resume path")
    _require(resume.get("sha256") == warmup_digest, "target direct resume SHA changed")
    _require(resume.get("training_total_timesteps") == WARMUP_TIMESTEPS, "target direct resume boundary changed")
    expected_optimizer_step = expected_step // 820 * 4 - (0 if final else 4)
    _require(metadata.get("optimizer_adam_step") == expected_optimizer_step, "target optimizer did not continue across the stage boundary")
    _require(
        metadata.get("training_config") == canonical_training_config(),
        "target PPO configuration/seed changed",
    )


def _expected_training_contexts(mode, episode_groups):
    """Reproduce the four deterministic per-env context streams exactly."""

    _require(mode in (ALL_EQUAL_MODE, UNIFORM_MODE), "unknown trace sampler")
    # SB3 seeds the VecEnv as seed + rank during model setup, overriding the
    # constructor's rank-spaced seed before the first reset.
    rngs = [
        np.random.default_rng(1 + rank + 74_711)
        for rank in range(4)
    ]
    rows = []
    for _ in range(int(episode_groups)):
        for rng in rngs:
            if mode == ALL_EQUAL_MODE:
                shared = np.float32(rng.uniform(0.0, 1.0))
                context = np.full(5, shared, dtype=np.float32)
            else:
                context = np.asarray(
                    rng.uniform(0.0, 1.0, size=5), dtype=np.float32
                )
            # Each reset draws the inner bilateral-environment seed after its
            # context.  Consume it so the following episode remains exact.
            rng.integers(0, 2 ** 31 - 1)
            rows.append(tuple(float(value) for value in context))
    return rows


def _validate_training_trace(path, *, mode, first_step, last_step, expected_episodes, expected_optimizers, checkpoint):
    path = Path(path).expanduser().resolve()
    episode_steps = []
    optimizer_steps = []
    observed_steps = []
    observed_contexts = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            step = int(row.get("train/total_timesteps", -1))
            observed_steps.append(step)
            _require(first_step <= step <= last_step, f"trace row {line_number} crosses a stage boundary")
            _same_path(row.get("checkpoint_path"), checkpoint, label="trace checkpoint path")
            _require(row.get("seed") == 1, f"trace row {line_number} has another seed")
            _require(row.get("algorithm") == "PPO", f"trace row {line_number} has another algorithm")
            for key, raw in row.items():
                if isinstance(raw, float):
                    _require(math.isfinite(raw), f"trace row {line_number} has nonfinite {key}")
            record_kind = row.get("record_kind")
            _require(
                record_kind in {"episode", "optimizer"},
                f"trace row {line_number} has an invalid record_kind",
            )
            if record_kind == "optimizer":
                optimizer_steps.append(step)
                _require(
                    row.get("train/game_active_rows") == 800.0
                    and row.get("train/economic_active_rows") == 20.0
                    and row.get("train/inactive_actor_rows") == 0.0,
                    f"trace optimizer row {line_number} is not 800+20+0 balanced",
                )
            else:
                episode_steps.append(step)
                thresholds = []
                for event_index in range(1, 6):
                    raw = row.get(f"train/event_{event_index}/threshold")
                    _require(
                        isinstance(raw, (int, float))
                        and not isinstance(raw, bool)
                        and math.isfinite(float(raw))
                        and 0.0 <= float(raw) <= 1.0,
                        f"trace row {line_number} has an invalid event "
                        f"{event_index} threshold",
                    )
                    thresholds.append(float(raw))
                observed_contexts.append(tuple(thresholds))
                _require(row.get("train/e1_sampler_mode") == mode, f"trace row {line_number} uses another sampler")
                _require(int(row.get("train/episode_length", 0)) == 205, f"trace row {line_number} has wrong horizon")
                expected_stratum = "all_equal" if mode == ALL_EQUAL_MODE else "uniform"
                _require(
                    row.get("train/e1_context_stratum") == expected_stratum,
                    f"trace row {line_number} has another context stratum",
                )
                if mode == ALL_EQUAL_MODE:
                    shared = row.get("train/e1_context_shared_value")
                    _require(
                        row.get("train/e1_context_entries_all_equal") == 1.0
                        and isinstance(shared, (int, float))
                        and math.isfinite(float(shared))
                        and 0.0 <= float(shared) <= 1.0
                        and row.get(
                            "train/e1_context_stratum_one_hot_all_equal"
                        ) == 1.0,
                        f"trace row {line_number} lacks all-equal evidence",
                    )
                    _require(
                        all(
                            abs(value - float(shared)) <= 1.0e-7
                            for value in thresholds
                        ),
                        f"trace row {line_number} thresholds are not the "
                        "reported shared all-equal context",
                    )
                else:
                    _require(
                        row.get(
                            "train/e1_context_stratum_one_hot_uniform"
                        ) == 1.0
                        and row.get(
                            "train/e1_context_stratum_one_hot_all_equal"
                        ) == 0.0
                        and len(set(thresholds)) > 1,
                        f"trace row {line_number} lacks independent-uniform "
                        "context evidence",
                    )
                expected_episode_count = (len(episode_steps) - 1) // 4 + 1
                _require(
                    row.get("train/e1_sampler_episode_count_per_env")
                    == float(expected_episode_count),
                    f"trace row {line_number} has the wrong per-env sampler "
                    "episode count",
                )
    _require(len(episode_steps) == expected_episodes, "training trace has the wrong episode count")
    _require(len(optimizer_steps) == expected_optimizers, "training trace has the wrong optimizer count")
    cadence = list(range(first_step, last_step + 1, 820))
    _require(
        Counter(episode_steps) == Counter({step: 4 for step in cadence}),
        "training trace does not contain exactly four episodes per 820-step clock",
    )
    _require(
        optimizer_steps == cadence,
        "training trace optimizer clocks are not exact monotone 820 cadence",
    )
    _require(
        observed_steps == sorted(observed_steps),
        "training trace clock is nonmonotone",
    )
    expected_contexts = _expected_training_contexts(
        mode, expected_episodes // 4
    )
    _require(
        len(observed_contexts) == len(expected_contexts)
        and all(
            np.allclose(
                observed, expected, rtol=0.0, atol=1.0e-7
            )
            for observed, expected in zip(
                observed_contexts, expected_contexts
            )
        ),
        "training trace event thresholds do not match the exact seeded "
        f"{mode} sampler stream",
    )
    return {"path": str(path), "sha256": sha256_file(path), "episode_rows": len(episode_steps), "optimizer_rows": len(optimizer_steps), "first_total_timesteps": min(episode_steps), "last_total_timesteps": max(episode_steps)}


def _canonical_builtin_evaluation_draw(*, seed, fixed_value=None):
    """Reproduce one built-in E1 evaluation context and event schedule."""

    from stackelberg_pomdp.atari.schedule import ExactFiveEventSchedule

    rng = np.random.default_rng(int(seed) + 74_711)
    if fixed_value is None:
        context = np.asarray(
            rng.uniform(0.0, 1.0, size=5), dtype=np.float32
        )
    else:
        context = np.full(5, float(fixed_value), dtype=np.float32)
    inner_seed = int(rng.integers(0, 2 ** 31 - 1))
    schedule = ExactFiveEventSchedule(
        gameplay_horizon=200, tail_steps=0
    ).sample(np.random.default_rng(inner_seed))
    return (
        tuple(float(value) for value in context),
        tuple(int(value) for value in schedule),
    )


def _validate_builtin_evaluation_design(
        rows, *, episodes, seed_start, fixed_value=None, label,
):
    _require(len(rows) == episodes, f"{label} has the wrong episode count")
    _require(
        [row.get("evaluation_episode") for row in rows]
        == list(range(episodes)),
        f"{label} episode indices changed",
    )
    for episode, row in enumerate(rows):
        expected_context, expected_schedule = (
            _canonical_builtin_evaluation_draw(
                seed=seed_start + episode, fixed_value=fixed_value
            )
        )
        actual_context = row.get("opponent_commitment", [])
        _require(
            len(actual_context) == 5
            and np.allclose(
                actual_context, expected_context, rtol=0.0, atol=1.0e-7
            ),
            f"{label} episode {episode} commitment differs from its exact "
            "seeded stream",
        )
        _require(
            tuple(row.get("event_steps", ())) == expected_schedule,
            f"{label} episode {episode} schedule differs from its exact "
            "seeded stream",
        )


def _recompute_builtin_evaluation_summary(rows):
    from replication.atari import train_atari_meta_response_sb3 as trainer
    from replication.atari.sb3_common import summarize_episodes

    summary = summarize_episodes(rows)
    summary.update(trainer._trade_diagnostics(
        rows, gameplay_horizon=200
    ))
    return _canonical_json(summary)


def _validate_builtin_evaluation_summary(reported, rows, *, label):
    recomputed = _recompute_builtin_evaluation_summary(rows)
    _require(
        reported == recomputed,
        f"{label} summary does not recompute from episode rows",
    )
    return recomputed


def _validate_evaluation(
        path, *, current, history, expected_step, checkpoint_metadata,
        random_episodes, fixed_episodes,
):
    path = Path(path).expanduser().resolve()
    checkpoint_path = Path(checkpoint_metadata["path"]).expanduser().resolve()
    expected_path = checkpoint_path.with_name(
        f"{checkpoint_path.stem}.evaluation.json"
    )
    _same_path(str(path), expected_path, label="evaluation sidecar")
    _require(
        checkpoint_metadata.get("training_timesteps") == expected_step,
        "evaluation endpoint metadata has the wrong timestep",
    )
    _require(
        sha256_file(checkpoint_path) == checkpoint_metadata.get("sha256"),
        "evaluation endpoint checkpoint bytes changed",
    )
    value = load_json(path)
    provenance = value.get("provenance", {})
    _require(provenance.get("training_sampler") == current, "evaluation current sampler changed")
    _require(provenance.get("training_sampler_history") == history, "evaluation sampler history changed")
    _require(provenance.get("evaluation_sampler") == canonical_uniform_sampler(), "evaluation is not canonical uniform")
    _require(provenance.get("actor_loss_mode") == ACTOR_LOSS_MODE, "evaluation actor-loss mode changed")
    _require(
        provenance.get("e0b_source", {}).get("sha256")
        == CANONICAL_E0B_SHA256,
        "evaluation E0b source changed",
    )
    _require(
        provenance.get("economic_head_initialization")
        == {"mean": 0.5, "concentration": 2.0},
        "evaluation economic initialization changed",
    )
    _require(provenance.get("target_kl") is None, "evaluation target KL changed")
    summary = value.get("summary")
    random = value.get("random", {})
    _require(isinstance(summary, dict), "evaluation has no summary")
    _require(random.get("summary") == summary, "evaluation random summary changed")
    random_rows = random.get("episode_rows", [])
    _validate_builtin_evaluation_design(
        random_rows,
        episodes=random_episodes,
        seed_start=300_001,
        label="evaluation random",
    )
    _validate_builtin_evaluation_summary(
        summary, random_rows, label="evaluation random"
    )
    fixed_summaries = value.get("fixed_contexts", [])
    fixed = value.get("fixed_context_evaluations", [])
    _require(len(fixed_summaries) == len(fixed) == 11, "evaluation fixed grid is incomplete")
    expected_values = [index / 10 for index in range(11)]
    _require(
        [row.get("opponent_value") for row in fixed] == expected_values,
        "evaluation fixed-grid values changed",
    )
    from replication.atari import evaluate_atari_meta_response_sb3 as evaluator

    all_rows = list(random_rows)
    reference_event_steps = None
    for index, (evaluation, compact, opponent_value) in enumerate(
            zip(fixed, fixed_summaries, expected_values)
    ):
        rows = evaluation.get("episode_rows", [])
        _validate_builtin_evaluation_design(
            rows,
            episodes=fixed_episodes,
            seed_start=400_001,
            fixed_value=opponent_value,
            label=f"evaluation fixed {opponent_value:.1f}",
        )
        _validate_builtin_evaluation_summary(
            evaluation.get("summary"), rows,
            label="evaluation fixed-grid",
        )
        _require(
            compact == {"opponent_value": opponent_value, **evaluation.get("summary", {})},
            "evaluation fixed compact/full summaries disagree",
        )
        schedules = [row.get("event_steps") for row in rows]
        if reference_event_steps is None:
            reference_event_steps = schedules
        else:
            _require(
                schedules == reference_event_steps,
                "evaluation fixed grid is not paired on event schedules",
            )
        all_rows.extend(rows)
    violations = [
        violation
        for row in all_rows
        for violation in evaluator.audit_episode(row, role=ROLE)
    ]
    _require(not violations, f"evaluation mechanics failed: {violations[:3]}")
    _require(summary.get("episodes") == random_episodes, "evaluation summary episode count changed")
    for record in (summary, *fixed_summaries):
        for key, raw in record.items():
            if isinstance(raw, float):
                _require(math.isfinite(raw), f"evaluation contains nonfinite {key}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "evidence_separation": (
            "evaluation JSON binds provenance/counts/mechanics; adjacent loaded "
            "checkpoint metadata independently binds path/SHA/timestep"
        ),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_metadata["sha256"],
            "training_timesteps": expected_step,
        },
        "random_episodes": random_episodes,
        "fixed_values": expected_values,
        "fixed_episodes_per_value": fixed_episodes,
        "mechanics_passed": True,
    }


def build_training_family(args):
    activation_path = Path(args.activation).expanduser().resolve()
    activation = validate_activation(activation_path, code_root=args.code_root)
    warmup = _load_candidate_metadata(args.warmup_checkpoint, activation=activation, device=args.device)
    warmup["resume_source"] = warmup.get("resume_source")
    _validate_warmup_metadata(warmup, activation)
    warmup_digest = warmup["sha256"]
    warmup_probe = validate_warmup_probe(
        activation=activation,
        probe=args.warmup_probe,
        checkpoint=args.warmup_checkpoint,
    )
    _require(
        warmup_probe["checkpoint_sha256"] == warmup_digest,
        "warm-up probe is bound to different checkpoint bytes",
    )
    candidate_paths = [str(Path(path).expanduser().resolve()) for path in args.candidate]
    _require(candidate_paths == activation["protocol"]["target"]["candidate_paths"], "target family differs from preregistration")
    _require(str(Path(args.warmup_checkpoint).resolve()) not in candidate_paths, "warm-up entered selectable candidates")
    candidates = []
    for index, (path, step) in enumerate(zip(candidate_paths, TARGET_CANDIDATE_TIMESTEPS)):
        metadata = _load_candidate_metadata(path, activation=activation, device=args.device)
        _validate_target_metadata(
            metadata, expected_path=path, expected_step=step,
            activation=activation, warmup_digest=warmup_digest,
            final=index == len(candidate_paths) - 1,
        )
        candidates.append(metadata)
    hashes = [item["sha256"] for item in candidates]
    _require(len(set(hashes)) == 6, "target candidates are not byte-distinct")
    warmup_trace = _validate_training_trace(
        args.warmup_training_log, mode=ALL_EQUAL_MODE, first_step=820,
        last_step=WARMUP_TIMESTEPS, expected_episodes=1_952,
        expected_optimizers=488, checkpoint=args.warmup_checkpoint,
    )
    target_trace = _validate_training_trace(
        args.target_training_log, mode=UNIFORM_MODE,
        first_step=WARMUP_TIMESTEPS + 820, last_step=FINAL_TIMESTEPS,
        expected_episodes=9_760, expected_optimizers=2_440,
        checkpoint=args.target_checkpoint,
    )
    warmup_eval = _validate_evaluation(
        args.warmup_evaluation,
        current=canonical_all_equal_sampler(),
        history=warmup["atari_e1_sampler_history"],
        expected_step=WARMUP_TIMESTEPS,
        checkpoint_metadata=warmup,
        random_episodes=20,
        fixed_episodes=5,
    )
    target_eval = _validate_evaluation(
        args.target_evaluation,
        current=canonical_uniform_sampler(),
        history=candidates[0]["atari_e1_sampler_history"],
        expected_step=FINAL_TIMESTEPS,
        checkpoint_metadata=candidates[-1],
        random_episodes=100,
        fixed_episodes=20,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": FAMILY_KIND,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "activation": {"path": str(activation_path), "sha256": sha256_file(activation_path)},
        "warmup": {
            "selectable": False,
            "metadata": warmup,
            "conditioning_probe": warmup_probe,
            "training_trace": warmup_trace,
            "evaluation": warmup_eval,
        },
        "target": {"candidate_metadata": candidates, "candidate_sha256": hashes, "training_trace": target_trace, "evaluation": target_eval},
    }


def _validate_family_contract(value, activation):
    _require(value.get("schema_version") == SCHEMA_VERSION and value.get("kind") == FAMILY_KIND, "unknown seller-recovery family")
    _require(value.get("passed") is True, "seller-recovery family did not pass")
    warmup = value.get("warmup", {})
    _require(warmup.get("selectable") is False, "warm-up became selectable")
    warmup_metadata = warmup.get("metadata", {})
    _validate_warmup_metadata(warmup_metadata, activation)
    _require(sha256_file(warmup_metadata.get("path", "")) == warmup_metadata.get("sha256"), "warm-up checkpoint changed")
    probe_record = warmup.get("conditioning_probe", {})
    probe = validate_warmup_probe(
        activation=activation,
        probe=probe_record.get("path", ""),
        checkpoint=warmup_metadata.get("path", ""),
    )
    _require(probe_record == probe, "family warm-up probe record changed")
    _require(
        probe["checkpoint_sha256"] == warmup_metadata.get("sha256"),
        "family probe/checkpoint SHA mismatch",
    )
    candidates = value.get("target", {}).get("candidate_metadata", [])
    hashes = value.get("target", {}).get("candidate_sha256", [])
    _require(len(candidates) == len(hashes) == 6 and len(set(hashes)) == 6, "target family is not six byte-distinct checkpoints")
    expected_paths = activation["protocol"]["target"]["candidate_paths"]
    _require([item.get("path") for item in candidates] == expected_paths, "family target paths changed")
    _require([item.get("training_timesteps") for item in candidates] == list(TARGET_CANDIDATE_TIMESTEPS), "family target clocks changed")
    _require(str(Path(warmup_metadata["path"]).resolve()) not in expected_paths, "warm-up entered family")
    for index, (item, path, step, digest) in enumerate(zip(candidates, expected_paths, TARGET_CANDIDATE_TIMESTEPS, hashes)):
        _require(sha256_file(path) == item.get("sha256") == digest, "target candidate bytes changed")
        _validate_target_metadata(
            item, expected_path=path, expected_step=step, activation=activation,
            warmup_digest=warmup_metadata["sha256"], final=index == 5,
        )
    warmup_trace = _validate_training_trace(
        warmup.get("training_trace", {}).get("path", ""),
        mode=ALL_EQUAL_MODE, first_step=820,
        last_step=WARMUP_TIMESTEPS, expected_episodes=1_952,
        expected_optimizers=488, checkpoint=warmup_metadata["path"],
    )
    _require(
        warmup.get("training_trace") == warmup_trace,
        "family warm-up trace record changed",
    )
    target = value["target"]
    target_trace = _validate_training_trace(
        target.get("training_trace", {}).get("path", ""),
        mode=UNIFORM_MODE, first_step=WARMUP_TIMESTEPS + 820,
        last_step=FINAL_TIMESTEPS, expected_episodes=9_760,
        expected_optimizers=2_440,
        checkpoint=candidates[-1]["path"],
    )
    _require(
        target.get("training_trace") == target_trace,
        "family target trace record changed",
    )
    warmup_evaluation = _validate_evaluation(
        warmup.get("evaluation", {}).get("path", ""),
        current=canonical_all_equal_sampler(),
        history=warmup_metadata["atari_e1_sampler_history"],
        expected_step=WARMUP_TIMESTEPS,
        checkpoint_metadata=warmup_metadata,
        random_episodes=20,
        fixed_episodes=5,
    )
    _require(
        warmup.get("evaluation") == warmup_evaluation,
        "family warm-up evaluation record changed",
    )
    target_evaluation = _validate_evaluation(
        target.get("evaluation", {}).get("path", ""),
        current=canonical_uniform_sampler(),
        history=candidates[0]["atari_e1_sampler_history"],
        expected_step=FINAL_TIMESTEPS,
        checkpoint_metadata=candidates[-1],
        random_episodes=100,
        fixed_episodes=20,
    )
    _require(
        target.get("evaluation") == target_evaluation,
        "family target evaluation record changed",
    )
    return value


def validate_training_family(path):
    path = Path(path).expanduser().resolve()
    value = load_json(path)
    activation_record = value.get("activation", {})
    _require(sha256_file(activation_record.get("path", "")) == activation_record.get("sha256"), "family activation changed")
    activation = validate_activation(activation_record.get("path", ""))
    return _validate_family_contract(value, activation)


def write_training_family(args):
    output = Path(args.output).expanduser().resolve()
    _require(not os.path.lexists(output), f"refusing to overwrite recovery family: {output}")
    value = build_training_family(args)
    atomic_write_json(output, value)
    return validate_training_family(output)


def canonical_selector_environment(activation):
    return {
        "gameplay_horizon": 200,
        "event_tail_steps": 0,
        "random_event_steps": None,
        "fixed_grid_event_steps": [20, 50, 80, 110, 140],
        "paired_timing_first_four_event_steps": [20, 50, 80, 110],
        "paired_timing_fifth_event_steps": {"early": 140, "late": 195},
        "paired_timing_fifth_prices": [0.5, 0.75, 0.9],
        "paired_timing_calibration_price": 0.75,
        "seller_game_reward_scale": 0.1,
        "buyer_game_reward_scale": 1.0,
        "noop_max": 30,
        "frame_skip": 4,
        "frame_stack": 4,
        "episodic_life": True,
        "clip_game_rewards": True,
        "max_frames": 100_000,
        "rom_path": str(Path(activation["rom"]["path"]).resolve()),
        "rom_sha256": CANONICAL_ROM_SHA256,
    }


def _validate_selector_environment(value, activation):
    _require(
        value == canonical_selector_environment(activation),
        "selector environment differs from the canonical Atari recovery "
        "configuration",
    )


def validate_selection(args):
    family = validate_training_family(args.family)
    report_path = Path(args.report).expanduser().resolve()
    _require(report_path.name == REPORT_NAME, "recovery selector report name changed")
    report = load_json(report_path)
    _require(report.get("role") == ROLE and report.get("evaluator") == EVALUATOR, "recovery selector role/evaluator changed")
    protocol = report.get("protocol", {})
    expected = {
        "screen_episodes": 20, "screen_seed_start": SCREEN_SEED_START,
        "confirmation_episodes": 100, "confirmation_seed_start": CONFIRMATION_SEED_START,
        "fixed_context_episodes": 20, "fixed_context_seed_start": FIXED_SEED_START,
        "confirmation_policy": "screen_winner_only_no_fallback",
    }
    for key, expected_value in expected.items():
        _require(protocol.get(key) == expected_value, f"recovery selector protocol changed: {key}")
    activation = validate_activation(family["activation"]["path"])
    _require(report.get("immutable_evaluation", {}).get("selector_code_revision") == activation["code_revision"], "selector code revision differs from activation")
    _require(report.get("immutable_evaluation", {}).get("e0b_sha256") == CANONICAL_E0B_SHA256, "selector E0b changed")
    _require(report.get("immutable_evaluation", {}).get("rom_sha256") == CANONICAL_ROM_SHA256, "selector ROM changed")
    _validate_selector_environment(report.get("environment"), activation)
    from replication.atari import evaluate_atari_meta_response_sb3 as evaluator

    screen = report.get("screen", {})
    common = screen.get("common_pairing", {})
    _require(common.get("passed") is True and common.get("candidates_checked") == 6, "recovery common screen failed")
    _require([row.get("evaluation_seed") for row in common.get("seed_context_pairs", [])] == list(range(SCREEN_SEED_START, SCREEN_SEED_START + 20)), "recovery screen seeds changed")
    results = screen.get("results", [])
    family_hashes = family["target"]["candidate_sha256"]
    _require(len(results) == 6, "recovery selector did not screen six candidates")
    _require([row.get("metadata", {}).get("sha256") for row in results] == family_hashes, "selector screened another family")
    _require([row.get("metadata", {}).get("training_timesteps") for row in results] == list(TARGET_CANDIDATE_TIMESTEPS), "selector included a warm-up or wrong target clock")
    _require(report.get("immutable_evaluation", {}).get("candidate_sha256") == family_hashes, "selector immutable family changed")
    screen_seeds = list(range(SCREEN_SEED_START, SCREEN_SEED_START + 20))
    screen_contexts = [evaluator.random_context(seed) for seed in screen_seeds]
    recomputed_common = evaluator.validate_common_screen(
        results, seeds=screen_seeds, contexts=screen_contexts
    )
    _require(common == recomputed_common, "recovery common screen does not recompute")
    screen_violations = []
    expected_metadata = family["target"]["candidate_metadata"]
    for index, (result, family_metadata) in enumerate(
            zip(results, expected_metadata)
    ):
        metadata = result.get("metadata", {})
        expected = {
            key: value for key, value in family_metadata.items()
            if key not in {"optimizer_adam_step", "resume_source"}
        }
        _require(metadata == expected, f"screen candidate {index} metadata changed")
        rows = result.get("episode_rows", [])
        _require(len(rows) == 20, f"screen candidate {index} lacks 20 rows")
        violations = [
            violation for row in rows
            for violation in evaluator.audit_episode(row, role=ROLE)
        ]
        screen_violations.extend(violations)
        _require(
            result.get("protocol")
            == {"passed": not violations, "violations": violations},
            f"screen candidate {index} protocol does not recompute",
        )
        _require(not violations, f"screen candidate {index} mechanics failed")
        _require(
            result.get("summary") == evaluator._summary(rows, role=ROLE),
            f"screen candidate {index} summary does not recompute",
        )
    _require(not screen_violations, "recovery screen mechanics failed")
    first = family["target"]["candidate_metadata"][0]
    report_family = report.get("training_family", {})
    _require(report_family.get("common_sampler_provenance") == first["atari_e1_sampler_provenance"], "selector current sampler changed")
    _require(report_family.get("common_sampler_history") == first["atari_e1_sampler_history"], "selector mixed history changed")
    ranking = report.get("ranking", [])
    _, recomputed_ranking = evaluator.rank_candidates(results)
    _require(ranking == recomputed_ranking, "selector ranking does not recompute")
    _require(len(ranking) == 6 and [row.get("rank") for row in ranking] == list(range(1, 7)), "selector ranking is malformed")
    winner = ranking[0].get("checkpoint_sha256")
    attempts = report.get("confirmation_attempts", [])
    _require(len(attempts) == 1 and attempts[0].get("metadata", {}).get("sha256") == winner, "selector did not confirm only the screen winner")
    _require([row.get("evaluation_seed") for row in attempts[0].get("random", {}).get("episode_rows", [])] == list(range(CONFIRMATION_SEED_START, CONFIRMATION_SEED_START + 100)), "recovery confirmation seeds changed")
    random_result = attempts[0].get("random", {})
    random_rows = random_result.get("episode_rows", [])
    expected_random_contexts = {
        seed: tuple(float(value) for value in evaluator.random_context(seed))
        for seed in range(
            CONFIRMATION_SEED_START,
            CONFIRMATION_SEED_START + 100,
        )
    }
    _require(
        {
            int(row.get("evaluation_seed")): tuple(row.get("opponent_commitment", []))
            for row in random_rows
        } == expected_random_contexts,
        "recovery confirmation random contexts changed",
    )
    random_violations = [
        violation for row in random_rows
        for violation in evaluator.audit_episode(row, role=ROLE)
    ]
    _require(
        random_result.get("protocol")
        == {"passed": not random_violations, "violations": random_violations},
        "recovery random confirmation protocol does not recompute",
    )
    _require(not random_violations, "recovery random confirmation mechanics failed")
    _require(
        random_result.get("summary") == evaluator._summary(random_rows, role=ROLE),
        "recovery random confirmation summary does not recompute",
    )
    fixed = attempts[0].get("fixed_contexts", [])
    _require(len(fixed) == 11, "recovery fixed grid is incomplete")
    for value_index, row in enumerate(fixed):
        value = value_index / 10
        _require(row.get("opponent_value") == value, "recovery fixed-grid value changed")
        _require([episode.get("evaluation_seed") for episode in row.get("episode_rows", [])] == list(range(FIXED_SEED_START, FIXED_SEED_START + 20)), "recovery fixed-grid seeds changed")
        rows = row.get("episode_rows", [])
        _require(
            all(
                all(abs(float(item) - value) <= 1.0e-6 for item in episode.get("opponent_commitment", []))
                and episode.get("event_steps") == [20, 50, 80, 110, 140]
                for episode in rows
            ),
            "recovery fixed-grid contexts/schedule changed",
        )
        violations = [
            violation for episode in rows
            for violation in evaluator.audit_episode(episode, role=ROLE)
        ]
        _require(
            row.get("protocol")
            == {"passed": not violations, "violations": violations},
            "recovery fixed-grid protocol does not recompute",
        )
        _require(not violations, "recovery fixed-grid mechanics failed")
        _require(
            row.get("summary") == evaluator._summary(rows, role=ROLE),
            "recovery fixed-grid summary does not recompute",
        )
    recomputed_gate = evaluator.behavioral_gate(
        role=ROLE,
        random_result=random_result,
        fixed_results=fixed,
        timing_results=[],
    )
    _require(
        attempts[0].get("behavioral_gate") == recomputed_gate,
        "seller behavioral gate does not recompute from confirmation data",
    )
    selection = report.get("selection", {})
    _require(selection.get("fallback_allowed") is False, "recovery selector permits fallback")
    _require(selection.get("screen_selected_checkpoint_sha256") == winner, "recovery screen winner record changed")
    selected = Path(args.selected).expanduser().resolve()
    if report.get("passed") is True:
        _require(recomputed_gate.get("passed") is True, "passing selector has a failed seller gate")
        _require(selection.get("selected_checkpoint_sha256") == winner, "passing selector chose a nonwinner")
        _require(sha256_file(selected) == winner, "selected recovery alias changed")
        _same_path(report.get("selected_alias", {}).get("pinned_path"), selected, label="recovery selected alias")
    else:
        _require(report.get("passed") is False, "selector report has no Boolean outcome")
        _require(recomputed_gate.get("passed") is False, "failed selector contains a passing gate")
        _require(report.get("selected_alias") is None and not os.path.lexists(selected), "failed selector retained an alias")
    _same_path(report.get("artifacts", {}).get("json"), report_path, label="recovery report artifact")
    return report


def build_selection_gate(args, report):
    _require(report.get("passed") is True, "only a passing recovery selection can open E2")
    family_path = Path(args.family).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    selected = Path(args.selected).expanduser().resolve()
    family = validate_training_family(family_path)
    activation_path = Path(family["activation"]["path"]).resolve()
    activation = validate_activation(activation_path)
    winner = report["selection"]["screen_selected_checkpoint_sha256"]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": GATE_KIND,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "role": ROLE,
        "actor_loss_mode": ACTOR_LOSS_MODE,
        "source_kind": SOURCE_KIND,
        "sampler_mode": UNIFORM_MODE,
        "report": {"path": str(report_path), "sha256": sha256_file(report_path), "evaluator": EVALUATOR},
        "selected_checkpoint": {"path": str(selected), "sha256": winner},
        "training_family": {"path": str(family_path), "sha256": sha256_file(family_path), "candidate_sha256": family["target"]["candidate_sha256"]},
        "activation": {"path": str(activation_path), "sha256": sha256_file(activation_path), "code_revision": activation["code_revision"]},
        "prerequisite_failure": dict(activation["prerequisite_failure"]),
        "seller_release": dict(activation["seller_release"]),
        "warmup_probe": dict(family["warmup"]["conditioning_probe"]),
        "selection": {
            "screen_selected_checkpoint_sha256": winner,
            "confirmed_checkpoint_sha256": report["confirmation_attempts"][0]["metadata"]["sha256"],
            "selected_checkpoint_sha256": report["selection"]["selected_checkpoint_sha256"],
            "confirmation_attempts": 1,
            "confirmation_policy": "screen_winner_only_no_fallback",
            "fallback_allowed": False,
        },
    }


def validate_selection_gate(path, *, family=None, report=None, selected=None):
    path = Path(path).expanduser().resolve()
    _require(path.name == GATE_NAME, "seller-recovery gate name changed")
    value = load_json(path)
    _require(value.get("schema_version") == SCHEMA_VERSION and value.get("kind") == GATE_KIND, "unknown seller-recovery gate")
    _require(value.get("passed") is True and value.get("role") == ROLE, "seller-recovery gate did not pass")
    _require(value.get("actor_loss_mode") == ACTOR_LOSS_MODE and value.get("source_kind") == SOURCE_KIND and value.get("sampler_mode") == UNIFORM_MODE, "seller-recovery gate identity changed")
    family_path = Path(value.get("training_family", {}).get("path", "")).resolve()
    report_path = Path(value.get("report", {}).get("path", "")).resolve()
    selected_path = Path(value.get("selected_checkpoint", {}).get("path", "")).resolve()
    if family is not None:
        _same_path(str(family_path), family, label="gate family")
    if report is not None:
        _same_path(str(report_path), report, label="gate report")
    if selected is not None:
        _same_path(str(selected_path), selected, label="gate selected checkpoint")
    _require(sha256_file(family_path) == value["training_family"]["sha256"], "gate family changed")
    _require(sha256_file(report_path) == value["report"]["sha256"], "gate report changed")
    _require(sha256_file(selected_path) == value["selected_checkpoint"]["sha256"], "gate selected checkpoint changed")
    family_value = validate_training_family(family_path)
    _require(value["training_family"]["candidate_sha256"] == family_value["target"]["candidate_sha256"], "gate candidate family changed")
    activation_path = Path(value["activation"]["path"]).resolve()
    activation = validate_activation(activation_path)
    _require(sha256_file(activation_path) == value["activation"]["sha256"], "gate activation changed")
    _require(activation["code_revision"] == value["activation"]["code_revision"], "gate activation revision changed")
    _require(value["prerequisite_failure"] == activation["prerequisite_failure"], "gate prerequisite failure changed")
    _require(value["seller_release"] == activation["seller_release"], "gate seller release changed")
    _require(
        value.get("warmup_probe")
        == family_value["warmup"]["conditioning_probe"],
        "gate warm-up probe changed",
    )
    report_value = validate_selection(SimpleNamespace(family=str(family_path), report=str(report_path), selected=str(selected_path)))
    expected = build_selection_gate(SimpleNamespace(family=str(family_path), report=str(report_path), selected=str(selected_path)), report_value)
    for key in (
        "passed", "role", "actor_loss_mode", "source_kind", "sampler_mode",
        "report", "selected_checkpoint", "training_family", "activation",
        "prerequisite_failure", "seller_release", "warmup_probe", "selection",
    ):
        _require(value.get(key) == expected.get(key), f"seller-recovery gate differs in {key}")
    return value


def write_or_validate_selection_gate(args, report):
    output = Path(args.gate_output).expanduser().resolve()
    if report.get("passed") is False:
        _require(not os.path.lexists(output), "failed recovery selector retained a gate")
        return None
    expected = build_selection_gate(args, report)
    if output.exists():
        existing = validate_selection_gate(output, family=args.family, report=args.report, selected=args.selected)
        for key in expected:
            if key != "created_utc":
                _require(existing.get(key) == expected.get(key), f"existing recovery gate differs in {key}")
        return existing
    atomic_write_json(output, expected)
    return validate_selection_gate(output, family=args.family, report=args.report, selected=args.selected)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    activation = subparsers.add_parser("activation")
    for name in (
        "failed-report", "seller-release", "e0b", "rom",
        "warmup-checkpoint", "warmup-probe", "target-checkpoint",
        "code-root", "output",
    ):
        activation.add_argument(f"--{name}", required=True)
    validate = subparsers.add_parser("validate-activation")
    validate.add_argument("--activation", required=True)
    validate.add_argument("--code-root")
    family = subparsers.add_parser("training-family")
    family.add_argument("--activation", required=True)
    family.add_argument("--code-root", required=True)
    family.add_argument("--warmup-checkpoint", required=True)
    family.add_argument("--warmup-probe", required=True)
    family.add_argument("--target-checkpoint", required=True)
    family.add_argument("--candidate", action="append", required=True)
    family.add_argument("--warmup-training-log", required=True)
    family.add_argument("--target-training-log", required=True)
    family.add_argument("--warmup-evaluation", required=True)
    family.add_argument("--target-evaluation", required=True)
    family.add_argument("--output", required=True)
    family.add_argument("--device", default="cpu")
    validate_family = subparsers.add_parser("validate-training-family")
    validate_family.add_argument("--family", required=True)
    probe = subparsers.add_parser("validate-warmup-probe")
    probe.add_argument("--activation", required=True)
    probe.add_argument("--probe", required=True)
    probe.add_argument("--checkpoint")
    warmup_checkpoint = subparsers.add_parser("validate-warmup-checkpoint")
    warmup_checkpoint.add_argument("--activation", required=True)
    warmup_checkpoint.add_argument("--checkpoint", required=True)
    warmup_checkpoint.add_argument("--device", default="cpu")
    warmup_stage = subparsers.add_parser("validate-warmup-stage")
    warmup_stage.add_argument("--activation", required=True)
    warmup_stage.add_argument("--probe", required=True)
    warmup_stage.add_argument("--checkpoint", required=True)
    warmup_stage.add_argument("--training-log", required=True)
    warmup_stage.add_argument("--evaluation", required=True)
    warmup_stage.add_argument("--device", default="cpu")
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
    if args.command == "activation":
        result = write_or_validate_activation(args)
    elif args.command == "validate-activation":
        result = validate_activation(args.activation, code_root=args.code_root)
    elif args.command == "training-family":
        result = write_training_family(args)
    elif args.command == "validate-training-family":
        result = validate_training_family(args.family)
    elif args.command == "validate-warmup-probe":
        result = validate_warmup_probe(
            activation=args.activation,
            probe=args.probe,
            checkpoint=args.checkpoint,
        )
    elif args.command == "validate-warmup-checkpoint":
        result = validate_warmup_checkpoint(
            activation=args.activation,
            checkpoint=args.checkpoint,
            device=args.device,
        )
    elif args.command == "validate-warmup-stage":
        result = validate_warmup_stage(
            activation=args.activation,
            probe=args.probe,
            checkpoint=args.checkpoint,
            training_log=args.training_log,
            evaluation=args.evaluation,
            device=args.device,
        )
    elif args.command == "selection":
        report = validate_selection(args)
        result = (
            write_or_validate_selection_gate(args, report)
            if args.gate_output else report
        )
        if result is None:
            result = report
    else:
        result = validate_selection_gate(
            args.gate, family=args.family, report=args.report,
            selected=args.selected,
        )
    print(json.dumps({
        "command": args.command,
        "passed": bool(result.get("passed", True)),
        "kind": result.get("kind"),
        "value": result,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
