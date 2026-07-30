#!/Users/gbrero/miniconda3/envs/stackelbergPOMDP/bin/python
"""Certify and select the six formal Atari seller-v5 checkpoints.

The diagnostics gate is an authorization to start training, not an E1
release.  This validator separately binds the completed formal family, screens
all six immutable checkpoints on common held-out episodes, and confirms only
the screen winner on fresh seeds.  A selected alias and E2-visible gate are
published only after the unchanged formal seller-v5 scientific gate passes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

import numpy as np

from replication.atari import evaluate_atari_meta_response_sb3 as evaluator
from replication.atari import probe_atari_e1_seller_shared_context as probe
from replication.atari.automation import (
    validate_atari_e1_seller_shared_context_v5 as diagnostics,
)


SCHEMA_VERSION = 1
FAMILY_KIND = "stackpomdp.atari.e1_seller_shared_context_formal_family.v5"
SELECTION_GATE_KIND = (
    "stackpomdp.atari.e1_seller_shared_context_selection_gate.v5"
)
ROLE = "seller"
ACTOR_LOSS_MODE = "balanced"
EVALUATOR = evaluator.EVALUATOR_NAME
FORMAL_TIMESTEPS = diagnostics.FORMAL_TIMESTEPS
FORMAL_STEP_TIMESTEPS = diagnostics.FORMAL_STEP_TIMESTEPS
FORMAL_CANDIDATE_TIMESTEPS = (*FORMAL_STEP_TIMESTEPS, FORMAL_TIMESTEPS)
FORMAL_CANDIDATE_SNAPSHOT_PHASES = (
    *(
        diagnostics.RETAINED_PRE_UPDATE_SNAPSHOT
        for _ in FORMAL_STEP_TIMESTEPS
    ),
    diagnostics.POST_UPDATE_SNAPSHOT,
)

SELECTION_PROTOCOLS = {
    diagnostics.STANDARD_PROTOCOL: {
        "screen_seed_start": 12_000_001,
        "confirmation_seed_start": 12_100_001,
        "fixed_seed_start": 12_200_001,
        "timing_seed_start": 12_300_001,
    },
    diagnostics.EXPOSURE_PROTOCOL: {
        "screen_seed_start": 13_000_001,
        "confirmation_seed_start": 13_100_001,
        "fixed_seed_start": 13_200_001,
        "timing_seed_start": 13_300_001,
    },
}

_require = diagnostics._require
sha256_file = diagnostics.sha256_file
load_json = diagnostics.load_json
atomic_write_json = diagnostics.atomic_write_json
_same_path = diagnostics._same_path
_canonical_json = diagnostics._canonical_json


def configure_protocol(name):
    """Select one exact v5 training/selection artifact namespace."""

    diagnostics.configure_protocol(name)
    if name not in SELECTION_PROTOCOLS:
        raise ValueError(f"unknown seller-v5 selection protocol: {name}")
    config = SELECTION_PROTOCOLS[name]
    global ACTIVE_PROTOCOL
    global SOURCE_KIND
    global TOKEN
    global DIAGNOSTICS_GATE_NAME
    global FORMAL_CHECKPOINT_NAME
    global FAMILY_NAME
    global REPORT_NAME
    global SELECTION_GATE_NAME
    global SELECTED_NAME
    global SCREEN_SEED_START
    global CONFIRMATION_SEED_START
    global FIXED_SEED_START
    global TIMING_SEED_START

    ACTIVE_PROTOCOL = name
    SOURCE_KIND = diagnostics.SOURCE_KIND
    TOKEN = diagnostics.TOKEN
    DIAGNOSTICS_GATE_NAME = diagnostics.GATE_NAME
    FORMAL_CHECKPOINT_NAME = diagnostics.FORMAL_CHECKPOINT_NAME
    FAMILY_NAME = f"e1_seller_{TOKEN}_formal_family.json"
    REPORT_NAME = f"e1_seller_{TOKEN}_all6_selector_v1.json"
    SELECTION_GATE_NAME = f"e1_seller_{TOKEN}_all6_selector_v1.gate.json"
    SELECTED_NAME = (
        f"meta_seller_e1_ppo_balanced_{TOKEN}_seed1_selected.zip"
    )
    SCREEN_SEED_START = config["screen_seed_start"]
    CONFIRMATION_SEED_START = config["confirmation_seed_start"]
    FIXED_SEED_START = config["fixed_seed_start"]
    TIMING_SEED_START = config["timing_seed_start"]
    return dict(config)


configure_protocol(diagnostics.DEFAULT_PROTOCOL)


def _git_revision(code_root):
    root = Path(code_root).expanduser().resolve()
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _require(
        re.fullmatch(r"[0-9a-f]{40}", revision) is not None,
        "seller-v5 selector code revision is invalid",
    )
    return revision


def _git_scoped_clean(code_root):
    root = Path(code_root).expanduser().resolve()
    module_root = Path(__file__).resolve().parents[3]
    _require(
        os.path.samefile(root, module_root),
        "seller-v5 selection validator is not executing from --code-root",
    )
    paths = (
        "replication/atari/train_atari_meta_response_sb3.py",
        "replication/atari/evaluate_atari_meta_response_sb3.py",
        "replication/atari/probe_atari_e1_seller_shared_context.py",
        "replication/atari/sb3_common.py",
        "replication/atari/automation/atari_e1_seller_shared_context_v5_common.zsh",
        "replication/atari/automation/atari_e1_seller_shared_context_v5_selection_common.zsh",
        "replication/atari/automation/run_e1_seller_shared_context_v5_selector.sh",
        "replication/atari/automation/run_e1_seller_shared_context_v5_exposure_v2_selector.sh",
        "replication/atari/automation/validate_atari_e1_seller_shared_context_v5.py",
        "replication/atari/automation/validate_atari_e1_seller_shared_context_v5_selection.py",
        "stackelberg_pomdp/atari",
    )
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--", *paths],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _require(not dirty, f"refusing dirty seller-v5 selection code: {dirty}")


def _exact_name(path, expected, label):
    _require(Path(path).name == expected, f"{label} filename changed")


def _formal_evaluation(path, *, metadata):
    record = diagnostics.shared._validate_evaluation(
        path,
        current=diagnostics.canonical_sampler(),
        history=diagnostics.canonical_sampler_history(),
        expected_step=FORMAL_TIMESTEPS,
        checkpoint_metadata=metadata,
        random_episodes=100,
        fixed_episodes=20,
    )
    value = load_json(path)
    provenance = value.get("provenance", {})
    _require(
        provenance.get("economic_architecture")
        == diagnostics.canonical_architecture(),
        "seller-v5 formal evaluation architecture changed",
    )
    _require(
        provenance.get("shared_context_initialization")
        == diagnostics.canonical_initialization(),
        "seller-v5 formal evaluation initialization changed",
    )
    _require(
        provenance.get("e1_training_code_revision")
        == metadata.get("e1_training_code_revision"),
        "seller-v5 formal evaluation training revision changed",
    )
    return record


def _candidate_metadata(
        path, *, e0b, training_revision, timesteps, snapshot_phase,
):
    metadata = diagnostics._load_metadata(
        path, e0b=e0b, expected_revision=training_revision
    )
    diagnostics._validate_checkpoint_metadata(
        metadata,
        checkpoint=path,
        timesteps=timesteps,
        expected_revision=training_revision,
        snapshot_phase=snapshot_phase,
    )
    metadata["checkpoint_snapshot_phase"] = snapshot_phase
    metadata["expected_optimizer_adam_step"] = (
        diagnostics.expected_optimizer_adam_step(
            timesteps, snapshot_phase=snapshot_phase,
        )
    )
    return metadata


def build_formal_family(args):
    _git_scoped_clean(args.code_root)
    selector_revision = _git_revision(args.code_root)
    diagnostics_gate_path = Path(args.diagnostics_gate).expanduser().resolve()
    _exact_name(
        diagnostics_gate_path,
        DIAGNOSTICS_GATE_NAME,
        "seller-v5 diagnostics gate",
    )
    gate = diagnostics.validate_gate(diagnostics_gate_path)
    training_revision = gate["code_revision"]
    e0b = gate["prerequisites"]["e0b"]["path"]
    formal = gate["formal_release"]
    candidate_paths = [
        str(Path(path).expanduser().resolve()) for path in args.candidate
    ]
    _require(
        candidate_paths == formal["candidate_paths"],
        "seller-v5 formal family differs from its diagnostics gate",
    )
    _require(
        len(candidate_paths) == len(FORMAL_CANDIDATE_TIMESTEPS) == 6,
        "seller-v5 formal family must contain exactly six candidates",
    )
    candidates = [
        _candidate_metadata(
            path,
            e0b=e0b,
            training_revision=training_revision,
            timesteps=timesteps,
            snapshot_phase=snapshot_phase,
        )
        for path, timesteps, snapshot_phase in zip(
            candidate_paths,
            FORMAL_CANDIDATE_TIMESTEPS,
            FORMAL_CANDIDATE_SNAPSHOT_PHASES,
        )
    ]
    hashes = [record["sha256"] for record in candidates]
    _require(
        len(set(hashes)) == 6,
        "seller-v5 formal candidates are not six byte-distinct checkpoints",
    )
    base = Path(formal["checkpoint"]).expanduser().resolve()
    _same_path(candidate_paths[-1], base, label="seller-v5 formal base")
    trace_path = Path(args.training_log).expanduser().resolve()
    evaluation_path = Path(args.evaluation).expanduser().resolve()
    _same_path(
        str(trace_path),
        base.with_name(f"{base.stem}.training.jsonl"),
        label="seller-v5 formal trace",
    )
    _same_path(
        str(evaluation_path),
        base.with_name(f"{base.stem}.evaluation.json"),
        label="seller-v5 formal evaluation",
    )
    trace = diagnostics._validate_trace(
        trace_path, checkpoint=base, timesteps=FORMAL_TIMESTEPS
    )
    evaluation = _formal_evaluation(evaluation_path, metadata=candidates[-1])
    console_log = Path(args.console_log).expanduser().resolve()
    _require(
        console_log.is_file() and not console_log.is_symlink(),
        "seller-v5 formal console log is missing or unsafe",
    )
    _git_scoped_clean(args.code_root)
    _require(
        _git_revision(args.code_root) == selector_revision,
        "seller-v5 selector code changed while building family",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": FAMILY_KIND,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "role": ROLE,
        "actor_loss_mode": ACTOR_LOSS_MODE,
        "source_kind": SOURCE_KIND,
        "exposure_protocol": diagnostics.canonical_protocol_provenance(),
        "training_code_revision": training_revision,
        "selector_code_revision": selector_revision,
        "diagnostics_gate": {
            "path": str(diagnostics_gate_path),
            "sha256": sha256_file(diagnostics_gate_path),
        },
        "seller_release": dict(gate["prerequisites"]["buyer_release"]),
        "e0b_source": dict(gate["prerequisites"]["e0b"]),
        "rom": dict(gate["prerequisites"]["rom"]),
        "economic_architecture": diagnostics.canonical_architecture(),
        "shared_context_initialization": diagnostics.canonical_initialization(),
        "training_config": diagnostics.canonical_training_config(),
        "sampler": diagnostics.canonical_sampler(),
        "formal_release": dict(formal),
        "candidate_snapshot_phases": list(
            FORMAL_CANDIDATE_SNAPSHOT_PHASES
        ),
        "candidate_metadata": candidates,
        "candidate_sha256": hashes,
        "training_trace": trace,
        "evaluation": evaluation,
        "console_log": {
            "path": str(console_log),
            "sha256": sha256_file(console_log),
        },
    }


def _validate_family_contract(value, *, code_root=None):
    _require(
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("kind") == FAMILY_KIND
        and value.get("passed") is True,
        "unknown or failed seller-v5 formal family",
    )
    _require(
        value.get("role") == ROLE
        and value.get("actor_loss_mode") == ACTOR_LOSS_MODE
        and value.get("source_kind") == SOURCE_KIND,
        "seller-v5 formal-family identity changed",
    )
    _require(
        value.get("exposure_protocol")
        == diagnostics.canonical_protocol_provenance(),
        "seller-v5 formal-family exposure protocol changed",
    )
    gate_record = value.get("diagnostics_gate", {})
    gate_path = Path(gate_record.get("path", "")).expanduser().resolve()
    _require(
        sha256_file(gate_path) == gate_record.get("sha256"),
        "seller-v5 diagnostics gate bytes changed",
    )
    gate = diagnostics.validate_gate(gate_path)
    _require(
        value.get("training_code_revision") == gate["code_revision"],
        "seller-v5 family training revision changed",
    )
    selector_revision = value.get("selector_code_revision")
    _require(
        isinstance(selector_revision, str)
        and re.fullmatch(r"[0-9a-f]{40}", selector_revision) is not None,
        "seller-v5 family selector revision is invalid",
    )
    if code_root is not None:
        _require(
            _git_revision(code_root) == selector_revision,
            "seller-v5 family selector runtime revision changed",
        )
        _git_scoped_clean(code_root)
    for key, gate_key in (
        ("seller_release", "buyer_release"),
        ("e0b_source", "e0b"),
        ("rom", "rom"),
    ):
        _require(
            value.get(key) == gate["prerequisites"][gate_key]
            and sha256_file(value[key]["path"]) == value[key]["sha256"],
            f"seller-v5 family prerequisite changed: {key}",
        )
    _require(
        value.get("economic_architecture")
        == diagnostics.canonical_architecture()
        and value.get("shared_context_initialization")
        == diagnostics.canonical_initialization()
        and value.get("training_config")
        == diagnostics.canonical_training_config()
        and value.get("sampler") == diagnostics.canonical_sampler(),
        "seller-v5 family architecture/training contract changed",
    )
    formal = gate["formal_release"]
    _require(
        value.get("formal_release") == formal,
        "seller-v5 formal-release record changed",
    )
    _require(
        value.get("candidate_snapshot_phases")
        == list(FORMAL_CANDIDATE_SNAPSHOT_PHASES),
        "seller-v5 formal checkpoint snapshot phases changed",
    )
    candidates = value.get("candidate_metadata", [])
    hashes = value.get("candidate_sha256", [])
    paths = formal["candidate_paths"]
    _require(
        len(candidates) == len(hashes) == len(paths) == 6
        and len(set(hashes)) == 6,
        "seller-v5 family no longer contains six byte-distinct candidates",
    )
    _require(
        [record.get("path") for record in candidates] == paths
        and [record.get("training_timesteps") for record in candidates]
        == list(FORMAL_CANDIDATE_TIMESTEPS),
        "seller-v5 formal candidate paths/clocks changed",
    )
    for record, path, timesteps, snapshot_phase, digest in zip(
            candidates,
            paths,
            FORMAL_CANDIDATE_TIMESTEPS,
            FORMAL_CANDIDATE_SNAPSHOT_PHASES,
            hashes,
    ):
        fresh = _candidate_metadata(
            path,
            e0b=value["e0b_source"]["path"],
            training_revision=value["training_code_revision"],
            timesteps=timesteps,
            snapshot_phase=snapshot_phase,
        )
        _require(
            record == fresh and sha256_file(path) == digest == fresh["sha256"],
            "seller-v5 formal candidate metadata/bytes changed",
        )
    base = paths[-1]
    trace = diagnostics._validate_trace(
        value["training_trace"]["path"],
        checkpoint=base,
        timesteps=FORMAL_TIMESTEPS,
    )
    _require(
        value.get("training_trace") == trace,
        "seller-v5 formal trace changed",
    )
    evaluation = _formal_evaluation(
        value["evaluation"]["path"], metadata=candidates[-1]
    )
    _require(
        value.get("evaluation") == evaluation,
        "seller-v5 formal evaluation changed",
    )
    console = value.get("console_log", {})
    _require(
        sha256_file(console.get("path", "")) == console.get("sha256"),
        "seller-v5 formal console log changed",
    )
    return value


def validate_formal_family(path, *, code_root=None):
    path = Path(path).expanduser().resolve()
    _exact_name(path, FAMILY_NAME, "seller-v5 formal family")
    return _validate_family_contract(load_json(path), code_root=code_root)


def write_formal_family(args):
    output = Path(args.output).expanduser().resolve()
    _exact_name(output, FAMILY_NAME, "seller-v5 formal-family output")
    _require(
        not os.path.lexists(output),
        "refusing to overwrite immutable seller-v5 formal family",
    )
    atomic_write_json(output, build_formal_family(args))
    return validate_formal_family(output, code_root=args.code_root)


def canonical_selector_environment(family):
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
        "rom_path": str(Path(family["rom"]["path"]).resolve()),
        "rom_sha256": diagnostics.CANONICAL_ROM_SHA256,
    }


def _selector_metadata(record):
    return {
        key: value for key, value in record.items()
        if key not in {
            "optimizer_adam_step", "resume_source",
            "frozen_gameplay_actor_sha256",
            "checkpoint_snapshot_phase", "expected_optimizer_adam_step",
        }
    }


def _event_rows(result, *, phase, metadata, ablated):
    rows = []
    for episode in result.get("episode_rows", []):
        for event in episode.get("events", []):
            rows.append({
                "phase": phase,
                "checkpoint_path": metadata["path"],
                "checkpoint_sha256": metadata["sha256"],
                "training_timesteps": metadata["training_timesteps"],
                "evaluation_seed": episode["evaluation_seed"],
                "fifth_economic_override": episode.get(
                    "fifth_economic_override"
                ),
                "all_trade_economic_override": episode.get(
                    "all_trade_economic_override"
                ),
                "v5_context_ablation": bool(ablated),
                **event,
            })
    return _canonical_json(rows)


def _validate_result(
        result, *, metadata, seeds, contexts, phase, ablated=False,
        fixed_event_steps=None, all_trade_override=None,
):
    seeds = [int(seed) for seed in seeds]
    contexts = [tuple(float(value) for value in row) for row in contexts]
    rows = result.get("episode_rows", [])
    _require(
        len(rows) == len(seeds)
        and [row.get("evaluation_seed") for row in rows] == seeds,
        f"seller-v5 {phase} seed schedule changed",
    )
    for row, context in zip(rows, contexts):
        _require(
            row.get("phase") == phase
            and row.get("checkpoint_sha256") == metadata["sha256"]
            and row.get("training_timesteps")
            == metadata["training_timesteps"],
            f"seller-v5 {phase} episode identity changed",
        )
        _same_path(
            row.get("checkpoint_path"),
            metadata["path"],
            label=f"seller-v5 {phase} checkpoint",
        )
        _require(
            len(row.get("opponent_commitment", [])) == 5
            and np.allclose(
                row["opponent_commitment"], context,
                rtol=0.0,
                atol=1.0e-7,
            ),
            f"seller-v5 {phase} context changed",
        )
        _require(
            bool(row.get("v5_context_ablation", False)) is bool(ablated),
            f"seller-v5 {phase} ablation flag changed",
        )
        _require(
            row.get("all_trade_economic_override") == all_trade_override,
            f"seller-v5 {phase} forced-price control changed",
        )
        if fixed_event_steps is not None:
            _require(
                tuple(row.get("event_steps", ()))
                == tuple(fixed_event_steps),
                f"seller-v5 {phase} event schedule changed",
            )
    violations = [
        violation
        for row in rows
        for violation in evaluator.audit_episode(row, role=ROLE)
    ]
    _require(
        not violations
        and result.get("protocol") == {"passed": True, "violations": []},
        f"seller-v5 {phase} mechanics failed: {violations[:3]}",
    )
    _require(
        result.get("summary") == evaluator._summary(rows, role=ROLE),
        f"seller-v5 {phase} summary does not recompute",
    )
    _require(
        result.get("event_rows")
        == _event_rows(
            result, phase=phase, metadata=metadata, ablated=ablated
        ),
        f"seller-v5 {phase} event rows do not recompute",
    )


def _validate_fixed_results(results, *, metadata, seeds):
    _require(len(results) == 11, "seller-v5 fixed grid is incomplete")
    event_steps = (20, 50, 80, 110, 140)
    for index, result in enumerate(results):
        value = index / 10.0
        _require(
            result.get("opponent_value") == value,
            "seller-v5 fixed-grid value changed",
        )
        _validate_result(
            result,
            metadata=metadata,
            seeds=seeds,
            contexts=[(value,) * 5] * len(seeds),
            phase=f"fixed_{value:.2f}",
            fixed_event_steps=event_steps,
        )


def validate_selection(args):
    family = validate_formal_family(args.family)
    report_path = Path(args.report).expanduser().resolve()
    selected = Path(args.selected).expanduser().resolve()
    _exact_name(report_path, REPORT_NAME, "seller-v5 selector report")
    _exact_name(selected, SELECTED_NAME, "seller-v5 selected alias")
    report = load_json(report_path)
    _require(
        report.get("role") == ROLE and report.get("evaluator") == EVALUATOR,
        "seller-v5 selector identity changed",
    )
    expected_protocol = {
        "screen_episodes": 20,
        "screen_seed_start": SCREEN_SEED_START,
        "confirmation_episodes": 100,
        "confirmation_seed_start": CONFIRMATION_SEED_START,
        "fixed_context_episodes": 20,
        "fixed_context_seed_start": FIXED_SEED_START,
        "confirmation_policy": "screen_winner_only_no_fallback",
    }
    protocol = report.get("protocol", {})
    for key, expected in expected_protocol.items():
        _require(
            protocol.get(key) == expected,
            f"seller-v5 selector protocol changed: {key}",
        )
    immutable = report.get("immutable_evaluation", {})
    hashes = family["candidate_sha256"]
    _require(
        immutable.get("selector_code_revision")
        == family["selector_code_revision"]
        and immutable.get("e0b_sha256") == diagnostics.CANONICAL_E0B_SHA256
        and immutable.get("rom_sha256") == diagnostics.CANONICAL_ROM_SHA256
        and immutable.get("candidate_sha256") == hashes,
        "seller-v5 immutable evaluation provenance changed",
    )
    _require(
        report.get("environment") == canonical_selector_environment(family),
        "seller-v5 selector environment changed",
    )
    screen_seeds = list(range(SCREEN_SEED_START, SCREEN_SEED_START + 20))
    screen_contexts = [evaluator.random_context(seed) for seed in screen_seeds]
    screen_record = report.get("screen", {})
    screen = screen_record.get("results", [])
    _require(len(screen) == 6, "seller-v5 selector did not screen six candidates")
    expected_metadata = [
        _selector_metadata(record) for record in family["candidate_metadata"]
    ]
    _require(
        [item.get("metadata") for item in screen] == expected_metadata,
        "seller-v5 selector screened another formal family",
    )
    common = evaluator.validate_common_screen(
        screen, seeds=screen_seeds, contexts=screen_contexts
    )
    _require(
        screen_record.get("common_pairing") == common,
        "seller-v5 common screen pairing changed",
    )
    for result, metadata in zip(screen, expected_metadata):
        _validate_result(
            result,
            metadata=metadata,
            seeds=screen_seeds,
            contexts=screen_contexts,
            phase="screen",
        )
    training_directory, training_stem = evaluator.checkpoint_family(
        expected_metadata[0]["path"]
    )
    expected_training_family = {
        "directory": training_directory,
        "stem": training_stem,
        **evaluator.common_training_family(screen),
    }
    _require(
        report.get("training_family") == expected_training_family,
        "seller-v5 report training family changed",
    )
    _, ranking = evaluator.rank_candidates(screen)
    _require(
        report.get("ranking") == ranking and len(ranking) == 6,
        "seller-v5 screen ranking does not recompute",
    )
    winner = ranking[0]["checkpoint_sha256"]
    attempts = report.get("confirmation_attempts", [])
    _require(
        len(attempts) == 1
        and attempts[0].get("metadata", {}).get("sha256") == winner,
        "seller-v5 selector did not confirm only the screen winner",
    )
    attempt = attempts[0]
    metadata = attempt["metadata"]
    confirmation_seeds = list(range(
        CONFIRMATION_SEED_START, CONFIRMATION_SEED_START + 100
    ))
    confirmation_contexts = [
        evaluator.random_context(seed) for seed in confirmation_seeds
    ]
    random_result = attempt.get("random", {})
    _validate_result(
        random_result,
        metadata=metadata,
        seeds=confirmation_seeds,
        contexts=confirmation_contexts,
        phase="confirmation_random",
    )
    fixed_seeds = list(range(FIXED_SEED_START, FIXED_SEED_START + 20))
    fixed_results = attempt.get("fixed_contexts", [])
    _validate_fixed_results(
        fixed_results, metadata=metadata, seeds=fixed_seeds
    )
    ablated = attempt.get("context_ablated_random", {})
    _validate_result(
        ablated,
        metadata=metadata,
        seeds=confirmation_seeds,
        contexts=confirmation_contexts,
        phase="confirmation_random_context_ablated",
        ablated=True,
    )
    _require(
        [row.get("event_steps") for row in random_result["episode_rows"]]
        == [row.get("event_steps") for row in ablated["episode_rows"]],
        "seller-v5 full and ablated confirmations are not paired",
    )
    raw_forced = attempt.get("forced_constant_controls", {})
    forced = {float(key): value for key, value in raw_forced.items()}
    _require(
        set(round(key, 6) for key in forced)
        == set(round(value, 6) for value in evaluator.CANONICAL_FIXED_VALUES),
        "seller-v5 forced-price grid is incomplete",
    )
    for value in evaluator.CANONICAL_FIXED_VALUES:
        result = forced[float(value)]
        _validate_result(
            result,
            metadata=metadata,
            seeds=confirmation_seeds,
            contexts=confirmation_contexts,
            phase=f"confirmation_random_forced_constant_{value:.1f}",
            all_trade_override=float(value),
        )
    model, loaded = evaluator.load_candidate(
        metadata["path"],
        role=ROLE,
        e0b_sha256=diagnostics.CANONICAL_E0B_SHA256,
        device="cpu",
    )
    try:
        conditioning = probe.collect_conditioning_report(model)
    finally:
        del model
    _require(loaded["sha256"] == winner, "seller-v5 winner bytes changed")
    _require(
        attempt.get("conditioning_probe") == conditioning,
        "seller-v5 winner conditioning probe does not recompute",
    )
    gate = evaluator.seller_v5_behavioral_gate(
        random_result=random_result,
        fixed_results=fixed_results,
        ablated_random_result=ablated,
        forced_constant_results=forced,
        formal=True,
    )
    gate["conditioning_probe_gate"] = conditioning.get("warmup_gate")
    gate["passed"] = bool(
        gate["passed"]
        and conditioning.get("warmup_gate", {}).get("passed") is True
    )
    _require(
        attempt.get("behavioral_gate") == gate,
        "seller-v5 formal scientific gate does not recompute",
    )
    selection = report.get("selection", {})
    _require(
        selection.get("fallback_allowed") is False
        and selection.get("screen_selected_checkpoint_sha256") == winner,
        "seller-v5 selector permits post-screen fallback",
    )
    if report.get("passed") is True:
        _require(
            gate.get("passed") is True
            and selection.get("selected_checkpoint_sha256") == winner,
            "passing seller-v5 report has a failed confirmation gate",
        )
        _require(
            sha256_file(selected) == winner,
            "seller-v5 selected alias bytes changed",
        )
        _same_path(
            report.get("selected_alias", {}).get("pinned_path"),
            selected,
            label="seller-v5 selected alias",
        )
    else:
        _require(
            report.get("passed") is False
            and gate.get("passed") is False
            and report.get("selected_alias") is None
            and not os.path.lexists(selected),
            "failed seller-v5 selector retained a release artifact",
        )
    _same_path(
        report.get("artifacts", {}).get("json"),
        report_path,
        label="seller-v5 report artifact",
    )
    return report


def _artifact_hashes(report):
    records = {}
    for name, raw_path in sorted(report.get("artifacts", {}).items()):
        path = Path(raw_path).expanduser().resolve()
        records[name] = {"path": str(path), "sha256": sha256_file(path)}
    _require("json" in records, "seller-v5 selection has no JSON artifact")
    return records


def build_selection_gate(args, report):
    _require(
        report.get("passed") is True,
        "only a passing seller-v5 confirmation can release E2",
    )
    family_path = Path(args.family).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    selected = Path(args.selected).expanduser().resolve()
    family = validate_formal_family(family_path)
    winner = report["selection"]["screen_selected_checkpoint_sha256"]
    attempt = report["confirmation_attempts"][0]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": SELECTION_GATE_KIND,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "role": ROLE,
        "actor_loss_mode": ACTOR_LOSS_MODE,
        "source_kind": SOURCE_KIND,
        "sampler_mode": diagnostics.SAMPLER_MODE,
        "exposure_protocol": diagnostics.canonical_protocol_provenance(),
        "economic_architecture": diagnostics.canonical_architecture(),
        "shared_context_initialization": diagnostics.canonical_initialization(),
        "e1_training_code_revision": family["training_code_revision"],
        "selector_code_revision": family["selector_code_revision"],
        "report": {
            "path": str(report_path),
            "sha256": sha256_file(report_path),
            "evaluator": EVALUATOR,
            "artifacts": _artifact_hashes(report),
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
        "diagnostics_gate": dict(family["diagnostics_gate"]),
        "seller_release": dict(family["seller_release"]),
        "selected_conditioning_probe": dict(attempt["conditioning_probe"]),
        "selected_behavioral_gate": dict(attempt["behavioral_gate"]),
        "selection": {
            "screen_selected_checkpoint_sha256": winner,
            "confirmed_checkpoint_sha256": attempt["metadata"]["sha256"],
            "selected_checkpoint_sha256": report["selection"][
                "selected_checkpoint_sha256"
            ],
            "confirmation_attempts": 1,
            "confirmation_policy": "screen_winner_only_no_fallback",
            "fallback_allowed": False,
        },
    }


def validate_selection_gate(
        path, *, family=None, report=None, selected=None,
):
    path = Path(path).expanduser().resolve()
    _exact_name(path, SELECTION_GATE_NAME, "seller-v5 selection gate")
    value = load_json(path)
    _require(
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("kind") == SELECTION_GATE_KIND
        and value.get("passed") is True,
        "unknown or failed seller-v5 selection gate",
    )
    _require(
        value.get("role") == ROLE
        and value.get("actor_loss_mode") == ACTOR_LOSS_MODE
        and value.get("source_kind") == SOURCE_KIND
        and value.get("sampler_mode") == diagnostics.SAMPLER_MODE,
        "seller-v5 selection-gate identity changed",
    )
    family_path = Path(value["training_family"]["path"]).resolve()
    report_path = Path(value["report"]["path"]).resolve()
    selected_path = Path(value["selected_checkpoint"]["path"]).resolve()
    if family is not None:
        _same_path(family_path, family, label="seller-v5 gate family")
    if report is not None:
        _same_path(report_path, report, label="seller-v5 gate report")
    if selected is not None:
        _same_path(selected_path, selected, label="seller-v5 gate selected")
    _require(
        sha256_file(family_path) == value["training_family"]["sha256"]
        and sha256_file(report_path) == value["report"]["sha256"]
        and sha256_file(selected_path)
        == value["selected_checkpoint"]["sha256"],
        "seller-v5 selection-gate artifact bytes changed",
    )
    family_value = validate_formal_family(family_path)
    _require(
        value["training_family"]["candidate_sha256"]
        == family_value["candidate_sha256"],
        "seller-v5 gate candidate family changed",
    )
    report_value = validate_selection(SimpleNamespace(
        family=str(family_path), report=str(report_path),
        selected=str(selected_path),
    ))
    expected = build_selection_gate(
        SimpleNamespace(
            family=str(family_path), report=str(report_path),
            selected=str(selected_path),
        ),
        report_value,
    )
    for key, expected_value in expected.items():
        if key != "created_utc":
            _require(
                value.get(key) == expected_value,
                f"seller-v5 selection gate differs in {key}",
            )
    return value


def write_or_validate_selection_gate(args, report):
    if report.get("passed") is False:
        if args.gate_output:
            _require(
                not os.path.lexists(args.gate_output),
                "failed seller-v5 selector retained a gate",
            )
        return None
    _require(
        args.gate_output,
        "passing seller-v5 selection requires a gate output",
    )
    output = Path(args.gate_output).expanduser().resolve()
    _exact_name(output, SELECTION_GATE_NAME, "seller-v5 selection gate")
    expected = build_selection_gate(args, report)
    if output.exists():
        existing = validate_selection_gate(
            output,
            family=args.family,
            report=args.report,
            selected=args.selected,
        )
        for key, expected_value in expected.items():
            if key != "created_utc":
                _require(
                    existing.get(key) == expected_value,
                    f"existing seller-v5 gate differs in {key}",
                )
        return existing
    _require(
        not output.is_symlink(),
        "refusing symlink seller-v5 selection-gate output",
    )
    atomic_write_json(output, expected)
    return validate_selection_gate(
        output,
        family=args.family,
        report=args.report,
        selected=args.selected,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        choices=tuple(SELECTION_PROTOCOLS),
        default=diagnostics.DEFAULT_PROTOCOL,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    family = commands.add_parser("formal-family")
    for name in (
            "diagnostics-gate", "training-log", "evaluation",
            "console-log", "code-root", "output",
    ):
        family.add_argument(f"--{name}", required=True)
    family.add_argument("--candidate", action="append", required=True)

    validate_family = commands.add_parser("validate-formal-family")
    validate_family.add_argument("--family", required=True)
    validate_family.add_argument("--code-root")

    selection = commands.add_parser("selection")
    for name in ("family", "report", "selected"):
        selection.add_argument(f"--{name}", required=True)
    selection.add_argument("--gate-output")

    gate = commands.add_parser("validate-selection-gate")
    gate.add_argument("--gate", required=True)
    gate.add_argument("--family")
    gate.add_argument("--report")
    gate.add_argument("--selected")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    configure_protocol(args.protocol)
    if args.command == "formal-family":
        result = write_formal_family(args)
    elif args.command == "validate-formal-family":
        result = validate_formal_family(args.family, code_root=args.code_root)
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
            args.gate,
            family=args.family,
            report=args.report,
            selected=args.selected,
        )
    print(json.dumps({
        "command": args.command,
        "passed": bool(result.get("passed", True)),
        "kind": result.get("kind"),
        "value": result,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
