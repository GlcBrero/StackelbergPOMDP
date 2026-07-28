#!/Users/gbrero/miniconda3/envs/stackelbergPOMDP/bin/python
"""Integrity checks and collision-safe manifests for clean Atari E1 -> E2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile


REPOSITORY_ROOT = Path(os.environ.get(
    "STACKPOMDP_CODE_ROOT",
    "/Users/gbrero/active-research/StackelbergPOMDP/code/StackelbergPOMDP",
)).resolve()
E1_EVALUATOR = "clean_atari_e1_selector_v2"
E2_EVALUATOR = "clean_atari_e2_selector_v2"
E2_STEPS = (400_680, 800_520, 1_200_360, 1_600_200, 2_000_040)

sys.path.insert(0, str(REPOSITORY_ROOT))


def fail(message: str) -> None:
    raise RuntimeError(message)


def load_json(path: Path) -> dict:
    path = path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        fail(f"expected a regular, non-symlink JSON file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        fail(f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    path = path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        fail(f"expected a regular, non-symlink file: {path}")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    identities = (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns),
        (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
    )
    if identities[0] != identities[1]:
        fail(f"file changed while hashing: {path}")
    return digest.hexdigest()


def validate_zip(path: Path) -> str:
    path = path.expanduser().resolve()
    digest = sha256_file(path)
    with zipfile.ZipFile(path, "r") as archive:
        broken = archive.testzip()
        if broken is not None:
            fail(f"corrupt ZIP member {broken!r} in {path}")
    if sha256_file(path) != digest:
        fail(f"ZIP changed during validation: {path}")
    return digest


def same_path(actual: object, expected: Path, *, label: str) -> None:
    if not isinstance(actual, str):
        fail(f"{label} is not a path string")
    if Path(actual).expanduser().resolve() != expected.expanduser().resolve():
        fail(f"{label} does not name {expected}: {actual}")


def expect_equal(actual: object, expected: object, *, label: str) -> None:
    if actual != expected:
        fail(f"{label}: expected {expected!r}, observed {actual!r}")


def validate_code_root() -> str:
    try:
        head = subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        fail(f"cannot validate E2 code root {REPOSITORY_ROOT}: {error}")
    expected = "7a193ba14b91f6ab116da29ff288e3e577d73b88"
    expect_equal(head, expected, label="E2 code HEAD")
    return head


def validate_e1_gate(args: argparse.Namespace) -> dict:
    report_path = Path(args.report).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    report = load_json(report_path)
    if report.get("passed") is not True:
        fail("E1 selector report is not strictly passing")
    expect_equal(report.get("role"), args.role, label="E1 report role")
    expect_equal(
        report.get("evaluator"), E1_EVALUATOR, label="E1 evaluator"
    )
    alias = report.get("selected_alias")
    if not isinstance(alias, dict):
        fail("passing E1 report has no selected_alias object")
    same_path(alias.get("path"), checkpoint, label="E1 selected alias")
    digest = validate_zip(checkpoint)
    expect_equal(alias.get("sha256"), digest, label="E1 alias SHA-256")

    config = report.get("training_family", {}).get(
        "common_training_config"
    )
    if not isinstance(config, dict):
        fail("E1 report has no common training configuration")
    expected_config = {
        "algorithm": "PPO",
        "actor_loss_mode": args.actor_loss_mode,
        "seed": 1,
        "n_steps": 205,
        "batch_size": 820,
        "n_epochs": 4,
        "learning_rate": 1.0e-4,
        "pretrained_lr_scale": 0.1,
        "gamma": 1.0,
        "gae_lambda": 1.0,
        "entropy_coefficient": 0.01,
        "clip_range_at_start": 0.1,
        "value_coefficient": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": None,
        "visual_features": 512,
        "state_features": 64,
        "economic_hidden": 64,
        "critic_hidden": 256,
        "policy_class": (
            "stackelberg_pomdp.atari.stackpomdp_policy."
            "StackPOMDPAtariPolicy"
        ),
    }
    for key, expected in expected_config.items():
        expect_equal(config.get(key), expected, label=f"E1 config {key}")

    from replication.atari.train_atari_stackpomdp_leader_sb3 import (
        checkpoint_policy_metadata,
    )

    metadata = checkpoint_policy_metadata(
        checkpoint, device="cpu", label=f"selected E1 {args.role}"
    )
    expect_equal(metadata.get("sha256"), digest, label="loaded E1 SHA-256")
    expect_equal(
        metadata.get("economic_role"), args.role, label="loaded E1 role"
    )
    expect_equal(
        metadata.get("economic_input_mode"),
        "full",
        label="loaded E1 economic input mode",
    )
    expect_equal(
        metadata.get("actor_loss_mode"),
        args.actor_loss_mode,
        label="loaded E1 actor loss mode",
    )
    return {
        "kind": "e1_gate",
        "role": args.role,
        "report": str(report_path),
        "checkpoint": str(checkpoint),
        "sha256": digest,
        "actor_loss_mode": args.actor_loss_mode,
        "passed": True,
    }


def discover_e1_gate(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.override_report:
        override = Path(args.override_report).expanduser().resolve()
        if not override.exists():
            return {"kind": "e1_gate_discovery", "found": False}
        reports = [override]
    else:
        reports = sorted(output_dir.glob(f"e1_{args.role}_*selector_v2.json"))
        reports = [
            path for path in reports
            if not re.search(r"_step[0-9]+_selector_v2[.]json$", path.name)
        ]

    passing = []
    for report_path in reports:
        report = load_json(report_path)
        if report.get("passed") is not True:
            if args.override_report:
                fail(f"explicit E1 gate report is not passing: {report_path}")
            continue
        if report.get("role") != args.role:
            fail(f"E1 gate report has the wrong role: {report_path}")
        config = report.get("training_family", {}).get(
            "common_training_config", {}
        )
        mode = config.get("actor_loss_mode")
        if mode not in ("balanced", "standard"):
            fail(f"E1 gate has an unsupported actor-loss mode: {report_path}")
        if args.require_mode and mode != args.require_mode:
            if args.override_report:
                fail(
                    f"explicit E1 gate must use {args.require_mode}, got {mode}"
                )
            continue
        alias = report.get("selected_alias")
        if not isinstance(alias, dict) or not isinstance(alias.get("path"), str):
            fail(f"passing E1 gate has no selected checkpoint: {report_path}")
        checkpoint = Path(alias["path"]).expanduser().resolve()
        validated = validate_e1_gate(argparse.Namespace(
            report=str(report_path),
            checkpoint=str(checkpoint),
            role=args.role,
            actor_loss_mode=mode,
        ))
        passing.append({
            **validated,
            "priority": 0 if mode == "balanced" else 1,
        })

    if not passing:
        return {"kind": "e1_gate_discovery", "found": False}
    passing.sort(key=lambda row: (row["priority"], row["report"]))
    best_priority = passing[0]["priority"]
    equally_preferred = [
        row for row in passing if row["priority"] == best_priority
    ]
    if len(equally_preferred) > 1 and not args.override_report:
        paths = [row["report"] for row in equally_preferred]
        fail(
            "multiple equally preferred passing E1 gates require an explicit "
            f"override: {paths}"
        )
    chosen = passing[0]
    return {
        "kind": "e1_gate_discovery",
        "found": True,
        "role": args.role,
        "report": chosen["report"],
        "checkpoint": chosen["checkpoint"],
        "checkpoint_sha256": chosen["sha256"],
        "actor_loss_mode": chosen["actor_loss_mode"],
    }


def validated_pipeline_inputs(path: Path, *, role: str) -> dict:
    value = load_json(path)
    expect_equal(
        value.get("schema"),
        "stackpomdp.atari.e2_pipeline_inputs.v1",
        label="E2 pipeline-input schema",
    )
    expect_equal(value.get("code_head"), validate_code_root(), label="code HEAD")
    expect_equal(value.get("leader_role"), role, label="input leader role")
    expect_equal(
        value.get("training_actor_loss_mode"),
        "balanced",
        label="E2 training actor-loss mode",
    )
    gates = value.get("e1_gates")
    if not isinstance(gates, dict):
        fail("E2 pipeline-input manifest has no E1 gates")
    for gate_role in ("buyer", "seller"):
        gate = gates.get(gate_role)
        if not isinstance(gate, dict):
            fail(f"E2 input manifest has no {gate_role} gate")
        report = Path(gate.get("report", "")).expanduser().resolve()
        checkpoint = Path(gate.get("checkpoint", "")).expanduser().resolve()
        expect_equal(
            sha256_file(report), gate.get("report_sha256"),
            label=f"{gate_role} gate-report SHA-256",
        )
        expect_equal(
            validate_zip(checkpoint), gate.get("checkpoint_sha256"),
            label=f"{gate_role} checkpoint SHA-256",
        )
        validate_e1_gate(argparse.Namespace(
            report=str(report),
            checkpoint=str(checkpoint),
            role=gate_role,
            actor_loss_mode=gate.get("actor_loss_mode"),
        ))
    cohort_path = Path(value.get("e1_gate_cohort", "")).expanduser().resolve()
    expect_equal(
        sha256_file(cohort_path),
        value.get("e1_gate_cohort_sha256"),
        label="shared E1 gate-cohort SHA-256",
    )
    cohort = validated_e1_gate_cohort(cohort_path)
    expect_equal(
        gates, cohort["e1_gates"], label="shared E1 gate-cohort records"
    )
    return value


def validated_e1_gate_cohort(path: Path) -> dict:
    value = load_json(path)
    expect_equal(
        value.get("schema"),
        "stackpomdp.atari.e2_e1_gate_cohort.v1",
        label="E1 gate-cohort schema",
    )
    expect_equal(value.get("code_head"), validate_code_root(), label="code HEAD")
    gates = value.get("e1_gates")
    if not isinstance(gates, dict) or set(gates) != {"buyer", "seller"}:
        fail("E1 gate cohort must contain exactly buyer and seller records")
    for role, gate in gates.items():
        if not isinstance(gate, dict):
            fail(f"E1 gate cohort has no {role} record")
        report = Path(gate.get("report", "")).expanduser().resolve()
        checkpoint = Path(gate.get("checkpoint", "")).expanduser().resolve()
        expect_equal(
            sha256_file(report), gate.get("report_sha256"),
            label=f"cohort {role} report SHA-256",
        )
        expect_equal(
            validate_zip(checkpoint), gate.get("checkpoint_sha256"),
            label=f"cohort {role} checkpoint SHA-256",
        )
        validate_e1_gate(argparse.Namespace(
            report=str(report), checkpoint=str(checkpoint), role=role,
            actor_loss_mode=gate.get("actor_loss_mode"),
        ))
    if gates["seller"]["actor_loss_mode"] != "balanced":
        fail("E2 gate cohort requires the balanced E1 seller")
    return value


def write_e1_gate_cohort(args: argparse.Namespace) -> dict:
    output = Path(args.output).expanduser().resolve()
    if os.path.lexists(output):
        fail(f"refusing to overwrite shared E1 gate cohort: {output}")
    entries = {}
    for role in ("buyer", "seller"):
        report = Path(getattr(args, f"{role}_report")).expanduser().resolve()
        checkpoint = Path(
            getattr(args, f"{role}_checkpoint")
        ).expanduser().resolve()
        mode = getattr(args, f"{role}_actor_loss_mode")
        validated = validate_e1_gate(argparse.Namespace(
            report=str(report), checkpoint=str(checkpoint), role=role,
            actor_loss_mode=mode,
        ))
        entries[role] = {
            "report": str(report),
            "report_sha256": sha256_file(report),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": validated["sha256"],
            "actor_loss_mode": mode,
        }
    if entries["seller"]["actor_loss_mode"] != "balanced":
        fail("E2 gate cohort requires the balanced E1 seller")
    result = {
        "schema": "stackpomdp.atari.e2_e1_gate_cohort.v1",
        "code_root": str(REPOSITORY_ROOT),
        "code_head": validate_code_root(),
        "buyer_preference": "balanced_then_standard",
        "e1_gates": entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {"kind": "e1_gate_cohort", "path": str(output), **result}


def read_e1_gate_cohort(args: argparse.Namespace) -> dict:
    path = Path(args.cohort_manifest).expanduser().resolve()
    return {
        "kind": "e1_gate_cohort",
        "path": str(path),
        **validated_e1_gate_cohort(path),
    }


def validate_rom(args: argparse.Namespace) -> dict:
    path = Path(args.rom).expanduser().resolve()
    digest = sha256_file(path)
    expect_equal(digest, args.sha256.lower(), label="canonical Atari ROM SHA-256")
    return {
        "kind": "canonical_rom",
        "path": str(path),
        "sha256": digest,
        "passed": True,
    }


def write_e2_input_manifest(args: argparse.Namespace) -> dict:
    output = Path(args.output).expanduser().resolve()
    if os.path.lexists(output):
        fail(f"refusing to overwrite E2 input manifest: {output}")
    cohort_path = Path(args.cohort_manifest).expanduser().resolve()
    cohort = validated_e1_gate_cohort(cohort_path)
    entries = cohort["e1_gates"]
    response_role = "seller" if args.role == "buyer" else "buyer"
    result = {
        "schema": "stackpomdp.atari.e2_pipeline_inputs.v1",
        "code_root": str(REPOSITORY_ROOT),
        "code_head": validate_code_root(),
        "leader_role": args.role,
        "frozen_response_role": response_role,
        "same_role_initialization_role": args.role,
        "training_actor_loss_mode": "balanced",
        "e1_gate_cohort": str(cohort_path),
        "e1_gate_cohort_sha256": sha256_file(cohort_path),
        "protocol": {
            "retained_step_timesteps": list(E2_STEPS),
            "include_post_update_final_base": True,
            "candidate_count": 6,
            "common_screen_episodes": 20,
            "common_screen_seed_start": 4_000_001,
            "confirmation_episodes": 100,
            "confirmation_seed_start": 5_000_001,
            "confirmation_policy": "screen_winner_only_no_fallback",
        },
        "e1_gates": entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {"kind": "e2_input_manifest", "path": str(output), **result}


def read_e2_input_manifest(args: argparse.Namespace) -> dict:
    path = Path(args.input_manifest).expanduser().resolve()
    value = validated_pipeline_inputs(path, role=args.role)
    return {"kind": "e2_input_manifest", "path": str(path), **value}


def inspect_e2_checkpoint(args: argparse.Namespace) -> tuple[dict, str]:
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    response = Path(args.response).expanduser().resolve()
    leader_e1 = Path(args.leader_e1).expanduser().resolve()
    digest = validate_zip(checkpoint)
    response_digest = validate_zip(response)
    leader_e1_digest = validate_zip(leader_e1)
    inputs = validated_pipeline_inputs(
        Path(args.input_manifest).expanduser().resolve(), role=args.role
    )

    from replication.atari.evaluate_atari_stackpomdp_leader_sb3 import (
        load_e2_checkpoint,
    )
    from replication.atari.sb3_common import model_actor_loss_mode
    from replication.atari.train_atari_stackpomdp_leader_sb3 import (
        E2_PROVENANCE_ATTRIBUTE,
        validate_e2_provenance_manifest,
    )

    model = load_e2_checkpoint(
        checkpoint,
        leader_role=args.role,
        device="cpu",
        expected_sha256=digest,
    )
    try:
        expect_equal(
            int(getattr(model, "num_timesteps", -1)),
            args.timesteps,
            label="E2 num_timesteps",
        )
        expect_equal(
            model_actor_loss_mode(model),
            "balanced",
            label="E2 actor loss mode",
        )
        manifest = validate_e2_provenance_manifest(
            getattr(model, E2_PROVENANCE_ATTRIBUTE, None)
        )
    finally:
        del model

    scientific = manifest.get("scientific_config", {})
    environment = scientific.get("environment", {})
    protocol = scientific.get("protocol", {})
    optimization = scientific.get("optimization", {})
    leader_policy = scientific.get("leader_policy", {})
    expect_equal(scientific.get("stage"), "e2", label="E2 stage")
    expect_equal(
        scientific.get("leader_role"), args.role, label="E2 leader role"
    )
    expect_equal(
        scientific.get("follower_role"),
        "seller" if args.role == "buyer" else "buyer",
        label="E2 follower role",
    )
    for key, expected in {
        "gameplay_horizon": 200,
        "event_tail_steps": 0,
        "fixed_event_steps": None,
        "noop_max": 30,
        "frame_skip": 4,
        "frame_stack": 4,
        "episodic_life": True,
        "clip_game_rewards": True,
        "max_frames": 100_000,
    }.items():
        expect_equal(environment.get(key), expected, label=f"E2 env {key}")
    for key, expected in {
        "trade_events": 5,
        "query_transitions": 5,
        "cached_trade_replays": 5,
        "outer_episode_transitions": 210,
        "policy_action_cache": True,
        "leader_economic_input": "event_only",
        "response_economic_input": "full",
        "response_algorithm": "frozen_meta_policy",
    }.items():
        expect_equal(protocol.get(key), expected, label=f"E2 protocol {key}")
    for key, expected in {
        "algorithm": "PPO",
        "actor_loss_mode": "balanced",
        "seed": 1,
        "num_envs": 4,
        "start_method": "spawn",
        "n_steps": 210,
        "batch_size": 840,
        "n_epochs": 4,
        "learning_rate": 1.0e-4,
        "pretrained_lr_scale": 0.1,
        "gamma": 1.0,
        "gae_lambda": 1.0,
        "clip_range": 0.1,
        "entropy_coefficient": 0.01,
        "value_coefficient": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": None,
    }.items():
        expect_equal(
            optimization.get(key), expected, label=f"E2 optimization {key}"
        )
    expect_equal(
        leader_policy.get("economic_role"), args.role, label="E2 policy role"
    )
    expect_equal(
        leader_policy.get("economic_input_mode"),
        "event_only",
        label="E2 policy input mode",
    )
    artifacts = manifest.get("artifacts", {})
    expect_equal(
        artifacts.get("frozen_response", {}).get("sha256"),
        response_digest,
        label="E2 frozen-response SHA-256",
    )
    expect_equal(
        artifacts.get("same_role_e1_initialization", {}).get("sha256"),
        leader_e1_digest,
        label="E2 E1-initialization SHA-256",
    )
    response_role = "seller" if args.role == "buyer" else "buyer"
    expect_equal(
        inputs["e1_gates"][response_role]["checkpoint_sha256"],
        response_digest,
        label="chosen response in pipeline-input manifest",
    )
    expect_equal(
        inputs["e1_gates"][args.role]["checkpoint_sha256"],
        leader_e1_digest,
        label="chosen initialization in pipeline-input manifest",
    )
    expect_equal(
        artifacts.get("frozen_response", {}).get("policy", {}).get(
            "actor_loss_mode"
        ),
        inputs["e1_gates"][response_role]["actor_loss_mode"],
        label="embedded response actor-loss mode",
    )
    expect_equal(
        artifacts.get("same_role_e1_initialization", {}).get("policy", {}).get(
            "actor_loss_mode"
        ),
        inputs["e1_gates"][args.role]["actor_loss_mode"],
        label="embedded initialization actor-loss mode",
    )
    return manifest, digest


def validate_e2_checkpoint(args: argparse.Namespace) -> dict:
    manifest, digest = inspect_e2_checkpoint(args)
    return {
        "kind": "e2_checkpoint",
        "role": args.role,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "sha256": digest,
        "timesteps": args.timesteps,
        "provenance_fingerprint": manifest["fingerprint_sha256"],
        "passed": True,
    }


def validate_e2_output(args: argparse.Namespace) -> dict:
    manifest, digest = inspect_e2_checkpoint(args)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    evaluation_path = checkpoint.with_name(
        f"{checkpoint.stem}.evaluation.json"
    )
    provenance_path = checkpoint.with_name(
        f"{checkpoint.stem}.provenance.json"
    )
    training_path = checkpoint.with_name(f"{checkpoint.stem}.training.jsonl")
    evaluation = load_json(evaluation_path)
    provenance = load_json(provenance_path)
    expect_equal(provenance, manifest, label="E2 provenance sidecar")
    expect_equal(
        evaluation.get("e2_provenance_manifest"),
        manifest,
        label="E2 evaluation provenance",
    )
    if not isinstance(evaluation.get("summary"), dict):
        fail("E2 evaluation has no summary")
    if not training_path.is_file() or training_path.is_symlink():
        fail(f"E2 training JSONL is absent or not regular: {training_path}")
    rows = []
    with training_path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                fail(f"invalid training JSONL line {number}: {error}")
            if not isinstance(row, dict):
                fail(f"training JSONL line {number} is not an object")
            rows.append(row)
    if not rows:
        fail("E2 training JSONL is empty")
    observed_steps = [
        int(row["train/total_timesteps"])
        for row in rows
        if "train/total_timesteps" in row
    ]
    if not observed_steps or max(observed_steps) != args.timesteps:
        fail(
            "E2 local training trace does not end at the expected timestep: "
            f"expected {args.timesteps}, max={max(observed_steps, default=None)}"
        )
    return {
        "kind": "e2_output",
        "role": args.role,
        "checkpoint": str(checkpoint),
        "sha256": digest,
        "timesteps": args.timesteps,
        "evaluation": str(evaluation_path),
        "provenance": str(provenance_path),
        "training_log": str(training_path),
        "provenance_fingerprint": manifest["fingerprint_sha256"],
        "passed": True,
    }


def validate_e2_family(args: argparse.Namespace) -> dict:
    if len(args.step_checkpoint) != len(E2_STEPS):
        fail(
            f"E2 family requires exactly {len(E2_STEPS)} retained step ZIPs"
        )
    manifests = []
    hashes = []
    candidates = []
    for timesteps, checkpoint in zip(E2_STEPS, args.step_checkpoint):
        local = argparse.Namespace(
            role=args.role,
            checkpoint=checkpoint,
            response=args.response,
            leader_e1=args.leader_e1,
            timesteps=timesteps,
            input_manifest=args.input_manifest,
        )
        manifest, digest = inspect_e2_checkpoint(local)
        manifests.append(manifest)
        hashes.append(digest)
        candidates.append({
            "kind": "retained_step",
            "timesteps": timesteps,
            "checkpoint": str(Path(checkpoint).expanduser().resolve()),
            "sha256": digest,
        })
    final_args = argparse.Namespace(
        role=args.role,
        checkpoint=args.base_checkpoint,
        response=args.response,
        leader_e1=args.leader_e1,
        timesteps=2_000_040,
        input_manifest=args.input_manifest,
    )
    final_manifest, final_digest = inspect_e2_checkpoint(final_args)
    manifests.append(final_manifest)
    hashes.append(final_digest)
    candidates.append({
        "kind": "post_update_final_base",
        "timesteps": 2_000_040,
        "checkpoint": str(Path(args.base_checkpoint).expanduser().resolve()),
        "sha256": final_digest,
    })
    if len(set(hashes)) != 6:
        fail("E2 selection family does not contain six byte-distinct candidates")
    fingerprints = {manifest["fingerprint_sha256"] for manifest in manifests}
    lineages = {manifest["run_lineage_id"] for manifest in manifests}
    identities = {
        manifest["scientific_identity_sha256"] for manifest in manifests
    }
    if len(fingerprints) != 1 or len(lineages) != 1 or len(identities) != 1:
        fail("E2 candidates do not belong to one exact training lineage/config")
    return {
        "kind": "e2_family",
        "role": args.role,
        "candidate_count": 6,
        "candidates": candidates,
        "provenance_fingerprint": next(iter(fingerprints)),
        "run_lineage_id": next(iter(lineages)),
        "passed": True,
    }


def validate_e2_report(args: argparse.Namespace) -> dict:
    report_path = Path(args.report).expanduser().resolve()
    selected = Path(args.selected).expanduser().resolve()
    response = Path(args.response).expanduser().resolve()
    report = load_json(report_path)
    expect_equal(report.get("schema_version"), 2, label="E2 report schema")
    passed = report.get("passed")
    if type(passed) is not bool:
        fail("E2 report passed field is not Boolean")
    if args.expect != "either":
        expect_equal(
            passed, args.expect == "passed", label="E2 selection outcome"
        )
    expect_equal(report.get("evaluator"), E2_EVALUATOR, label="E2 evaluator")
    environment = report.get("environment_config", {})
    expect_equal(
        environment.get("leader_role"), args.role, label="E2 report role"
    )
    expect_equal(
        environment.get("gameplay_horizon"), 200, label="E2 report horizon"
    )
    expect_equal(
        environment.get("event_tail_steps"), 0, label="E2 report tail"
    )
    response_digest = validate_zip(response)
    inputs = validated_pipeline_inputs(
        Path(args.input_manifest).expanduser().resolve(), role=args.role
    )
    same_path(
        report.get("response_checkpoint"),
        response,
        label="E2 report response checkpoint",
    )
    expect_equal(
        report.get("response_checkpoint_sha256"),
        response_digest,
        label="E2 report response SHA-256",
    )
    response_role = "seller" if args.role == "buyer" else "buyer"
    expect_equal(
        inputs["e1_gates"][response_role]["checkpoint_sha256"],
        response_digest,
        label="E2 report chosen response",
    )
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict):
        fail("E2 report does not declare its artifact set")
    same_path(
        artifacts.get("report_json"), report_path, label="E2 report artifact"
    )
    for label, raw in artifacts.items():
        if not isinstance(raw, str):
            fail(f"E2 artifact path {label} is not a string")
        artifact = Path(raw).expanduser().resolve()
        if not artifact.is_file() or artifact.is_symlink():
            fail(f"E2 artifact {label} is absent or not regular: {artifact}")

    if len(args.candidate) != 6:
        fail("E2 report validation requires exactly six declared candidates")
    expected_candidates = []
    for raw in args.candidate:
        path = Path(raw).expanduser().resolve()
        expected_candidates.append({
            "path": path,
            "sha256": validate_zip(path),
        })
    expected_hashes = [item["sha256"] for item in expected_candidates]
    if len(set(expected_hashes)) != 6:
        fail("declared E2 family is not byte-distinct")
    screen = report.get("screen")
    if not isinstance(screen, dict):
        fail("E2 report has no common screen")
    for key, expected in {
        "episodes_per_checkpoint": 20,
        "seed_start": 4_000_001,
        "seed_end": 4_000_020,
    }.items():
        expect_equal(screen.get(key), expected, label=f"E2 screen {key}")
    pairing = screen.get("common_seed_schedule_check")
    if not isinstance(pairing, dict) or pairing.get("passed") is not True:
        fail("E2 common screen seed/schedule check did not pass")
    expect_equal(pairing.get("episodes"), 20, label="common screen episodes")
    pairs = pairing.get("seed_schedule_pairs")
    if not isinstance(pairs, list) or [
        row.get("evaluation_seed") for row in pairs
    ] != list(range(4_000_001, 4_000_021)):
        fail("E2 common screen does not contain the exact seed schedule")
    results = screen.get("checkpoint_results")
    if not isinstance(results, list) or len(results) != 6:
        fail("E2 common screen must contain exactly six checkpoint results")
    observed = []
    fingerprints = set()
    for result in results:
        if not isinstance(result, dict):
            fail("E2 screen result is not an object")
        path = Path(result.get("checkpoint_path", "")).expanduser().resolve()
        observed.append((path, result.get("checkpoint_sha256")))
        fingerprints.add(result.get("e2_provenance_fingerprint"))
        expect_equal(
            result.get("response_checkpoint_sha256"),
            response_digest,
            label="screen response SHA-256",
        )
        expect_equal(result.get("seed_start"), 4_000_001, label="screen seed")
        expect_equal(result.get("seed_end"), 4_000_020, label="screen seed end")
    expected_pairs = {
        (item["path"], item["sha256"]) for item in expected_candidates
    }
    if set(observed) != expected_pairs or len(set(observed)) != 6:
        fail("E2 report screen does not match the exact declared family")
    if len(fingerprints) != 1 or None in fingerprints:
        fail("E2 screen candidates do not share one provenance fingerprint")

    selection = report.get("selection")
    if not isinstance(selection, dict):
        fail("E2 report has no selection record")
    screen_winner = selection.get("screen_selected_checkpoint_sha256")
    if screen_winner is not None and screen_winner not in expected_hashes:
        fail("E2 screen winner is outside the declared family")
    confirmation = report.get("confirmation")
    attempts = report.get("confirmation_attempts")
    if not isinstance(attempts, list) or len(attempts) > 1:
        fail("E2 selector attempted confirmation fallback")
    if confirmation is None:
        if attempts or screen_winner is not None:
            fail("E2 report omitted confirmation for a screen winner")
    else:
        if len(attempts) != 1 or attempts[0] != confirmation:
            fail("E2 report must retain exactly its sole confirmation attempt")
        for key, expected in {
            "screen_rank": 1,
            "episodes": 100,
            "seed_start": 5_000_001,
            "seed_end": 5_000_100,
            "disjoint_from_screen": True,
        }.items():
            expect_equal(
                confirmation.get(key), expected, label=f"confirmation {key}"
            )
        confirmed = confirmation.get("result")
        if not isinstance(confirmed, dict):
            fail("E2 confirmation has no result")
        expect_equal(
            confirmed.get("checkpoint_sha256"),
            screen_winner,
            label="confirmed screen-winner SHA-256",
        )
        expect_equal(
            confirmed.get("seed_start"), 5_000_001,
            label="confirmation result seed start",
        )
        expect_equal(
            confirmed.get("seed_end"), 5_000_100,
            label="confirmation result seed end",
        )
        checks = confirmation.get("checks")
        if not isinstance(checks, dict):
            fail("E2 confirmation has no screen-match checks")
        expect_equal(
            checks.get("checkpoint_hash_matches_screen"),
            True,
            label="confirmation winner identity",
        )
    selected_hash = selection.get("selected_checkpoint_sha256")
    if passed:
        expect_equal(selected_hash, screen_winner, label="selected winner hash")
        if confirmation is None or confirmation.get("checks", {}).get(
                "passed"
        ) is not True:
            fail("passing E2 report has no passing sole confirmation")
    else:
        expect_equal(selected_hash, None, label="failed selected hash")

    alias = report.get("selected_alias")
    if passed:
        if not isinstance(alias, dict):
            fail("passing E2 report has no selected alias")
        same_path(
            alias.get("selected_checkpoint_path"),
            selected,
            label="E2 selected alias",
        )
        expect_equal(
            alias.get("created_by_selection_run"),
            True,
            label="E2 alias ownership",
        )
        expect_equal(
            alias.get("copy_verified"), True, label="E2 alias copy check"
        )
        digest = validate_zip(selected)
        expect_equal(
            alias.get("checkpoint_sha256"), digest, label="E2 alias SHA-256"
        )
    else:
        expect_equal(alias, None, label="failed E2 alias")
        if os.path.lexists(selected):
            fail(f"failed E2 gate unexpectedly left a selected alias: {selected}")

    return {
        "kind": "e2_report",
        "role": args.role,
        "report": str(report_path),
        "selected": str(selected) if passed else None,
        "passed": passed,
    }


def add_e2_checkpoint_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--role", choices=("buyer", "seller"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--response", required=True)
    parser.add_argument("--leader-e1", required=True)
    parser.add_argument("--timesteps", type=int, required=True)
    parser.add_argument("--input-manifest", required=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    e1 = subparsers.add_parser("e1-gate")
    e1.add_argument("--report", required=True)
    e1.add_argument("--role", choices=("buyer", "seller"), required=True)
    e1.add_argument("--checkpoint", required=True)
    e1.add_argument(
        "--actor-loss-mode", choices=("balanced", "standard"), required=True
    )
    e1.set_defaults(handler=validate_e1_gate)

    discovery = subparsers.add_parser("discover-e1-gate")
    discovery.add_argument("--role", choices=("buyer", "seller"), required=True)
    discovery.add_argument("--output-dir", required=True)
    discovery.add_argument("--override-report")
    discovery.add_argument("--require-mode", choices=("balanced", "standard"))
    discovery.set_defaults(handler=discover_e1_gate)

    cohort = subparsers.add_parser("write-e1-gate-cohort")
    cohort.add_argument("--output", required=True)
    for role in ("buyer", "seller"):
        cohort.add_argument(f"--{role}-report", required=True)
        cohort.add_argument(f"--{role}-checkpoint", required=True)
        cohort.add_argument(
            f"--{role}-actor-loss-mode",
            choices=("balanced", "standard"),
            required=True,
        )
    cohort.set_defaults(handler=write_e1_gate_cohort)

    read_cohort = subparsers.add_parser("read-e1-gate-cohort")
    read_cohort.add_argument("--cohort-manifest", required=True)
    read_cohort.set_defaults(handler=read_e1_gate_cohort)

    rom = subparsers.add_parser("validate-rom")
    rom.add_argument("--rom", required=True)
    rom.add_argument("--sha256", required=True)
    rom.set_defaults(handler=validate_rom)

    inputs = subparsers.add_parser("write-e2-input-manifest")
    inputs.add_argument("--role", choices=("buyer", "seller"), required=True)
    inputs.add_argument("--output", required=True)
    inputs.add_argument("--cohort-manifest", required=True)
    inputs.set_defaults(handler=write_e2_input_manifest)

    read_inputs = subparsers.add_parser("read-e2-input-manifest")
    read_inputs.add_argument("--role", choices=("buyer", "seller"), required=True)
    read_inputs.add_argument("--input-manifest", required=True)
    read_inputs.set_defaults(handler=read_e2_input_manifest)

    e2 = subparsers.add_parser("e2-checkpoint")
    add_e2_checkpoint_arguments(e2)
    e2.set_defaults(handler=validate_e2_checkpoint)

    output = subparsers.add_parser("e2-output")
    add_e2_checkpoint_arguments(output)
    output.set_defaults(handler=validate_e2_output)

    family = subparsers.add_parser("e2-family")
    family.add_argument("--role", choices=("buyer", "seller"), required=True)
    family.add_argument("--response", required=True)
    family.add_argument("--leader-e1", required=True)
    family.add_argument("--input-manifest", required=True)
    family.add_argument("--step-checkpoint", action="append", required=True)
    family.add_argument("--base-checkpoint", required=True)
    family.set_defaults(handler=validate_e2_family)

    report = subparsers.add_parser("e2-report")
    report.add_argument("--report", required=True)
    report.add_argument("--role", choices=("buyer", "seller"), required=True)
    report.add_argument("--selected", required=True)
    report.add_argument("--response", required=True)
    report.add_argument("--input-manifest", required=True)
    report.add_argument("--candidate", action="append", required=True)
    report.add_argument(
        "--expect", choices=("passed", "failed", "either"), required=True
    )
    report.set_defaults(handler=validate_e2_report)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_code_root()
    result = args.handler(args)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
