import argparse
from copy import deepcopy
import json
from pathlib import Path

import pytest

from replication.atari.automation import (
    validate_atari_e1_seller_conditioning_recovery as recovery,
)
from replication.atari.automation import (
    validate_atari_e2_pipeline_artifact as downstream,
)


def _metadata(path, *, step, optimizer_step, warmup, activation):
    history = (
        [{
            "start_total_timesteps": 0,
            "sampler": recovery.canonical_all_equal_sampler(),
            "inferred_for_legacy_checkpoint": False,
            "resume_sources": [],
        }]
        if warmup else recovery._expected_history(
            activation, "b" * 64
        )
    )
    return {
        "path": str(Path(path).resolve()),
        "sha256": "b" * 64 if warmup else "c" * 64,
        "training_timesteps": step,
        "role": "seller",
        "training_config": deepcopy(recovery.canonical_training_config()),
        "atari_e1_sampler_provenance": (
            recovery.canonical_all_equal_sampler()
            if warmup else recovery.canonical_uniform_sampler()
        ),
        "atari_e1_sampler_history": history,
        "resume_source": (
            None if warmup else {
                "path": str(Path(activation["protocol"]["warmup"]["checkpoint"]).resolve()),
                "sha256": "b" * 64,
                "training_total_timesteps": recovery.WARMUP_TIMESTEPS,
            }
        ),
        "optimizer_adam_step": optimizer_step,
    }


def _activation(tmp_path):
    warmup = tmp_path / "warmup.zip"
    target = tmp_path / "target.zip"
    probe = tmp_path / "warmup_probe.json"
    return {
        "protocol": {
            "warmup": {
                "checkpoint": str(warmup.resolve()),
                "conditioning_probe": {
                    "path": str(probe.resolve()),
                },
            },
            "target": {
                "candidate_paths": recovery.expected_target_candidates(target),
            },
        },
        "e0b_source": {
            "path": str((tmp_path / "e0b.zip").resolve()),
            "sha256": recovery.CANONICAL_E0B_SHA256,
        },
    }


def test_recovery_schedule_excludes_warmup_and_uses_global_target_clocks(tmp_path):
    warmup = tmp_path / "warmup.zip"
    target = tmp_path / "target.zip"
    candidates = recovery.expected_target_candidates(target)
    assert str(warmup.resolve()) not in candidates
    assert candidates == [
        str(tmp_path / f"target_step{step}.zip")
        for step in recovery.TARGET_STEP_TIMESTEPS
    ] + [str(target.resolve())]
    assert recovery.TARGET_CANDIDATE_TIMESTEPS == (
        800_320, 1_200_480, 1_600_640, 2_000_800, 2_400_960,
        2_400_960,
    )


def test_activation_preregisters_two_stages_probe_and_fresh_holdout(
        monkeypatch, tmp_path,
):
    failed = tmp_path / recovery.FAILED_REPORT_NAME
    release = tmp_path / "release.json"
    e0b = tmp_path / "e0b.zip"
    rom = tmp_path / "rom.bin"
    for path in (failed, release, e0b, rom):
        path.write_bytes(path.name.encode())

    monkeypatch.setattr(recovery, "_git_scoped_clean", lambda path: None)
    monkeypatch.setattr(recovery, "_git_revision", lambda path: "a" * 40)
    monkeypatch.setattr(
        recovery, "validate_failed_seller_report",
        lambda path: {
            "path": str(failed.resolve()), "sha256": "1" * 64,
            "passed": False,
            "e0b_source": {"path": str(e0b), "sha256": recovery.CANONICAL_E0B_SHA256},
        },
    )
    monkeypatch.setattr(recovery, "validate_seller_release", lambda path: {})
    real_sha = recovery.sha256_file

    def digest(path):
        resolved = Path(path).resolve()
        if resolved == e0b.resolve():
            return recovery.CANONICAL_E0B_SHA256
        if resolved == rom.resolve():
            return recovery.CANONICAL_ROM_SHA256
        return real_sha(resolved)

    monkeypatch.setattr(recovery, "sha256_file", digest)
    args = argparse.Namespace(
        code_root=str(tmp_path), failed_report=str(failed),
        seller_release=str(release), e0b=str(e0b), rom=str(rom),
        warmup_checkpoint=str(tmp_path / "warmup.zip"),
        warmup_probe=str(tmp_path / "probe.json"),
        target_checkpoint=str(tmp_path / "target.zip"),
    )
    value = recovery.build_activation(args)
    protocol = value["protocol"]
    assert value["prerequisite_failure"]["passed"] is False
    assert protocol["stage_order"] == [
        "all_equal_warmup", "independent_uniform_target"
    ]
    assert protocol["warmup"]["sampler"]["mode"] == "all-equal-v1"
    assert protocol["warmup"]["additional_timesteps"] == 400_160
    assert protocol["warmup"]["selectable"] is False
    probe = protocol["warmup"]["conditioning_probe"]
    assert probe["probe"] == recovery.PROBE_NAME
    assert probe["minimum_endpoint_response"] == 0.25
    assert probe["maximum_adjacent_reversal"] == 0.15
    assert probe["require_all_outputs_finite_and_unit"] is True
    assert protocol["target"]["sampler"]["mode"] == "uniform"
    assert protocol["target"]["start_total_timesteps"] == 400_160
    assert protocol["target"]["additional_timesteps"] == 2_000_800
    assert protocol["target"]["expected_total_timesteps"] == 2_400_960
    assert protocol["evaluation"]["screen"]["seed_start"] == 9_000_001


def test_warmup_metadata_requires_fresh_single_all_equal_stage(tmp_path):
    activation = _activation(tmp_path)
    metadata = _metadata(
        tmp_path / "warmup.zip", step=400_160,
        optimizer_step=1_952, warmup=True, activation=activation,
    )
    recovery._validate_warmup_metadata(metadata, activation)

    changed = deepcopy(metadata)
    changed["training_timesteps"] += 820
    with pytest.raises(ValueError, match="400160"):
        recovery._validate_warmup_metadata(changed, activation)

    changed = deepcopy(metadata)
    changed["atari_e1_sampler_history"].append(deepcopy(
        changed["atari_e1_sampler_history"][0]
    ))
    with pytest.raises(ValueError, match="mixed or resumed"):
        recovery._validate_warmup_metadata(changed, activation)

    changed = deepcopy(metadata)
    changed["resume_source"] = {"sha256": "0" * 64}
    with pytest.raises(ValueError, match="initialized fresh"):
        recovery._validate_warmup_metadata(changed, activation)


def test_target_history_binds_order_boundary_source_hash_and_optimizer(tmp_path):
    activation = _activation(tmp_path)
    path = Path(
        activation["protocol"]["target"]["candidate_paths"][0]
    )
    metadata = _metadata(
        path, step=800_320, optimizer_step=3_900,
        warmup=False, activation=activation,
    )
    recovery._validate_target_metadata(
        metadata, expected_path=path, expected_step=800_320,
        activation=activation, warmup_digest="b" * 64, final=False,
    )

    changed = deepcopy(metadata)
    changed["atari_e1_sampler_history"].reverse()
    with pytest.raises(ValueError, match="mixed sampler history"):
        recovery._validate_target_metadata(
            changed, expected_path=path, expected_step=800_320,
            activation=activation, warmup_digest="b" * 64, final=False,
        )

    changed = deepcopy(metadata)
    changed["atari_e1_sampler_history"][1]["start_total_timesteps"] = 0
    with pytest.raises(ValueError, match="mixed sampler history"):
        recovery._validate_target_metadata(
            changed, expected_path=path, expected_step=800_320,
            activation=activation, warmup_digest="b" * 64, final=False,
        )

    changed = deepcopy(metadata)
    changed["resume_source"]["sha256"] = "d" * 64
    with pytest.raises(ValueError, match="resume SHA"):
        recovery._validate_target_metadata(
            changed, expected_path=path, expected_step=800_320,
            activation=activation, warmup_digest="b" * 64, final=False,
        )

    changed = deepcopy(metadata)
    changed["optimizer_adam_step"] = 1_948
    with pytest.raises(ValueError, match="optimizer did not continue"):
        recovery._validate_target_metadata(
            changed, expected_path=path, expected_step=800_320,
            activation=activation, warmup_digest="b" * 64, final=False,
        )


def _probe_report(checkpoint, checkpoint_sha):
    from replication.atari import probe_atari_e1_seller_conditioning as probe

    all_equal_prices = [
        [0.20 + 0.31 * threshold] * 5
        for threshold in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    coordinate_low = [0.30] * 5
    coordinate_high = [0.45] * 5
    return {
        "probe": recovery.PROBE_NAME,
        "execution": {
            "read_only_checkpoint": True,
            "ale_instantiated": False,
            "environment_steps": 0,
            "deterministic_statistic": "Beta mean",
            "device": "cpu",
        },
        "checkpoint": {
            "path": str(Path(checkpoint).resolve()),
            "sha256": checkpoint_sha,
            "training_timesteps": 400_160,
            "role": "seller",
            "atari_e1_sampler_provenance": recovery.canonical_all_equal_sampler(),
            "atari_e1_sampler_history": [{
                "start_total_timesteps": 0,
                "sampler": recovery.canonical_all_equal_sampler(),
                "inferred_for_legacy_checkpoint": False,
                "resume_sources": [],
            }],
        },
        "e0b_source": {"sha256": recovery.CANONICAL_E0B_SHA256},
        "all_equal_thresholds": [
            {
                "threshold": threshold,
                "event_beta_mean_prices": prices,
            }
            for threshold, prices in zip(
                (0.0, 0.25, 0.5, 0.75, 1.0), all_equal_prices
            )
        ],
        "current_coordinate_only_sensitivity": {
            "rows": [
                {
                    "event_index": index,
                    "low_beta_mean_price": coordinate_low[index],
                    "high_beta_mean_price": coordinate_high[index],
                }
                for index in range(5)
            ]
        },
        "warmup_gate": probe.warmup_diagnostic_gate(
            all_equal_prices, coordinate_low, coordinate_high
        ),
    }


def test_probe_gate_binds_sha_history_thresholds_and_no_ale(monkeypatch, tmp_path):
    from replication.atari import probe_atari_e1_seller_conditioning as probe_module

    checkpoint = tmp_path / "warmup.zip"
    checkpoint.write_bytes(b"warmup")
    digest = recovery.sha256_file(checkpoint)
    probe = tmp_path / "warmup_probe.json"
    activation = _activation(tmp_path)
    probe.write_text(json.dumps(_probe_report(checkpoint, digest)), encoding="utf-8")
    monkeypatch.setattr(
        recovery, "_validate_activation_contract", lambda value: value
    )
    monkeypatch.setattr(
        probe_module,
        "run_probe_from_checkpoints",
        lambda **kwargs: {
            "created_utc": "fresh timestamp is intentionally ignored",
            **_probe_report(checkpoint, digest),
        },
    )
    result = recovery.validate_warmup_probe(
        activation=activation, probe=probe, checkpoint=checkpoint
    )
    assert result["passed"] is True
    assert result["checkpoint_sha256"] == digest
    assert result["no_ale"] is True

    value = _probe_report(checkpoint, digest)
    value["checkpoint"]["atari_e1_sampler_history"][0][
        "start_total_timesteps"
    ] = 1
    probe.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one fresh"):
        recovery.validate_warmup_probe(
            activation=activation, probe=probe, checkpoint=checkpoint
        )

    value = _probe_report(checkpoint, digest)
    value["warmup_gate"]["checks"][
        "minimum_all_one_minus_all_zero_beta_mean_price"
    ]["actual"] = 0.99
    probe.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="does not recompute"):
        recovery.validate_warmup_probe(
            activation=activation, probe=probe, checkpoint=checkpoint
        )

    value = _probe_report(checkpoint, digest)
    value["all_equal_thresholds"][2]["event_beta_mean_prices"][0] += 0.01
    raw_all_equal = [
        row["event_beta_mean_prices"]
        for row in value["all_equal_thresholds"]
    ]
    coordinate_rows = value["current_coordinate_only_sensitivity"]["rows"]
    value["warmup_gate"] = probe_module.warmup_diagnostic_gate(
        raw_all_equal,
        [row["low_beta_mean_price"] for row in coordinate_rows],
        [row["high_beta_mean_price"] for row in coordinate_rows],
    )
    probe.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="fresh checkpoint inference"):
        recovery.validate_warmup_probe(
            activation=activation, probe=probe, checkpoint=checkpoint
        )


def test_training_metadata_rejects_any_ppo_or_seed_mutation(tmp_path):
    activation = _activation(tmp_path)
    metadata = _metadata(
        tmp_path / "warmup.zip", step=400_160,
        optimizer_step=1_952, warmup=True, activation=activation,
    )
    for key, value in (
        ("seed", 2),
        ("learning_rate", 2.0e-4),
        ("gamma", 0.99),
        ("n_epochs", 3),
    ):
        changed = deepcopy(metadata)
        changed["training_config"][key] = value
        with pytest.raises(ValueError, match="PPO configuration/seed"):
            recovery._validate_warmup_metadata(changed, activation)


def _trace_rows(mode, checkpoint, *, step):
    contexts = recovery._expected_training_contexts(mode, 1)
    rows = []
    for context in contexts:
        row = {
            "record_kind": "episode",
            "train/total_timesteps": step,
            "checkpoint_path": str(Path(checkpoint).resolve()),
            "seed": 1,
            "algorithm": "PPO",
            "train/episode_length": 205,
            "train/e1_sampler_mode": mode,
            "train/e1_context_stratum": (
                "all_equal"
                if mode == recovery.ALL_EQUAL_MODE else "uniform"
            ),
            "train/e1_sampler_episode_count_per_env": 1.0,
        }
        for index, value in enumerate(context, start=1):
            row[f"train/event_{index}/threshold"] = value
        if mode == recovery.ALL_EQUAL_MODE:
            row.update({
                "train/e1_context_entries_all_equal": 1.0,
                "train/e1_context_shared_value": context[0],
                "train/e1_context_stratum_one_hot_all_equal": 1.0,
            })
        else:
            row.update({
                "train/e1_context_stratum_one_hot_uniform": 1.0,
                "train/e1_context_stratum_one_hot_all_equal": 0.0,
            })
        rows.append(row)
    rows.append({
        "record_kind": "optimizer",
        "train/total_timesteps": step,
        "checkpoint_path": str(Path(checkpoint).resolve()),
        "seed": 1,
        "algorithm": "PPO",
        "train/game_active_rows": 800.0,
        "train/economic_active_rows": 20.0,
        "train/inactive_actor_rows": 0.0,
    })
    return rows


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("mode", "step"),
    ((recovery.ALL_EQUAL_MODE, 820),
     (recovery.UNIFORM_MODE, recovery.WARMUP_TIMESTEPS + 820)),
)
def test_trace_binds_exact_record_kinds_and_seeded_event_contexts(
        tmp_path, mode, step,
):
    checkpoint = tmp_path / f"{mode}.zip"
    trace = tmp_path / f"{mode}.jsonl"
    rows = _trace_rows(mode, checkpoint, step=step)
    _write_jsonl(trace, rows)
    recovery._validate_training_trace(
        trace, mode=mode, first_step=step, last_step=step,
        expected_episodes=4, expected_optimizers=1,
        checkpoint=checkpoint,
    )

    changed = deepcopy(rows)
    changed[0]["record_kind"] = "maybe_episode"
    _write_jsonl(trace, changed)
    with pytest.raises(ValueError, match="record_kind"):
        recovery._validate_training_trace(
            trace, mode=mode, first_step=step, last_step=step,
            expected_episodes=4, expected_optimizers=1,
            checkpoint=checkpoint,
        )

    changed = deepcopy(rows)
    changed[0]["train/event_3/threshold"] = (
        0.0 if changed[0]["train/event_3/threshold"] > 0.5 else 1.0
    )
    if mode == recovery.ALL_EQUAL_MODE:
        changed[0]["train/e1_context_shared_value"] = changed[0][
            "train/event_3/threshold"
        ]
        for index in range(1, 6):
            changed[0][f"train/event_{index}/threshold"] = changed[0][
                "train/e1_context_shared_value"
            ]
    _write_jsonl(trace, changed)
    with pytest.raises(ValueError, match="exact seeded|sampler stream"):
        recovery._validate_training_trace(
            trace, mode=mode, first_step=step, last_step=step,
            expected_episodes=4, expected_optimizers=1,
            checkpoint=checkpoint,
        )


def test_builtin_evaluation_design_and_summary_reject_row_mutations():
    rows = []
    for episode in range(2):
        context, schedule = recovery._canonical_builtin_evaluation_draw(
            seed=300_001 + episode
        )
        rows.append({
            "evaluation_episode": episode,
            "opponent_commitment": list(context),
            "event_steps": list(schedule),
        })
    recovery._validate_builtin_evaluation_design(
        rows, episodes=2, seed_start=300_001,
        label="test random",
    )
    changed = deepcopy(rows)
    changed[0]["opponent_commitment"][0] = 0.0
    with pytest.raises(ValueError, match="exact seeded stream"):
        recovery._validate_builtin_evaluation_design(
            changed, episodes=2, seed_start=300_001,
            label="test random",
        )
    changed = deepcopy(rows)
    changed[1]["event_steps"][0] += 1
    with pytest.raises(ValueError, match="schedule"):
        recovery._validate_builtin_evaluation_design(
            changed, episodes=2, seed_start=300_001,
            label="test random",
        )

    summary_rows = [{
        "evaluation_episode": 0,
        "evaluation_return": 1.0,
        "evaluation_steps": 205,
        "events": [
            {
                "event_index": index,
                "game_step": 20 + 30 * index,
                "price": 0.4,
                "threshold": 0.5,
                "accepted": True,
            }
            for index in range(5)
        ],
    }]
    summary = recovery._recompute_builtin_evaluation_summary(summary_rows)
    recovery._validate_builtin_evaluation_summary(
        summary, summary_rows, label="test summary"
    )
    changed_summary = deepcopy(summary)
    changed_summary["episodes"] = 2
    with pytest.raises(ValueError, match="does not recompute"):
        recovery._validate_builtin_evaluation_summary(
            changed_summary, summary_rows, label="test summary"
        )


def test_selector_environment_is_exact_and_release_sha_is_pinned(tmp_path):
    activation = {
        "rom": {"path": str((tmp_path / "space_invaders.bin").resolve())}
    }
    environment = recovery.canonical_selector_environment(activation)
    recovery._validate_selector_environment(environment, activation)
    changed = deepcopy(environment)
    changed["seller_game_reward_scale"] = 1.0
    with pytest.raises(ValueError, match="canonical Atari recovery"):
        recovery._validate_selector_environment(changed, activation)

    release = tmp_path / "release.json"
    release.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="exact canonical seller-release"):
        recovery.validate_seller_release(release)
    assert recovery.CANONICAL_SELLER_RELEASE_SHA256 == (
        "a1f4e02ce0a99246e1bcb24dd9f9680c8ff934531518bbc069836740bca3ebb7"
    )


def test_authoritative_recovery_keeps_e2_closed_until_gate(
        monkeypatch, tmp_path,
):
    activation = tmp_path / downstream.E1_SELLER_RECOVERY_ACTIVATION_NAME
    activation.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        downstream, "_run_seller_recovery_validator", lambda args: {}
    )
    pending = downstream._discover_authoritative_seller_recovery_gate(
        output_dir=tmp_path, override_report=None
    )
    assert pending["found"] is False
    assert pending["state"] == "recovery_training_or_selection_pending"

    report = tmp_path / downstream.E1_SELLER_RECOVERY_REPORT_NAME
    report.write_text(json.dumps({
        "passed": False, "role": "seller",
        "evaluator": downstream.E1_EVALUATOR,
    }), encoding="utf-8")
    with pytest.raises(RuntimeError, match="recovery confirmation failed"):
        downstream._discover_authoritative_seller_recovery_gate(
            output_dir=tmp_path, override_report=None
        )

    report.write_text(json.dumps({
        "passed": True, "role": "seller",
        "evaluator": downstream.E1_EVALUATOR,
    }), encoding="utf-8")
    pending = downstream._discover_authoritative_seller_recovery_gate(
        output_dir=tmp_path, override_report=None
    )
    assert pending["found"] is False
    assert pending["state"] == "recovery_gate_publication_pending"

    selected = tmp_path / "selected.zip"
    selected.write_bytes(b"selected")
    gate = tmp_path / downstream.E1_SELLER_RECOVERY_GATE_NAME
    gate.write_text(json.dumps({
        "selected_checkpoint": {"path": str(selected)}
    }), encoding="utf-8")
    monkeypatch.setattr(
        downstream, "validate_e1_gate",
        lambda args: {
            "report": str(report), "checkpoint": str(selected),
            "sha256": "f" * 64, "actor_loss_mode": "balanced",
            "source_kind": downstream.E1_SELLER_RECOVERY_SOURCE_KIND,
            "sampler_mode": "uniform", "support_artifacts": {},
        },
    )
    released = downstream._discover_authoritative_seller_recovery_gate(
        output_dir=tmp_path, override_report=None
    )
    assert released["found"] is True
    assert released["source_kind"] == "seller_conditioning_recovery_v1"


def test_threshold_residual_recovery_keeps_e2_closed_until_gate(
        monkeypatch, tmp_path,
):
    activation = (
        tmp_path / downstream.E1_SELLER_THRESHOLD_RESIDUAL_ACTIVATION_NAME
    )
    activation.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        downstream,
        "_run_seller_threshold_residual_validator",
        lambda args: {},
    )
    pending = downstream._discover_authoritative_seller_threshold_residual_gate(
        output_dir=tmp_path, override_report=None,
    )
    assert pending == {
        "kind": "e1_gate_discovery",
        "found": False,
        "authoritative_source": (
            downstream.E1_SELLER_THRESHOLD_RESIDUAL_SOURCE_KIND
        ),
        "state": "threshold_residual_training_or_selection_pending",
    }

    with pytest.raises(RuntimeError, match="forbids overriding"):
        downstream._discover_authoritative_seller_threshold_residual_gate(
            output_dir=tmp_path,
            override_report=str(tmp_path / "legacy_seller.json"),
        )

    report = tmp_path / downstream.E1_SELLER_THRESHOLD_RESIDUAL_REPORT_NAME
    report.write_text(json.dumps({
        "passed": False,
        "role": "seller",
        "evaluator": downstream.E1_EVALUATOR,
    }), encoding="utf-8")
    with pytest.raises(
            RuntimeError,
            match="threshold-residual confirmation failed",
    ):
        downstream._discover_authoritative_seller_threshold_residual_gate(
            output_dir=tmp_path, override_report=None,
        )

    report.write_text(json.dumps({
        "passed": True,
        "role": "seller",
        "evaluator": downstream.E1_EVALUATOR,
    }), encoding="utf-8")
    pending = downstream._discover_authoritative_seller_threshold_residual_gate(
        output_dir=tmp_path, override_report=None,
    )
    assert pending["found"] is False
    assert pending["state"] == "threshold_residual_gate_publication_pending"

    selected = tmp_path / "residual_selected.zip"
    selected.write_bytes(b"selected")
    gate = tmp_path / downstream.E1_SELLER_THRESHOLD_RESIDUAL_GATE_NAME
    gate.write_text(json.dumps({
        "selected_checkpoint": {"path": str(selected.resolve())},
    }), encoding="utf-8")
    monkeypatch.setattr(
        downstream,
        "validate_e1_gate",
        lambda args: {
            "report": str(report.resolve()),
            "checkpoint": str(selected.resolve()),
            "sha256": "f" * 64,
            "actor_loss_mode": "balanced",
            "source_kind": (
                downstream.E1_SELLER_THRESHOLD_RESIDUAL_SOURCE_KIND
            ),
            "sampler_mode": "uniform",
            "support_artifacts": {},
        },
    )
    released = downstream._discover_authoritative_seller_threshold_residual_gate(
        output_dir=tmp_path, override_report=None,
    )
    assert released["found"] is True
    assert released["source_kind"] == (
        "seller_conditioning_recovery_v3_direct_threshold_residual_v1"
    )


@pytest.mark.parametrize(
    "artifact_name",
    (
        downstream.E1_SELLER_THRESHOLD_RESIDUAL_REPORT_NAME,
        downstream.E1_SELLER_THRESHOLD_RESIDUAL_GATE_NAME,
    ),
)
def test_orphan_threshold_residual_artifacts_never_fall_back(
        monkeypatch, tmp_path, artifact_name,
):
    (tmp_path / artifact_name).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        downstream,
        "_discover_authoritative_seller_recovery_gate",
        lambda **kwargs: pytest.fail("orphan v2 artifact fell back to v1"),
    )
    with pytest.raises(
            RuntimeError,
            match="without their authoritative activation",
    ):
        downstream.discover_e1_gate(argparse.Namespace(
            role="seller",
            output_dir=str(tmp_path),
            override_report=None,
            require_mode="balanced",
        ))


def test_threshold_residual_discovery_has_priority_over_v1(monkeypatch, tmp_path):
    expected = {
        "kind": "e1_gate_discovery",
        "found": False,
        "authoritative_source": (
            downstream.E1_SELLER_THRESHOLD_RESIDUAL_SOURCE_KIND
        ),
        "state": "threshold_residual_training_or_selection_pending",
    }
    monkeypatch.setattr(
        downstream,
        "_discover_authoritative_seller_threshold_residual_gate",
        lambda **kwargs: expected,
    )
    monkeypatch.setattr(
        downstream,
        "_discover_authoritative_seller_recovery_gate",
        lambda **kwargs: pytest.fail("v1 discovery ran after v2 activation"),
    )
    discovered = downstream.discover_e1_gate(argparse.Namespace(
        role="seller",
        output_dir=str(tmp_path),
        override_report=None,
        require_mode="balanced",
    ))
    assert discovered is expected


def test_preexisting_e2_cohort_cannot_bypass_active_recovery(
        monkeypatch, tmp_path,
):
    seller_report = tmp_path / "old_seller.json"
    seller_report.write_text("{}", encoding="utf-8")
    (tmp_path / downstream.E1_SELLER_RECOVERY_ACTIVATION_NAME).write_text(
        "{}", encoding="utf-8"
    )
    recovery_report = tmp_path / "recovery.json"
    recovery_report.write_text("{}", encoding="utf-8")
    old_seller = {
        "report": str(seller_report.resolve()),
        "checkpoint": str((tmp_path / "old.zip").resolve()),
        "checkpoint_sha256": "a" * 64,
        "actor_loss_mode": "balanced",
        "source_kind": "legacy_uniform",
        "sampler_mode": "uniform",
        "support_artifacts": {},
    }
    buyer = {
        "source_kind": downstream.E1_PRIMARY_SOURCE_KIND,
        "actor_loss_mode": "balanced",
    }
    cohort = tmp_path / "cohort.json"
    cohort.write_text(json.dumps({
        "schema": "stackpomdp.atari.e2_e1_gate_cohort.v3",
        "code_head": "h" * 40,
        "e1_gates": {"buyer": buyer, "seller": old_seller},
    }), encoding="utf-8")
    monkeypatch.setattr(downstream, "validate_code_root", lambda: "h" * 40)
    monkeypatch.setattr(
        downstream, "validate_e1_gate_record",
        lambda gate, role: deepcopy(gate),
    )
    monkeypatch.setattr(
        downstream, "_validate_cohort_seller_release_binding",
        lambda value, gates: {},
    )
    monkeypatch.setattr(
        downstream, "_discover_authoritative_seller_recovery_gate",
        lambda **kwargs: {
            "found": True,
            "report": str(recovery_report.resolve()),
            "checkpoint": str((tmp_path / "recovery.zip").resolve()),
            "checkpoint_sha256": "b" * 64,
            "actor_loss_mode": "balanced",
            "source_kind": downstream.E1_SELLER_RECOVERY_SOURCE_KIND,
            "sampler_mode": "uniform",
            "support_artifacts": {},
        },
    )
    with pytest.raises(RuntimeError, match="authoritative recovery"):
        downstream.validated_e1_gate_cohort(cohort)


def test_preexisting_e2_cohort_cannot_bypass_pending_threshold_residual(
        monkeypatch, tmp_path,
):
    seller_report = tmp_path / "v1_seller.json"
    seller_report.write_text("{}", encoding="utf-8")
    (
        tmp_path / downstream.E1_SELLER_THRESHOLD_RESIDUAL_ACTIVATION_NAME
    ).write_text("{}", encoding="utf-8")
    old_seller = {
        "report": str(seller_report.resolve()),
        "checkpoint": str((tmp_path / "v1.zip").resolve()),
        "checkpoint_sha256": "a" * 64,
        "actor_loss_mode": "balanced",
        "source_kind": downstream.E1_SELLER_RECOVERY_SOURCE_KIND,
        "sampler_mode": "uniform",
        "support_artifacts": {},
    }
    buyer = {
        "source_kind": downstream.E1_PRIMARY_SOURCE_KIND,
        "actor_loss_mode": "balanced",
    }
    cohort = tmp_path / "cohort.json"
    cohort.write_text(json.dumps({
        "schema": "stackpomdp.atari.e2_e1_gate_cohort.v3",
        "code_head": "h" * 40,
        "e1_gates": {"buyer": buyer, "seller": old_seller},
    }), encoding="utf-8")
    monkeypatch.setattr(downstream, "validate_code_root", lambda: "h" * 40)
    monkeypatch.setattr(
        downstream,
        "validate_e1_gate_record",
        lambda gate, role: deepcopy(gate),
    )
    monkeypatch.setattr(
        downstream,
        "_run_seller_threshold_residual_validator",
        lambda args: {},
    )
    monkeypatch.setattr(
        downstream,
        "_discover_authoritative_seller_recovery_gate",
        lambda **kwargs: pytest.fail("pending residual authority fell back to v1"),
    )
    with pytest.raises(
            RuntimeError,
            match="threshold-residual recovery v2 has not released",
    ):
        downstream.validated_e1_gate_cohort(cohort)


def test_preexisting_e2_cohort_accepts_exact_authoritative_recovery(
        monkeypatch, tmp_path,
):
    report = tmp_path / "recovery.json"
    report.write_text("{}", encoding="utf-8")
    (tmp_path / downstream.E1_SELLER_RECOVERY_ACTIVATION_NAME).write_text(
        "{}", encoding="utf-8"
    )
    seller = {
        "report": str(report.resolve()),
        "report_sha256": downstream.sha256_file(report),
        "checkpoint": str((tmp_path / "recovery.zip").resolve()),
        "checkpoint_sha256": "b" * 64,
        "actor_loss_mode": "balanced",
        "source_kind": downstream.E1_SELLER_RECOVERY_SOURCE_KIND,
        "sampler_mode": "uniform",
        "support_artifacts": {},
    }
    buyer = {
        "source_kind": downstream.E1_PRIMARY_SOURCE_KIND,
        "actor_loss_mode": "balanced",
    }
    cohort = tmp_path / "cohort.json"
    cohort.write_text(json.dumps({
        "schema": "stackpomdp.atari.e2_e1_gate_cohort.v3",
        "code_head": "h" * 40,
        "e1_gates": {"buyer": buyer, "seller": seller},
    }), encoding="utf-8")
    monkeypatch.setattr(downstream, "validate_code_root", lambda: "h" * 40)
    monkeypatch.setattr(
        downstream, "validate_e1_gate_record",
        lambda gate, role: deepcopy(gate),
    )
    monkeypatch.setattr(
        downstream, "_validate_cohort_seller_release_binding",
        lambda value, gates: {},
    )
    monkeypatch.setattr(
        downstream, "_discover_authoritative_seller_recovery_gate",
        lambda **kwargs: {
            "found": True,
            "report": seller["report"],
            "checkpoint": seller["checkpoint"],
            "checkpoint_sha256": seller["checkpoint_sha256"],
            "actor_loss_mode": seller["actor_loss_mode"],
            "source_kind": seller["source_kind"],
            "sampler_mode": seller["sampler_mode"],
            "support_artifacts": seller["support_artifacts"],
        },
    )
    assert downstream.validated_e1_gate_cohort(cohort)["e1_gates"][
        "seller"
    ] == seller


def test_v1_launchers_remain_historical_and_master_uses_residual_v2():
    root = Path(__file__).resolve().parents[1]
    automation = root / "replication/atari/automation"
    train = (
        automation / "run_atari_clean_e1_seller_conditioning_recovery.sh"
    ).read_text(encoding="utf-8")
    selector = (
        automation / "run_e1_seller_conditioning_recovery_selector.sh"
    ).read_text(encoding="utf-8")
    master = (
        automation / "run_atari_clean_e1_primary_economic_to_e2_sequential.sh"
    ).read_text(encoding="utf-8")

    warmup_mode = train.index("--e1-sampler-mode all-equal-v1")
    warmup_budget = train.index("--timesteps 400160", warmup_mode)
    probe = train.index("e1r_validate_warmup_probe", warmup_budget)
    resume = train.index('--resume "$E1R_WARMUP_BASE"', probe)
    target_mode = train.index("--e1-sampler-mode uniform", resume)
    target_budget = train.index("--timesteps 2000800", target_mode)
    family = train.index("e1r_validate_family", target_budget)
    selector_exec = train.index(
        'run_e1_seller_conditioning_recovery_selector.sh', family
    )
    assert warmup_mode < warmup_budget < probe < resume
    assert resume < target_mode < target_budget < family < selector_exec
    assert "--require-pass" in train
    assert "replication.atari.probe_atari_e1_seller_conditioning" in train
    assert "--wandb-project StackPOMDP" in train
    assert "--wandb-group atari_clean_curriculum" in train

    common_source = (
        automation / "atari_e1_seller_conditioning_recovery_common.zsh"
    ).read_text(encoding="utf-8")
    assert "E1R_TARGET_STEPS=(800320 1200480 1600640 2000800 2400960)" in common_source
    assert "replication/atari/probe_atari_e1_seller_conditioning.py" in common_source
    assert 'checkpoint_arguments+=(--checkpoint "$candidate")' in selector
    assert 'checkpoint_arguments+=(--checkpoint "$E1R_WARMUP_BASE")' not in selector
    assert "--screen-seed-start 9000001" in selector
    assert "--confirmation-seed-start 9100001" in selector
    assert "--fixed-seed-start 9200001" in selector
    assert "run_atari_clean_e1_seller_conditioning_recovery.sh" not in master
    assert (
        "run_atari_clean_e1_seller_direct_threshold_residual_recovery.sh"
        in master
    )
    assert master.index("threshold-residual recovery and selection") < master.index(
        "sequential E2 buyer/seller training and selection"
    )


def test_threshold_residual_launchers_enforce_stages_family_and_wandb():
    root = Path(__file__).resolve().parents[1]
    automation = root / "replication/atari/automation"
    train = (
        automation
        / "run_atari_clean_e1_seller_threshold_residual_recovery.sh"
    ).read_text(encoding="utf-8")
    selector = (
        automation / "run_e1_seller_threshold_residual_recovery_selector.sh"
    ).read_text(encoding="utf-8")
    common = (
        automation / "atari_e1_seller_threshold_residual_recovery_common.zsh"
    ).read_text(encoding="utf-8")

    preflight = train.index("run_v2_pure64_preflight\ne1r2_activate")
    activation = train.index("e1r2_activate", preflight)
    runtime = train.index("e1r2_prepare_runtime", activation)
    warmup_start = train.index('print "starting 400160-step', runtime)
    warmup_mode = train.index("--e1-sampler-mode all-equal-v1", warmup_start)
    warmup_budget = train.index("--timesteps 400160", warmup_mode)
    warmup_probe = train.index("run_v2_warmup_probe", warmup_budget)
    warmup_gate = train.index("e1r2_validate_warmup_stage", warmup_probe)
    resume = train.index('--resume "$E1R2_WARMUP_BASE"', warmup_gate)
    target_mode = train.index("--e1-sampler-mode uniform", resume)
    target_budget = train.index("--timesteps 2000800", target_mode)
    family = train.index("e1r2_validate_family", target_budget)
    selector_exec = train.index("$E1R2_SELECTOR_SCRIPT", family)
    assert preflight < activation < runtime < warmup_start
    assert warmup_start < warmup_mode < warmup_budget < warmup_probe
    assert warmup_probe < warmup_gate < resume < target_mode
    assert target_mode < target_budget < family < selector_exec
    assert train.count('"${E1R2_ARCHITECTURE_FLAGS[@]}"') == 3
    assert '"$E1R2_PROBE_MODULE"' in train
    assert "--wandb-project StackPOMDP" in train
    assert "--wandb-group atari_clean_curriculum" in train
    assert "--wandb-job-type \"$E1R2_WARMUP_JOB_TYPE\"" in train
    assert "--wandb-job-type \"$E1R2_TARGET_JOB_TYPE\"" in train

    assert (
        "E1R2_TOKEN=conditioning_recovery_v2_threshold_residual_v1" in common
    )
    assert "E1R2_TARGET_STEPS=(800320 1200480 1600640 2000800 2400960)" in common
    assert "E1R2_LEARNING_RATE=0.0001" in common
    assert "replication/atari/probe_atari_e1_seller_conditioning.py" in common
    assert "replication/atari/probe_atari_e1_seller_threshold_residual.py" in common
    assert (
        "replication/atari/probe_atari_e1_seller_direct_threshold_residual.py"
        in common
    )
    assert 'checkpoint_arguments+=(--checkpoint "$candidate")' in selector
    assert (
        'checkpoint_arguments+=(--checkpoint "$E1R2_WARMUP_BASE")'
        not in selector
    )
    assert "--screen-seed-start 9000001" in selector
    assert "--confirmation-seed-start 9100001" in selector
    assert "--fixed-seed-start 9200001" in selector


def test_gate_schema_binds_probe_failure_release_and_family():
    source = Path(
        recovery.__file__
    ).read_text(encoding="utf-8")
    assert '"prerequisite_failure": dict(activation["prerequisite_failure"])' in source
    assert '"seller_release": dict(activation["seller_release"])' in source
    assert '"warmup_probe": dict(family["warmup"]["conditioning_probe"])' in source
    assert '"training_family": {' in source
