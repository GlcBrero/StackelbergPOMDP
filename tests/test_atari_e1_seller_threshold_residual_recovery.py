import argparse
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from replication.atari import probe_atari_e1_seller_threshold_residual as probe
from replication.atari.automation import (
    validate_atari_e1_seller_threshold_residual_recovery as recovery,
)
from replication.atari.automation import (
    validate_atari_e2_pipeline_artifact as e2,
)


def _activation_args(tmp_path):
    return argparse.Namespace(
        v1_activation=str(tmp_path / "v1_activation.json"),
        v1_warmup_checkpoint=str(tmp_path / "v1_warmup.zip"),
        v1_warmup_probe=str(tmp_path / "v1_probe.json"),
        v1_warmup_training_log=str(tmp_path / "v1.training.jsonl"),
        v1_warmup_evaluation=str(tmp_path / "v1.evaluation.json"),
        seller_release=str(tmp_path / "seller_release.json"),
        e0b=str(tmp_path / "e0b.zip"),
        rom=str(tmp_path / "space_invaders.bin"),
        warmup_checkpoint=str(tmp_path / recovery.WARMUP_CHECKPOINT_NAME),
        warmup_probe=str(tmp_path / recovery.PROBE_ARTIFACT_NAME),
        target_checkpoint=str(tmp_path / recovery.TARGET_CHECKPOINT_NAME),
        preflight_checkpoint=str(tmp_path / recovery.PREFLIGHT_CHECKPOINT_NAME),
        preflight_probe=str(tmp_path / recovery.PREFLIGHT_PROBE_NAME),
        preflight_evaluation=str(tmp_path / "pure64_preflight.evaluation.json"),
        code_root=str(tmp_path),
        output=str(tmp_path / recovery.ACTIVATION_NAME),
    )


def _patch_activation_prerequisites(monkeypatch, args):
    prior = {
        "passed": True,
        "v1_warmup_gate_passed": False,
        "activation": {"path": args.v1_activation, "sha256": "1" * 64},
        "warmup_checkpoint": {
            "path": args.v1_warmup_checkpoint,
            "sha256": "2" * 64,
        },
        "probe": {"path": args.v1_warmup_probe, "sha256": "3" * 64},
        "training_trace": {"path": args.v1_warmup_training_log},
        "evaluation": {"path": args.v1_warmup_evaluation},
    }
    preflight = {
        "passed": True,
        "formal_training_family": False,
        "code_revision": "a" * 40,
        "checkpoint": {
            "path": str(Path(args.preflight_checkpoint).resolve()),
            "sha256": "4" * 64,
        },
        "conditioning_probe": {
            "path": str(Path(args.preflight_probe).resolve()),
            "sha256": "5" * 64,
        },
        "evaluation": {
            "path": str(Path(args.preflight_evaluation).resolve()),
            "behavioral_readiness": {"passed": True},
        },
    }
    monkeypatch.setattr(recovery, "_git_scoped_clean", lambda path: None)
    monkeypatch.setattr(recovery, "_git_revision", lambda path: "a" * 40)
    monkeypatch.setattr(
        recovery, "validate_v1_warmup_failure", lambda **kwargs: prior,
    )
    monkeypatch.setattr(
        recovery, "validate_preflight_evidence", lambda **kwargs: preflight,
    )
    monkeypatch.setattr(recovery.v1, "validate_seller_release", lambda path: {})

    def digest(path):
        resolved = Path(path).resolve()
        if resolved == Path(args.seller_release).resolve():
            return recovery.CANONICAL_SELLER_RELEASE_SHA256
        if resolved == Path(args.e0b).resolve():
            return recovery.CANONICAL_E0B_SHA256
        if resolved == Path(args.rom).resolve():
            return recovery.CANONICAL_ROM_SHA256
        return "f" * 64

    monkeypatch.setattr(recovery, "sha256_file", digest)
    return prior, preflight


def test_activation_binds_pure64_preflight_schedule_and_exact_lr(
        monkeypatch, tmp_path,
):
    args = _activation_args(tmp_path)
    prior, preflight = _patch_activation_prerequisites(monkeypatch, args)
    value = recovery.build_activation(args)

    architecture = recovery.canonical_economic_architecture()
    assert architecture["base_head_input_features"] == 64
    assert architecture["direct_extra_input_features"] == 0
    assert architecture["parameterization"] == (
        "seller_threshold_residual_beta_v1"
    )
    assert architecture["current_threshold"]["new_observation_fields"] == []
    assert architecture["mean_transform"]["threshold_weight"] == 0.5
    assert architecture["mean_transform"]["base_weight"] == 0.5

    assert value["prerequisite_v1_failure"] == prior
    assert value["pure64_preflight"] == preflight
    assert value["code_revision"] == preflight["code_revision"]
    change = value["controlled_change"]
    assert "64-input seller economic head" in (
        change["only_architectural_change_from_v1"]
    )
    assert change["economic_architecture"] == architecture
    assert change["new_observation_fields"] == []
    assert change["environment_unchanged"]
    assert change["samplers_unchanged"]
    assert change["ppo_hyperparameters_unchanged"]
    assert change["learning_rate"] == recovery.LEARNING_RATE == 1.0e-4
    assert change["fixed_anchor_is_inductive_bias"]
    assert change["learned_threshold_slope_claim"] is False

    protocol = value["protocol"]
    assert protocol["economic_architecture"] == architecture
    assert protocol["training_config"] == recovery.canonical_training_config()
    assert protocol["training_config"]["state_features"] == 64
    assert protocol["training_config"]["learning_rate"] == 1.0e-4
    assert protocol["training_config"]["economic_threshold_residual"] is True
    assert protocol["stage_order"] == [
        "all_equal_warmup", "independent_uniform_target",
    ]
    assert protocol["warmup"]["additional_timesteps"] == 400_160
    assert protocol["warmup"]["selectable"] is False
    assert protocol["target"]["additional_timesteps"] == 2_000_800
    assert protocol["target"]["candidate_count"] == 6
    assert protocol["target"]["candidate_timesteps"] == list(
        recovery.TARGET_CANDIDATE_TIMESTEPS
    )
    assert protocol["warmup"]["checkpoint"] not in (
        protocol["target"]["candidate_paths"]
    )


def test_activation_fails_closed_before_preflight_evidence(
        monkeypatch, tmp_path,
):
    args = _activation_args(tmp_path)
    _patch_activation_prerequisites(monkeypatch, args)

    def reject(**kwargs):
        raise ValueError("pure-64 real-ALE preflight did not pass")

    monkeypatch.setattr(recovery, "validate_preflight_evidence", reject)
    with pytest.raises(ValueError, match="preflight did not pass"):
        recovery.build_activation(args)


def test_activation_rejects_revision_change_while_evidence_is_bound(
        monkeypatch, tmp_path,
):
    args = _activation_args(tmp_path)
    _patch_activation_prerequisites(monkeypatch, args)
    revisions = iter(["a" * 40, "b" * 40])
    monkeypatch.setattr(recovery, "_git_revision", lambda path: next(revisions))
    with pytest.raises(ValueError, match="changed while building activation"):
        recovery.build_activation(args)


def test_activation_rechecks_scoped_cleanliness_after_evidence_validation(
        monkeypatch, tmp_path,
):
    args = _activation_args(tmp_path)
    _patch_activation_prerequisites(monkeypatch, args)
    calls = 0

    def clean_then_dirty(path):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("scoped code became dirty")

    monkeypatch.setattr(recovery, "_git_scoped_clean", clean_then_dirty)
    with pytest.raises(ValueError, match="became dirty"):
        recovery.build_activation(args)
    assert calls == 2


def test_formal_candidate_revision_must_equal_activation():
    activation = {
        "code_revision": "a" * 40,
        "protocol": {"training_config": recovery.canonical_training_config()},
    }
    metadata = {
        "economic_architecture": recovery.canonical_economic_architecture(),
        "training_config": recovery.canonical_training_config(),
        "e1_training_code_revision": "b" * 40,
    }
    with pytest.raises(ValueError, match="differs from activation"):
        recovery._validate_architecture_metadata(metadata, activation)


def test_exact_v1_negative_artifact_hashes_are_pinned():
    assert recovery.V1_CODE_REVISION == (
        "ca57c6daeb1b7282fb8f91018474d1fae8c48752"
    )
    for digest in (
        recovery.V1_ACTIVATION_SHA256,
        recovery.V1_WARMUP_SHA256,
        recovery.V1_PROBE_SHA256,
        recovery.V1_TRAINING_LOG_SHA256,
        recovery.V1_EVALUATION_SHA256,
    ):
        assert len(digest) == 64
        int(digest, 16)


def _patch_v1_failure_validation(monkeypatch, tmp_path, gate):
    activation_path = tmp_path / "v1_activation.json"
    warmup = tmp_path / "v1_warmup.zip"
    probe_path = tmp_path / "v1_probe.json"
    trace = tmp_path / "v1.training.jsonl"
    evaluation = tmp_path / "v1.evaluation.json"
    for path in (activation_path, warmup, probe_path, trace, evaluation):
        path.write_bytes(path.name.encode())
    activation = {
        "code_revision": recovery.V1_CODE_REVISION,
        "protocol": {
            "warmup": {
                "checkpoint": str(warmup.resolve()),
                "conditioning_probe": {"path": str(probe_path.resolve())},
            },
            "target": {"candidate_paths": []},
        },
        "e0b_source": {"path": str((tmp_path / "e0b.zip").resolve())},
    }
    reported = {
        "probe": recovery.v1_probe.PROBE_NAME,
        "checkpoint": {
            "path": str(warmup.resolve()),
            "sha256": recovery.V1_WARMUP_SHA256,
        },
        "all_equal_thresholds": [{"event_beta_mean_prices": [0.5] * 5}],
        "current_coordinate_only_sensitivity": {
            "rows": [{
                "low_beta_mean_price": 0.49,
                "high_beta_mean_price": 0.51,
            } for _ in range(5)],
        },
        "warmup_gate": gate,
    }
    monkeypatch.setattr(recovery, "_exact_hash", lambda *args: None)
    monkeypatch.setattr(
        recovery.v1, "validate_activation", lambda path: deepcopy(activation),
    )
    monkeypatch.setattr(
        recovery.v1, "_load_candidate_metadata",
        lambda *args, **kwargs: {"atari_e1_sampler_history": []},
    )
    monkeypatch.setattr(recovery.v1, "_validate_warmup_metadata", lambda *a: None)
    monkeypatch.setattr(
        recovery.v1, "_validate_training_trace",
        lambda path, **kwargs: {"path": str(Path(path).resolve())},
    )
    monkeypatch.setattr(
        recovery.v1, "_validate_evaluation",
        lambda path, **kwargs: {"path": str(Path(path).resolve())},
    )
    monkeypatch.setattr(recovery, "load_json", lambda path: deepcopy(reported))
    monkeypatch.setattr(
        recovery.v1_probe, "warmup_diagnostic_gate", lambda *args: deepcopy(gate),
    )
    monkeypatch.setattr(
        recovery.v1_probe, "run_probe_from_checkpoints",
        lambda **kwargs: deepcopy(reported),
    )
    return {
        "activation": activation_path,
        "warmup_checkpoint": warmup,
        "probe": probe_path,
        "training_log": trace,
        "evaluation": evaluation,
    }


def test_v1_failure_must_be_endpoint_only_not_a_second_gate_failure(
        monkeypatch, tmp_path,
):
    gate = {
        "passed": False,
        "checks": {
            "all_outputs_finite_and_in_unit_interval": {"passed": True},
            "minimum_all_one_minus_all_zero_beta_mean_price": {
                "passed": False,
                "required": recovery.MINIMUM_ENDPOINT_RESPONSE,
                "actual": 0.01,
            },
            "largest_adjacent_threshold_price_reversal": {"passed": False},
        },
    }
    paths = _patch_v1_failure_validation(monkeypatch, tmp_path, gate)
    with pytest.raises(ValueError, match="other than endpoint response"):
        recovery.validate_v1_warmup_failure(**paths)

    gate["checks"]["largest_adjacent_threshold_price_reversal"][
        "passed"
    ] = True
    paths = _patch_v1_failure_validation(monkeypatch, tmp_path, gate)
    result = recovery.validate_v1_warmup_failure(**paths)
    assert result["passed"] is True
    assert result["v1_warmup_gate_passed"] is False
    assert result["probe"]["minimum_endpoint_response"] == 0.01


def test_scoped_clean_includes_the_residual_probe(monkeypatch, tmp_path):
    monkeypatch.setattr(recovery.v1, "_git_scoped_clean", lambda path: None)
    observed = {}

    def run(command, **kwargs):
        observed["command"] = command
        return SimpleNamespace(stdout=" M residual_probe.py\n")

    monkeypatch.setattr(recovery.subprocess, "run", run)
    with pytest.raises(ValueError, match="clean residual probe"):
        recovery._git_scoped_clean(tmp_path)
    assert (
        "replication/atari/probe_atari_e1_seller_threshold_residual.py"
        in observed["command"]
    )


def _passing_probe_arrays():
    thresholds = np.linspace(0.0, 1.0, 5, dtype=np.float64)[:, None]
    event_offsets = 0.001 * np.arange(5, dtype=np.float64)[None, :]
    base_all_equal = 0.45 + 0.02 * thresholds + event_offsets
    final_all_equal = 0.5 * base_all_equal + 0.5 * thresholds
    base_low = np.full(5, 0.45, dtype=np.float64)
    base_high = np.full(5, 0.47, dtype=np.float64)
    final_low = 0.5 * base_low
    final_high = 0.5 * base_high + 0.5
    return {
        "final_all_equal": final_all_equal,
        "final_coordinate_low": final_low,
        "final_coordinate_high": final_high,
        "base_all_equal": base_all_equal,
        "base_coordinate_low": base_low,
        "base_coordinate_high": base_high,
    }


def test_probe_gates_final_and_learned_base_responses_separately():
    arrays = _passing_probe_arrays()
    gate = probe.threshold_residual_warmup_gate(**arrays)
    assert gate["passed"]
    assert gate["fixed_anchor_is_inductive_bias"]
    assert gate["learned_threshold_slope_claim"] is False
    checks = gate["checks"]
    assert checks[
        "minimum_all_one_minus_all_zero_beta_mean_price"
    ]["passed"]
    assert checks[
        "minimum_current_coordinate_final_beta_mean_response"
    ]["passed"]
    assert checks[
        "minimum_learned_base_all_one_minus_all_zero_mean_response"
    ]["passed"]
    assert checks[
        "minimum_learned_base_current_coordinate_mean_response"
    ]["passed"]

    no_base_endpoint = dict(arrays)
    no_base_endpoint["base_all_equal"] = np.repeat(
        arrays["base_all_equal"][:1], 5, axis=0
    )
    failed = probe.threshold_residual_warmup_gate(**no_base_endpoint)
    assert failed["passed"] is False
    assert failed["checks"][
        "minimum_learned_base_all_one_minus_all_zero_mean_response"
    ]["passed"] is False

    no_base_coordinate = dict(arrays)
    no_base_coordinate["base_coordinate_high"] = arrays[
        "base_coordinate_low"
    ]
    failed = probe.threshold_residual_warmup_gate(**no_base_coordinate)
    assert failed["passed"] is False
    assert failed["checks"][
        "minimum_learned_base_current_coordinate_mean_response"
    ]["passed"] is False


def test_probe_rejects_missing_pure64_model_architecture_provenance(monkeypatch):
    architecture = probe.canonical_economic_architecture()

    class Policy:
        economic_threshold_residual = True
        economic_head = [SimpleNamespace(in_features=64)]

        @staticmethod
        def economic_architecture_provenance():
            return architecture

    model = SimpleNamespace(policy=Policy())
    metadata = {
        "economic_architecture": architecture,
        "training_config": {"economic_threshold_residual": True},
    }
    monkeypatch.setattr(
        probe.base_probe, "validate_loaded_seller",
        lambda *args: {"checks": {"base_contract": True}},
    )
    with pytest.raises(ValueError, match="model_economic_architecture"):
        probe.validate_loaded_seller(model, metadata, {})
    setattr(model, probe.trainer.ECONOMIC_ARCHITECTURE_ATTRIBUTE, architecture)
    with pytest.raises(ValueError, match="training_code_revision_is_full_sha"):
        probe.validate_loaded_seller(model, metadata, {})
    revision = "c" * 40
    setattr(
        model, probe.trainer.E1_TRAINING_CODE_REVISION_ATTRIBUTE, revision
    )
    metadata["e1_training_code_revision"] = revision
    result = probe.validate_loaded_seller(model, metadata, {})
    assert result["passed"]
    assert result["checks"]["pure_64_input_base_head"]
    assert result["economic_architecture"] == architecture
    assert result["e1_training_code_revision"] == revision


def test_probe_is_read_only_no_ale_and_require_pass_exits_two(
        monkeypatch, tmp_path,
):
    checkpoint = tmp_path / "seller.zip"
    checkpoint.write_bytes(b"checkpoint")
    fake_model = SimpleNamespace(policy=object())
    metadata = {"sha256": "a" * 64}
    from replication.atari import evaluate_atari_meta_response_sb3 as evaluator

    monkeypatch.setattr(evaluator, "checkpoint_path", lambda path: checkpoint)
    monkeypatch.setattr(evaluator, "checkpoint_sha256", lambda path: "a" * 64)
    monkeypatch.setattr(
        evaluator, "validate_e0b", lambda *args, **kwargs: {"sha256": "b" * 64},
    )
    monkeypatch.setattr(
        evaluator, "load_candidate", lambda *args, **kwargs: (fake_model, metadata),
    )
    monkeypatch.setattr(
        probe, "validate_loaded_seller",
        lambda *args: {"passed": True, "checks": {}},
    )
    monkeypatch.setattr(
        probe, "collect_conditioning_report",
        lambda model: {"warmup_gate": {"passed": False}},
    )
    result = probe.run_probe_from_checkpoints(
        checkpoint=checkpoint, e0b_checkpoint=tmp_path / "e0b.zip"
    )
    assert result["execution"] == {
        "read_only_checkpoint": True,
        "ale_instantiated": False,
        "environment_steps": 0,
        "deterministic_statistics": [
            "learned base Beta mean", "final residual Beta mean",
        ],
        "device": "cpu",
    }
    monkeypatch.setattr(
        probe, "run_probe_from_checkpoints", lambda **kwargs: result,
    )
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)
    assert probe.main([
        "--checkpoint", str(checkpoint),
        "--e0b-checkpoint", str(tmp_path / "e0b.zip"),
        "--require-pass",
    ]) == 2


def _behavioral_fixture():
    fixed = []
    for index in range(11):
        threshold = index / 10.0
        price = 0.25 + 0.5 * threshold
        purchases = 5 if price <= threshold else 0
        payments = purchases * price
        fixed.append({
            "opponent_value": threshold,
            "protocol": {"passed": True},
            "summary": {
                "mean_price": price,
                "mean_payments": payments,
                "mean_seller_shots_fired": 5.0 if purchases == 0 else 0.0,
                "mean_seller_game_reward": 5.0 if purchases == 0 else 0.0,
                "mean_controlled_payoff": payments + (
                    0.5 if purchases == 0 else 0.0
                ),
            },
            "episode_rows": [{"purchases": purchases} for _ in range(20)],
        })
    random_result = {
        "protocol": {"passed": True},
        "summary": {"mean_controlled_payoff": 2.0},
        "episode_rows": [{"seller_reward": 2.0} for _ in range(20)],
    }
    return random_result, fixed


def test_real_ale_behavioral_readiness_requires_retention_and_high_sales():
    from replication.atari import evaluate_atari_meta_response_sb3 as evaluator

    random_result, fixed = _behavioral_fixture()
    gate = evaluator.seller_threshold_residual_behavioral_gate(
        random_result=random_result,
        fixed_results=fixed,
        mechanics_verified=True,
    )
    assert gate["passed"]
    assert gate["mechanics_passed"]
    assert gate["learned_threshold_slope_claim"] is False
    assert gate["economic_constant_price_baseline"]["regret_reduction"] > 0.05

    fixed[0]["episode_rows"][0]["purchases"] = 1
    failed = evaluator.seller_threshold_residual_behavioral_gate(
        random_result=random_result,
        fixed_results=fixed,
        mechanics_verified=True,
    )
    assert failed["passed"] is False
    check = next(
        row for row in failed["checks"]
        if row["name"] == "threshold 0 maximum per-episode purchases"
    )
    assert check["passed"] is False


def test_launchers_are_preflighted_gated_fixed_lr_and_wandb_visible():
    root = Path(__file__).resolve().parents[1]
    automation = root / "replication/atari/automation"
    common = (automation / (
        "atari_e1_seller_threshold_residual_recovery_common.zsh"
    )).read_text(encoding="utf-8")
    train = (automation / (
        "run_atari_clean_e1_seller_threshold_residual_recovery.sh"
    )).read_text(encoding="utf-8")
    selector = (automation / (
        "run_e1_seller_threshold_residual_recovery_selector.sh"
    )).read_text(encoding="utf-8")

    assert "typeset -gr E1R2_LEARNING_RATE=0.0001" in common
    assert "${STACKPOMDP_E1R2_LEARNING_RATE" not in common
    assert "E1R2_PREFLIGHT_BASE" in common
    assert "E1R2_PREFLIGHT_PROBE" in common
    assert "E1R2_V1_GUARD_LOCK" in common
    assert "E1R2_TARGET_STEPS=(800320 1200480 1600640 2000800 2400960)" in common
    assert "threshold_residual_v1" in common
    assert 'E1R2_ARCHITECTURE_FLAGS=(--economic-threshold-residual)' in common
    assert train.count('"${E1R2_ARCHITECTURE_FLAGS[@]}"') == 3
    assert train.count('--learning-rate "$E1R2_LEARNING_RATE"') == 3
    assert '--wandb-job-type "$E1R2_WARMUP_JOB_TYPE"' in train
    assert '--wandb-job-type "$E1R2_TARGET_JOB_TYPE"' in train
    assert "starting no-W&B 20500-step $E1R2_PROFILE_LABEL real-ALE preflight" in train
    assert train.index("\nrun_v2_pure64_preflight\n") < train.index(
        "\ne1r2_activate\n"
    )
    target = train.index('--resume "$E1R2_WARMUP_BASE"')
    assert train.rindex("e1r2_validate_warmup_stage", 0, target) < target
    assert "--require-pass" in train
    assert 'checkpoint_arguments+=(--checkpoint "$candidate")' in selector
    selection_slice = selector[
        selector.index("checkpoint_arguments=()"):
        selector.index("revision=")
    ]
    assert "E1R2_WARMUP_BASE" not in selection_slice


def test_failed_selection_never_publishes_alias_or_gate(tmp_path):
    gate = tmp_path / recovery.GATE_NAME
    args = SimpleNamespace(gate_output=str(gate))
    assert recovery.write_or_validate_selection_gate(
        args, {"passed": False}
    ) is None
    assert not gate.exists()

    gate.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="failed v2 selector retained gate"):
        recovery.write_or_validate_selection_gate(args, {"passed": False})
    with pytest.raises(ValueError, match="requires gate output"):
        recovery.write_or_validate_selection_gate(
            SimpleNamespace(gate_output=None), {"passed": True}
        )


def test_residual_report_dispatch_has_priority(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(
        e2, "_validate_seller_threshold_residual_gate",
        lambda args: called.append("v2") or {"passed": True},
    )
    monkeypatch.setattr(
        e2, "_validate_seller_recovery_gate",
        lambda args: called.append("v1") or {"passed": True},
    )
    result = e2.validate_e1_gate(argparse.Namespace(
        role="seller",
        report=str(tmp_path / e2.E1_SELLER_THRESHOLD_RESIDUAL_REPORT_NAME),
        checkpoint=str(tmp_path / "selected.zip"),
        actor_loss_mode="balanced",
    ))
    assert result["passed"]
    assert called == ["v2"]


def test_residual_authority_is_fail_closed_for_orphans_and_pending_gate(
        monkeypatch, tmp_path,
):
    report = tmp_path / e2.E1_SELLER_THRESHOLD_RESIDUAL_REPORT_NAME
    report.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="without their authoritative activation"):
        e2._discover_authoritative_seller_threshold_residual_gate(
            output_dir=tmp_path, override_report=None,
        )

    activation = tmp_path / e2.E1_SELLER_THRESHOLD_RESIDUAL_ACTIVATION_NAME
    activation.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        e2, "_run_seller_threshold_residual_validator", lambda args: {},
    )
    report.unlink()
    pending = e2._discover_authoritative_seller_threshold_residual_gate(
        output_dir=tmp_path, override_report=None,
    )
    assert pending["found"] is False
    assert pending["state"] == "threshold_residual_training_or_selection_pending"

    report.write_text(json.dumps({
        "passed": True, "role": "seller", "evaluator": e2.E1_EVALUATOR,
    }), encoding="utf-8")
    pending = e2._discover_authoritative_seller_threshold_residual_gate(
        output_dir=tmp_path, override_report=None,
    )
    assert pending["found"] is False
    assert pending["state"] == "threshold_residual_gate_publication_pending"


def _cohort(tmp_path, seller):
    buyer = {
        "source_kind": e2.E1_PRIMARY_SOURCE_KIND,
        "actor_loss_mode": "balanced",
    }
    path = tmp_path / "cohort.json"
    path.write_text(json.dumps({
        "schema": "stackpomdp.atari.e2_e1_gate_cohort.v3",
        "code_head": "h" * 40,
        "e1_gates": {"buyer": buyer, "seller": seller},
    }), encoding="utf-8")
    return path


def test_cached_cohort_cannot_bypass_residual_authority(
        monkeypatch, tmp_path,
):
    report = tmp_path / e2.E1_SELLER_THRESHOLD_RESIDUAL_REPORT_NAME
    report.write_text("{}", encoding="utf-8")
    authoritative = {
        "found": True,
        "report": str(report.resolve()),
        "checkpoint": str((tmp_path / "v2.zip").resolve()),
        "checkpoint_sha256": "b" * 64,
        "actor_loss_mode": "balanced",
        "source_kind": e2.E1_SELLER_THRESHOLD_RESIDUAL_SOURCE_KIND,
        "sampler_mode": "uniform",
        "support_artifacts": {"economic_architecture": (
            e2.canonical_seller_threshold_residual_architecture()
        )},
    }
    seller = {
        "report": str((tmp_path / "old.json").resolve()),
        "report_sha256": "a" * 64,
        "checkpoint": str((tmp_path / "old.zip").resolve()),
        "checkpoint_sha256": "a" * 64,
        "actor_loss_mode": "balanced",
        "source_kind": "legacy_uniform",
        "sampler_mode": "uniform",
        "support_artifacts": {},
    }
    monkeypatch.setattr(e2, "validate_code_root", lambda: "h" * 40)
    monkeypatch.setattr(
        e2, "validate_e1_gate_record", lambda gate, role: deepcopy(gate),
    )
    monkeypatch.setattr(
        e2, "_validate_cohort_seller_release_binding", lambda value, gates: {},
    )
    monkeypatch.setattr(
        e2, "_discover_authoritative_seller_threshold_residual_gate",
        lambda **kwargs: deepcopy(authoritative),
    )
    stale = _cohort(tmp_path, seller)
    with pytest.raises(RuntimeError, match="authoritative threshold-residual"):
        e2.validated_e1_gate_cohort(stale)

    stale.unlink()
    exact = {
        "report": authoritative["report"],
        "report_sha256": e2.sha256_file(report),
        "checkpoint": authoritative["checkpoint"],
        "checkpoint_sha256": authoritative["checkpoint_sha256"],
        "actor_loss_mode": authoritative["actor_loss_mode"],
        "source_kind": authoritative["source_kind"],
        "sampler_mode": authoritative["sampler_mode"],
        "support_artifacts": authoritative["support_artifacts"],
    }
    cohort = _cohort(tmp_path, exact)
    assert e2.validated_e1_gate_cohort(cohort)["e1_gates"]["seller"] == exact
