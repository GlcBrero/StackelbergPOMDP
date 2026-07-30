"""Regression tests for the gated seller-v5 automation namespace."""

import inspect
import json
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import gym

from replication.atari import evaluate_atari_meta_response_sb3 as evaluator
from replication.atari import probe_atari_e1_seller_shared_context as probe
from replication.atari import train_atari_meta_response_sb3 as trainer
from replication.atari.automation import (
    validate_atari_e1_seller_shared_context_v5 as validator,
)
from stackelberg_pomdp.atari.protocol import action_space, observation_space
from stackelberg_pomdp.atari.stackpomdp_policy import StackPOMDPAtariPolicy


ROOT = Path(__file__).resolve().parents[1]
AUTOMATION = ROOT / "replication/atari/automation"
COMMON = AUTOMATION / "atari_e1_seller_shared_context_v5_common.zsh"
LAUNCHER = AUTOMATION / "run_atari_clean_e1_seller_shared_context_v5.sh"
EXPOSURE_LAUNCHER = (
    AUTOMATION
    / "run_atari_clean_e1_seller_shared_context_v5_exposure_v2.sh"
)


def _shell_function(path, name):
    lines = path.read_text(encoding="utf-8").splitlines()
    start = lines.index(f"function {name}() {{")
    for stop in range(start + 1, len(lines)):
        if lines[stop] == "}":
            return "\n".join(lines[start:stop + 1])
    raise AssertionError(f"unterminated shell function {name}")


def test_v5_automation_contract_is_exact_and_versioned():
    assert validator.TOKEN == "conditioning_recovery_v5_shared_context_v1"
    assert validator.ARCHITECTURE == "seller_shared_context_beta_v5"
    assert validator.SOURCE_KIND == (
        "seller_conditioning_recovery_v5_shared_context_v1"
    )
    assert validator.canonical_architecture() == (
        probe.canonical_economic_architecture()
    )
    initialization = validator.canonical_initialization()
    assert initialization["optimizer_groups"] == [
        {
            "name": "seller_v5_live",
            "learning_rate": 5.0e-4,
            "gradient_clip_norm": 0.5,
        },
        {
            "name": "seller_v5_context",
            "learning_rate": 2.0e-3,
            "gradient_clip_norm": 0.5,
        },
        {
            "name": "seller_v5_critic",
            "learning_rate": 1.0e-4,
            "gradient_clip_norm": 0.5,
        },
    ]
    config = validator.canonical_training_config()
    assert config["economic_architecture"] == validator.ARCHITECTURE
    assert config["actor_loss_mode"] == "balanced"
    assert config["learning_rate"] == 5.0e-4
    assert config["max_grad_norm"] == 0.5
    assert (validator.SMOKE_TIMESTEPS, validator.PREFLIGHT_TIMESTEPS) == (
        20_500,
        82_000,
    )
    assert validator.FORMAL_TIMESTEPS == 2_000_800
    assert validator.WANDB_PROJECT == "StackPOMDP"
    assert validator.WANDB_JOB_TYPE == "atari_e1_seller_v5_standard_v1_formal"
    assert len(validator.WANDB_JOB_TYPE) <= 64
    assert validator.WANDB_NAME == (
        "atari_clean_e1_seller_conditioning_recovery_v5_shared_context_v1_"
        "uniform_seed1_2000800_local"
    )


def test_v5_exposure_v2_changes_only_fresh_exposure_and_holdout_namespace():
    validator.configure_protocol(validator.STANDARD_PROTOCOL)
    standard = {
        "architecture": validator.canonical_architecture(),
        "initialization": validator.canonical_initialization(),
        "training": validator.canonical_training_config(),
        "sampler": validator.canonical_sampler(),
        "smoke_timesteps": validator.SMOKE_TIMESTEPS,
        "formal_timesteps": validator.FORMAL_TIMESTEPS,
    }
    try:
        validator.configure_protocol(validator.EXPOSURE_PROTOCOL)
        assert validator.ACTIVE_PROTOCOL == "exposure_v2"
        assert validator.TOKEN == (
            "conditioning_recovery_v5_shared_context_exposure_v2"
        )
        assert validator.SOURCE_KIND == (
            "seller_conditioning_recovery_v5_shared_context_exposure_v2"
        )
        assert validator.PREFLIGHT_TIMESTEPS == 400_160
        assert validator.PREFLIGHT_BEHAVIOR_SEED_START == 11_400_001
        assert validator.PREFLIGHT_FIXED_SEED_START == 11_500_001
        assert validator.EXPOSURE_V2_DIAGNOSTIC_REVISION == (
            "c4a7dcd92b621c0884f3dcef0b170961e1ec625b"
        )
        assert validator.SMOKE_TIMESTEPS == standard["smoke_timesteps"]
        assert validator.FORMAL_TIMESTEPS == standard["formal_timesteps"]
        assert validator.canonical_architecture() == standard["architecture"]
        assert validator.canonical_initialization() == standard["initialization"]
        assert validator.canonical_training_config() == standard["training"]
        assert validator.canonical_sampler() == standard["sampler"]
        provenance = validator.canonical_protocol_provenance()
        assert provenance[
            "only_preflight_exposure_and_holdout_seed_namespace_changed"
        ] is True
        assert provenance["architecture_changed"] is False
        assert provenance["behavioral_gate_changed"] is False
        assert provenance["fresh_from_canonical_e0b_not_resume"] is True
        assert provenance["expected_preflight_episodes"] == 1_952
        assert provenance["expected_preflight_rollout_iterations"] == 488
        assert provenance["expected_preflight_adam_step"] == 1_952
        assert provenance["predecessor_negative_evidence"] == (
            validator.V1_NEGATIVE_EVIDENCE
        )
        assert validator.WANDB_JOB_TYPE == (
            "atari_e1_seller_v5_exposure_v2_formal"
        )
        assert len(validator.WANDB_JOB_TYPE) <= 64
        assert validator.WANDB_NAME == (
            "atari_clean_e1_seller_conditioning_recovery_v5_shared_context_"
            "exposure_v2_uniform_seed1_2000800_local"
        )
    finally:
        validator.configure_protocol(validator.STANDARD_PROTOCOL)


def test_scoped_clean_checks_cover_the_base_conditioning_probe():
    dependency = "replication/atari/probe_atari_e1_seller_conditioning.py"
    assert dependency in _shell_function(COMMON, "e1v5_require_scoped_clean")
    assert dependency in inspect.getsource(validator._git_scoped_clean)
    exposure_wrapper = (
        "replication/atari/automation/"
        "run_atari_clean_e1_seller_shared_context_v5_exposure_v2.sh"
    )
    assert exposure_wrapper in _shell_function(
        COMMON, "e1v5_require_scoped_clean"
    )
    assert exposure_wrapper in inspect.getsource(validator._git_scoped_clean)


def test_v5_probe_accepts_only_the_exact_shared_context_checkpoint_contract():
    image = gym.spaces.Box(0, 255, shape=(84, 84, 4), dtype=np.uint8)
    policy = StackPOMDPAtariPolicy(
        observation_space(image, 6),
        action_space(6),
        lambda _: 5.0e-4,
        economic_role="seller",
        economic_input_mode="full",
        economic_architecture=validator.ARCHITECTURE,
        pretrained_lr_scale=0.1,
    )
    frozen = trainer.gameplay_actor_sha256(policy)
    revision = "a" * 40
    e0b_sha256 = "b" * 64
    architecture = validator.canonical_architecture()
    initialization = validator.canonical_initialization()
    model = SimpleNamespace(
        policy=policy,
        num_timesteps=validator.PREFLIGHT_TIMESTEPS,
        n_steps=205,
        gamma=1.0,
        gae_lambda=1.0,
        max_grad_norm=0.5,
        atari_e1_source_provenance={
            "frozen_gameplay_actor_sha256": frozen,
        },
    )
    setattr(model, trainer.ECONOMIC_ARCHITECTURE_ATTRIBUTE, architecture)
    setattr(
        model,
        trainer.SHARED_CONTEXT_INITIALIZATION_ATTRIBUTE,
        initialization,
    )
    setattr(model, trainer.E1_TRAINING_CODE_REVISION_ATTRIBUTE, revision)
    metadata = {
        "role": "seller",
        "economic_input_mode": "full",
        "sha256": "c" * 64,
        "training_timesteps": validator.PREFLIGHT_TIMESTEPS,
        "training_config": {
            "algorithm": "PPO",
            "economic_architecture": validator.ARCHITECTURE,
        },
        "e0b_source_provenance": {
            "sha256": e0b_sha256,
            "zero_initialized_actor_state_indices": [9, 10, 11, 12, 13],
            "frozen_gameplay_actor_sha256": frozen,
        },
        "atari_e1_sampler_provenance": validator.canonical_sampler(),
        "atari_e1_sampler_history": validator.canonical_sampler_history(),
        "economic_architecture": architecture,
        "shared_context_initialization": initialization,
        "e1_training_code_revision": revision,
    }
    result = probe.validate_loaded_seller(
        model, metadata, {"sha256": e0b_sha256}
    )
    assert result["passed"] is True
    assert all(result["checks"].values())

    setattr(model, trainer.SHARED_CONTEXT_INITIALIZATION_ATTRIBUTE, None)
    with pytest.raises(ValueError, match="initialization_provenance_exact"):
        probe.validate_loaded_seller(model, metadata, {"sha256": e0b_sha256})


def _optimizer_row():
    row = {
        "record_kind": "optimizer",
        **validator.V5_LEARNING_RATES,
    }
    for key in validator.V5_GRADIENT_NORM_KEYS:
        row[key] = 0.25 if key.endswith("_pre") else 0.20
    return row


def _write_rows(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_v5_optimizer_telemetry_gate_accepts_only_exact_rates_and_group_norms(
        tmp_path,
):
    path = tmp_path / "trace.jsonl"
    row = _optimizer_row()
    _write_rows(path, [row])
    record = validator._validate_v5_optimizer_telemetry(
        path, expected_optimizers=1
    )
    assert record == {
        "optimizer_rows": 1,
        "learning_rates": validator.V5_LEARNING_RATES,
        "gradient_norm_keys": list(validator.V5_GRADIENT_NORM_KEYS),
        "independent_group_gradient_clip_norm": 0.5,
    }

    float32_roundoff = dict(row)
    float32_roundoff["train/seller_v5_critic_grad_norm_pre"] = (
        0.34057314693927765
    )
    float32_roundoff["train/seller_v5_critic_grad_norm_post"] = (
        0.3405742794275284
    )
    _write_rows(path, [float32_roundoff])
    validator._validate_v5_optimizer_telemetry(
        path, expected_optimizers=1
    )

    material_increase = dict(row)
    material_increase["train/seller_v5_critic_grad_norm_pre"] = 0.25
    material_increase["train/seller_v5_critic_grad_norm_post"] = 0.25001
    _write_rows(path, [material_increase])
    with pytest.raises(ValueError, match="independently at 0.5"):
        validator._validate_v5_optimizer_telemetry(
            path, expected_optimizers=1
        )

    cap_violation = dict(row)
    cap_violation["train/seller_v5_critic_grad_norm_pre"] = 0.8
    cap_violation["train/seller_v5_critic_grad_norm_post"] = 0.50001
    _write_rows(path, [cap_violation])
    with pytest.raises(ValueError, match="independently at 0.5"):
        validator._validate_v5_optimizer_telemetry(
            path, expected_optimizers=1
        )

    bad_rate = dict(row)
    bad_rate["train/context_learning_rate"] = 5.0e-4
    _write_rows(path, [bad_rate])
    with pytest.raises(ValueError, match="context_learning_rate"):
        validator._validate_v5_optimizer_telemetry(
            path, expected_optimizers=1
        )

    missing_norm = dict(row)
    missing_norm.pop("train/seller_v5_context_grad_norm_post")
    _write_rows(path, [missing_norm])
    with pytest.raises(ValueError, match="lacks exact group norms"):
        validator._validate_v5_optimizer_telemetry(
            path, expected_optimizers=1
        )

    unclipped = dict(row)
    unclipped["train/seller_v5_context_grad_norm_pre"] = 2.0
    unclipped["train/seller_v5_context_grad_norm_post"] = 0.6
    _write_rows(path, [unclipped])
    with pytest.raises(ValueError, match="independently at 0.5"):
        validator._validate_v5_optimizer_telemetry(
            path, expected_optimizers=1
        )


class _Policy:
    def __init__(self):
        self.calls = []

    def set_v4_context_ablation(self, value):
        self.calls.append(("v4", bool(value)))

    def set_v5_context_ablation(self, value):
        self.calls.append(("v5", bool(value)))


class _Model:
    def __init__(self):
        self.policy = _Policy()

    def predict(self, observation, deterministic):
        assert deterministic is True
        return np.asarray([0.0, 0.5], dtype=np.float32), None


class _OneStepEnv:
    def __init__(self):
        self.closed = False

    def reset(self):
        return {}

    def step(self, action):
        return {}, 1.0, True, {"episode": {"ignored": True}}

    def close(self):
        self.closed = True


def test_evaluator_v5_ablation_is_joint_reversible_and_version_exclusive(
        monkeypatch,
):
    environment = _OneStepEnv()
    monkeypatch.setattr(
        evaluator.trainer, "make_env", lambda *args, **kwargs: environment
    )
    model = _Model()
    row = evaluator._episode(
        model,
        SimpleNamespace(fixed_event_steps=None),
        seed=1,
        context=np.zeros(5),
        checkpoint_metadata={
            "path": "/tmp/v5.zip",
            "sha256": "0" * 64,
            "training_timesteps": 82_000,
        },
        phase="test",
        v5_context_ablation=True,
    )
    assert model.policy.calls == [("v5", True), ("v5", False)]
    assert row["v5_context_ablation"] is True
    assert "v4_context_ablation" not in row
    assert environment.closed is True

    with pytest.raises(ValueError, match="mutually exclusive"):
        evaluator._episode(
            model,
            SimpleNamespace(fixed_event_steps=None),
            seed=1,
            context=np.zeros(5),
            checkpoint_metadata={
                "path": "/tmp/v5.zip",
                "sha256": "0" * 64,
                "training_timesteps": 82_000,
            },
            phase="test",
            v4_context_ablation=True,
            v5_context_ablation=True,
        )


def test_v5_behavior_gate_changes_only_the_versioned_name(monkeypatch):
    captured = {}
    v4_result = {
        "name": "seller_two_branch_behavioral_gate_v4",
        "passed": True,
        "checks": [{"actual": 0.25, "required": 0.10, "passed": True}],
        "context_ablation": {"mean_absolute_price_difference": 0.12},
    }

    def fake_v4(**kwargs):
        captured.update(kwargs)
        return v4_result

    monkeypatch.setattr(evaluator, "seller_v4_behavioral_gate", fake_v4)
    arguments = {
        "random_result": object(),
        "fixed_results": object(),
        "ablated_random_result": object(),
        "forced_constant_results": object(),
        "formal": True,
    }
    result = evaluator.seller_v5_behavioral_gate(**arguments)
    assert captured == arguments
    assert result == {
        **v4_result,
        "name": "seller_shared_context_behavioral_gate_v5",
    }
    assert v4_result["name"] == "seller_two_branch_behavioral_gate_v4"


def test_evaluator_event_rows_use_only_the_active_ablation_version(monkeypatch):
    row = {
        "evaluation_seed": 7,
        "fifth_economic_override": None,
        "events": [{"event_index": 0, "price": 0.5}],
    }
    monkeypatch.setattr(evaluator, "_episode", lambda *args, **kwargs: row)
    monkeypatch.setattr(evaluator, "audit_episode", lambda *args, **kwargs: [])
    monkeypatch.setattr(evaluator, "_summary", lambda *args, **kwargs: {})
    common = {
        "model": object(),
        "args": SimpleNamespace(role="seller"),
        "seeds": [7],
        "contexts": [np.zeros(5)],
        "phase": "test",
    }
    base_metadata = {
        "path": "/tmp/model.zip",
        "sha256": "0" * 64,
        "training_timesteps": 82_000,
    }
    v5 = evaluator.evaluate_rows(
        metadata={
            **base_metadata,
            "economic_architecture": {
                "parameterization": validator.ARCHITECTURE,
            },
        },
        **common,
    )["event_rows"][0]
    assert v5["v5_context_ablation"] is False
    assert "v4_context_ablation" not in v5

    v4 = evaluator.evaluate_rows(
        metadata={
            **base_metadata,
            "economic_architecture": {
                "parameterization": "seller_two_branch_beta_v4",
            },
        },
        **common,
    )["event_rows"][0]
    assert v4["v4_context_ablation"] is False
    assert "v5_context_ablation" not in v4


def test_launcher_is_fresh_ordered_fail_closed_and_wandb_only_formal():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    common = COMMON.read_text(encoding="utf-8")
    assert "probe_atari_e1_seller_two_branch" not in launcher + common
    assert "probe_atari_e1_seller_shared_context" in launcher + common
    assert launcher.count("--no-wandb") == 2
    assert len(re.findall(r"^\s+--wandb\s*\\?$", launcher, re.MULTILINE)) == 1
    smoke = launcher.index("mechanics smoke")
    preflight = launcher.index("conditioning preflight", smoke + 1)
    gate = launcher.index('"${E1V5_VALIDATOR_PROTOCOL_ARGS[@]}" gate')
    formal = launcher.index("2000800-step online W&B formal run")
    assert smoke < preflight < gate < formal
    # One use initializes all three fresh training stages; the second binds
    # the read-only conditioning probe to the same certified E0b bytes.
    assert launcher.count("--e0b-checkpoint \"$E1V5_E0B\"") == 2
    assert "--resume" not in launcher
    assert "conditioning_recovery_v5_shared_context_v1" in common
    assert "seller_shared_context_beta_v5" in common
    assert "StackPOMDP" in launcher


def test_v5_exposure_v2_launcher_is_explicit_and_inherits_fresh_fail_closed_path():
    wrapper = EXPOSURE_LAUNCHER.read_text(encoding="utf-8")
    common = COMMON.read_text(encoding="utf-8")
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert "STACKPOMDP_E1V5_PROTOCOL=exposure_v2" in wrapper
    assert "exec \"${0:A:h}/run_atari_clean_e1_seller_shared_context_v5.sh\"" in wrapper
    assert "conditioning_recovery_v5_shared_context_exposure_v2" in common
    assert "E1V5_PREFLIGHT_TIMESTEPS=400160" in common
    assert "E1V5_VALIDATOR_PROTOCOL_ARGS=(--protocol exposure_v2)" in common
    assert "E1V5_EVIDENCE_REVISION=c4a7dcd92b621c" in common
    assert '--evidence-code-revision "$E1V5_DIAGNOSTIC_REVISION"' in launcher
    assert (
        '"${E1V5_VALIDATOR_PROTOCOL_ARGS[@]}" validate-gate' in common
    )
    assert "${E1V5_PREFLIGHT_TIMESTEPS}" in launcher
    assert "--resume" not in launcher + wrapper
    assert launcher.count("--no-wandb") == 2
    assert len(re.findall(r"^\s+--wandb\s*\\?$", launcher, re.MULTILINE)) == 1


def test_v5_exposure_v2_rejects_standard_v1_artifact_names():
    validator.configure_protocol(validator.STANDARD_PROTOCOL)
    standard_smoke = validator.SMOKE_CHECKPOINT_NAME
    standard_preflight = validator.PREFLIGHT_CHECKPOINT_NAME
    try:
        validator.configure_protocol(validator.EXPOSURE_PROTOCOL)
        with pytest.raises(ValueError, match="filename changed"):
            validator._exact_name(
                Path("/tmp") / standard_smoke,
                validator.SMOKE_CHECKPOINT_NAME,
                "v5 exposure smoke",
            )
        with pytest.raises(ValueError, match="filename changed"):
            validator._exact_name(
                Path("/tmp") / standard_preflight,
                validator.PREFLIGHT_CHECKPOINT_NAME,
                "v5 exposure preflight",
            )
    finally:
        validator.configure_protocol(validator.STANDARD_PROTOCOL)


def test_v5_launcher_preserves_failure_status_and_releases_owned_lock(tmp_path):
    lock = tmp_path / "seller-v5.lock"
    release = _shell_function(LAUNCHER, "release_e1v5_lock")
    program = "\n".join((
        "set -euo pipefail",
        'source "$LOCK_COMMON"',
        'typeset -g E1V5_LOCK="$PROFILE"',
        "typeset -g STACKPOMDP_E1V5_LOCK_TOKEN=test-token",
        'stackpomdp_claim_owned_lock "$E1V5_LOCK" '
        '"$STACKPOMDP_E1V5_LOCK_TOKEN" test',
        'typeset -g E1V5_LOCK_OWNED="$STACKPOMDP_LOCK_RESULT_OWNED"',
        release,
        "trap 'release_e1v5_lock' EXIT",
        "exit 2",
    ))
    result = subprocess.run(
        ["zsh", "-c", program],
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin",
            "LOCK_COMMON": str(
                AUTOMATION / "atari_e2_pipeline_common.zsh"
            ),
            "PROFILE": str(lock),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2, result.stderr
    assert not lock.exists()
