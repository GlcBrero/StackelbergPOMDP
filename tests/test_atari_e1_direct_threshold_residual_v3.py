from collections import OrderedDict
from pathlib import Path

import gym
import numpy as np
import pytest
import torch

from stable_baselines3 import PPO

from replication.atari import evaluate_atari_meta_response_sb3 as evaluator
from replication.atari import (
    probe_atari_e1_seller_direct_threshold_residual as probe
)
from replication.atari import train_atari_meta_response_sb3 as trainer
from stackelberg_pomdp.atari.protocol import (
    ACTOR_STATE,
    CRITIC_STATE_DIM,
    FOLLOWER_TRADE,
    action_space,
    actor_state,
    observation,
    observation_space,
)
from stackelberg_pomdp.atari.stackpomdp_policy import (
    StackPOMDPAtariPolicy,
    direct_threshold_residual_architecture_provenance,
)


def _spaces():
    image = gym.spaces.Box(
        0, 255, shape=(84, 84, 4), dtype=np.uint8
    )
    return observation_space(image, 6), action_space(6)


def _policy(*, direct, seed=19):
    observations, actions = _spaces()
    torch.manual_seed(seed)
    return StackPOMDPAtariPolicy(
        observations,
        actions,
        lambda _: 1.0e-4,
        economic_role="seller",
        economic_input_mode="full",
        economic_threshold_residual=True,
        economic_threshold_residual_direct_input=direct,
    )


def _observation(*, event_index, commitment):
    return observation(
        image=np.zeros((84, 84, 4), dtype=np.uint8),
        state=actor_state(
            ammo_fraction=0.2,
            projectile_active=0.0,
            normalized_time=0.4,
            trade_mode=1.0,
            event_index=event_index,
            opponent_commitment=commitment,
        ),
        action_mask=np.ones(6, dtype=np.float32),
        decision_kind=FOLLOWER_TRADE,
        critic_state=np.zeros(CRITIC_STATE_DIM, dtype=np.float32),
    )


def _batch(*items):
    return OrderedDict(
        (
            key,
            torch.as_tensor(np.stack([item[key] for item in items])),
        )
        for key in items[0]
    )


def _state_dict_equal(first, second):
    return all(torch.equal(first[key], second[key]) for key in first)


def test_v3_is_seller_only_and_requires_the_residual_anchor():
    observations, actions = _spaces()
    with pytest.raises(ValueError, match="requires the threshold-residual"):
        StackPOMDPAtariPolicy(
            observations,
            actions,
            lambda _: 1.0e-4,
            economic_role="seller",
            economic_threshold_residual_direct_input=True,
        )
    with pytest.raises(ValueError, match="reserved for full-input E1 seller"):
        StackPOMDPAtariPolicy(
            observations,
            actions,
            lambda _: 1.0e-4,
            economic_role="buyer",
            economic_threshold_residual=True,
            economic_threshold_residual_direct_input=True,
        )


def test_v3_preserves_canonical_rng_and_initializes_only_new_column_zero():
    pure64 = _policy(direct=False)
    direct = _policy(direct=True)

    for name in ("features_extractor", "game_action_net", "value_net"):
        assert _state_dict_equal(
            getattr(pure64, name).state_dict(),
            getattr(direct, name).state_dict(),
        )
    assert pure64.economic_head[0].in_features == 64
    assert direct.economic_head[0].in_features == 65
    torch.testing.assert_close(
        direct.economic_head[0].weight[:, :64],
        pure64.economic_head[0].weight,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        direct.economic_head[0].weight[:, 64],
        torch.zeros(64),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        direct.economic_head[0].bias,
        pure64.economic_head[0].bias,
        rtol=0,
        atol=0,
    )
    for index in (2, 4):
        assert _state_dict_equal(
            pure64.economic_head[index].state_dict(),
            direct.economic_head[index].state_dict(),
        )


def test_v3_post_construction_rng_state_matches_canonical_policy():
    _policy(direct=False, seed=211)
    canonical_state = torch.random.get_rng_state().clone()
    _policy(direct=True, seed=211)
    direct_state = torch.random.get_rng_state().clone()
    torch.testing.assert_close(
        direct_state, canonical_state, rtol=0, atol=0
    )


def test_v3_direct_scalar_enters_base_head_only_and_starts_neutral():
    policy = _policy(direct=True)
    low = _observation(
        event_index=2,
        commitment=[0.5, 0.5, 0.0, 0.5, 0.5],
    )
    high = _observation(
        event_index=2,
        commitment=[0.5, 0.5, 1.0, 0.5, 0.5],
    )
    values = _batch(low, high)
    processed, _, state_features, actor_features = policy._actor_features(values)
    economic = policy._economic_state_features(processed, state_features)
    assert economic.shape == (2, 65)
    torch.testing.assert_close(
        economic[:, -1], torch.tensor([0.0, 1.0]), rtol=0, atol=0
    )
    with torch.no_grad():
        base_parameters = policy.economic_head(economic)
        distribution = policy.get_distribution(values)
        critic_values = policy.predict_values(values)
        game_logits = distribution.game.logits
    torch.testing.assert_close(
        base_parameters[0], base_parameters[1], rtol=0, atol=0
    )
    torch.testing.assert_close(
        distribution.economic_mean,
        torch.tensor([0.25, 0.75]),
        rtol=0,
        atol=1.0e-7,
    )
    assert actor_features.shape[1] == 576
    assert game_logits.shape == (2, 6)
    assert critic_values.shape == (2, 1)
    assert policy.game_action_net.in_features == 576
    assert policy.value_net[0].in_features == 576 + CRITIC_STATE_DIM


def _economic_update(policy, values, economic_action):
    actions = torch.tensor(
        [[0.0, economic_action]], dtype=torch.float32
    )
    policy.optimizer.zero_grad()
    loss = -policy.get_distribution(values).log_prob(actions).mean()
    loss.backward()
    policy.optimizer.step()


def test_zero_direct_column_becomes_trainable_on_the_second_update():
    policy = _policy(direct=True, seed=223)
    values = _batch(_observation(
        event_index=2,
        commitment=[0.5, 0.5, 1.0, 0.5, 0.5],
    ))
    direct_column = policy.economic_head[0].weight[:, 64]
    assert torch.count_nonzero(direct_column).item() == 0
    _economic_update(policy, values, 0.9)
    # The final layer starts at zero, so the first update cannot yet transmit
    # a gradient into either hidden layer.
    assert torch.count_nonzero(direct_column).item() == 0
    _economic_update(policy, values, 0.9)
    assert torch.count_nonzero(direct_column).item() > 0


def test_v3_architecture_and_initialization_provenance_are_exact():
    policy = _policy(direct=True)
    architecture = direct_threshold_residual_architecture_provenance(
        state_features=64
    )
    assert policy.economic_architecture_provenance() == architecture
    assert architecture["base_head_input_features"] == 65
    assert architecture["direct_extra_input_features"] == 1
    assert architecture["mean_transform"]["threshold_weight"] == 0.5
    direct = architecture["direct_current_threshold_input"]
    assert direct["first_linear_column_index"] == 64
    assert direct["initialization"] == "exact_zero"
    assert direct["new_observation_fields"] == []
    initialization = trainer.direct_threshold_initialization_provenance(
        policy
    )
    assert initialization == {
        "schema": "stackpomdp.atari.e1_direct_threshold_initialization.v1",
        "architecture_parameterization": (
            "seller_direct_threshold_residual_beta_v3"
        ),
        "parameter": "economic_head.0.weight",
        "first_linear_shape": [64, 65],
        "ordinary_prefix_columns": [0, 64],
        "new_direct_column_index": 64,
        "new_direct_column_width": 1,
        "new_direct_parameter_count": 64,
        "new_direct_column_initialized_exact_zero": True,
        "initial_learned_base_threshold_slope": 0.0,
        "canonical_64_input_head_prefix_copied_exactly": True,
        "canonical_rng_stream_preserved": True,
        "non_economic_weights_same_seed_invariant": True,
    }


def test_v3_flag_is_separately_slugged_and_requires_both_cli_switches(tmp_path):
    args = trainer.parse_args([
        "--role", "seller",
        "--e0b-checkpoint", str(tmp_path / "e0b.zip"),
        "--economic-threshold-residual",
        "--economic-threshold-residual-direct-input",
        "--no-wandb",
    ])
    assert args.economic_threshold_residual
    assert args.economic_threshold_residual_direct_input
    assert "_direct_threshold_residual_v3_seed1.zip" in args.checkpoint
    with pytest.raises(SystemExit):
        trainer.parse_args([
            "--role", "seller",
            "--e0b-checkpoint", str(tmp_path / "e0b.zip"),
            "--economic-threshold-residual-direct-input",
            "--no-wandb",
        ])


def test_evaluator_rejects_missing_or_mismatched_v3_initialization_record():
    policy = _policy(direct=True)
    architecture = policy.economic_architecture_provenance()
    model = type("Model", (), {"policy": policy})()
    with pytest.raises(ValueError, match="lacks exact zero-column"):
        evaluator.candidate_direct_threshold_initialization(
            model, architecture
        )
    expected = trainer.direct_threshold_initialization_provenance(policy)
    setattr(model, trainer.DIRECT_THRESHOLD_INITIALIZATION_ATTRIBUTE, expected)
    assert evaluator.candidate_direct_threshold_initialization(
        model, architecture
    ) == expected
    corrupted = dict(expected)
    corrupted["new_direct_column_initialized_exact_zero"] = False
    setattr(model, trainer.DIRECT_THRESHOLD_INITIALIZATION_ATTRIBUTE, corrupted)
    with pytest.raises(ValueError, match="lacks exact zero-column"):
        evaluator.candidate_direct_threshold_initialization(
            model, architecture
        )


def test_full_ppo_v3_save_load_roundtrip_keeps_optimizer_and_provenance(
        tmp_path,
):
    observations, actions = _spaces()
    model = PPO(
        StackPOMDPAtariPolicy,
        None,
        policy_kwargs={
            "economic_role": "seller",
            "economic_input_mode": "full",
            "economic_threshold_residual": True,
            "economic_threshold_residual_direct_input": True,
        },
        learning_rate=1.0e-4,
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        gamma=1.0,
        gae_lambda=1.0,
        seed=229,
        device="cpu",
        verbose=0,
        _init_setup_model=False,
    )
    # SB3 cannot infer spaces without an environment when setup is deferred.
    model.observation_space = observations
    model.action_space = actions
    model.n_envs = 1
    model._setup_model()
    architecture = trainer._attach_economic_architecture_contract(model)
    initialization = (
        trainer._attach_direct_threshold_initialization_provenance(
            model, initialize=True
        )
    )
    revision = trainer._attach_training_code_revision(
        model, initialize=True
    )
    values = _batch(_observation(
        event_index=1,
        commitment=[0.5, 1.0, 0.5, 0.5, 0.5],
    ))
    _economic_update(model.policy, values, 0.8)
    before_optimizer = model.policy.optimizer.state_dict()
    assert before_optimizer["state"]
    checkpoint = tmp_path / "v3_roundtrip.zip"
    model.save(checkpoint)

    restored = PPO.load(checkpoint, device="cpu")
    assert restored.policy.economic_threshold_residual is True
    assert restored.policy.economic_threshold_residual_direct_input is True
    assert restored.policy.economic_head[0].in_features == 65
    assert restored.policy.economic_architecture_provenance() == architecture
    assert getattr(
        restored, trainer.ECONOMIC_ARCHITECTURE_ATTRIBUTE
    ) == architecture
    assert getattr(
        restored, trainer.DIRECT_THRESHOLD_INITIALIZATION_ATTRIBUTE
    ) == initialization
    assert getattr(
        restored, trainer.E1_TRAINING_CODE_REVISION_ATTRIBUTE
    ) == revision
    assert evaluator.candidate_direct_threshold_initialization(
        restored, architecture
    ) == initialization
    assert evaluator.candidate_e1_training_code_revision(
        restored, architecture
    ) == revision
    after_optimizer = restored.policy.optimizer.state_dict()
    assert len(after_optimizer["param_groups"]) == 2
    assert after_optimizer["param_groups"] == before_optimizer["param_groups"]
    assert set(after_optimizer["state"]) == set(before_optimizer["state"])
    for key in before_optimizer["state"]:
        for field, before in before_optimizer["state"][key].items():
            after = after_optimizer["state"][key][field]
            if torch.is_tensor(before):
                torch.testing.assert_close(after, before, rtol=0, atol=0)
            else:
                assert after == before


def test_v2_and_ordinary_defaults_remain_without_the_v3_field():
    pure64 = _policy(direct=False)
    assert pure64.economic_head[0].in_features == 64
    assert trainer.direct_threshold_initialization_provenance(pure64) is None
    assert (
        pure64.economic_architecture_provenance()["parameterization"]
        == "seller_threshold_residual_beta_v1"
    )
    observations, actions = _spaces()
    ordinary = StackPOMDPAtariPolicy(
        observations, actions, lambda _: 1.0e-4
    )
    assert ordinary.economic_threshold_residual is False
    assert ordinary.economic_threshold_residual_direct_input is False
    assert ordinary.economic_architecture_provenance() is None
    assert ordinary.economic_head[0].in_features == 64


def test_shared_recovery_entrypoints_are_executable():
    root = Path(__file__).resolve().parents[1]
    for relative in (
            "replication/atari/automation/"
            "run_atari_clean_e1_seller_threshold_residual_recovery.sh",
            "replication/atari/automation/"
            "run_e1_seller_threshold_residual_recovery_selector.sh",
            "replication/atari/automation/"
            "run_atari_clean_e1_seller_direct_threshold_residual_recovery.sh",
            "replication/atari/automation/"
            "run_e1_seller_direct_threshold_residual_recovery_selector.sh",
    ):
        path = root / relative
        assert path.is_file()
        assert path.stat().st_mode & 0o111


def _passing_probe_arrays():
    thresholds = np.linspace(0.0, 1.0, 5, dtype=np.float64)[:, None]
    event_offsets = 0.001 * np.arange(5, dtype=np.float64)[None, :]
    base_all_equal = 0.45 + 0.03 * thresholds + event_offsets
    final_all_equal = 0.5 * base_all_equal + 0.5 * thresholds
    base_low = np.full(5, 0.45, dtype=np.float64)
    base_high = np.full(5, 0.48, dtype=np.float64)
    final_low = 0.5 * base_low
    final_high = 0.5 * base_high + 0.5
    noncurrent_low = np.full(5, 0.46, dtype=np.float64)
    noncurrent_high = np.full(5, 0.462, dtype=np.float64)
    return {
        "final_all_equal": final_all_equal,
        "final_coordinate_low": final_low,
        "final_coordinate_high": final_high,
        "base_all_equal": base_all_equal,
        "base_coordinate_low": base_low,
        "base_coordinate_high": base_high,
        "base_noncurrent_low": noncurrent_low,
        "base_noncurrent_high": noncurrent_high,
    }


def test_v3_probe_requires_response_and_noncurrent_specificity():
    arrays = _passing_probe_arrays()
    gate = probe.direct_threshold_residual_warmup_gate(**arrays)
    assert gate["passed"]
    assert gate["learned_threshold_slope_claim"]
    check = gate["checks"][
        "learned_base_current_vs_noncurrent_specificity_margin"
    ]
    assert check["passed"]
    assert check["minimum_current_response"] == pytest.approx(0.03)
    assert check["maximum_absolute_noncurrent_response"] == pytest.approx(
        0.002
    )

    leakage = dict(arrays)
    leakage["base_noncurrent_high"] = np.full(
        5, 0.49, dtype=np.float64
    )
    failed = probe.direct_threshold_residual_warmup_gate(**leakage)
    assert failed["passed"] is False
    assert failed["checks"][
        "learned_base_current_vs_noncurrent_specificity_margin"
    ]["passed"] is False

    no_learned_response = dict(arrays)
    no_learned_response["base_coordinate_high"] = arrays[
        "base_coordinate_low"
    ]
    failed = probe.direct_threshold_residual_warmup_gate(
        **no_learned_response
    )
    assert failed["passed"] is False
    assert failed["checks"][
        "minimum_learned_base_current_coordinate_mean_response"
    ]["passed"] is False
