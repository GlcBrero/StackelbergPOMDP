from collections import OrderedDict
from types import SimpleNamespace

import gym
import numpy as np
import pytest
import torch
from torch.nn import functional as functional

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from replication.atari import evaluate_atari_meta_response_sb3 as evaluator
from replication.atari import train_atari_meta_response_sb3 as trainer
from stackelberg_pomdp.atari.protocol import (
    ACTOR_STATE,
    CRITIC_STATE_DIM,
    FOLLOWER_TRADE,
    GAMEPLAY,
    action_space,
    actor_state,
    observation,
    observation_space,
)
from stackelberg_pomdp.atari.stackpomdp_policy import (
    BETA_PARAMETER_EPSILON,
    THRESHOLD_RESIDUAL_BLEND_WEIGHT,
    THRESHOLD_RESIDUAL_EPSILON,
    StackPOMDPAtariPolicy,
    current_event_threshold,
    threshold_residual_architecture_provenance,
)


def _spaces():
    image = gym.spaces.Box(0, 255, shape=(84, 84, 4), dtype=np.uint8)
    return observation_space(image, 6), action_space(6)


def _policy(*, residual, role="seller", input_mode="full"):
    observations, actions = _spaces()
    return StackPOMDPAtariPolicy(
        observations,
        actions,
        lambda _: 1.0e-4,
        economic_role=role,
        economic_input_mode=input_mode,
        economic_threshold_residual=residual,
    )


def _observation(
        *, event_index, commitment, kind=FOLLOWER_TRADE,
        critic_state=None,
):
    return observation(
        image=np.zeros((84, 84, 4), dtype=np.uint8),
        state=actor_state(
            ammo_fraction=0.2,
            projectile_active=0.0,
            normalized_time=0.4,
            trade_mode=float(kind == FOLLOWER_TRADE),
            event_index=event_index,
            opponent_commitment=commitment,
        ),
        action_mask=np.ones(6, dtype=np.float32),
        decision_kind=kind,
        critic_state=(
            np.zeros(CRITIC_STATE_DIM, dtype=np.float32)
            if critic_state is None else critic_state
        ),
    )


def _batch(*items):
    return OrderedDict(
        (
            key,
            torch.as_tensor(np.stack([item[key] for item in items])),
        )
        for key in items[0]
    )


class _TinyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space, self.action_space = _spaces()

    def reset(self):
        return _observation(
            event_index=None,
            commitment=np.zeros(5, dtype=np.float32),
            kind=GAMEPLAY,
        )

    def step(self, action):
        del action
        return self.reset(), 0.0, True, {}


def test_current_event_threshold_selects_each_existing_context_coordinate():
    states = torch.as_tensor(np.stack([
        actor_state(
            ammo_fraction=0.2,
            projectile_active=0.0,
            normalized_time=0.3,
            trade_mode=1.0,
            event_index=index,
            opponent_commitment=[0.11, 0.22, 0.33, 0.44, 0.55],
        )
        for index in range(5)
    ]))
    torch.testing.assert_close(
        current_event_threshold(states).reshape(-1),
        torch.tensor([0.11, 0.22, 0.33, 0.44, 0.55]),
        rtol=0,
        atol=0,
    )
    with pytest.raises(ValueError, match="final dimension 14"):
        current_event_threshold(torch.zeros(2, 13))


def test_residual_is_pure64_seller_full_only_and_has_exact_provenance():
    policy = _policy(residual=True)
    assert policy.economic_head[0].in_features == 64
    assert policy.economic_architecture_provenance() == (
        threshold_residual_architecture_provenance(state_features=64)
    )
    provenance = policy.economic_architecture_provenance()
    assert provenance["direct_extra_input_features"] == 0
    assert provenance["mean_transform"]["threshold_weight"] == 0.5
    assert provenance["mean_transform"]["unit_interval_clamp_epsilon"] == 1e-4

    for role, input_mode in (("buyer", "full"), ("seller", "event_only")):
        with pytest.raises(ValueError, match="reserved for full-input E1 seller"):
            _policy(residual=True, role=role, input_mode=input_mode)
    observations, actions = _spaces()
    with pytest.raises(ValueError, match="exactly 64"):
        StackPOMDPAtariPolicy(
            observations,
            actions,
            lambda _: 1.0e-4,
            economic_role="seller",
            economic_input_mode="full",
            state_features=32,
            economic_threshold_residual=True,
        )


def test_residual_switch_is_same_seed_invariant_before_post_head_transform():
    torch.manual_seed(9182)
    ordinary = _policy(residual=False)
    ordinary_rng = torch.get_rng_state().clone()
    torch.manual_seed(9182)
    residual = _policy(residual=True)
    residual_rng = torch.get_rng_state().clone()

    assert ordinary.state_dict().keys() == residual.state_dict().keys()
    for name, value in ordinary.state_dict().items():
        torch.testing.assert_close(value, residual.state_dict()[name], rtol=0, atol=0)
    torch.testing.assert_close(ordinary_rng, residual_rng, rtol=0, atol=0)
    assert [
        [parameter.numel() for parameter in group["params"]]
        for group in ordinary.optimizer.param_groups
    ] == [
        [parameter.numel() for parameter in group["params"]]
        for group in residual.optimizer.param_groups
    ]

    values = _batch(_observation(
        event_index=3,
        commitment=[0.1, 0.2, 0.3, 0.87, 0.5],
    ))
    with torch.no_grad():
        ordinary_distribution = ordinary.get_distribution(values)
        residual_distribution = residual.get_distribution(values)
        ordinary_value = ordinary.predict_values(values)
        residual_value = residual.predict_values(values)
    torch.testing.assert_close(
        ordinary_distribution.game.logits,
        residual_distribution.game.logits,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(ordinary_value, residual_value, rtol=0, atol=0)


def test_residual_beta_mean_formula_clamp_and_concentration_are_exact():
    policy = _policy(residual=True)
    low = _observation(event_index=1, commitment=[0.5, 0.0, 0.5, 0.5, 0.5])
    high = _observation(event_index=1, commitment=[0.5, 1.0, 0.5, 0.5, 0.5])
    values = _batch(low, high)
    with torch.no_grad():
        processed, _, state, _ = policy._actor_features(values)
        parameters = policy.economic_head(
            policy._economic_state_features(processed, state)
        )
        base_alpha = (
            functional.softplus(parameters[:, 0]) + BETA_PARAMETER_EPSILON
        )
        base_beta = (
            functional.softplus(parameters[:, 1]) + BETA_PARAMETER_EPSILON
        )
        base_concentration = base_alpha + base_beta
        base_mean = base_alpha / base_concentration
        expected = (
            (1.0 - THRESHOLD_RESIDUAL_BLEND_WEIGHT) * base_mean
            + THRESHOLD_RESIDUAL_BLEND_WEIGHT * torch.tensor([0.0, 1.0])
        ).clamp(
            THRESHOLD_RESIDUAL_EPSILON,
            1.0 - THRESHOLD_RESIDUAL_EPSILON,
        )
        distribution = policy.get_distribution(values)
    torch.testing.assert_close(
        distribution.economic.mean, expected, rtol=0, atol=1.0e-7
    )
    torch.testing.assert_close(
        (
            distribution.economic.concentration1
            + distribution.economic.concentration0
        ),
        base_concentration,
        rtol=0,
        atol=1.0e-6,
    )


def test_trade_logprob_uses_residual_beta_and_gameplay_gates_it_out():
    policy = _policy(residual=True)
    trade = _batch(_observation(
        event_index=2,
        commitment=[0.2, 0.3, 0.8, 0.4, 0.5],
    ))
    distribution = policy.get_distribution(trade)
    action = torch.tensor([[0.0, 0.7]])
    expected = distribution.economic.log_prob(action[:, 1])
    torch.testing.assert_close(
        distribution.log_prob(action), expected, rtol=0, atol=0
    )
    sampled, _, sampled_log_prob = policy.forward(trade, deterministic=False)
    _, evaluated_log_prob, _ = policy.evaluate_actions(trade, sampled)
    torch.testing.assert_close(
        sampled_log_prob, evaluated_log_prob, rtol=0, atol=1.0e-6
    )
    policy.zero_grad(set_to_none=True)
    (-distribution.log_prob(action).mean()).backward()
    trade_gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in policy.economic_head.parameters()
        if parameter.grad is not None
    )
    assert np.isfinite(trade_gradient) and trade_gradient > 0.0

    gameplay = _batch(_observation(
        event_index=None,
        commitment=np.zeros(5, dtype=np.float32),
        kind=GAMEPLAY,
    ))
    gameplay_distribution = policy.get_distribution(gameplay)
    policy.zero_grad(set_to_none=True)
    (-gameplay_distribution.log_prob(torch.tensor([[0.0, 0.7]]))).backward()
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for parameter in policy.economic_head.parameters()
    )


def test_residual_mean_clamps_only_at_declared_unit_boundaries():
    policy = _policy(residual=True)
    low = _batch(_observation(
        event_index=0,
        commitment=[0.0, 0.5, 0.5, 0.5, 0.5],
    ))
    high = _batch(_observation(
        event_index=0,
        commitment=[1.0, 0.5, 0.5, 0.5, 0.5],
    ))
    with torch.no_grad():
        policy.economic_head[-1].weight.zero_()
        policy.economic_head[-1].bias.copy_(torch.tensor([-100.0, 100.0]))
        low_mean = policy.get_distribution(low).economic_mean.item()
        policy.economic_head[-1].bias.copy_(torch.tensor([100.0, -100.0]))
        high_mean = policy.get_distribution(high).economic_mean.item()
    assert low_mean == pytest.approx(THRESHOLD_RESIDUAL_EPSILON)
    assert high_mean == pytest.approx(1.0 - THRESHOLD_RESIDUAL_EPSILON)


def test_residual_constructor_and_architecture_roundtrip(tmp_path):
    vec_env = DummyVecEnv([_TinyEnv])
    try:
        model = PPO(
            StackPOMDPAtariPolicy,
            vec_env,
            policy_kwargs={
                "economic_role": "seller",
                "economic_input_mode": "full",
                "economic_threshold_residual": True,
            },
            n_steps=2,
            batch_size=2,
            n_epochs=1,
            seed=37,
            device="cpu",
            verbose=0,
        )
        expected = trainer._attach_economic_architecture_contract(model)
        revision = trainer._attach_training_code_revision(
            model, initialize=True
        )
        checkpoint = tmp_path / "residual_e1.zip"
        model.save(checkpoint)
        restored = PPO.load(checkpoint, device="cpu")
        assert restored.policy.economic_threshold_residual is True
        assert restored.policy.economic_head[0].in_features == 64
        assert evaluator.candidate_economic_architecture_contract(restored) == expected
        assert evaluator.candidate_e1_training_code_revision(
            restored, expected
        ) == revision
    finally:
        vec_env.close()


def test_ordinary_e0b_actor_transfer_into_residual_keeps_fresh_economic_head(
        tmp_path,
):
    vec_env = DummyVecEnv([_TinyEnv])
    try:
        source = PPO(
            StackPOMDPAtariPolicy,
            vec_env,
            policy_kwargs={
                "economic_role": "gameplay",
                "economic_input_mode": "full",
                "pretrained_lr_scale": 0.1,
            },
            n_steps=2,
            batch_size=2,
            n_epochs=1,
            seed=13,
            device="cpu",
            verbose=0,
        )
        checkpoint = tmp_path / "ordinary_e0b.zip"
        source.save(checkpoint)
        target = _policy(residual=True)
        economic_before = {
            name: value.clone()
            for name, value in target.economic_head.state_dict().items()
        }
        provenance = target.load_actor_checkpoint(
            checkpoint, include_economic=False, device="cpu"
        )
        assert provenance["source_economic_threshold_residual"] is False
        assert provenance["modules"] == (
            "features_extractor", "game_action_net"
        )
        for name, value in economic_before.items():
            torch.testing.assert_close(
                value, target.economic_head.state_dict()[name], rtol=0, atol=0
            )
        with pytest.raises(ValueError, match="matching seller economic"):
            target.load_actor_checkpoint(
                checkpoint, include_economic=True, device="cpu"
            )
    finally:
        vec_env.close()


def test_resume_rejects_a_threshold_residual_mode_mismatch(tmp_path):
    vec_env = DummyVecEnv([_TinyEnv])
    try:
        model = PPO(
            StackPOMDPAtariPolicy,
            vec_env,
            policy_kwargs={
                "economic_role": "seller",
                "economic_input_mode": "full",
                "economic_threshold_residual": True,
            },
            n_steps=2,
            batch_size=2,
            n_epochs=1,
            seed=19,
            device="cpu",
            verbose=0,
        )
        checkpoint = tmp_path / "residual_resume.zip"
        model.save(checkpoint)
        args = SimpleNamespace(
            actor_loss_mode="standard",
            resume=str(checkpoint),
            device="cpu",
            learning_rate=1.0e-4,
            n_steps=2,
            batch_size=2,
            n_epochs=1,
            clip_range=0.1,
            entropy_coeff=0.01,
            value_coefficient=0.5,
            max_grad_norm=0.5,
            role="seller",
            economic_threshold_residual=False,
        )
        with pytest.raises(ValueError, match="must match the saved E1"):
            trainer._resumed_model(args, vec_env)
    finally:
        vec_env.close()


def test_residual_cli_is_seller_only_fixed_slug_and_default_lr(tmp_path):
    args = trainer.parse_args([
        "--role", "seller",
        "--e0b-checkpoint", str(tmp_path / "e0b.zip"),
        "--economic-threshold-residual",
        "--no-wandb",
    ])
    assert args.economic_threshold_residual
    assert args.learning_rate == 1.0e-4
    assert "_threshold_residual_v1_seed1.zip" in args.checkpoint
    with pytest.raises(SystemExit):
        trainer.parse_args([
            "--role", "buyer",
            "--e0b-checkpoint", str(tmp_path / "e0b.zip"),
            "--economic-threshold-residual",
            "--no-wandb",
        ])


def test_candidate_architecture_contract_requires_exact_saved_provenance():
    policy = _policy(residual=True)
    architecture = policy.economic_architecture_provenance()
    model = SimpleNamespace(policy=policy)
    with pytest.raises(ValueError, match="lacks exact"):
        evaluator.candidate_economic_architecture_contract(model)
    setattr(model, trainer.ECONOMIC_ARCHITECTURE_ATTRIBUTE, architecture)
    assert evaluator.candidate_economic_architecture_contract(model) == architecture
    with pytest.raises(ValueError, match="training code revision"):
        evaluator.candidate_e1_training_code_revision(model, architecture)
    setattr(model, trainer.E1_TRAINING_CODE_REVISION_ATTRIBUTE, "a" * 40)
    assert evaluator.candidate_e1_training_code_revision(
        model, architecture
    ) == "a" * 40


def test_residual_training_revision_initializes_once_and_resume_is_exact(
        monkeypatch,
):
    revision = "1" * 40
    monkeypatch.setattr(
        trainer, "_current_training_code_revision", lambda: revision
    )
    residual = SimpleNamespace(policy=_policy(residual=True))
    assert trainer._attach_training_code_revision(
        residual, initialize=True
    ) == revision
    assert trainer._attach_training_code_revision(
        residual, initialize=False
    ) == revision

    setattr(
        residual, trainer.E1_TRAINING_CODE_REVISION_ATTRIBUTE, "2" * 40
    )
    with pytest.raises(ValueError, match="differs from the current"):
        trainer._attach_training_code_revision(residual, initialize=False)
    delattr(residual, trainer.E1_TRAINING_CODE_REVISION_ATTRIBUTE)
    with pytest.raises(ValueError, match="missing its training code revision"):
        trainer._attach_training_code_revision(residual, initialize=False)

    ordinary = SimpleNamespace(policy=_policy(residual=False))
    assert trainer._attach_training_code_revision(
        ordinary, initialize=True
    ) is None


def test_residual_candidate_family_rejects_mixed_training_revisions():
    sampler = {
        "schema": "test",
        "mode": "uniform",
    }
    base_metadata = {
        "training_config": {"algorithm": "PPO"},
        "atari_e1_sampler_provenance": sampler,
        "atari_e1_sampler_history": [{"sampler": sampler}],
        "sampler_contract_inferred_for_legacy_checkpoint": False,
        "economic_architecture": threshold_residual_architecture_provenance(
            state_features=64
        ),
    }
    results = [
        {
            "metadata": {
                **base_metadata,
                "e1_training_code_revision": revision * 40,
            },
            "episode_rows": [{}],
        }
        for revision in ("a", "b")
    ]
    with pytest.raises(ValueError, match="one training code revision"):
        evaluator.common_training_family(results)


def _seller_gate_result(*, threshold=None):
    random = threshold is None
    price = 0.6 if random else 0.05 + 0.9 * float(threshold)
    purchases = 3.0 if random else float(threshold >= 0.5) * 5.0
    shots = 4.5 if random or purchases == 0 else 0.0
    game_reward = 4.5 if random or purchases == 0 else 0.0
    payment = purchases * price
    seller_reward = payment + 0.1 * game_reward
    rows = [
        {
            "purchases": purchases,
            "seller_shots_fired": shots,
            "seller_game_reward": game_reward,
            "seller_reward": seller_reward,
        }
        for _ in range(20)
    ]
    summary = {
        "episodes": 20,
        "mean_controlled_payoff": seller_reward,
        "mean_purchases": purchases,
        "mean_payments": payment,
        "mean_seller_game_reward": game_reward,
        "mean_seller_shots_fired": shots,
        "mean_price": price,
    }
    result = {
        "summary": summary,
        "protocol": {"passed": True, "violations": []},
        "episode_rows": rows,
    }
    if not random:
        result["opponent_value"] = float(threshold)
    return result


def test_residual_readiness_and_final_gate_bind_behavior_and_learned_probe():
    random_result = _seller_gate_result()
    fixed_results = [
        _seller_gate_result(threshold=index / 10.0) for index in range(11)
    ]
    readiness = evaluator.seller_threshold_residual_behavioral_gate(
        random_result=random_result,
        fixed_results=fixed_results,
    )
    assert readiness["passed"]
    assert readiness["learned_threshold_slope_claim"] is False
    assert readiness["economic_constant_price_baseline"][
        "regret_reduction"
    ] > 0.05

    conditioning = {"warmup_gate": {"passed": True, "checks": {}}}
    final = evaluator.seller_threshold_residual_final_gate(
        random_result=random_result,
        fixed_results=fixed_results,
        conditioning_probe=conditioning,
    )
    assert final["passed"]
    conditioning["warmup_gate"]["passed"] = False
    assert not evaluator.seller_threshold_residual_final_gate(
        random_result=random_result,
        fixed_results=fixed_results,
        conditioning_probe=conditioning,
    )["passed"]

    fixed_results[0]["episode_rows"][0]["purchases"] = 1.0
    assert not evaluator.seller_threshold_residual_behavioral_gate(
        random_result=random_result,
        fixed_results=fixed_results,
    )["passed"]
