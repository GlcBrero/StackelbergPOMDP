from collections import OrderedDict
from pathlib import Path

import gym
import numpy as np
import pytest
import torch

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure

from replication.atari import evaluate_atari_meta_response_sb3 as evaluator
from replication.atari import train_atari_meta_response_sb3 as trainer
from replication.atari.sb3_common import PhaseBalancedPPO
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
    SELLER_TWO_BRANCH_BETA_V4,
    StackPOMDPAtariPolicy,
    seller_two_branch_architecture_provenance,
)


def _spaces():
    image = gym.spaces.Box(0, 255, shape=(84, 84, 4), dtype=np.uint8)
    return observation_space(image, 6), action_space(6)


def _policy(seed=41):
    observations, actions = _spaces()
    torch.manual_seed(seed)
    return StackPOMDPAtariPolicy(
        observations,
        actions,
        lambda _: 5.0e-4,
        economic_role="seller",
        economic_input_mode="full",
        economic_architecture=SELLER_TWO_BRANCH_BETA_V4,
    )


def _observation(*, event_index, commitment, ammo=0.2, time=0.4):
    return observation(
        image=np.zeros((84, 84, 4), dtype=np.uint8),
        state=actor_state(
            ammo_fraction=ammo,
            projectile_active=0.0,
            normalized_time=time,
            trade_mode=1.0,
            event_index=event_index,
            opponent_commitment=commitment,
        ),
        action_mask=np.ones(6, dtype=np.float32),
        decision_kind=FOLLOWER_TRADE,
        critic_state=np.zeros(CRITIC_STATE_DIM, dtype=np.float32),
    )


def _gameplay_observation():
    values = _observation(event_index=0, commitment=np.zeros(5))
    values["critic:action_credit"] = np.array([1.0, 0.0], dtype=np.float32)
    return values


def _batch(*items):
    return OrderedDict(
        (
            key,
            torch.as_tensor(np.stack([item[key] for item in items])),
        )
        for key in items[0]
    )


def _state_dict_equal(left, right):
    return all(torch.equal(left[key], right[key]) for key in left)


def test_v4_is_an_enum_only_full_input_seller_architecture():
    observations, actions = _spaces()
    with pytest.raises(ValueError, match="reserved for full-input E1 seller"):
        StackPOMDPAtariPolicy(
            observations,
            actions,
            lambda _: 5.0e-4,
            economic_role="buyer",
            economic_architecture=SELLER_TWO_BRANCH_BETA_V4,
        )
    with pytest.raises(ValueError, match="legacy threshold-residual"):
        StackPOMDPAtariPolicy(
            observations,
            actions,
            lambda _: 5.0e-4,
            economic_role="seller",
            economic_architecture=SELLER_TWO_BRANCH_BETA_V4,
            economic_threshold_residual=True,
        )


def test_v4_neutral_initialization_has_no_fixed_context_anchor():
    policy = _policy()
    values = _batch(
        _observation(event_index=0, commitment=np.zeros(5)),
        _observation(event_index=2, commitment=np.ones(5)),
        _observation(
            event_index=4,
            commitment=[0.1, 0.9, 0.2, 0.8, 0.3],
        ),
    )
    with torch.no_grad():
        distribution = policy.get_distribution(values)
        processed = policy._processed(values)
        live = processed[ACTOR_STATE][:, :9]
        base = policy.economic_live_output(
            policy.economic_live_encoder(live)
        )
        alpha, beta = policy._v4_economic_parameters(processed)
    torch.testing.assert_close(
        distribution.economic_mean,
        torch.full((3,), 0.5),
        rtol=0,
        atol=1.0e-7,
    )
    torch.testing.assert_close(base[:, 0], torch.zeros(3), rtol=0, atol=0)
    torch.testing.assert_close(
        alpha + beta,
        torch.full((3,), 2.0),
        rtol=0,
        atol=2.0e-7,
    )
    torch.testing.assert_close(
        policy.economic_context_output.weight,
        torch.zeros_like(policy.economic_context_output.weight),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        policy.economic_context_output.bias,
        torch.zeros_like(policy.economic_context_output.bias),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        policy.economic_current_slopes,
        torch.zeros(5),
        rtol=0,
        atol=0,
    )
    provenance = seller_two_branch_architecture_provenance()
    assert policy.economic_architecture_provenance() == provenance
    assert provenance["fixed_threshold_anchor"] is False


def test_v4_live_branch_masks_commitment_and_context_gradients_are_first_order():
    policy = _policy(seed=53)
    low = _observation(event_index=2, commitment=[0.5, 0.5, 0.0, 0.5, 0.5])
    high = _observation(event_index=2, commitment=[0.5, 0.5, 1.0, 0.5, 0.5])
    values = _batch(low, high)
    processed = policy._processed(values)
    live = processed[ACTOR_STATE][:, :9]
    torch.testing.assert_close(live[0], live[1], rtol=0, atol=0)

    actions = torch.tensor([[0.0, 0.1], [0.0, 0.9]])
    loss = -policy.get_distribution(values).log_prob(actions).mean()
    policy.optimizer.zero_grad()
    loss.backward()
    assert policy.economic_current_slopes.grad is not None
    assert torch.count_nonzero(policy.economic_current_slopes.grad).item() > 0
    assert policy.economic_context_output.weight.grad is not None
    assert torch.count_nonzero(
        policy.economic_context_output.weight.grad
    ).item() > 0
    for module in policy.gameplay_actor_modules():
        for parameter in module.parameters():
            assert parameter.requires_grad is False
            assert parameter.grad is None


def test_v4_event_selection_current_skip_full_context_and_ablation_math():
    policy = _policy(seed=59)
    context = [0.8, 0.7, 0.6, 0.2, 0.1]
    event_one = _batch(_observation(event_index=1, commitment=context))
    event_three = _batch(_observation(event_index=3, commitment=context))
    with torch.no_grad():
        policy.economic_context_output.weight.zero_()
        policy.economic_context_output.bias.copy_(
            torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        )
        policy.economic_current_slopes.zero_()
        mean_one = policy.get_distribution(event_one).economic_mean
        mean_three = policy.get_distribution(event_three).economic_mean
    assert mean_three.item() > mean_one.item()

    with torch.no_grad():
        policy.economic_context_output.bias.zero_()
        policy.economic_current_slopes.copy_(
            torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        )
        low = _batch(_observation(
            event_index=2,
            commitment=[0.5, 0.5, 0.0, 0.5, 0.5],
        ))
        high = _batch(_observation(
            event_index=2,
            commitment=[0.5, 0.5, 1.0, 0.5, 0.5],
        ))
        low_mean = policy.get_distribution(low).economic_mean.item()
        high_mean = policy.get_distribution(high).economic_mean.item()
    assert low_mean < 0.5 < high_mean

    # A noncurrent coordinate reaches event 2 only through the learned
    # full-context branch, never through the current-threshold skip.
    with torch.no_grad():
        policy.economic_current_slopes.zero_()
        policy.economic_context_encoder[0].weight.zero_()
        policy.economic_context_encoder[0].bias.zero_()
        policy.economic_context_encoder[0].weight[0, 4] = 1.0
        policy.economic_context_output.weight.zero_()
        policy.economic_context_output.bias.zero_()
        policy.economic_context_output.weight[2, 0] = 1.0
        noncurrent_low = _batch(_observation(
            event_index=2,
            commitment=[0.5, 0.5, 0.5, 0.5, 0.0],
        ))
        noncurrent_high = _batch(_observation(
            event_index=2,
            commitment=[0.5, 0.5, 0.5, 0.5, 1.0],
        ))
        low_distribution = policy.get_distribution(noncurrent_low)
        high_distribution = policy.get_distribution(noncurrent_high)
        assert high_distribution.economic_mean.item() > (
            low_distribution.economic_mean.item()
        )
        torch.testing.assert_close(
            (
                low_distribution.economic.concentration1
                + low_distribution.economic.concentration0
            ),
            (
                high_distribution.economic.concentration1
                + high_distribution.economic.concentration0
            ),
            rtol=0,
            atol=1.0e-7,
        )
        policy.set_v4_context_ablation(True)
        ablated_low = policy.get_distribution(noncurrent_low).economic_mean
        ablated_high = policy.get_distribution(noncurrent_high).economic_mean
        policy.set_v4_context_ablation(False)
    torch.testing.assert_close(ablated_low, ablated_high, rtol=0, atol=0)


def test_v4_optimizer_groups_partition_only_economics_and_critic():
    policy = _policy()
    groups = policy.optimizer.param_groups
    assert [group["group_name"] for group in groups] == [
        "seller_v4_economic",
        "seller_v4_critic",
    ]
    assert [group["lr"] for group in groups] == pytest.approx([5.0e-4, 1.0e-4])
    grouped = {
        id(parameter) for group in groups for parameter in group["params"]
    }
    assert grouped == {
        id(parameter)
        for parameter in policy.parameters()
        if parameter.requires_grad
    }
    assert all(
        id(parameter) not in grouped
        for module in policy.gameplay_actor_modules()
        for parameter in module.parameters()
    )


def test_v4_stochastic_rollout_always_uses_deterministic_game_argmax():
    policy = _policy(seed=67)
    gameplay = policy.get_distribution(_batch(_gameplay_observation()))
    expected = torch.argmax(gameplay.game.logits, dim=1)
    sampled_game = torch.stack([
        gameplay.sample()[:, 0].long() for _ in range(64)
    ])
    assert torch.equal(sampled_game, expected.expand_as(sampled_game))

    trade = policy.get_distribution(_batch(_observation(
        event_index=2, commitment=np.full(5, 0.5)
    )))
    expected_trade_game = torch.argmax(trade.game.logits, dim=1)
    sampled_trade = torch.stack([trade.sample()[0] for _ in range(64)])
    assert torch.equal(
        sampled_trade[:, 0].long(), expected_trade_game.expand(64)
    )
    assert torch.unique(sampled_trade[:, 1]).numel() > 1


class _TinyAtariEnv(gym.Env):
    def __init__(self):
        self.observation_space, self.action_space = _spaces()
        self._steps = 0

    def reset(self):
        self._steps = 0
        return _observation(event_index=0, commitment=np.zeros(5))

    def step(self, action):
        self._steps += 1
        done = self._steps >= 2
        return (
            _observation(event_index=self._steps % 5, commitment=np.zeros(5)),
            0.0,
            done,
            {},
        )


def test_v4_freeze_and_optimizer_groups_survive_ppo_save_load(tmp_path):
    env = DummyVecEnv([_TinyAtariEnv])
    model = PhaseBalancedPPO(
        StackPOMDPAtariPolicy,
        env,
        policy_kwargs={
            "economic_role": "seller",
            "economic_input_mode": "full",
            "economic_architecture": SELLER_TWO_BRANCH_BETA_V4,
        },
        learning_rate=5.0e-4,
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        gamma=1.0,
        gae_lambda=1.0,
    )
    before = [
        {key: value.detach().clone() for key, value in module.state_dict().items()}
        for module in model.policy.gameplay_actor_modules()
    ]
    model.set_logger(configure(str(tmp_path / "sb3_logs"), []))
    model._update_learning_rate(model.policy.optimizer)
    assert [group["lr"] for group in model.policy.optimizer.param_groups] == (
        pytest.approx([5.0e-4, 1.0e-4])
    )
    values = _batch(_observation(event_index=1, commitment=np.ones(5)))
    values["image"] = values["image"].permute(0, 3, 1, 2)
    actions = torch.tensor([[0.0, 0.8]])
    model.policy.optimizer.zero_grad()
    loss = -model.policy.get_distribution(values).log_prob(actions).mean()
    loss.backward()
    model.policy.optimizer.step()
    saved_optimizer_state = model.policy.optimizer.state_dict()
    checkpoint = tmp_path / "seller_v4.zip"
    model.save(checkpoint)
    restored = PhaseBalancedPPO.load(checkpoint, env=env)
    assert restored.policy.economic_architecture == SELLER_TWO_BRANCH_BETA_V4
    assert [
        group["group_name"] for group in restored.policy.optimizer.param_groups
    ] == ["seller_v4_economic", "seller_v4_critic"]
    assert [
        group["lr_scale"] for group in restored.policy.optimizer.param_groups
    ] == pytest.approx([1.0, 0.2])
    restored_optimizer_state = restored.policy.optimizer.state_dict()
    assert (
        restored_optimizer_state["param_groups"]
        == saved_optimizer_state["param_groups"]
    )
    assert restored_optimizer_state["state"].keys() == (
        saved_optimizer_state["state"].keys()
    )
    for parameter_id, expected_state in saved_optimizer_state["state"].items():
        actual_state = restored_optimizer_state["state"][parameter_id]
        for key, expected in expected_state.items():
            actual = actual_state[key]
            if torch.is_tensor(expected):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            else:
                assert actual == expected
    for expected, module in zip(before, restored.policy.gameplay_actor_modules()):
        assert _state_dict_equal(expected, module.state_dict())
        assert all(
            not parameter.requires_grad for parameter in module.parameters()
        )


def test_v4_trainer_cli_requires_uniform_balanced_and_economic_lr(tmp_path):
    common = [
        "--role", "seller",
        "--e0b-checkpoint", str(tmp_path / "unused.zip"),
        "--economic-architecture", SELLER_TWO_BRANCH_BETA_V4,
        "--actor-loss-mode", "balanced",
        "--learning-rate", "5e-4",
    ]
    args = trainer.parse_args(common)
    assert args.economic_architecture == SELLER_TWO_BRANCH_BETA_V4
    assert args.e1_sampler_mode == "uniform"
    assert "two_branch_v4" in Path(args.checkpoint).stem
    with pytest.raises(SystemExit):
        trainer.parse_args(common + ["--e1-sampler-mode", "all-equal-v1"])
    with pytest.raises(SystemExit):
        trainer.parse_args([
            item for item in common if item not in ("5e-4",)
        ] + ["1e-4"])


def test_v4_evaluator_metadata_uses_enum_without_legacy_flags():
    policy = _policy()
    architecture = policy.economic_architecture_provenance()
    assert evaluator.economic_architecture_training_flags(
        policy, architecture
    ) == {"economic_architecture": SELLER_TWO_BRANCH_BETA_V4}
    assert trainer.two_branch_initialization_provenance(policy) == (
        trainer.two_branch_initialization_contract()
    )


def _synthetic_result(
        *, reward, price_difference=0.0, episodes=100,
        payments=3.5, commitment=None, commitments=None,
):
    if commitments is None:
        commitment = (
            [0.8] * 5 if commitment is None else list(commitment)
        )
        commitments = [commitment] * episodes
    if np.isscalar(payments):
        payments = [float(payments)] * episodes
    rows = [
        {
            "evaluation_seed": seed,
            "seller_reward": float(reward),
            "payments": float(episode_payments),
            "opponent_commitment": list(episode_commitment),
            "event_steps": [20, 50, 80, 110, 140],
        }
        for seed, (episode_payments, episode_commitment) in enumerate(zip(
            payments, commitments
        ))
    ]
    events = [
        {
            "evaluation_seed": seed,
            "event_index": event,
            "price": float(0.6 - price_difference),
        }
        for seed in range(episodes)
        for event in range(5)
    ]
    return {
        "summary": {
            "mean_controlled_payoff": float(reward),
        },
        "protocol": {"passed": True},
        "episode_rows": rows,
        "event_rows": events,
    }


def test_v4_formal_behavior_gate_uses_paired_total_and_transfer_controls():
    episodes = 100
    commitments = [
        [0.43] * 5 if seed % 2 == 0 else [0.87] * 5
        for seed in range(episodes)
    ]
    payments = [
        2.15 if seed % 2 == 0 else 4.35
        for seed in range(episodes)
    ]
    random_result = _synthetic_result(
        reward=2.0,
        episodes=episodes,
        payments=payments,
        commitments=commitments,
    )
    ablated = _synthetic_result(
        reward=1.5,
        price_difference=0.1,
        commitments=commitments,
    )
    fixed = []
    for index in range(11):
        value = index / 10.0
        purchases = 0.0 if value == 0.0 else 5.0
        payments = 5.0 * value if value > 0.0 else 0.0
        fixed.append({
            "opponent_value": value,
            "summary": {
                "mean_price": value,
                "mean_payments": payments,
                "mean_purchases": purchases,
                "mean_seller_shots_fired": 5.0 if value == 0.0 else 0.0,
                "mean_seller_game_reward": 5.0 if value == 0.0 else 0.0,
            },
            "protocol": {"passed": True},
            "episode_rows": [
                {
                    "evaluation_seed": seed,
                    "payments": payments,
                }
                for seed in range(20)
            ],
        })
    forced = {
        index / 10.0: _synthetic_result(
            reward=1.0,
            commitments=commitments,
        )
        for index in range(11)
    }
    gate = evaluator.seller_v4_behavioral_gate(
        random_result=random_result,
        fixed_results=fixed,
        ablated_random_result=ablated,
        forced_constant_results=forced,
        formal=True,
    )
    assert gate["passed"]
    assert gate["transfer_payoff_baseline"]["label"] == (
        "transfer payoff only; not total seller payoff"
    )
    assert gate["transfer_payoff_baseline"]["best_constant_price"] == (
        pytest.approx(0.87)
    )
    assert gate["transfer_payoff_baseline"]["improvement"] == (
        pytest.approx(1.075)
    )
    assert gate["context_ablation"]["paired_total_payoff"]["mean"] == (
        pytest.approx(0.5)
    )
    assert gate["forced_constant_total_payoff_control"]["mean"] == (
        pytest.approx(1.0)
    )

    mismatched_context = _synthetic_result(reward=1.5)
    with pytest.raises(ValueError, match="different commitments"):
        evaluator.seller_v4_behavioral_gate(
            random_result=random_result,
            fixed_results=fixed,
            ablated_random_result=mismatched_context,
            forced_constant_results=forced,
            formal=True,
        )
    mismatched_schedule = _synthetic_result(
        reward=1.5,
        commitments=commitments,
    )
    mismatched_schedule["episode_rows"][0]["event_steps"][0] = 21
    with pytest.raises(ValueError, match="different event schedules"):
        evaluator.seller_v4_behavioral_gate(
            random_result=random_result,
            fixed_results=fixed,
            ablated_random_result=mismatched_schedule,
            forced_constant_results=forced,
            formal=True,
        )

    incomplete = evaluator.seller_v4_behavioral_gate(
        random_result=random_result,
        fixed_results=fixed,
        ablated_random_result=ablated,
        forced_constant_results={0.5: forced[0.5]},
        formal=True,
    )
    assert not incomplete["passed"]
