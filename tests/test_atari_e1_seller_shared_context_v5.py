from collections import OrderedDict
from types import SimpleNamespace

import gym
import numpy as np
import pytest
import torch

from stable_baselines3.common.vec_env import DummyVecEnv

from replication.atari import evaluate_atari_meta_response_sb3 as evaluator
from replication.atari import train_atari_meta_response_sb3 as trainer
from replication.atari import train_atari_stackpomdp_leader_sb3 as leader_trainer
from stackelberg_pomdp.atari.training import (
    PhaseBalancedPPO,
    SELLER_V5_OPTIMIZER_GROUPS,
)
from stackelberg_pomdp.atari.protocol import (
    CRITIC_STATE_DIM,
    FOLLOWER_TRADE,
    GAMEPLAY,
    action_space,
    actor_state,
    observation,
    observation_space,
)
from stackelberg_pomdp.atari.policies.composite import (
    SELLER_SHARED_CONTEXT_BETA_V5,
    SELLER_TWO_BRANCH_BETA_V4,
    StackPOMDPAtariPolicy,
    seller_shared_context_architecture_provenance,
)


def _spaces():
    image = gym.spaces.Box(0, 255, shape=(84, 84, 4), dtype=np.uint8)
    return observation_space(image, 6), action_space(6)


def _policy(seed=73, *, architecture=SELLER_SHARED_CONTEXT_BETA_V5):
    observations, actions = _spaces()
    torch.manual_seed(seed)
    return StackPOMDPAtariPolicy(
        observations,
        actions,
        lambda _: 5.0e-4,
        economic_role="seller",
        economic_input_mode="full",
        economic_architecture=architecture,
    )


def _observation(
        *, event_index, commitment, ammo=0.2, time=0.4,
        decision_kind=FOLLOWER_TRADE,
):
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
        decision_kind=decision_kind,
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


class _TinyAtariEnv(gym.Env):
    def __init__(self):
        self.observation_space, self.action_space = _spaces()
        self._steps = 0

    def reset(self):
        self._steps = 0
        return _observation(
            event_index=0,
            commitment=np.full(5, 0.5),
            decision_kind=GAMEPLAY,
        )

    def step(self, action):
        self._steps += 1
        return (
            _observation(event_index=1, commitment=np.full(5, 0.5)),
            0.0,
            self._steps >= 2,
            {},
        )


def _model(*, architecture=SELLER_SHARED_CONTEXT_BETA_V5):
    env = DummyVecEnv([_TinyAtariEnv])
    return PhaseBalancedPPO(
        StackPOMDPAtariPolicy,
        env,
        policy_kwargs={
            "economic_role": "seller",
            "economic_input_mode": "full",
            "economic_architecture": architecture,
        },
        learning_rate=5.0e-4,
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        gamma=1.0,
        gae_lambda=1.0,
        max_grad_norm=0.5,
    )


def test_v5_is_enum_only_full_input_seller_and_neutral_at_init():
    observations, actions = _spaces()
    with pytest.raises(ValueError, match="reserved for full-input E1 seller"):
        StackPOMDPAtariPolicy(
            observations,
            actions,
            lambda _: 5.0e-4,
            economic_role="buyer",
            economic_architecture=SELLER_SHARED_CONTEXT_BETA_V5,
        )
    policy = _policy()
    values = _batch(*[
        _observation(
            event_index=event,
            commitment=np.linspace(0.05 * event, 1.0 - 0.05 * event, 5),
        )
        for event in range(5)
    ])
    with torch.no_grad():
        distribution = policy.get_distribution(values)
    torch.testing.assert_close(
        distribution.economic_mean,
        torch.full((5,), 0.5),
        rtol=0,
        atol=1.0e-7,
    )
    torch.testing.assert_close(
        (
            distribution.economic.concentration1
            + distribution.economic.concentration0
        ),
        torch.full((5,), 2.0),
        rtol=0,
        atol=2.0e-7,
    )
    for values in (
            policy.economic_live_output.weight,
            policy.economic_context_output.weight,
            policy.economic_context_output.bias,
            policy.economic_current_slope,
    ):
        assert torch.equal(values, torch.zeros_like(values))


def test_v5_all_events_send_first_order_gradient_into_the_shared_route():
    policy = _policy(seed=79)
    output_gradients = []
    skip_gradients = []
    for event in range(5):
        context = np.linspace(0.15, 0.95, 5)
        values = _batch(_observation(
            event_index=event,
            commitment=context,
        ))
        actions = torch.tensor([[0.0, 0.9]])
        policy.optimizer.zero_grad()
        loss = -policy.get_distribution(values).log_prob(actions).mean()
        loss.backward()
        output_gradient = policy.economic_context_output.weight.grad
        skip_gradient = policy.economic_current_slope.grad
        assert output_gradient is not None
        assert skip_gradient is not None
        assert torch.count_nonzero(output_gradient).item() > 0
        assert torch.count_nonzero(skip_gradient).item() > 0
        output_gradients.append(output_gradient.detach().clone())
        skip_gradients.append(skip_gradient.detach().clone())
    assert len(output_gradients) == len(skip_gradients) == 5
    for module in policy.gameplay_actor_modules():
        for parameter in module.parameters():
            assert parameter.requires_grad is False
            assert parameter.grad is None


def test_v5_shared_context_route_expresses_full_context_and_event_identity():
    policy = _policy(seed=83)
    with torch.no_grad():
        first = policy.economic_context_encoder[0]
        first.weight.zero_()
        first.bias.zero_()
        # Hidden unit zero reads noncurrent commitment zero.  Hidden unit one
        # reads event one's one-hot coordinate.
        first.weight[0, 0] = 1.0
        first.weight[1, 5 + 1] = 1.0
        policy.economic_context_output.weight.zero_()
        policy.economic_context_output.bias.zero_()
        policy.economic_context_output.weight[0, 0] = 1.0
        policy.economic_context_output.weight[0, 1] = 0.5
        policy.economic_current_slope.zero_()

        noncurrent_low = _batch(_observation(
            event_index=3,
            commitment=[0.0, 0.5, 0.5, 0.5, 0.5],
        ))
        noncurrent_high = _batch(_observation(
            event_index=3,
            commitment=[1.0, 0.5, 0.5, 0.5, 0.5],
        ))
        assert policy.get_distribution(
            noncurrent_high
        ).economic_mean.item() > policy.get_distribution(
            noncurrent_low
        ).economic_mean.item()

        same_context = [0.5] * 5
        event_one = _batch(_observation(
            event_index=1, commitment=same_context
        ))
        event_three = _batch(_observation(
            event_index=3, commitment=same_context
        ))
        assert policy.get_distribution(
            event_one
        ).economic_mean.item() > policy.get_distribution(
            event_three
        ).economic_mean.item()


def test_v5_joint_context_ablation_is_reversible_without_mutation():
    policy = _policy(seed=89)
    values = _batch(_observation(
        event_index=2,
        commitment=[0.2, 0.4, 0.9, 0.3, 0.7],
    ))
    with torch.no_grad():
        policy.economic_context_output.weight.fill_(0.25)
        policy.economic_context_output.bias.fill_(0.4)
        policy.economic_current_slope.fill_(0.8)
        before_parameters = {
            name: value.detach().clone()
            for name, value in policy.state_dict().items()
        }
        original = policy.get_distribution(values).economic_mean.clone()
        policy.set_v5_context_ablation(True)
        ablated = policy.get_distribution(values).economic_mean.clone()
        policy.set_v5_context_ablation(False)
        restored = policy.get_distribution(values).economic_mean.clone()
    assert not torch.equal(original, ablated)
    torch.testing.assert_close(restored, original, rtol=0, atol=0)
    for name, expected in before_parameters.items():
        assert torch.equal(policy.state_dict()[name], expected)


def _set_group_gradient_norm(group, norm):
    parameters = list(group["params"])
    total = sum(parameter.numel() for parameter in parameters)
    value = float(norm) / np.sqrt(float(total))
    for parameter in parameters:
        parameter.grad = torch.full_like(parameter, value)


def test_v5_optimizer_partition_rates_and_independent_clipping_telemetry():
    model = _model()
    policy = model.policy
    groups = policy.optimizer.param_groups
    assert tuple(group["group_name"] for group in groups) == (
        SELLER_V5_OPTIMIZER_GROUPS
    )
    assert [group["lr_scale"] for group in groups] == pytest.approx(
        [1.0, 4.0, 0.2]
    )
    assert [group["lr"] for group in groups] == pytest.approx(
        [5.0e-4, 2.0e-3, 1.0e-4]
    )
    flattened = [
        parameter for group in groups for parameter in group["params"]
    ]
    assert len({id(parameter) for parameter in flattened}) == len(flattened)
    assert {id(parameter) for parameter in flattened} == {
        id(parameter)
        for parameter in policy.parameters()
        if parameter.requires_grad
    }

    for group, norm in zip(groups, (0.25, 2.0, 3.0)):
        _set_group_gradient_norm(group, norm)
    metrics = model._clip_policy_gradients()
    for group_name, expected_pre, expected_post in zip(
            SELLER_V5_OPTIMIZER_GROUPS,
            (0.25, 2.0, 3.0),
            (0.25, 0.5, 0.5),
    ):
        assert metrics[f"{group_name}_grad_norm_pre"] == pytest.approx(
            expected_pre, rel=5.0e-5, abs=5.0e-5
        )
        assert metrics[f"{group_name}_grad_norm_post"] == pytest.approx(
            expected_post, rel=5.0e-5, abs=5.0e-5
        )

    # The same exact keys are made durable by the PhaseBalancedPPO telemetry.
    model.learn(total_timesteps=2)
    optimizer_metrics = getattr(model, "atari_last_optimizer_metrics")
    for group_name in SELLER_V5_OPTIMIZER_GROUPS:
        for when in ("pre", "post"):
            key = f"train/{group_name}_grad_norm_{when}"
            assert key in optimizer_metrics
            assert np.isfinite(optimizer_metrics[key])

    # The old architecture still takes the unchanged global-clipping path.
    v4_model = _model(architecture=SELLER_TWO_BRANCH_BETA_V4)
    assert v4_model._clip_policy_gradients() == {}


def test_v5_optimizer_step_cannot_change_frozen_gameplay_hash():
    policy = _policy(seed=97)
    before = trainer.gameplay_actor_sha256(policy)
    values = _batch(_observation(
        event_index=4,
        commitment=[0.1, 0.3, 0.5, 0.7, 0.9],
    ))
    policy.optimizer.zero_grad()
    loss = -policy.get_distribution(values).log_prob(
        torch.tensor([[0.0, 0.85]])
    ).mean()
    loss.backward()
    policy.optimizer.step()
    assert trainer.gameplay_actor_sha256(policy) == before


def test_v5_provenance_is_exact_and_cross_version_resume_is_rejected(
        tmp_path,
):
    model = _model()
    policy = model.policy
    architecture = seller_shared_context_architecture_provenance()
    assert policy.economic_architecture_provenance() == architecture
    assert evaluator.economic_architecture_training_flags(
        policy, architecture
    ) == {"economic_architecture": SELLER_SHARED_CONTEXT_BETA_V5}
    assert trainer.shared_context_initialization_provenance(policy) == (
        trainer.shared_context_initialization_contract()
    )
    model.atari_e1_source_provenance = {
        "frozen_gameplay_actor_sha256": trainer.gameplay_actor_sha256(policy),
    }
    trainer._attach_shared_context_initialization_provenance(
        model, initialize=True
    )
    assert evaluator.candidate_shared_context_initialization(
        model, architecture
    ) == trainer.shared_context_initialization_contract()

    v4_model = _model(architecture=SELLER_TWO_BRANCH_BETA_V4)
    checkpoint = tmp_path / "seller_v4.zip"
    v4_model.save(checkpoint)
    args = SimpleNamespace(
        actor_loss_mode="balanced",
        resume=str(checkpoint),
        device="cpu",
        learning_rate=5.0e-4,
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        clip_range=0.1,
        entropy_coeff=0.01,
        value_coefficient=0.5,
        max_grad_norm=0.5,
        role="seller",
        economic_threshold_residual=False,
        economic_threshold_residual_direct_input=False,
        economic_architecture=SELLER_SHARED_CONTEXT_BETA_V5,
    )
    with pytest.raises(ValueError, match="economic-architecture must match"):
        trainer._resumed_model(args, v4_model.env)


def test_v5_cli_contract_requires_uniform_balanced_rates_and_clip(tmp_path):
    common = [
        "--role", "seller",
        "--e0b-checkpoint", str(tmp_path / "unused.zip"),
        "--economic-architecture", SELLER_SHARED_CONTEXT_BETA_V5,
        "--actor-loss-mode", "balanced",
        "--learning-rate", "5e-4",
        "--max-grad-norm", "0.5",
    ]
    args = trainer.parse_args(common)
    assert args.economic_architecture == SELLER_SHARED_CONTEXT_BETA_V5
    assert "shared_context_v5" in args.checkpoint
    with pytest.raises(SystemExit):
        trainer.parse_args(common + ["--e1-sampler-mode", "all-equal-v1"])
    with pytest.raises(SystemExit):
        trainer.parse_args(common + ["--max-grad-norm", "0.4"])


def test_v5_main_persists_initialization_in_wandb_and_evaluation(
        monkeypatch, tmp_path,
):
    initialization = trainer.shared_context_initialization_contract()
    architecture = seller_shared_context_architecture_provenance()
    model = SimpleNamespace(
        policy=SimpleNamespace(
            economic_architecture=SELLER_SHARED_CONTEXT_BETA_V5,
        ),
        num_timesteps=82_000,
        target_kl=None,
        atari_e1_source_provenance={"sha256": "0" * 64},
        atari_e1_sampler_provenance={"mode": "uniform"},
        atari_e1_sampler_history=[],
        atari_e1_resume_source_provenance=None,
        atari_e1_economic_architecture_provenance=architecture,
        atari_e1_threshold_residual_training_code_revision="1" * 40,
        atari_e1_shared_context_initialization_provenance=initialization,
        atari_economic_head_initialization={
            "mean": 0.5,
            "concentration": 2.0,
        },
        atari_actor_loss_mode="balanced",
    )
    args = SimpleNamespace(
        role="seller",
        seed=1,
        checkpoint=str(tmp_path / "seller_v5.zip"),
        num_envs=1,
        start_method="spawn",
        eval_only=True,
        resume=str(tmp_path / "resume.zip"),
        gameplay_horizon=200,
        event_tail_steps=0,
    )

    class _Config:
        def __init__(self):
            self.values = {}

        def update(self, values, *, allow_val_change):
            assert allow_val_change is True
            self.values.update(values)

    run = SimpleNamespace(config=_Config())
    vec_env = SimpleNamespace(close=lambda: None)
    captured = {}
    monkeypatch.setattr(trainer, "parse_args", lambda argv=None: args)
    monkeypatch.setattr(
        trainer, "make_vec_env", lambda *args, **kwargs: vec_env
    )
    monkeypatch.setattr(trainer, "init_wandb", lambda *args, **kwargs: run)
    monkeypatch.setattr(trainer, "build_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(
        trainer, "validate_frozen_gameplay_actor", lambda model: None
    )
    monkeypatch.setattr(
        trainer,
        "evaluate_response",
        lambda *args, **kwargs: {"summary": {}, "fixed_contexts": []},
    )
    monkeypatch.setattr(trainer, "write_csv", lambda *args, **kwargs: None)

    def finish_run(run, *, checkpoint, evaluation, total_timesteps):
        captured.update({
            "run": run,
            "checkpoint": checkpoint,
            "evaluation": evaluation,
            "total_timesteps": total_timesteps,
        })

    monkeypatch.setattr(trainer, "finish_run", finish_run)
    trainer.main([])

    assert run.config.values[
        "shared_context_initialization_provenance"
    ] == initialization
    assert captured["evaluation"]["provenance"][
        "shared_context_initialization"
    ] == initialization


def _attach_v5_e1_checkpoint_contract(model):
    architecture = trainer._attach_economic_architecture_contract(model)
    initialization = trainer._attach_shared_context_initialization_provenance(
        model, initialize=True
    )
    model.atari_e1_threshold_residual_training_code_revision = "2" * 40
    model.atari_e1_source_provenance = {
        "frozen_gameplay_actor_sha256": trainer.gameplay_actor_sha256(
            model.policy
        ),
    }
    return architecture, initialization


def test_e2_metadata_validates_v5_initialization_and_frozen_gameplay(
        tmp_path,
):
    model = _model()
    architecture, initialization = _attach_v5_e1_checkpoint_contract(model)
    checkpoint = tmp_path / "seller_v5.zip"
    model.save(checkpoint)

    metadata = leader_trainer.checkpoint_policy_metadata(
        checkpoint, label="frozen E1 response"
    )
    assert metadata["economic_architecture"] == architecture
    assert metadata["shared_context_initialization"] == initialization
    assert metadata["policy_metadata"][
        "shared_context_initialization"
    ] == initialization
    assert metadata["policy_metadata"][
        "frozen_gameplay_actor_sha256"
    ] == trainer.gameplay_actor_sha256(model.policy)

    delattr(model, trainer.SHARED_CONTEXT_INITIALIZATION_ATTRIBUTE)
    missing = tmp_path / "seller_v5_missing_initialization.zip"
    model.save(missing)
    with pytest.raises(ValueError, match="seller-v5.*initialization provenance"):
        leader_trainer.checkpoint_policy_metadata(missing)

    setattr(
        model,
        trainer.SHARED_CONTEXT_INITIALIZATION_ATTRIBUTE,
        initialization,
    )
    model.atari_e1_source_provenance["frozen_gameplay_actor_sha256"] = {
        "tampered": "3" * 64,
    }
    bad_hash = tmp_path / "seller_v5_bad_gameplay_hash.zip"
    model.save(bad_hash)
    with pytest.raises(RuntimeError, match="frozen seller gameplay actor changed"):
        leader_trainer.checkpoint_policy_metadata(bad_hash)


def test_v5_seller_e1_can_initialize_seller_leader_e2(monkeypatch, tmp_path):
    args = leader_trainer.parse_args([
        "--leader-role", "seller",
        "--response-checkpoint", str(tmp_path / "buyer_e1.zip"),
        "--leader-e1-checkpoint", str(tmp_path / "seller_v5.zip"),
        "--checkpoint", str(tmp_path / "seller_e2.zip"),
        "--num-envs", "1",
        "--no-wandb",
    ])
    architecture = seller_shared_context_architecture_provenance()
    source_policy = {
        "economic_role": "seller",
        "economic_input_mode": "full",
        "economic_architecture": architecture,
    }
    manifest = {
        "artifacts": {
            "same_role_e1_initialization": {
                "sha256": "4" * 64,
                "policy": source_policy,
            },
        },
    }
    captured = {}

    class _Policy:
        def load_actor_checkpoint(
                self, checkpoint, *, include_economic, device,
        ):
            captured["transfer"] = (checkpoint, include_economic, device)
            return {
                "sha256": "4" * 64,
                "source_economic_role": "seller",
                "source_economic_input_mode": "full",
                "source_economic_threshold_residual": False,
                "source_economic_threshold_residual_direct_input": False,
                "source_economic_architecture": (
                    SELLER_SHARED_CONTEXT_BETA_V5
                ),
                "modules": leader_trainer.E2_ACTOR_TRANSFER_MODULES,
                "critic_transferred": False,
            }

        def reset_economic_head(self, *, mean, concentration):
            captured["reset"] = (mean, concentration)

        def clear_obs_action_map(self):
            captured["cache_cleared"] = True

    class _PPO:
        def __init__(self, policy_class, env, **kwargs):
            self.policy = _Policy()

    monkeypatch.setattr(
        leader_trainer, "ppo_class_for_actor_loss_mode", lambda mode: _PPO
    )

    def attach(model, value):
        model.e2_provenance_manifest = value
        return value

    monkeypatch.setattr(leader_trainer, "attach_e2_provenance", attach)
    model = leader_trainer._new_model(
        args, vec_env="vec", provenance_manifest=manifest
    )

    assert captured["transfer"][1] is False
    assert captured["reset"] == (0.5, 2.0)
    assert captured["cache_cleared"] is True
    assert model.e2_provenance_manifest is manifest
