import json
from types import SimpleNamespace

import gym
import numpy as np
import pytest
import torch

from replication.atari import sb3_common
from replication.atari.sb3_common import (
    ECONOMIC_INIT_ATTRIBUTE,
    EpisodeCheckpointCallback,
    PHASE_BALANCED_ACTOR_LOSS_MODE,
    STANDARD_ACTOR_LOSS_MODE,
    PhaseBalancedPPO,
    ScaledLearningRatePPO,
    attach_atari_training_contract,
    model_actor_loss_mode,
    model_economic_initialization,
    phase_balanced_actor_terms,
)
from stackelberg_pomdp.atari.protocol import (
    ACTION_CREDIT,
    CACHED_TRADE_REPLAY,
    FOLLOWER_TRADE,
    GAMEPLAY,
    action_space,
    actor_state,
    observation,
    observation_space,
)
from stackelberg_pomdp.atari.stackpomdp_policy import StackPOMDPAtariPolicy


def _actor_terms(*, advantages, log_ratio, credit, entropy=None, clip=0.2):
    log_prob = torch.as_tensor(log_ratio, dtype=torch.float32)
    return phase_balanced_actor_terms(
        advantages=torch.as_tensor(advantages, dtype=torch.float32),
        log_prob=log_prob,
        old_log_prob=torch.zeros_like(log_prob),
        entropy=(
            None
            if entropy is None
            else torch.as_tensor(entropy, dtype=torch.float32)
        ),
        action_credit=torch.as_tensor(credit, dtype=torch.float32),
        clip_range=clip,
    )


def test_phase_balanced_terms_use_separate_active_means_and_ignore_cache_rows():
    terms = _actor_terms(
        advantages=[1.0, 3.0, 5.0, 1_000.0],
        log_ratio=[0.0, 0.0, 0.0, 9.0],
        entropy=[0.2, 0.4, 0.6, 1_000.0],
        credit=[[1, 0], [1, 0], [0, 1], [0, 0]],
    )

    assert terms["game_policy_loss"].item() == pytest.approx(-2.0)
    assert terms["economic_policy_loss"].item() == pytest.approx(-5.0)
    assert terms["policy_loss"].item() == pytest.approx(-7.0)
    assert terms["game_entropy_loss"].item() == pytest.approx(-0.3)
    assert terms["economic_entropy_loss"].item() == pytest.approx(-0.6)
    assert terms["entropy_loss"].item() == pytest.approx(-0.9)
    assert terms["game_active_rows"].item() == 2
    assert terms["economic_active_rows"].item() == 1
    assert terms["inactive_actor_rows"].item() == 1

    # Duplicating one phase does not dilute the other phase's actor term.
    duplicated = _actor_terms(
        advantages=[1.0, 3.0, 1.0, 3.0, 5.0, 1_000.0],
        log_ratio=[0.0, 0.0, 0.0, 0.0, 0.0, 9.0],
        entropy=[0.2, 0.4, 0.2, 0.4, 0.6, 1_000.0],
        credit=[[1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 0]],
    )
    assert duplicated["policy_loss"].item() == pytest.approx(
        terms["policy_loss"].item()
    )
    assert duplicated["entropy_loss"].item() == pytest.approx(
        terms["entropy_loss"].item()
    )


def test_phase_balanced_terms_clip_each_head_and_use_the_max_head_kl():
    terms = _actor_terms(
        advantages=[2.0, 4.0],
        log_ratio=np.log([1.5, 0.5]),
        credit=[[1, 0], [0, 1]],
    )

    assert terms["game_policy_loss"].item() == pytest.approx(-2.4)
    assert terms["economic_policy_loss"].item() == pytest.approx(-2.0)
    expected_game_kl = 1.5 - 1.0 - np.log(1.5)
    expected_economic_kl = 0.5 - 1.0 - np.log(0.5)
    assert terms["game_approx_kl"].item() == pytest.approx(expected_game_kl)
    assert terms["economic_approx_kl"].item() == pytest.approx(
        expected_economic_kl
    )
    assert terms["max_approx_kl"].item() == pytest.approx(
        max(expected_game_kl, expected_economic_kl)
    )
    assert terms["game_clip_fraction"].item() == pytest.approx(1.0)
    assert terms["economic_clip_fraction"].item() == pytest.approx(1.0)


def test_phase_balanced_cache_rows_have_exactly_zero_actor_gradient():
    log_prob = torch.tensor([0.0, 0.0, 0.7], requires_grad=True)
    terms = phase_balanced_actor_terms(
        advantages=torch.tensor([1.0, 2.0, 10_000.0]),
        log_prob=log_prob,
        old_log_prob=torch.zeros(3),
        entropy=None,
        action_credit=torch.tensor([[1, 0], [0, 1], [0, 0]]),
        clip_range=0.2,
    )
    terms["policy_loss"].backward()

    assert log_prob.grad[0].item() != 0.0
    assert log_prob.grad[1].item() != 0.0
    assert log_prob.grad[2].item() == 0.0


@pytest.mark.parametrize(
    ("credit", "message"),
    (
        ([[1, 0], [0.5, 0.5]], "exact binary"),
        ([[1, 0], [1, 1]], "cannot credit both"),
        ([[1, 0], [1, 0]], "economic row"),
        ([[0, 1], [0, 1]], "gameplay row"),
    ),
)
def test_phase_balanced_credit_contract_fails_fast(credit, message):
    with pytest.raises(ValueError, match=message):
        _actor_terms(
            advantages=[1.0, 1.0],
            log_ratio=[0.0, 0.0],
            credit=credit,
        )


def test_training_contract_uses_legacy_defaults_and_validates_new_metadata():
    legacy = SimpleNamespace()
    assert model_actor_loss_mode(legacy) == STANDARD_ACTOR_LOSS_MODE
    assert model_economic_initialization(
        legacy, default_mean=0.95, default_concentration=10.0
    ) == {"mean": 0.95, "concentration": 10.0}

    attach_atari_training_contract(
        legacy,
        actor_loss_mode=PHASE_BALANCED_ACTOR_LOSS_MODE,
        economic_init_mean=0.95,
        economic_init_concentration=10.0,
    )
    assert model_actor_loss_mode(legacy) == PHASE_BALANCED_ACTOR_LOSS_MODE
    assert getattr(legacy, ECONOMIC_INIT_ATTRIBUTE) == {
        "mean": 0.95,
        "concentration": 10.0,
    }

    setattr(legacy, ECONOMIC_INIT_ATTRIBUTE, {"mean": 1.0, "concentration": 10})
    with pytest.raises(ValueError, match="invalid"):
        model_economic_initialization(
            legacy, default_mean=0.95, default_concentration=10.0
        )

    setattr(
        legacy,
        ECONOMIC_INIT_ATTRIBUTE,
        {"mean": 0.95, "concentration": float("nan")},
    )
    with pytest.raises(ValueError, match="invalid"):
        model_economic_initialization(
            legacy, default_mean=0.95, default_concentration=10.0
        )
    with pytest.raises(ValueError, match="concentration must be positive"):
        attach_atari_training_contract(
            legacy,
            actor_loss_mode=STANDARD_ACTOR_LOSS_MODE,
            economic_init_mean=0.95,
            economic_init_concentration=float("nan"),
        )


class _AlternatingCreditEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Dict({
            "features": gym.spaces.Box(
                -1.0, 1.0, shape=(2,), dtype=np.float32
            ),
            ACTION_CREDIT: gym.spaces.Box(
                0.0, 1.0, shape=(2,), dtype=np.float32
            ),
        })
        self.action_space = gym.spaces.Discrete(2)
        self.step_index = 0

    def _observation(self):
        credit = (
            np.array([1.0, 0.0], dtype=np.float32)
            if self.step_index % 2 == 0
            else np.array([0.0, 1.0], dtype=np.float32)
        )
        return {
            "features": np.array([0.25, -0.25], dtype=np.float32),
            ACTION_CREDIT: credit,
        }

    def reset(self):
        self.step_index = 0
        return self._observation()

    def step(self, action):
        reward = float(action == self.step_index % 2)
        self.step_index += 1
        done = self.step_index == 4
        return self._observation(), reward, done, {}


def _small_phase_model(
        *, batch_size, learning_rate=1.0e-3, n_epochs=1, target_kl=None
):
    return PhaseBalancedPPO(
        "MultiInputPolicy",
        _AlternatingCreditEnv(),
        learning_rate=learning_rate,
        n_steps=4,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=1.0,
        gae_lambda=1.0,
        target_kl=target_kl,
        seed=3,
        device="cpu",
        verbose=0,
    )


def test_phase_balanced_ppo_trains_full_buffer_and_logs_each_head():
    model = _small_phase_model(batch_size=4)
    model.learn(total_timesteps=4)

    values = model.logger.name_to_value
    assert values["train/game_active_rows"] == pytest.approx(2.0)
    assert values["train/economic_active_rows"] == pytest.approx(2.0)
    assert values["train/inactive_actor_rows"] == pytest.approx(0.0)
    assert values["train/max_head_approx_kl"] == pytest.approx(
        max(
            values["train/game_approx_kl"],
            values["train/economic_approx_kl"],
        )
    )


def test_phase_balanced_advantage_normalization_is_global(monkeypatch):
    captured = []
    original = sb3_common.phase_balanced_actor_terms

    def capture(**kwargs):
        captured.append(kwargs["advantages"].detach().cpu().clone())
        return original(**kwargs)

    monkeypatch.setattr(sb3_common, "phase_balanced_actor_terms", capture)
    model = _small_phase_model(batch_size=4)
    model.learn(total_timesteps=4)

    assert len(captured) == 1
    assert captured[0].numel() == 4
    assert captured[0].mean().item() == pytest.approx(0.0, abs=1.0e-6)
    assert captured[0].std().item() == pytest.approx(1.0, abs=1.0e-6)


def test_phase_balanced_ppo_rejects_partial_minibatches_at_train_time():
    model = _small_phase_model(batch_size=2)
    with pytest.raises(ValueError, match="full-rollout minibatch"):
        model.learn(total_timesteps=4)


def test_phase_metrics_flush_at_the_next_rollout_and_final_update(tmp_path):
    class _Run:
        def __init__(self):
            self.rows = []

        def log(self, payload, step=None):
            metric_step = (
                payload["train/total_timesteps"] if step is None else step
            )
            self.rows.append((dict(payload), int(metric_step)))

    run = _Run()
    model = _small_phase_model(batch_size=4)
    callback = EpisodeCheckpointCallback(
        checkpoint=tmp_path / "two_rollouts.zip",
        checkpoint_every=1_000,
        seed=3,
        wandb_run=run,
    )
    model.learn(total_timesteps=8, callback=callback)

    optimizer_steps = [
        step
        for payload, step in run.rows
        if "train/game_policy_loss" in payload
    ]
    assert optimizer_steps == [4, 8]
    local_optimizer_steps = [
        row["train/optimizer_total_timesteps"]
        for row in (
            json.loads(line)
            for line in callback.training_log.read_text().splitlines()
        )
        if row.get("record_kind") == "optimizer"
    ]
    assert local_optimizer_steps == [4, 8]


def test_phase_balanced_target_kl_stops_on_the_larger_head_divergence():
    model = _small_phase_model(
        batch_size=4,
        learning_rate=0.1,
        n_epochs=4,
        target_kl=1.0e-12,
    )
    model.learn(total_timesteps=4)

    assert 1 < model._n_updates < 4
    values = model.logger.name_to_value
    assert values["train/max_head_approx_kl"] == pytest.approx(
        max(
            values["train/game_approx_kl"],
            values["train/economic_approx_kl"],
        )
    )


def test_phase_balanced_checkpoint_metadata_survives_standard_eval_load(tmp_path):
    model = _small_phase_model(batch_size=4)
    attach_atari_training_contract(
        model,
        actor_loss_mode=PHASE_BALANCED_ACTOR_LOSS_MODE,
        economic_init_mean=0.95,
        economic_init_concentration=10.0,
    )
    checkpoint = tmp_path / "phase_balanced.zip"
    model.save(checkpoint)

    restored = ScaledLearningRatePPO.load(checkpoint, device="cpu")
    assert model_actor_loss_mode(restored) == PHASE_BALANCED_ACTOR_LOSS_MODE
    assert model_economic_initialization(
        restored, default_mean=0.5, default_concentration=2.0
    ) == {"mean": 0.95, "concentration": 10.0}


class _CompositeCreditEnv(gym.Env):
    """One gameplay, one trade, and one cached row for policy integration."""

    decision_kinds = (GAMEPLAY, FOLLOWER_TRADE, CACHED_TRADE_REPLAY)

    def __init__(self):
        super().__init__()
        image = gym.spaces.Box(0, 255, shape=(84, 84, 4), dtype=np.uint8)
        self.observation_space = observation_space(image, 6)
        self.action_space = action_space(6)
        self.step_index = 0

    def _observation(self):
        index = min(self.step_index, len(self.decision_kinds) - 1)
        trade = index > 0
        return observation(
            image=np.zeros((84, 84, 4), dtype=np.uint8),
            state=actor_state(
                ammo_fraction=0.2,
                projectile_active=0.0,
                normalized_time=0.1,
                trade_mode=float(trade),
                event_index=0 if trade else None,
                opponent_commitment=np.zeros(5, dtype=np.float32),
            ),
            action_mask=np.ones(6, dtype=np.float32),
            decision_kind=self.decision_kinds[index],
        )

    def reset(self):
        self.step_index = 0
        return self._observation()

    def step(self, action):
        del action
        self.step_index += 1
        done = self.step_index == len(self.decision_kinds)
        reward = 1.0 if done else 0.0
        return self._observation(), reward, done, {}


def test_phase_balanced_ppo_runs_the_real_composite_policy_and_credit_protocol(
        tmp_path
):
    class _Run:
        def __init__(self):
            self.rows = []

        def log(self, payload, step=None):
            metric_step = (
                payload["train/total_timesteps"] if step is None else step
            )
            self.rows.append((dict(payload), int(metric_step)))

    run = _Run()
    model = PhaseBalancedPPO(
        StackPOMDPAtariPolicy,
        _CompositeCreditEnv(),
        policy_kwargs={
            "economic_role": "buyer",
            "economic_input_mode": "full",
            "pretrained_lr_scale": 0.1,
        },
        learning_rate=1.0e-3,
        n_steps=3,
        batch_size=3,
        n_epochs=1,
        gamma=1.0,
        gae_lambda=1.0,
        seed=3,
        device="cpu",
        verbose=0,
    )
    callback = EpisodeCheckpointCallback(
        checkpoint=tmp_path / "phase_balanced.zip",
        checkpoint_every=1_000,
        seed=3,
        wandb_run=run,
    )

    model.learn(total_timesteps=3, callback=callback)

    values = model.logger.name_to_value
    assert values["train/game_active_rows"] == pytest.approx(1.0)
    assert values["train/economic_active_rows"] == pytest.approx(1.0)
    assert values["train/inactive_actor_rows"] == pytest.approx(1.0)
    expected_full_value_loss = np.mean(
        (model.rollout_buffer.returns - model.rollout_buffer.values) ** 2
    )
    assert values["train/value_loss"] == pytest.approx(
        expected_full_value_loss
    )
    assert [group["lr"] for group in model.policy.optimizer.param_groups] == (
        pytest.approx([1.0e-4, 1.0e-3])
    )
    optimizer_payload, optimizer_step = next(
        (payload, step)
        for payload, step in run.rows
        if "train/game_policy_loss" in payload
    )
    assert optimizer_step == 3
    assert optimizer_payload["train/optimizer_total_timesteps"] == 3
    assert optimizer_payload["train/game_active_rows"] == pytest.approx(1.0)
    local_rows = [
        json.loads(line)
        for line in callback.training_log.read_text().splitlines()
    ]
    optimizer_row = next(
        row for row in local_rows if row.get("record_kind") == "optimizer"
    )
    assert optimizer_row["train/economic_active_rows"] == pytest.approx(1.0)
