import json
from pathlib import Path
from types import SimpleNamespace

import gym
import numpy as np
import pytest
import torch

from stable_baselines3.common.vec_env import DummyVecEnv

from replication.atari import train_atari_curriculum_sb3 as trainer
from replication.atari.sb3_common import (
    EpisodeCheckpointCallback,
    ScaledLearningRatePPO,
    json_path,
    step_checkpoint_path,
    target_checkpoint_path,
    target_selection_path,
)
from stackelberg_pomdp.atari.protocol import (
    GAMEPLAY,
    action_space,
    actor_state,
    observation,
    observation_space,
)
from stackelberg_pomdp.atari.stackpomdp_policy import StackPOMDPAtariPolicy


class _StableEnv(gym.Env):
    def __init__(self):
        super().__init__()
        image = gym.spaces.Box(0, 255, shape=(84, 84, 4), dtype=np.uint8)
        self.observation_space = observation_space(image, 6)
        self.action_space = action_space(6)

    def _obs(self):
        return observation(
            image=np.zeros((84, 84, 4), dtype=np.uint8),
            state=actor_state(
                ammo_fraction=1.0,
                projectile_active=0.0,
                normalized_time=0.0,
                trade_mode=0.0,
                event_index=None,
                opponent_commitment=np.zeros(5),
            ),
            action_mask=np.ones(6, dtype=np.float32),
            decision_kind=GAMEPLAY,
        )

    def reset(self):
        return self._obs()

    def step(self, action):
        del action
        return self._obs(), 0.0, True, {}


def _args(stage, *, init_checkpoint=None, resume=None):
    return SimpleNamespace(
        stage=stage,
        seed=1,
        init_checkpoint=init_checkpoint,
        resume=resume,
        learning_rate=2.5e-4,
        pretrained_lr_scale=0.1,
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        clip_range=0.1,
        entropy_coeff=0.01,
        value_coefficient=0.5,
        max_grad_norm=0.5,
        device="cpu",
    )


def _source_model(vec_env, *, scale=1.0):
    return ScaledLearningRatePPO(
        StackPOMDPAtariPolicy,
        vec_env,
        policy_kwargs={
            "economic_role": "gameplay",
            "economic_input_mode": "full",
            "pretrained_lr_scale": scale,
        },
        learning_rate=2.5e-4,
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        gamma=1.0,
        gae_lambda=1.0,
        seed=9,
        device="cpu",
        verbose=0,
    )


def _module_equal(left, right):
    return all(
        torch.equal(left.state_dict()[key], right.state_dict()[key])
        for key in left.state_dict()
    )


def test_e0_parser_separates_stage_initialization_from_true_resume(tmp_path):
    e0a = trainer.parse_args([
        "--stage", "e0a",
        "--checkpoint", str(tmp_path / "e0a.zip"),
        "--no-wandb",
    ])
    assert e0a.timesteps == 50_000_000
    assert e0a.checkpoint_every == 400_000
    assert e0a.target_eval_every == 400_000
    assert e0a.target_eval_episodes == 20
    assert e0a.target_confirm_episodes == 100
    assert e0a.target_consecutive_passes == 2

    with pytest.raises(SystemExit):
        trainer.parse_args(["--stage", "e0b", "--no-wandb"])

    e0b = trainer.parse_args([
        "--stage", "e0b",
        "--init-checkpoint", str(tmp_path / "e0a.zip"),
        "--no-wandb",
    ])
    assert e0b.n_steps == 205
    assert e0b.batch_size == 820

    with pytest.raises(SystemExit):
        trainer.parse_args([
            "--stage", "e0b",
            "--init-checkpoint", str(tmp_path / "e0a.zip"),
            "--resume", str(tmp_path / "e0b.zip"),
            "--no-wandb",
        ])


def test_e0b_transfers_actor_but_not_critic_and_uses_scaled_groups(tmp_path):
    vec_env = DummyVecEnv([_StableEnv])
    try:
        source = _source_model(vec_env, scale=1.0)
        with torch.no_grad():
            for parameter in source.policy.value_net.parameters():
                parameter.fill_(0.314159)
        checkpoint = tmp_path / "e0a.zip"
        source.save(checkpoint)

        target = trainer._new_model(
            _args("e0b", init_checkpoint=str(checkpoint)), vec_env
        )
        assert _module_equal(
            source.policy.features_extractor,
            target.policy.features_extractor,
        )
        assert _module_equal(
            source.policy.game_action_net,
            target.policy.game_action_net,
        )
        assert not _module_equal(source.policy.value_net, target.policy.value_net)
        assert [
            group["lr_scale"] for group in target.policy.optimizer.param_groups
        ] == [0.1, 1.0]
    finally:
        vec_env.close()


def test_e0_resume_restores_the_complete_stage_checkpoint(tmp_path):
    vec_env = DummyVecEnv([_StableEnv])
    try:
        source = _source_model(vec_env, scale=1.0)
        with torch.no_grad():
            for parameter in source.policy.value_net.parameters():
                parameter.fill_(0.271828)
            for parameter in source.policy.economic_head.parameters():
                parameter.fill_(-0.161803)
        checkpoint = tmp_path / "e0a_resume.zip"
        source.save(checkpoint)

        restored = trainer._resumed_model(
            _args("e0a", resume=str(checkpoint)), vec_env
        )
        assert _module_equal(source.policy.value_net, restored.policy.value_net)
        assert _module_equal(
            source.policy.economic_head,
            restored.policy.economic_head,
        )
        assert [
            group["lr_scale"]
            for group in restored.policy.optimizer.param_groups
        ] == [1.0, 1.0]
    finally:
        vec_env.close()


def _target_evaluation(
        *,
        episodes=20,
        reward_failures=None,
        shot_failures=None,
        gameplay_horizon=200,
        payment=0.0,
        accounting_error=0.0,
        return_error=0.0,
):
    if reward_failures is None:
        reward_failures = episodes // 10
    if shot_failures is None:
        shot_failures = episodes // 20
    rows = []
    for episode in range(episodes):
        reward = 4.0 if episode < reward_failures else 5.0
        fired = 4 if episode < shot_failures else 5
        rows.append({
            "evaluation_episode": episode,
            "evaluation_return": reward + float(return_error),
            "evaluation_steps": gameplay_horizon,
            "game_reward": reward,
            "payments": payment,
            "shots_fired": fired,
            "final_ammo": 5 - fired,
            "bullet_accounting_error": accounting_error,
        })
    return {"summary": {}, "episode_rows": rows}


def _target_metrics(evaluation, *, episodes=20):
    return trainer.e0a_target_metrics(
        evaluation,
        required_episodes=episodes,
        gameplay_horizon=200,
        target_reward=5.0,
        target_mean_reward=4.8,
        target_reward_rate=0.90,
        target_shots=5.0,
        target_mean_shots=4.95,
        target_fired_rate=0.95,
        target_final_ammo=0.05,
    )


def test_e0a_target_gate_includes_performance_horizon_and_accounting():
    metrics = _target_metrics(_target_evaluation())
    assert metrics["passed"]
    assert metrics["episodes"] == 20
    assert metrics["mean_game_reward"] == pytest.approx(4.9)
    assert metrics["reward_target_rate"] == pytest.approx(0.90)
    assert metrics["mean_shots_fired"] == pytest.approx(4.95)
    assert metrics["fired_all_rate"] == pytest.approx(0.95)
    assert metrics["mean_final_ammo"] == pytest.approx(0.05)
    assert metrics["complete_horizons"]
    assert metrics["exact_economics"]

    assert not _target_metrics(
        _target_evaluation(reward_failures=3)
    )["passed"]
    assert not _target_metrics(
        _target_evaluation(shot_failures=2)
    )["passed"]
    assert not _target_metrics(
        _target_evaluation(gameplay_horizon=199)
    )["passed"]
    assert not _target_metrics(
        _target_evaluation(payment=0.01)
    )["passed"]
    assert not _target_metrics(
        _target_evaluation(return_error=0.01)
    )["passed"]
    assert not _target_metrics(
        _target_evaluation(accounting_error=1.0)
    )["passed"]
    assert not _target_metrics(
        _target_evaluation(episodes=19, reward_failures=1, shot_failures=0)
    )["passed"]


def test_e0a_target_gate_checks_mean_reward_separately_from_hit_rate():
    evaluation = _target_evaluation()
    evaluation["episode_rows"][0]["game_reward"] = 0.0
    evaluation["episode_rows"][0]["evaluation_return"] = 0.0
    metrics = _target_metrics(evaluation)
    assert metrics["reward_target_rate"] == pytest.approx(0.90)
    assert metrics["mean_game_reward"] == pytest.approx(4.7)
    assert not metrics["passed"]


def test_selector_requires_two_screens_then_fresh_confirmation(
        tmp_path, monkeypatch
):
    checkpoint = tmp_path / "e0a.zip"
    args = trainer.parse_args([
        "--stage", "e0a",
        "--checkpoint", str(checkpoint),
        "--num-envs", "1",
        "--no-wandb",
    ])
    screen_one = step_checkpoint_path(checkpoint, 400_000)
    screen_two = step_checkpoint_path(checkpoint, 800_000)
    screen_one.write_bytes(b"screen-one")
    screen_two.write_bytes(b"screen-two")

    loaded = []
    saved = []

    class _Candidate:
        def __init__(self, source):
            self.source = Path(source)

        def save(self, path):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"selected from {self.source.name}", encoding="utf-8")
            saved.append((self.source, path))

    def fake_load(path, *, device):
        assert device == "cpu"
        loaded.append(Path(path))
        return _Candidate(path)

    evaluation_queue = [
        _target_evaluation(),
        _target_evaluation(),
        _target_evaluation(episodes=100),
    ]
    evaluation_calls = []

    def fake_evaluate(model, env_factory, *, episodes):
        seeds = [env_factory(episode) for episode in range(episodes)]
        evaluation_calls.append({
            "source": model.source,
            "episodes": episodes,
            "seeds": seeds,
        })
        return evaluation_queue.pop(0)

    monkeypatch.setattr(
        trainer.ScaledLearningRatePPO, "load", staticmethod(fake_load)
    )
    monkeypatch.setattr(trainer, "make_env", lambda _args, *, seed: seed)
    monkeypatch.setattr(trainer, "evaluate_model", fake_evaluate)

    selector = trainer.E0ATargetSelector(args)
    assert not selector.evaluate_checkpoint(screen_one, step=400_000)
    assert not target_checkpoint_path(checkpoint).exists()
    assert len(evaluation_calls) == 1
    assert evaluation_calls[0]["episodes"] == 20

    assert selector.evaluate_checkpoint(screen_two, step=800_000)
    assert loaded == [screen_one, screen_two]
    assert [call["episodes"] for call in evaluation_calls] == [20, 20, 100]
    validation_start = args.seed + trainer.VALIDATION_SEED_OFFSET
    confirmation_start = args.seed + trainer.CONFIRMATION_SEED_OFFSET
    assert evaluation_calls[0]["seeds"] == list(
        range(validation_start, validation_start + 20)
    )
    assert evaluation_calls[1]["seeds"] == evaluation_calls[0]["seeds"]
    assert evaluation_calls[2]["seeds"] == list(
        range(confirmation_start, confirmation_start + 100)
    )

    selected_checkpoint = target_checkpoint_path(checkpoint)
    selection_path = target_selection_path(checkpoint)
    assert selected_checkpoint.is_file()
    assert selection_path.is_file()
    assert json_path(screen_one, "validation.json").is_file()
    assert json_path(screen_two, "validation.json").is_file()
    assert json_path(screen_two, "confirmation_1.json").is_file()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    assert selection["kind"] == "confirmation"
    assert selection["episodes"] == 100
    assert selection["screening_checkpoint_path"] == str(screen_two)
    assert selection["selected_checkpoint_path"] == str(selected_checkpoint)
    assert (screen_two, selected_checkpoint) in saved


def test_e0a_training_chunks_end_after_complete_ppo_updates(
        tmp_path, monkeypatch
):
    checkpoint = tmp_path / "e0a.zip"
    args = SimpleNamespace(
        checkpoint=str(checkpoint),
        checkpoint_every=400_000,
        seed=1,
        resume=None,
        n_steps=200,
        num_envs=4,
        target_eval_every=400_000,
        target_stop=True,
        timesteps=1_200_000,
    )

    class _Model:
        def __init__(self):
            self.num_timesteps = 0
            self._n_updates = 0
            self.learn_calls = []

        def learn(self, *, total_timesteps, callback, reset_num_timesteps):
            del callback
            assert total_timesteps % (args.n_steps * args.num_envs) == 0
            self.num_timesteps += total_timesteps
            self._n_updates += 1
            self.learn_calls.append((total_timesteps, reset_num_timesteps))
            return self

        def save(self, path):
            Path(path).write_text(
                f"steps={self.num_timesteps};updates={self._n_updates}",
                encoding="utf-8",
            )

    selectors = []

    class _Selector:
        def __init__(self, _args, *, wandb_run=None):
            del _args, wandb_run
            self.evaluations = []
            self.finished_at = None
            self.state = {"selected": None}
            selectors.append(self)

        def evaluate_checkpoint(self, path, *, step):
            contents = Path(path).read_text(encoding="utf-8")
            self.evaluations.append((Path(path), step, contents))
            return len(self.evaluations) == 2

        def finish_at_cap(self, step):
            self.finished_at = step

        def _save_state(self):
            pass

        def final_checkpoint(self):
            return tmp_path / "selected.zip"

    callback = object()
    monkeypatch.setattr(
        trainer, "make_training_callback", lambda *args, **kwargs: callback
    )
    monkeypatch.setattr(trainer, "E0ATargetSelector", _Selector)

    model = _Model()
    selected, stopped = trainer.train_e0a_to_target(model, args)
    assert stopped
    assert selected == tmp_path / "selected.zip"
    assert model.num_timesteps == 800_000
    assert model.learn_calls == [(400_000, True), (400_000, False)]
    assert selectors[0].finished_at is None
    assert [step for _, step, _ in selectors[0].evaluations] == [
        400_000,
        800_000,
    ]
    assert selectors[0].evaluations[0][2] == "steps=400000;updates=1"
    assert selectors[0].evaluations[1][2] == "steps=800000;updates=2"


def test_resumed_checkpoint_schedule_advances_past_existing_clock(tmp_path):
    callback = EpisodeCheckpointCallback(
        checkpoint=tmp_path / "e0a.zip",
        checkpoint_every=100,
        seed=1,
        resume=True,
    )
    callback.model = SimpleNamespace(num_timesteps=250)
    callback._init_callback()
    assert callback.next_checkpoint == 300
    assert callback.started_timesteps == 250


def test_e0a_validation_confirmation_and_final_seed_sets_are_disjoint():
    base_seed = 7
    validation = set(range(
        base_seed + trainer.VALIDATION_SEED_OFFSET,
        base_seed + trainer.VALIDATION_SEED_OFFSET + 20,
    ))
    confirmation_one = set(range(
        base_seed + trainer.CONFIRMATION_SEED_OFFSET,
        base_seed + trainer.CONFIRMATION_SEED_OFFSET + 100,
    ))
    confirmation_two = set(range(
        base_seed
        + trainer.CONFIRMATION_SEED_OFFSET
        + trainer.CONFIRMATION_SEED_STRIDE,
        base_seed
        + trainer.CONFIRMATION_SEED_OFFSET
        + trainer.CONFIRMATION_SEED_STRIDE
        + 100,
    ))
    final = set(range(
        base_seed + trainer.FINAL_EVALUATION_SEED_OFFSET,
        base_seed + trainer.FINAL_EVALUATION_SEED_OFFSET + 20,
    ))
    seed_sets = [validation, confirmation_one, confirmation_two, final]
    for index, left in enumerate(seed_sets):
        for right in seed_sets[index + 1:]:
            assert left.isdisjoint(right)
