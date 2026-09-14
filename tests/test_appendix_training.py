"""Behavioral checks for the two appendix training recipes."""

import json

import numpy as np
import pytest
import torch

from stackelberg_pomdp.envs.matrix import (
    LegacyMatrixQLeaderEnv, MatrixMetaLeaderEnv, get_matrix_game,
)
from stackelberg_pomdp.experiments.matrix_ablations import (
    build_parser, train_leader,
)
from stackelberg_pomdp.matrix_ablations.reinforce import Reinforce


class CooperatingFollower:
    def predict(self, observation, deterministic=True):
        return np.asarray(0), None


def test_pg_preserves_commitments_and_can_condition_on_the_phase(tmp_path):
    # Removing caching, or ignoring the phase in its key, breaks this trace.
    env = MatrixMetaLeaderEnv(
        get_matrix_game("prisoners_dilemma"),
        response_model=CooperatingFollower(), phase_observable=True,
    )
    model = Reinforce(env=env, n_steps=10,
                      policy_kwargs={"cache_actions": True}, seed=1)
    policy = model.policy
    # The final two one-hot features encode the phase.
    with torch.no_grad():
        policy.action_net.weight.zero_()
        policy.action_net.weight[0, -2] = 20
        policy.action_net.weight[1, -1] = 20
    obs = env.reset()
    for _ in range(5):
        action, _ = model.predict(obs, deterministic=True)
        assert int(action) == 0
        obs, reward, done, info = env.step(action)
        assert reward == 0 and not done
    action, _ = model.predict(obs, deterministic=True)
    assert int(action) == 1
    _, _, _, info = env.step(action)
    assert info["commitment_consistent"] is False

    policy.clear_obs_action_map()
    hidden = {"base_environment": 0, env.PHASE_KEY: 0}
    first, _ = model.predict(hidden, deterministic=False)
    with torch.no_grad():
        policy.action_net.weight.mul_(-1)
    repeated, _ = model.predict(hidden, deterministic=False)
    assert int(repeated) == int(first)
    policy.clear_obs_action_map()
    after_reset, _ = model.predict(hidden, deterministic=True)
    assert int(after_reset) != int(first)

    model.save(tmp_path / "pg")
    restored = Reinforce.load(tmp_path / "pg.zip")
    assert restored.policy.fix_actions
    restored.policy.clear_obs_action_map()
    assert int(restored.predict(hidden, deterministic=True)[0]) == int(after_reset)
    model.get_env().close()


def test_pg_complete_episode_batch_contains_query_reward_to_go():
    # A batch cut, missing queries, or a learned baseline changes these returns.
    env = LegacyMatrixQLeaderEnv(
        get_matrix_game("battle_of_the_sexes"), response_episodes=2,
        q_epsilon=0, q_init="zero",
    )
    model = Reinforce(env=env, n_steps=6,
                      policy_kwargs={"cache_actions": True}, seed=1)
    with torch.no_grad():
        model.policy.action_net.weight.copy_(torch.tensor([[30.], [-30.]]))
    from stackelberg_pomdp.callbacks import FixPolicyActionsCallback
    model.learn(6, callback=FixPolicyActionsCallback())
    np.testing.assert_array_equal(model.rollout_buffer.rewards.ravel(), [0, 0, 2, 0, 0, 2])
    np.testing.assert_array_equal(model.rollout_buffer.returns.ravel(), [2] * 6)
    assert model._n_updates == 1
    assert model.policy.obs_action_map == {}
    model.get_env().close()


@pytest.mark.parametrize("figure,condition,algorithm", [
    ("fig_bots_leaderreward", "included", "SIMPLEQ"),
    ("fig_bots_leaderreward", "excluded", "SIMPLEQ"),
])
def test_appendix_recipe_trains_saves_and_reports_reward_play(
        tmp_path, figure, condition, algorithm):
    # Exercise the public entry point, real learner, and separate evaluation.
    args = build_parser().parse_args([
        "leader", "--figure", figure, "--condition", condition,
        "--timesteps", "110", "--eval-freq", "110", "--eval-episodes", "2",
        "--final-eval-episodes", "2", "--eval-warmup", "0",
        "--final-eval-seed-start", "2000001",
        "--output-root", str(tmp_path),
    ])
    run = train_leader(args)
    config = json.loads((run / "config.json").read_text())
    assert config["algorithm"] == algorithm
    assert config["q_protocol"]["exploration"] == "parameter_noise"
    assert (run / "model.zip").is_file()
    evaluation = json.loads((run / "evaluation.json").read_text())
    assert evaluation["per_stage_summary"]["n"] == 2
    manifest = json.loads((run / "run_manifest.json").read_text())
    assert manifest["status"] == "completed"
    rows = [json.loads(line) for line in (run / "progress.jsonl").read_text().splitlines()]
    assert rows[-1]["policy_updates"] > 0
    assert rows[-1]["evaluation_mean"] == evaluation["per_stage_summary"]["mean"]


def test_simpleq_noise_is_episode_fixed_and_never_changes_clean_weights(tmp_path):
    from stackelberg_pomdp.matrix_ablations.simple_q import SimpleQ
    from stackelberg_pomdp.callbacks import FixPolicyActionsCallback

    env = LegacyMatrixQLeaderEnv(get_matrix_game("coordination_zero_miscoordination"))
    model = SimpleQ(env=env, seed=3)
    clean = {key: value.clone() for key, value in model.q_net.state_dict().items()}
    obs = env.reset()
    actions = [int(model.predict(obs, deterministic=False)[0]) for _ in range(11)]
    assert len(set(actions)) == 1
    for key, value in model.q_net.state_dict().items():
        torch.testing.assert_close(value, clean[key])
    model.policy.clear_obs_action_map()
    assert model.policy.obs_action_map == {}
    # Replay must include all adaptation turns and the terminal reward turn.
    model.learn(110, callback=FixPolicyActionsCallback())
    assert model.replay_buffer.size() == 110
    assert model._n_updates > 0
    assert np.count_nonzero(model.replay_buffer.dones) == 10
    assert any(not torch.equal(value, clean[key])
               for key, value in model.q_net.state_dict().items())
    model.save(tmp_path / "q")
    restored = SimpleQ.load(tmp_path / "q.zip")
    assert int(restored.predict(obs, deterministic=True)[0]) == int(model.predict(obs, deterministic=True)[0])
    torch.testing.assert_close(restored.policy.parameter_noise_std, model.policy.parameter_noise_std)
    model.get_env().close()


def test_phase_figure_command_trains_both_treatments_from_a_real_response(tmp_path):
    from stackelberg_pomdp.experiments.matrix_ablations import train_meta_follower
    follower = train_meta_follower(build_parser().parse_args([
        "meta-follower", "--algorithm", "REINFORCE", "--timesteps", "50",
        "--output-root", str(tmp_path / "follower"),
    ]))
    for condition in ("visible", "hidden"):
        args = build_parser().parse_args([
            "leader", "--figure", "fig_memory_pg", "--condition", condition,
            "--response-checkpoint", str(follower / "model.zip"),
            "--allow-uncertified-response", "--timesteps", "100",
            "--eval-freq", "100", "--eval-episodes", "1", "--final-eval-episodes", "1",
            "--output-root", str(tmp_path / "leaders"),
        ])
        run = train_leader(args)
        config = json.loads((run / "config.json").read_text())
        assert config["algorithm"] == "PG"
        assert config["response_algorithm"] == "REINFORCE"
        assert config["leader_protocol"]["batch_steps"] == 100
        model = Reinforce.load(run / "model.zip")
        assert model._n_updates == 1


def test_public_targets_cover_all_appendix_curves_and_learning_rates():
    from replication.run import build_commands, load_targets, validate_command, DEFAULT_MANIFEST
    targets = load_targets(DEFAULT_MANIFEST)
    expected = {
        "fig_memory_phase_ablation": 6,
        "fig_reward_during_learning_ablation": 4,
    }
    for name, count in expected.items():
        commands = build_commands(targets[name], seed=1)
        assert len(commands) == count
        for _, command in commands:
            validate_command(command)


def test_plot_keeps_learning_rates_separate():
    import importlib.util
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt

    path = Path(__file__).resolve().parents[1] / "replication/matrix_ablations/plot.py"
    spec = importlib.util.spec_from_file_location("appendix_plot", path)
    plot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plot)
    rows = [dict(experiment="phase_observability", matrix="prisoners_dilemma", algorithm="PG",
                 condition=condition, learning_rate=lr, seed=seed,
                 profile_id="paper_joint_v1", status="completed",
                 evaluation_target_step=step, leader_reward=float(seed))
            for condition in ("visible", "hidden") for lr in (0.008, 0.015)
            for seed in (1, 2) for step in (0, 100)]
    history = pd.DataFrame(rows)
    runs = history.drop(columns=["evaluation_target_step", "leader_reward"]).drop_duplicates()
    plot.validate_runs(runs, {1, 2}, False)
    plot.validate_required_figure_cells(runs, "fig_phase_observability", 2, "test")
    figure = plot.plot_phase(plot.summarize(history))
    assert len(figure.axes[0].lines) == 4
    assert all(len(line.get_xdata()) == 2 for line in figure.axes[0].lines)
    plt.close(figure)


def test_trend_pairs_seeds_within_the_same_learning_rate():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "replication/matrix_ablations/trend.py"
    spec = importlib.util.spec_from_file_location("appendix_trend", path)
    trend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trend)
    rows = [dict(stage="leader", status="completed", experiment="phase_observability",
                 matrix="prisoners_dilemma", algorithm="PG", condition=condition,
                 learning_rate=rate, seed=seed, per_stage_mean=value)
            for rate in (0.008, 0.015) for seed in (1, 2)
            for condition, value in (("visible", 2), ("hidden", 1))]
    assert len(trend._cell_summaries(rows)) == 4
    deltas = trend.condition_deltas(rows)["phase_observability"]
    assert len(deltas) == 2
    assert all(row["mean"] == 1 and row["paired_seeds"] == [1, 2] for row in deltas)
