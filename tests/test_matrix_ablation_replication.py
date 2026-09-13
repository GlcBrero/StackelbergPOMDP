import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = REPO_ROOT / "replication/matrix_ablations/{}.py".format(name)
    spec = importlib.util.spec_from_file_location(
        "matrix_ablation_{}".format(name), path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plan_covers_six_phase_six_reset_and_four_reward_curves_per_seed(tmp_path):
    sweep = load_script("sweep")
    args = argparse.Namespace(
        results_root=str(tmp_path),
        sweep_id="test",
        seeds=[1, 2],
        e1_timesteps=10,
        leader_timesteps=20,
        eval_freq=5,
    )
    sweep.make_plan(args)
    with (tmp_path / "test/plan.json").open() as handle:
        payload = json.load(handle)
    assert payload["e1_runs"] == 2
    assert payload["leader_runs"] == 32
    assert len(payload["records"]) == 34
    meta = [row for row in payload["records"] if row["stage"] == "meta-follower"]
    assert {row["seed"] for row in meta} == {1, 2}
    response_runs = [
        row for row in payload["records"]
        if row.get("response_seed") is not None
    ]
    assert all(row["response_seed"] == row["seed"] for row in response_runs)
    from stackelberg_pomdp.envs.matrix import get_matrix_game
    by_seed = {row["seed"]: row for row in meta}
    for row in response_runs:
        response_game = get_matrix_game(by_seed[row["seed"]]["match"]["matrix"])
        leader_game = get_matrix_game(row["match"]["matrix"])
        np.testing.assert_array_equal(response_game.payoffs, leader_game.payoffs)
    assert all(row["match"]["sweep_id"] == "test" for row in payload["records"])
    assert all(
        row["match"]["record_key"] == row["key"]
        for row in payload["records"]
    )
    independent = sweep.select_records(payload, "leader", "independent")
    meta_dependent = sweep.select_records(payload, "leader", "meta-dependent")
    assert len(independent) == 20
    assert len(meta_dependent) == 12

    leaders = [row for row in payload["records"] if row["stage"] == "leader"]
    assert all("--learning-rate" in row["argv"] for row in leaders)
    assert all(
        not any(argument.startswith("--es-") for argument in row["argv"])
        for row in leaders
    )


def test_sweep_status_collapses_failed_attempts_without_hiding_duplicates():
    sweep = load_script("sweep")
    record = {"match": {"sweep_id": "s", "record_key": "cell"}}

    def item(attempt, status):
        config = {"sweep_id": "s", "record_key": "cell", "attempt": attempt}
        return config, {"status": status}, Path("attempt{}".format(attempt))

    status, directory = sweep.record_status(record, [item(0, "failed")])
    assert status == "failed"
    assert directory == Path("attempt0")
    status, directory = sweep.record_status(
        record, [item(0, "failed"), item(1, "completed")]
    )
    assert status == "completed"
    assert directory == Path("attempt1")
    status, directory = sweep.record_status(
        record, [item(0, "completed"), item(1, "completed")]
    )
    assert status == "duplicate"
    assert directory is None


def test_sweep_parser_accepts_one_array_record():
    sweep = load_script("sweep")
    args = sweep.build_parser().parse_args([
        "run", "--sweep-id", "test", "--stage", "leader",
        "--record-index", "17", "--record-group", "meta-dependent",
        "--wandb",
    ])
    assert args.record_index == 17
    assert args.record_group == "meta-dependent"
    assert args.wandb is True


def test_wandb_record_args_are_bound_to_logical_attempt(tmp_path):
    sweep = load_script("sweep")
    args = sweep.common_output(
        tmp_path, tmp_path / "plan.json", "qualitative-v1",
        "phase.prisoners_dilemma.a2c.visible.seed1", 2,
        wandb=True, wandb_project="StackPOMDP",
    )
    assert "--wandb" in args
    assert "--no-wandb" not in args
    name_index = args.index("--wandb-name") + 1
    assert args[name_index].endswith(".attempt2")


def test_meta_follower_parser_accepts_explicit_network_widths():
    from stackelberg_pomdp.experiments.matrix_ablations import build_parser

    args = build_parser().parse_args([
        "meta-follower", "--net-arch", "128,64", "--timesteps", "10",
    ])
    assert args.net_arch == (128, 64)


def test_meta_follower_parser_resolves_paper_reinforce_protocol():
    from stackelberg_pomdp.experiments.matrix_ablations import (
        build_parser,
        meta_config,
        validate_meta_args,
    )
    from stackelberg_pomdp.envs.matrix import get_matrix_game

    args = build_parser().parse_args([
        "meta-follower", "--algorithm", "REINFORCE",
    ])
    validate_meta_args(args)
    assert args.timesteps == 25_000
    assert args.learning_rate == 0.02
    assert args.ent_coef == 0.0
    assert args.episodes_per_batch == 10
    assert args.n_steps == 50
    assert args.net_arch == ()
    config = meta_config(args, get_matrix_game("prisoners_dilemma"))
    assert config["profile_id"] == "paper_joint_v1"
    assert config["reinforce_protocol"]["checkpoint_geometry"] == {
        "update_interval": 100,
        "follower_gradient_samples_interval": 5000,
        "executed_env_steps_interval": 10_000,
    }

    diagnostic_lr = build_parser().parse_args([
        "meta-follower", "--algorithm", "REINFORCE",
        "--learning-rate", "0.01",
    ])
    validate_meta_args(diagnostic_lr)
    assert diagnostic_lr.learning_rate == 0.01

    legacy_memory = build_parser().parse_args([
        "meta-follower", "--algorithm", "REINFORCE",
        "--memory-mode", "opponent",
    ])
    validate_meta_args(legacy_memory)
    assert legacy_memory.timesteps == 32_500
    assert legacy_memory.episodes_per_batch == 13
    assert legacy_memory.n_steps == 65
    assert meta_config(
        legacy_memory,
        get_matrix_game("prisoners_dilemma", memory_mode="opponent"),
    )["profile_id"] == "paper_opponent_sensitivity_v1"

    historical = build_parser().parse_args([
        "meta-follower", "--algorithm", "REINFORCE",
        "--profile-id", "legacy_opponent_v1",
    ])
    validate_meta_args(historical)
    historical_spec = get_matrix_game(
        "prisoners_dilemma", profile_id="legacy_opponent_v1"
    )
    historical_config = meta_config(historical, historical_spec)
    assert historical.timesteps == 32_500
    assert historical_spec.memory_mode == "opponent"
    assert historical_spec.reward_offset == -2.5
    assert historical_config["profile_id"] == "legacy_opponent_v1"

    leader = build_parser().parse_args([
        "leader", "--experiment", "phase_observability",
        "--condition", "visible", "--response-algorithm", "REINFORCE",
    ])
    assert leader.response_algorithm == "REINFORCE"


def test_matrix_parsers_accept_dqn_response_protocol():
    from stackelberg_pomdp.experiments.matrix_ablations import build_parser

    meta = build_parser().parse_args([
        "meta-follower", "--algorithm", "DQN",
        "--dqn-learning-starts", "10",
        "--dqn-target-update-interval", "250",
        "--dqn-exploration-steps", "2000",
        "--dqn-final-epsilon", "0.02",
    ])
    assert meta.algorithm == "DQN"
    assert meta.dqn_learning_starts == 10
    assert meta.dqn_target_update_interval == 250
    assert meta.dqn_exploration_steps == 2_000
    assert meta.dqn_final_epsilon == 0.02

    leader = build_parser().parse_args([
        "leader", "--experiment", "phase_observability",
        "--condition", "visible", "--response-algorithm", "DQN",
    ])
    assert leader.response_algorithm == "DQN"


def test_plot_collects_only_evaluation_rows_and_verifies_hashes(tmp_path):
    plot = load_script("plot")
    run_dir = tmp_path / "runs/q_reset/example"
    run_dir.mkdir(parents=True)
    config = {
        "stage": "leader",
        "experiment": "q_reset",
        "condition": "reset",
        "matrix": "battle_of_the_sexes",
        "algorithm": "A2C",
        "seed": 1,
        "learning_rate": 0.008,
    }
    config_path = run_dir / "config.json"
    progress_path = run_dir / "progress.jsonl"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    rows = [
        {"global_step": 10, "candidate_return_mean": 1.0},
        {
            "global_step": 20,
            "evaluation_target_step": 20,
            "evaluation_mean": 1.5,
            "evaluation_sem": 0.2,
            "evaluation_episodes": 5,
        },
    ]
    progress_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    manifest = {
        "status": "completed",
        "config_sha256": plot.config_sha256(config),
        "artifacts": {"progress": {"sha256": plot.sha256(progress_path)}},
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    runs, history, configs = plot.collect_runs(tmp_path)
    assert len(runs) == 1
    assert len(history) == 1
    assert len(configs) == 1
    assert history.iloc[0]["leader_reward"] == 1.5
    assert history.iloc[0]["evaluation_target_step"] == 20




def test_plot_parser_accepts_repeated_additional_inputs():
    plot = load_script("plot")
    args = plot.build_parser().parse_args([
        "--input", "seeds1to3",
        "--additional-input", "seeds4to10",
        "--additional-input", "seeds11to25",
        "--seeds", "1-25",
    ])
    assert args.input == Path("seeds1to3")
    assert args.additional_input == [
        Path("seeds4to10"), Path("seeds11to25")
    ]


def test_plot_rejects_missing_seeds_within_each_learning_rate():
    plot = load_script("plot")
    base = {
        "status": "completed",
        "profile_id": "paper_joint_v1",
        "experiment": "phase_observability",
        "condition": "visible",
        "matrix": "prisoners_dilemma",
        "algorithm": "A2C",
        "learning_rate": 0.008,
    }
    runs = pd.DataFrame([
        {**base, "seed": 1, "learning_rate": 0.008},
        {**base, "seed": 2, "learning_rate": 0.02},
    ])
    try:
        plot.validate_runs(runs, {1, 2}, allow_incomplete=False)
    except ValueError as error:
        assert "seed mismatch" in str(error)
    else:
        raise AssertionError("incompatible learning-rate cohorts must be rejected")


def test_plot_summary_uses_sample_sem_across_seed_level_means():
    plot = load_script("plot")
    history = pd.DataFrame([
        {
            "experiment": "q_reset",
            "matrix": "battle_of_the_sexes",
            "algorithm": "A2C",
            "condition": "reset",
            "learning_rate": 0.008,
            "evaluation_target_step": 100,
            "seed": 1,
            "leader_reward": 1.0,
        },
        {
            "experiment": "q_reset",
            "matrix": "battle_of_the_sexes",
            "algorithm": "A2C",
            "condition": "reset",
            "learning_rate": 0.008,
            "evaluation_target_step": 100,
            "seed": 2,
            "leader_reward": 3.0,
        },
    ])
    plot.validate_history(history, {1, 2}, allow_incomplete=False)
    row = plot.summarize(history).iloc[0]
    assert row["mean"] == 2.0
    assert row["n"] == 2
    assert np.isclose(row["std"], np.sqrt(2.0))
    assert np.isclose(row["sem"], 1.0)
