"""Defaults must reach the actual learners, not just their CLI help."""

import json
from inspect import signature

import pytest
from stable_baselines3 import A2C, DQN, PPO

from replication.cluster import cohort
from stackelberg_pomdp.env_setups import get_simple_allocation_env
from stackelberg_pomdp.experiments.common import finalized_config
from stackelberg_pomdp.experiments.simple_allocation import build_parser
from stackelberg_pomdp.experiments import matrix_ablations
from stackelberg_pomdp.rl_trainer_setup import get_custom_training_algorithm
from stackelberg_pomdp.run_setups import _write_optimizer_parameters


@pytest.mark.parametrize("algorithm,cls", [("PPO", PPO), ("A2C", A2C)])
def test_common_parser_defaults_reach_model_and_saved_metadata(algorithm, cls, tmp_path):
    args = build_parser().parse_args(["--algorithm", algorithm])
    config = finalized_config(args, "simple_allocation:3")
    config["logger"] = None
    env = get_simple_allocation_env(config)
    try:
        model = get_custom_training_algorithm(config, env)
        defaults = signature(cls.__init__).parameters
        for key in ("learning_rate", "gae_lambda", "vf_coef", "max_grad_norm"):
            assert getattr(model, key) == defaults[key].default
        assert model.gamma == 1.0  # Undiscounted economic objective.
        assert model.ent_coef == 0.01  # Documented commitment exploration.
        if algorithm == "PPO":
            assert model.batch_size == defaults["batch_size"].default
            assert model.n_epochs == defaults["n_epochs"].default
            assert model.clip_range(1.0) == defaults["clip_range"].default
        _write_optimizer_parameters(model, tmp_path)
        saved = json.loads((tmp_path / "optimizer_parameters.json").read_text())
        assert saved["learning_rate"] == model.learning_rate
        assert saved["n_steps"] == model.n_steps
        assert saved["stable_baselines3_version"]
    finally:
        env.close()


def test_all_210_main_target_configs_use_library_optimizer_defaults():
    records = cohort.make_records(cohort.load_targets(cohort.ROOT / "replication/targets.json"))
    assert len(records) == 210
    for row in records:
        config = row["resolved_config"]
        defaults = signature(PPO.__init__).parameters
        for key, parameter in (("learning_rate", "learning_rate"),
                               ("ppo_batch_size", "batch_size"),
                               ("ppo_n_epochs", "n_epochs")):
            assert config[key] == defaults[parameter].default
        assert config["ppo_rollout_geometry"] == "complete_episodes"
        if row["group"] != "spm":
            assert config["ppo_episodes_per_batch"] is None


@pytest.mark.parametrize("algorithm,cls", [("PPO", PPO), ("A2C", A2C), ("DQN", DQN)])
def test_optional_meta_follower_defaults_reach_the_model(algorithm, cls):
    from stackelberg_pomdp.envs.matrix import MatrixFixedCommitmentResponseEnv, get_matrix_game

    args = matrix_ablations.build_parser().parse_args(["meta-follower", "--algorithm", algorithm])
    matrix_ablations.validate_meta_args(args)
    spec = get_matrix_game(args.matrix)
    env = MatrixFixedCommitmentResponseEnv(spec, seed=args.seed)
    try:
        model = matrix_ablations._meta_model(args, env, spec.episode_length)
        defaults = signature(cls.__init__).parameters
        assert model.learning_rate == defaults["learning_rate"].default
        assert model.gamma == 1.0
        if algorithm == "DQN":
            for key in ("buffer_size", "learning_starts", "batch_size", "target_update_interval",
                        "exploration_fraction", "exploration_final_eps"):
                assert getattr(model, key) == defaults[key].default
        else:
            assert model.ent_coef == defaults["ent_coef"].default
            assert model.gae_lambda == defaults["gae_lambda"].default
            if algorithm == "PPO":
                assert model.batch_size == defaults["batch_size"].default
                assert model.n_epochs == defaults["n_epochs"].default
    finally:
        env.close()
