"""The two appendix recipes; figure names are stable public identifiers."""

FIGURE_EXPERIMENTS = {
    "fig_memory_pg": "phase_observability",
    "fig_bots_leaderreward": "response_reward",
}

LEADER_DEFAULTS = {
    "phase_observability": {"algorithm": "PG", "learning_rate": 0.008, "timesteps": 200_000, "seeds": 10},
    "response_reward": {"algorithm": "SIMPLEQ", "learning_rate": 0.1, "timesteps": 22_000, "seeds": 10},
}

# These are the learning-rate/treatment curves actually plotted in the paper.
# The phase figure has four visible-phase curves and two complete controls.
APPENDIX_CURVES = {
    "phase_observability": [
        (condition, "prisoners_dilemma", rate)
        for condition, rates in (("visible", (0.004, 0.008, 0.015, 0.03)),
                                 ("hidden", (0.008, 0.015))) for rate in rates
    ],
    "response_reward": [(condition, matrix, 0.1)
                        for matrix in ("coordination_zero_miscoordination",
                                       "coordination_penalized_miscoordination")
                        for condition in ("excluded", "included")],
}


def leader_protocol(args, episode_steps):
    """Record the resolved learner mechanics alongside every run."""
    if args.algorithm == "PG":
        batch = ((100 + episode_steps - 1) // episode_steps) * episode_steps
        return {
            "loss": "negative_mean_log_probability_times_reward_to_go",
            "gamma": 1.0, "baseline": "none", "entropy_coefficient": 0.0,
            "optimizer": "Adam", "adam_epsilon": 1e-8,
            "policy": "bias_free_linear_categorical",
            "batch_steps": batch, "complete_episodes": True,
            "action_cache": "one_action_per_observation_per_outer_episode",
        }
    if args.algorithm == "SIMPLEQ":
        return {
            "loss": "huber_one_step_terminal_masked_td", "gamma": 1.0,
            "optimizer": "Adam", "adam_epsilon": 1e-8,
            "policy": "bias_free_linear_q", "replay_capacity": 50_000,
            "replay_batch_size": 1024, "learning_starts": 100,
            "target_update_steps": 500, "gradient_clip_norm": 40,
            "updates_per_episode": 1, "parameter_noise_initial_std": 1.0,
            "parameter_noise_adaptation": "episode_kl_target_zero_factor_1.01",
            "action_cache": "one_action_per_observation_per_outer_episode",
        }
    return {"algorithm": args.algorithm, "action_cache": "enabled"}
