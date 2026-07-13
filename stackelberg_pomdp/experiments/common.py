import argparse

from stackelberg_pomdp.gym_envs.envs.wrappers import MWFollowersWrapper


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def add_common_training_args(parser, default_algorithm="PPO", default_max_steps=10000000):
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=default_max_steps)
    parser.add_argument("--algorithm", type=str, default=default_algorithm, choices=("PPO", "A2C"))
    parser.add_argument("--learning_rate", type=float, default=7e-4)
    parser.add_argument("--critic_obs", type=str, default="full", choices=("none", "flag", "full"))
    parser.add_argument("--fix_episode_actions", type=str_to_bool, default=True)
    parser.add_argument("--followers_algorithm", type=str, default="MW", choices=("MW", "Qlearning", "RoundRobin"))
    parser.add_argument("--mw_epsilon", type=float, default=MWFollowersWrapper.DEFAULT_EPS)
    parser.add_argument("--mw_reset_weights_each_episode", type=str_to_bool, default=True)
    parser.add_argument("--align_mw_response_phase", type=str_to_bool, default=True)
    parser.add_argument(
        "--pomdp_mode",
        type=str,
        default="stackelberg",
        choices=("stackelberg", "hidden_queries", "reward_during_response"),
    )
    parser.add_argument("--learning_method", type=str, default="RL:Standard")
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--ppo_episodes_per_batch", type=int, default=16)
    parser.add_argument("--ppo_batch_size", type=int, default=None)
    parser.add_argument("--ppo_n_epochs", type=int, default=4)
    parser.add_argument("--ppo_log_interval", type=int, default=100)
    parser.add_argument("--spm_exact_eval_freq", type=int, default=10000)
    parser.add_argument("--progress_freq", type=int, default=10000)
    parser.add_argument("--reward_print_freq", type=int, default=1)
    parser.add_argument("--eval_freq", type=int, default=10000)
    parser.add_argument("--eval_reward_episodes", type=int, default=100)
    parser.add_argument("--response_diagnostic_freq", type=int, default=0)
    parser.add_argument("--response_bcce_threshold", type=float, default=None)
    parser.add_argument("--response_bcce_min_records", type=int, default=1)
    parser.add_argument("--response_bcce_check_freq", type=int, default=1)
    parser.add_argument("--use_wandb", action="store_true", default=False)


def add_response_phase_args(parser, default_response_episodes, default_reward_episodes):
    parser.add_argument(
        "--tot_num_response_episodes",
        type=int,
        default=default_response_episodes,
        help="Number of follower-response sub-episodes.",
    )
    parser.add_argument("--tot_num_reward_episodes", type=int, default=default_reward_episodes)


def finalized_config(args, experiment_type):
    config = vars(args)
    config["experiment_type"] = experiment_type
    config["training_seed"] = config["seed"]
    return config


def run_training(config):
    from stackelberg_pomdp.run_setups import train_run

    train_run(config)
