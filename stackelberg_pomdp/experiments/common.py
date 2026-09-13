import argparse

from stackelberg_pomdp.wrappers.core import MWFollowersWrapper
from stackelberg_pomdp.training_defaults import resolve_common_optimizer_defaults

DEFAULT_MECHANISM_MW_CYCLES = 33


def mw_update_period_from_config(config):
    """Follower games in one complete MW joint-action sweep."""
    parts = config["experiment_type"].split(":")
    if parts[0] == "simple_allocation":
        return int(parts[1])
    if parts[0] == "mspm":
        return int(parts[3]) ** 2
    if parts[0] == "matrix_design":
        return 4
    if parts[0] == "normal_form":
        from stackelberg_pomdp.games import get_normal_form_game
        game = get_normal_form_game(parts[1])
        period = 1
        for follower in game.followers_list:
            period *= game.action_space(follower)
        return period
    return None


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
    parser.add_argument("--learning_rate", type=float,
                        help="Default: the selected SB3 algorithm's learning rate.")
    parser.add_argument("--critic_obs", type=str, default="full", choices=("none", "flag", "full"))
    parser.add_argument("--fix_episode_actions", type=str_to_bool, default=True)
    parser.add_argument("--followers_algorithm", type=str, default="MW", choices=("MW", "Qlearning"))
    parser.add_argument("--mw_epsilon", type=float, default=MWFollowersWrapper.DEFAULT_EPS)
    parser.add_argument("--mw_reset_weights_each_episode", type=str_to_bool, default=True)
    parser.add_argument(
        "--mw_fixed_seed",
        type=int,
        default=None,
        help=(
            "If set, restart MW's private-type sampling from this seed at "
            "every StackPOMDP episode. Joint messages are enumerated exactly."
        ),
    )
    parser.add_argument(
        "--pomdp_mode",
        type=str,
        default="stackelberg",
        choices=("stackelberg", "hidden_queries"),
    )
    parser.add_argument("--learning_method", type=str, default="RL:Standard")
    parser.add_argument("--ent_coef", type=float, default=0.01,
                        help="Commitment-exploration exception; see replication/PARAMETERS.md.")
    parser.add_argument(
        "--ppo_rollout_geometry",
        choices=("complete_episodes", "historical_ratio_scaled"),
        default="complete_episodes",
        help=(
            "PPO rollout sizing rule. 'complete_episodes' is the maintained "
            "default, rounding SB3's rollout budget to complete episodes. "
            "'historical_ratio_scaled' is an explicit legacy formula; "
            "it does not recover historical optimizer settings."
        ),
    )
    parser.add_argument("--ppo_episodes_per_batch", type=int,
                        help="Override automatic episode count derived from SB3's rollout budget.")
    parser.add_argument("--ppo_batch_size", type=int, default=None)
    parser.add_argument("--ppo_n_epochs", type=int)
    parser.add_argument("--ppo_log_interval", type=int, default=100)
    parser.add_argument("--spm_exact_eval_freq", type=int, default=10000)
    parser.add_argument("--progress_freq", type=int, default=10000)
    parser.add_argument("--reward_print_freq", type=int, default=1)
    parser.add_argument("--eval_freq", type=int, default=10000)
    parser.add_argument("--eval_episodes", type=int, default=1)
    parser.add_argument("--eval_reward_episodes", type=int, default=100)
    parser.add_argument("--response_diagnostic_freq", type=int, default=0)
    parser.add_argument(
        "--reward_trace_targets",
        type=str,
        default="",
        help="Comma-separated reward-phase averages to trace once each (for example: 0,-0.08).",
    )
    parser.add_argument("--reward_trace_tol", type=float, default=1e-6)
    parser.add_argument("--response_bcce_threshold", type=float, default=None)
    parser.add_argument("--response_bcce_min_records", type=int, default=1)
    parser.add_argument("--response_bcce_check_freq", type=int, default=1)
    parser.add_argument(
        "--response_bcce_max_extra_updates",
        type=int,
        default=10000,
        help=(
            "Fail loudly if certified MW needs more than this many complete "
            "updates beyond the fixed response prefix."
        ),
    )
    parser.add_argument(
        "--response_bcce_failure_reward",
        type=float,
        default=None,
        help=(
            "Deprecated. Certified MW no longer uses an artificial failure reward."
        ),
    )
    parser.add_argument("--use_wandb", action="store_true", default=False)


def add_response_phase_args(parser, default_response_episodes, default_reward_episodes,
                            default_response_cycles=None):
    response = parser.add_mutually_exclusive_group()
    response.add_argument(
        "--tot_num_response_episodes",
        type=int,
        help="Follower-game budget; MW rounds down to complete joint-action sweeps.",
    )
    response.add_argument(
        "--mw_response_cycles", type=int,
        help="Number of complete MW updates in the fixed response prefix.",
    )
    parser.set_defaults(_default_response_episodes=default_response_episodes,
                        _default_response_cycles=default_response_cycles)
    parser.add_argument("--tot_num_reward_episodes", type=int, default=default_reward_episodes)


def finalized_config(args, experiment_type):
    config = vars(args).copy()
    config["experiment_type"] = experiment_type
    config["training_seed"] = config["seed"]
    config["mw_action_update"] = "exact_expectation"
    default_games = config.pop("_default_response_episodes", None)
    default_cycles = config.pop("_default_response_cycles", None)
    if config.get("tot_num_response_episodes") is None and config.get("mw_response_cycles") is None:
        if config.get("followers_algorithm") == "MW" and default_cycles is not None:
            config["mw_response_cycles"] = default_cycles
        elif default_games is not None:
            config["tot_num_response_episodes"] = default_games
    cycles = config.get("mw_response_cycles")
    if cycles is not None:
        if config.get("followers_algorithm") != "MW" or cycles < 1:
            raise ValueError("mw_response_cycles requires MW followers and a positive cycle count")
        period = mw_update_period_from_config(config)
        if period is None:
            raise ValueError(f"MW cycle counts are not supported for {experiment_type}")
        config["tot_num_response_episodes"] = cycles * period
    return resolve_common_optimizer_defaults(config)


def run_training(config):
    from stackelberg_pomdp.run_setups import train_run

    train_run(config)
