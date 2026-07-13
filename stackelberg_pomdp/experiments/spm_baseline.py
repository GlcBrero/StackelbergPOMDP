import argparse

from stackelberg_pomdp.experiments.common import (
    add_common_training_args,
    finalized_config,
    run_training,
    str_to_bool,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Standard sequential posted-price mechanism PPO baseline."
    )
    parser.add_argument("--setting", type=str, default="MSGSpace", choices=("PI", "MSGSpace"))
    parser.add_argument("--num_types", type=int, required=True)
    parser.add_argument("--discrete_prices", type=str_to_bool, default=True)
    parser.add_argument("--spm_eval_deterministic", type=str_to_bool, default=True)
    parser.add_argument("--spm_eval_action_samples", type=int, default=1)
    add_common_training_args(parser, default_algorithm="PPO")
    parser.set_defaults(
        learning_method="RL:Standard",
        learning_rate=3e-4,
        critic_obs="none",
        ent_coef=0.0,
        ppo_episodes_per_batch=1024,
        ppo_batch_size=64,
        ppo_n_epochs=10,
        tot_num_reward_episodes=1,
        eval_reward_episodes=1,
        response_diagnostic_freq=0,
        response_bcce_threshold=None,
    )
    return parser


def main():
    args = build_parser().parse_args()
    config = finalized_config(args, f"spm:{args.setting}:{args.num_types}")
    run_training(config)


if __name__ == "__main__":
    main()
