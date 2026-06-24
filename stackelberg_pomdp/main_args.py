import argparse

def run(config_dict):
    try:
        from .run_setups import train_run
    except ImportError:
        from run_setups import train_run
    train_run(config_dict)


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="StackPOMDP")

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="The random number generator seed",
    )

    parser.add_argument(
        "--training_seed",
        type=int,
        default=None,
        help="The training seed, defaults to the value of --seed if not specified",
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=10000000,
        help="Number of training steps"
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        default="PPO",
        help="DeepRL algorithm used for leader. Options: 'A2C', 'PPO'"
    )

    parser.add_argument(
        "--tot_num_reward_episodes",
        type=int,
        default=1,
        help="Number of reward sub_episodes in Stackelberg POMDP"
    )

    parser.add_argument(
        "--tot_num_eq_episodes",
        type=int,
        default=100,
        help="Number of equilibrium sub_episodes in Stackelberg POMDP"
    )

    parser.add_argument(
        "--experiment_type",
        type=str,
        default="normal_form:game_1:False",
        help="Type of experiment to run. "
             "For normal_form, format is 'normal_form:game_name:randomized' where game_name is either 'game_1' or 'game_2' and randomized is either 'True' or 'False'. "
             "For simple_allocation, format is 'simple_allocation:num_followers_messages' where num_followers_messages is an integer. "
             "For mspm, format is 'mspm:mspm_setting:num_followers_types:num_followers_messages' where mspm_setting is either 'PI' or 'MSGSpace', and num_followers_types and num_followers_messages are integers. "
             "Other options are 'matrix_design'"
    )

    parser.add_argument(
        "--critic_obs",
        type=str,
        default="full",
        help="Determines which additional info is given to critic network. Options: none (critic has same observation as actor), full (critic observes POMDP states)"
    )

    parser.add_argument(
        "--fix_episode_actions",
        type=str_to_bool,
        default=True,
        help="If true, keep observation-action mapping so that leader policy behaves deterministically during each StackPOMDP episode"
    )

    parser.add_argument(
        "--followers_algorithm",
        type=str,
        default="MW",
        help="Determines the followers' learning algorithm. Options: MW, Qlearning"
    )

    parser.add_argument(
        "--response_phase_prob",
        type=float,
        default=1.0,
        help="Probability that each step in response phase is included in the RL buffer"
    )

    parser.add_argument(
        "--learning_method",
        type=str,
        default='RL:Standard',
        help="Options: RL:Standard, RL:StopOnThreshold, PolicyEnumeration:n"
    )

    parser.add_argument(
        "--use_wandb",
        type=bool,
        default=False,
        help="Use Weights & Biases for logging. Set to True to turn on."
    )

    # Bertrand competition arguments
    parser.add_argument(
        "--platform_intervention",
        type=str,
        default="learn_threshold",
        help="Buy-box intervention type. Options: 'no_intervene', 'pdp', 'dpdp', 'learn_threshold'"
    )

    parser.add_argument(
        "--platform_observation_space",
        type=str,
        default="price_profile",
        help="Leader observation type. Options: 'price_profile', 'no_observation'"
    )

    parser.add_argument(
        "--price_grid_length",
        type=int,
        default=15,
        help="Number of discrete prices in the Bertrand price grid"
    )

    parser.add_argument(
        "--price_min",
        "--grid_lower_bound",
        dest="price_min",
        type=float,
        default=1.05,
        help="Lower endpoint of the Bertrand price grid. --grid_lower_bound is a legacy alias."
    )

    parser.add_argument(
        "--price_max",
        "--grid_upper_bound",
        dest="price_max",
        type=float,
        default=1.7,
        help="Upper endpoint of the Bertrand price grid. --grid_upper_bound is a legacy alias."
    )

    parser.add_argument(
        "--marginal_cost",
        type=float,
        default=1.0,
        help="Marginal cost for pricing agents"
    )

    parser.add_argument(
        "--num_pricing_agents",
        type=int,
        default=2,
        help="Number of pricing agents in Bertrand competition"
    )

    parser.add_argument(
        "--q_restart_rate",
        type=float,
        default=-1,
        help="Exploration restart rate. -1 = restart each episode. >0 = expected restarts per episode"
    )

    parser.add_argument(
        "--cost_perturbation",
        action="store_true",
        default=False,
        help="Enable cost perturbation across episodes"
    )

    parser.add_argument(
        "--follower_alpha",
        type=float,
        default=0.15,
        help="Q-learning rate for follower agents"
    )

    parser.add_argument(
        "--follower_beta",
        type=float,
        default=4e-5,
        help="Exploration decay rate for follower agents"
    )

    parser.add_argument(
        "--q_tables_path",
        type=str,
        default=None,
        help="Path to pre-converged Q-tables pkl file for warm initialization"
    )

    parser.add_argument(
        "--intervention_lambda",
        type=float,
        default=0.0,
        help="Weight on buy-box exclusion penalty. 0 = no penalty (pure CS maximization)."
    )

    parser.add_argument(
        "--leader_k",
        type=int,
        default=1,
        help="Number of previous follower action profiles shown to a reactive leader."
    )

    parser.add_argument(
        "--sort_obs",
        action="store_true",
        default=False,
        help="Sort follower action observations before exposing them to the leader."
    )

    parser.add_argument(
        "--ent_coef",
        type=float,
        default=0.01,
        help="Entropy coefficient for leader RL. Nonzero entropy is important for StackPOMDP exploration."
    )

    parser.add_argument(
        "--ppo_episodes_per_batch",
        type=int,
        default=16,
        help="Number of complete StackPOMDP episodes per PPO rollout batch."
    )

    parser.add_argument(
        "--ppo_n_epochs",
        type=int,
        default=4,
        help="Number of PPO optimization epochs per batch."
    )

    parser.add_argument(
        "--warm_start_q",
        action="store_true",
        default=False,
        help="Preserve Q-tables across StackPOMDP episodes (warm-start)"
    )

    args = parser.parse_args()

    if args.training_seed is None:
        args.training_seed = args.seed

    config_dict = vars(args)

    run(config_dict)
