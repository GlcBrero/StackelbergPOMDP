from stackelberg_pomdp.gym_envs.envs.custom_envs import *
from games import *
import warnings

def get_bertrand_env(config_dict):

    log = config_dict["logger"]
    seed = config_dict["seed"]

    env = BertrandCompetitionEnv(
        num_agents=config_dict.get('num_pricing_agents', 2),
        c_i=config_dict.get('marginal_cost', 1),
        platform_intervention=config_dict.get('platform_intervention', 'learn_threshold'),
        m=config_dict.get('price_grid_length', 15),
        price_min=config_dict.get('price_min', 1.05),
        price_max=config_dict.get('price_max', 1.7),
        seed=seed,
        logger=log,
    )

    followers_alg = config_dict.get('followers_algorithm', 'Qlearning')
    if followers_alg == 'RoundRobin':
        env = RoundRobinFollowersWrapper(env)
    else:
        env = QLearningFollowersWrapper(
            env,
            alpha=config_dict.get('follower_alpha', 0.15),
            beta=config_dict.get('follower_beta', 4e-5),
            warm_start_q=config_dict.get('warm_start_q', False),
            q_tables_path=config_dict.get('q_tables_path'),
        )

    if config_dict.get('platform_observation_space', 'no_observation') == 'price_profile':
        env = ReactiveLeaderWrapper(env, leader_k=config_dict.get('leader_k', 1),
                                    sort_obs=config_dict.get('sort_obs', False))

    env = StackPOMDPWrapper(
        env,
        tot_num_eq_episodes=config_dict.get('tot_num_eq_episodes', 50000),
        tot_num_reward_episodes=config_dict.get('tot_num_reward_episodes', 30),
        critic_obs=config_dict.get('critic_obs', 'full'),
        response_phase_prob=config_dict.get('response_phase_prob', 1),
    )

    env = StationaryCycleRewardWrapper(env)

    intervention_lambda = config_dict.get('intervention_lambda', 0.0)
    if intervention_lambda > 0:
        env = OpennessEvaluationWrapper(
            env,
            intervention_lambda=intervention_lambda,
            m=config_dict.get('price_grid_length', 4),
        )

    env = LoggingWrapper(env, logger=config_dict.get('logger'))

    return env

def wrap_env(env, config_dict):
    if config_dict["followers_algorithm"] == "MW":
        env = MWFollowersWrapper(env)
    elif config_dict["followers_algorithm"] == "Qlearning":
        env = QLearningFollowersWrapper(env)

    env = StackPOMDPWrapper(
        env,
        tot_num_reward_episodes=config_dict['tot_num_reward_episodes'],
        tot_num_eq_episodes=config_dict['tot_num_eq_episodes'],
        critic_obs=config_dict['critic_obs'],
        response_phase_prob=config_dict['response_phase_prob'],
    )

    if "RL:StopOnThreshold" in config_dict["learning_method"] or "PolicyEnumeration" in config_dict["learning_method"]:
        env = StopOnThresholdWrapper(env)

    return env

def get_standard_matrix_env(config_dict):

    log = config_dict["logger"]
    seed = config_dict["seed"]

    game_type = config_dict["experiment_type"].split(":")[1]
    randomized = config_dict["experiment_type"].split(":")[2] == 'True'

    if game_type == "game_1":
        matrix_game = np.array(
            [
                [[15, 15], [10, 10], [0, 0]],
                [[10, 10], [10, 10], [0, 0]],
                [[0, 0], [0, 0], [30, 30]],
            ]
        )
    elif game_type == "game_2":
        matrix_game = np.array(
            [
                [[20, 15], [0, 0], [0, 0]],
                [[30, 0], [10, 5], [0, 0]],
                [[0, 0], [0, 0], [5, 10]],
            ]
        )
    elif game_type == "game_3":
        matrix_game = np.array([[[1, 1], [3, 0]], [[0, 0], [2, 1]]])
    elif game_type == "game_4":
        matrix_game = np.array([[[3, 2], [1, 3]], [[2, 0], [0, 1]]])

    game = StackelbergMatrixGame(matrix_game)

    env = BaseEnvSimpleMatrixGame(
        game,
        logger=log,
        seed=seed,
        randomized=randomized,
    )

    return wrap_env(env, config_dict)


def get_matrix_design_env(config_dict):
    seed = config_dict["seed"]
    log = config_dict["logger"]

    game = MatrixDesignGame()

    env = BaseEnvMatrixDesignGame(
        game,
        logger=log,
        seed=seed,
    )

    return wrap_env(env, config_dict)



def get_simple_allocation_env(config_dict):

    log = config_dict["logger"]
    seed = config_dict["seed"]

    # Extract the number of messages from the experiment_type argument
    num_messages = int(config_dict["experiment_type"].split(":")[1])

    game = SimpleAllocationGame(num_messages=num_messages)

    env = BaseSimpleAllocation(
        game=game,
        logger=log,
        seed=seed,
    )

    return wrap_env(env, config_dict)



def get_mspm_env(config_dict):

    log = config_dict["logger"]
    seed = config_dict["seed"]

    experiment_details = config_dict["experiment_type"].split(":")
    if len(experiment_details) != 4:
        raise Exception("Experiment details must have exactly 4 components: 'mspm', 'mspm_setting', 'num_followers_types', 'num_followers_messages'")

    mspm_setting = experiment_details[1]
    num_followers_types = int(experiment_details[2])
    num_followers_messages = int(experiment_details[3])

    if mspm_setting == "PI" and num_followers_types != 2:
        warnings.warn("For 'PI' setting, 'num_followers_types' should be 2. Overriding 'num_followers_types' to 2.")
        num_followers_types = 2

    setting_mapping = {
        "PI": lambda: PISetting(num_messages=num_followers_messages),
        "MSGSpace": lambda: MSGSpaceSetting(num_types=num_followers_types, num_messages=num_followers_messages)
    }

    setting_class = setting_mapping.get(mspm_setting)
    if not setting_class:
        raise Exception("SPM setting not recognized! Available options: PI, MSGSpace")

    setting = setting_class()

    env = BaseMessageSPM(
        game=setting,
        logger=log,
        seed=seed,
    )

    return wrap_env(env, config_dict)
