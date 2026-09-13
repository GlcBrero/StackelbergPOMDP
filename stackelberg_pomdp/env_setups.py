from stackelberg_pomdp.games import (
    MatrixDesignGame,
    SimpleAllocationGame,
    get_mspm_setting,
    get_normal_form_game,
)
from stackelberg_pomdp.envs.bertrand import BertrandCompetitionEnv
from stackelberg_pomdp.envs.matrix_design import BaseEnvMatrixDesignGame
from stackelberg_pomdp.envs.normal_form import BaseEnvSimpleMatrixGame
from stackelberg_pomdp.envs.simple_allocation import BaseSimpleAllocation
from stackelberg_pomdp.envs.spm import BaseMessageSPM, BaseSPM
from stackelberg_pomdp.wrappers.core import (
    ExpectedResponseRewardWrapper,
    LoggingWrapper,
    MWFollowersWrapper,
    OpennessEvaluationWrapper,
    QLearningFollowersWrapper,
    ReactiveLeaderWrapper,
    StackPOMDPWrapper,
    StationaryCycleRewardWrapper,
)


def _requested_response_episodes(config_dict, default=50000):
    requested = config_dict.get('tot_num_response_episodes')
    return default if requested is None else requested


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
        leader_observation_space=config_dict.get('platform_observation_space', 'no_observation'),
        leader_k=config_dict.get('leader_k', 1),
        sort_leader_observation=config_dict.get('sort_obs', False),
        seed=seed,
        logger=log,
    )

    return wrap_env(
        env,
        config_dict,
        use_reactive_leader=config_dict.get('platform_observation_space', 'no_observation') == 'price_profile',
        use_cycle_reward=True,
        use_openness_evaluation=True,
        use_logging=True,
    )


def get_matrix_design_env(config_dict):
    seed = config_dict["seed"]
    log = config_dict["logger"]

    game = MatrixDesignGame()

    env = BaseEnvMatrixDesignGame(
        game,
        logger=log,
        seed=seed,
    )

    return wrap_env(env, config_dict, use_reactive_leader=True, use_logging=True)


def get_mspm_env(config_dict):

    log = config_dict["logger"]
    seed = config_dict["seed"]

    setting = get_mspm_setting(
        config_dict["setting"],
        config_dict["num_types"],
        config_dict["num_messages"],
    )

    env = BaseMessageSPM(
        game=setting,
        logger=log,
        seed=seed,
    )
    env.include_zero_weight_reward_profiles = config_dict.get(
        'include_zero_weight_reward_profiles',
        True,
    )

    return wrap_env(
        env,
        config_dict,
        use_reactive_leader=True,
    )


def get_spm_env(config_dict):
    setting = get_mspm_setting(
        config_dict["setting"],
        config_dict["num_types"],
        config_dict.get("num_messages", 1),
    )
    return BaseSPM(
        game=setting,
        logger=config_dict["logger"],
        seed=config_dict["seed"],
        discrete_prices=config_dict.get("discrete_prices", False),
    )


def get_simple_allocation_env(config_dict):

    log = config_dict["logger"]
    seed = config_dict["seed"]

    game = SimpleAllocationGame(num_messages=config_dict["num_messages"])

    env = BaseSimpleAllocation(
        game=game,
        logger=log,
        seed=seed,
    )
    env.include_zero_weight_reward_profiles = config_dict.get(
        'include_zero_weight_reward_profiles',
        True,
    )

    return wrap_env(env, config_dict, use_reactive_leader=True)


def get_standard_matrix_env(config_dict):

    log = config_dict["logger"]
    seed = config_dict["seed"]

    game_type = config_dict["experiment_type"].split(":")[1]
    randomized = config_dict["experiment_type"].split(":")[2] == 'True'

    game = get_normal_form_game(game_type)

    env = BaseEnvSimpleMatrixGame(
        game,
        logger=log,
        seed=seed,
        randomized=randomized,
    )

    return wrap_env(env, config_dict)


def wrap_env(
        env,
        config_dict,
        *,
        use_reactive_leader=False,
        use_cycle_reward=False,
        use_openness_evaluation=False,
        use_logging=False,
):
    followers_alg = config_dict.get('followers_algorithm', 'Qlearning')
    tot_num_response_episodes = _requested_response_episodes(config_dict)
    if followers_alg == "MW":
        if config_dict.get('align_mw_response_phase') is False:
            raise ValueError("MW always uses complete updates; remove align_mw_response_phase=false")
        fixed_seed = config_dict.get('mw_fixed_seed')
        if (
                config_dict.get('response_bcce_threshold') is not None
                and fixed_seed is None
        ):
            fixed_seed = 0
            config_dict['mw_fixed_seed'] = fixed_seed
        env = MWFollowersWrapper(
            env,
            epsilon=config_dict.get('mw_epsilon', MWFollowersWrapper.DEFAULT_EPS),
            reset_weights_each_episode=config_dict.get('mw_reset_weights_each_episode', True),
            fixed_seed=fixed_seed,
            response_bcce_threshold=config_dict.get('response_bcce_threshold'),
            response_bcce_min_records=config_dict.get('response_bcce_min_records', 1),
            response_bcce_check_freq=config_dict.get('response_bcce_check_freq', 1),
            response_bcce_max_extra_updates=config_dict.get(
                'response_bcce_max_extra_updates',
                10000,
            ),
        )
        cycles = config_dict.get('mw_response_cycles')
        if cycles is not None:
            if int(cycles) != cycles or cycles < 1:
                raise ValueError("mw_response_cycles must be a positive integer")
            cycle_games = int(cycles) * env.response.response_games_per_update
            requested = config_dict.get('tot_num_response_episodes')
            if requested is not None and requested != cycle_games:
                raise ValueError("mw_response_cycles conflicts with tot_num_response_episodes")
            tot_num_response_episodes = cycle_games
            config_dict['tot_num_response_episodes'] = cycle_games
    elif followers_alg == "Qlearning":
        if config_dict.get('mw_response_cycles') is not None:
            raise ValueError("mw_response_cycles requires MW followers")
        env = QLearningFollowersWrapper(
            env,
            alpha=config_dict.get('follower_alpha', 0.15),
            beta=config_dict.get('follower_beta', 4e-5),
            warm_start_q=config_dict.get('warm_start_q', False),
            q_tables_path=config_dict.get('q_tables_path'),
        )

    else:
        raise ValueError(f"Unsupported followers_algorithm: {followers_alg}")

    if use_reactive_leader:
        env = ReactiveLeaderWrapper(env)

    pomdp_mode = config_dict.get('pomdp_mode', 'stackelberg')
    if pomdp_mode in ("stackelberg", "hidden_queries"):
        env = StackPOMDPWrapper(
            env,
            tot_num_response_episodes=tot_num_response_episodes,
            tot_num_reward_episodes=config_dict.get('tot_num_reward_episodes', 30),
            critic_obs=config_dict.get('critic_obs', 'full'),
            response_variant=pomdp_mode,
        )
    else:
        raise ValueError(f"Unsupported pomdp_mode: {pomdp_mode}")

    config_dict['effective_tot_num_response_episodes'] = env.tot_num_response_episodes

    if config_dict.get('response_bcce_threshold') is not None and followers_alg != "MW":
        raise ValueError("Certified response stopping is supported only for MW followers.")
    if config_dict.get('response_bcce_failure_reward') is not None:
        raise ValueError(
            "response_bcce_failure_reward is obsolete: certified MW now extends "
            "the response phase until it finds a valid response."
        )
    if followers_alg == "MW":
        env = ExpectedResponseRewardWrapper(env)

    if use_cycle_reward:
        env = StationaryCycleRewardWrapper(env)

    intervention_lambda = config_dict.get('intervention_lambda', 0.0)
    if use_openness_evaluation and intervention_lambda > 0:
        env = OpennessEvaluationWrapper(
            env,
            intervention_lambda=intervention_lambda,
            m=config_dict.get('price_grid_length', 4),
        )

    if use_logging:
        env = LoggingWrapper(
            env,
            logger=config_dict.get('logger'),
            log_regular_done=use_logging,
        )

    return env
