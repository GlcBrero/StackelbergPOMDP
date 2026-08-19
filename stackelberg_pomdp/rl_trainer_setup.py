from stackelberg_pomdp.algorithms.on_policy import CustomA2C, CustomPPO
from stackelberg_pomdp.envs.base import BertrandCompetitionEnv
from stackelberg_pomdp.policies.generic import CustomPolicy
from stackelberg_pomdp.wrappers.core import StackPOMDPWrapper
from stackelberg_pomdp.utils import get_all_wrappers


PPO_ROLLOUT_GEOMETRIES = ("complete_episodes", "historical_ratio_scaled")
HISTORICAL_PPO_SCALE = 2048
HISTORICAL_RESPONSE_RATIO_CAP = 100


def _historical_response_phase_probability(config_dict):
    """Return the legacy probability that response queries entered PPO.

    The final paper cohorts used only the two endpoints of the old
    ``response_phase_prob`` argument: StackPOMDP stored response queries
    (probability one), whereas Basic POMDP hid all of them (probability zero).
    """
    return 0.0 if config_dict.get("pomdp_mode") == "hidden_queries" else 1.0


def ppo_rollout_geometry(config_dict, max_episode_transitions):
    """Resolve PPO rollout and minibatch sizes for one training run.

    ``complete_episodes`` is the scientifically cleaner maintained default.
    ``historical_ratio_scaled`` exactly preserves the batching formula used by
    the plotted Simple Allocation and Matrix Design cohorts.  The legacy
    ``ppo_episodes_per_batch`` name is retained because it was the multiplier
    in those runs, although the resulting rollout did not contain exactly that
    many complete outer episodes.
    """
    mode = config_dict.get("ppo_rollout_geometry", "complete_episodes")
    if mode not in PPO_ROLLOUT_GEOMETRIES:
        raise ValueError(
            f"Unknown PPO rollout geometry {mode!r}; "
            f"expected one of {PPO_ROLLOUT_GEOMETRIES}."
        )

    rollout_multiplier = int(config_dict.get("ppo_episodes_per_batch", 16))
    if rollout_multiplier <= 0:
        raise ValueError("ppo_episodes_per_batch must be positive")

    if mode == "complete_episodes":
        block_steps = int(max_episode_transitions)
        minimum_completed_episodes = rollout_multiplier
    else:
        response_games = int(
            config_dict.get(
                "effective_tot_num_response_episodes",
                config_dict["tot_num_response_episodes"],
            )
        )
        reward_games = int(config_dict["tot_num_reward_episodes"])
        if response_games < 0:
            raise ValueError("tot_num_response_episodes must be non-negative")
        if reward_games <= 0:
            raise ValueError("tot_num_reward_episodes must be positive")
        response_probability = _historical_response_phase_probability(config_dict)
        ratio_steps = int(
            1 + response_games * response_probability / reward_games
        )
        block_steps = (
            min(ratio_steps, HISTORICAL_RESPONSE_RATIO_CAP)
            * HISTORICAL_PPO_SCALE
        )
        # Historical collection stopped at the fixed stored-transition budget;
        # it did not enforce a minimum number of completed outer episodes.
        minimum_completed_episodes = 0

    batch_size = int(config_dict.get("ppo_batch_size") or block_steps)
    if batch_size <= 0:
        raise ValueError("ppo_batch_size must be positive")
    return {
        "mode": mode,
        "block_steps": block_steps,
        "n_steps": block_steps * rollout_multiplier,
        "batch_size": batch_size,
        "minimum_completed_episodes": minimum_completed_episodes,
    }


def _is_critic_key(key):
    return key.split(":")[0] == "critic"


def get_observation_split(env):
    from stable_baselines3.common.preprocessing import get_flattened_obs_dim

    critic_feature_dim = 0
    actor_obs_keys = []
    seen_critic = False
    for key in env.observation_space.spaces.keys():
        if _is_critic_key(key):
            seen_critic = True
            critic_feature_dim += get_flattened_obs_dim(env.observation_space[key])
        else:
            if seen_critic:
                raise ValueError(
                    f"Actor-visible observation key {key!r} appears after critic-only keys. "
                    "Use a prefix that sorts before 'critic:', such as 'base:'."
                )
            actor_obs_keys.append(key)

    return critic_feature_dim, tuple(actor_obs_keys)


def _stack_pomdp_wrapper(env):
    for wrapper in get_all_wrappers(env):
        if isinstance(wrapper, StackPOMDPWrapper):
            return wrapper
    return None


def get_custom_training_algorithm(config_dict, env, tensorboard_folder=None):

    algorithm = config_dict['algorithm']
    seed = config_dict['training_seed']
    learning_rate = config_dict.get('learning_rate', 7e-4)

    # Compute the actor/critic feature split from the final wrapped observation
    # space. Actor sees every non-critic key; critic additionally sees critic:*.
    cutoff_entry, actor_obs_keys = get_observation_split(env)

    stack_env = _stack_pomdp_wrapper(env)
    if stack_env is not None:
        max_episode_transitions = stack_env.max_episode_transitions()
    elif hasattr(env.unwrapped, "max_episode_transitions"):
        max_episode_transitions = env.unwrapped.max_episode_transitions()
    else:
        max_episode_transitions = config_dict['tot_num_reward_episodes']

    if isinstance(env.unwrapped, BertrandCompetitionEnv) and config_dict.get('intervention_lambda', 0) > 0:
        m = config_dict.get('price_grid_length', 4)
        max_episode_transitions += m * m

    ent_coef = config_dict.get('ent_coef', 0.01)

    if algorithm == "PPO":
        geometry = ppo_rollout_geometry(config_dict, max_episode_transitions)
        ppo_n_epochs = config_dict.get('ppo_n_epochs', 4)
        m = CustomPPO(env=env, policy=CustomPolicy, gamma=1, learning_rate=learning_rate, seed=seed, n_steps=geometry["n_steps"],
                ent_coef=ent_coef, batch_size=geometry["batch_size"], n_epochs=ppo_n_epochs,
                policy_kwargs={
                    "cutoff_entry": cutoff_entry,
                    "actor_obs_keys": actor_obs_keys,
                },
                tensorboard_log=tensorboard_folder)
        m.min_completed_episodes_per_rollout = geometry["minimum_completed_episodes"]
        print(
            f"[ppo_geometry] mode={geometry['mode']} "
            f"max_episode_transitions={max_episode_transitions} "
            f"block_steps={geometry['block_steps']} "
            f"required_completed_episodes>={geometry['minimum_completed_episodes']} "
            f"n_steps={geometry['n_steps']} batch_size={geometry['batch_size']}",
            flush=True,
        )

    elif algorithm == "A2C":
        m = CustomA2C(env=env, policy=CustomPolicy, gamma=1, learning_rate=learning_rate, seed=seed, n_steps=max_episode_transitions,
                 ent_coef=ent_coef,
                 policy_kwargs={
                     "cutoff_entry": cutoff_entry,
                     "actor_obs_keys": actor_obs_keys,
                 },
                 tensorboard_log=tensorboard_folder)

    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    return m
