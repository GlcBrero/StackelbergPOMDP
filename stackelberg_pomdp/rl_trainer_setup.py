from stackelberg_pomdp.gym_envs.envs.base_envs import BertrandCompetitionEnv
from stackelberg_pomdp.gym_envs.envs.wrappers import StackPOMDPWrapper
from stackelberg_pomdp.baselines_utils import CustomPolicy, CustomA2C, CustomPPO
from stackelberg_pomdp.utils import get_all_wrappers


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
        # A fixed SB3 rollout cannot exactly match a variable-length episode.
        # Using the maximum episode length guarantees that every rollout
        # contains at least one completed episode and therefore observed reward.
        ppo_episodes_per_batch = config_dict.get('ppo_episodes_per_batch', 16)
        ppo_n_epochs = config_dict.get('ppo_n_epochs', 4)
        ppo_n_steps = max_episode_transitions * ppo_episodes_per_batch
        ppo_batch_size = config_dict.get('ppo_batch_size') or max_episode_transitions
        m = CustomPPO(env=env, policy=CustomPolicy, gamma=1, learning_rate=learning_rate, seed=seed, n_steps=ppo_n_steps,
                ent_coef=ent_coef, batch_size=ppo_batch_size, n_epochs=ppo_n_epochs,
                policy_kwargs={
                    "cutoff_entry": cutoff_entry,
                    "actor_obs_keys": actor_obs_keys,
                },
                tensorboard_log=tensorboard_folder)
        m.min_completed_episodes_per_rollout = ppo_episodes_per_batch
        print(
            f"[ppo_geometry] max_episode_transitions={max_episode_transitions} "
            f"guaranteed_completed_episodes_per_rollout>={ppo_episodes_per_batch} "
            f"n_steps={ppo_n_steps} batch_size={ppo_batch_size}",
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
