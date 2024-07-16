from stable_baselines3.common.policies import register_policy

from stackelberg_pomdp.gym_envs.envs.custom_envs import BaseMessageSPM
from stackelberg_pomdp.baselines_utils import CustomPolicy, CustomA2C, CustomPPO


def get_cutoff_entry(env):
    from stable_baselines3.common.preprocessing import preprocess_obs
    from stable_baselines3.common.utils import obs_as_tensor, get_device
    obs = env.observation_space.sample()
    tensor_obs = obs_as_tensor(obs, get_device())
    cutoff_entry = 0
    for key in obs.keys():
        if key.split(":")[0] == "critic":
            cutoff_entry = cutoff_entry + preprocess_obs(tensor_obs[key], env.observation_space[key]).flatten().shape[0]
    return cutoff_entry

def get_custom_training_algorithm(config_dict, env, tensorboard_folder=None):

    algorithm = config_dict['algorithm']
    seed = config_dict['training_seed']

    register_policy("CustomPolicy", CustomPolicy)

    if isinstance(env.unwrapped, BaseMessageSPM):
        learning_rate = 3e-6
        decay_rate_value = 0
    else:
        learning_rate = 3e-4 # Default learning rate
        decay_rate_value = 0 # Default decay rate

    cutoff_entry = get_cutoff_entry(env)  # Determines which part of the observation is shared between actor and critic

    # Scale n_steps to match expected reward steps in default POMG
    n_steps = int(1 + config_dict['tot_num_eq_episodes'] * config_dict['response_phase_prob'] / config_dict['tot_num_reward_episodes'])
    n_steps = min(n_steps, 100) # Limit to factor 100 to avoid too long training times

    if algorithm == "PPO":
        n_steps *= 2048  # Default n_steps for PPO
        m = CustomPPO(env=env, policy="CustomPolicy", gamma=1, learning_rate=learning_rate, seed=seed, n_steps=n_steps,
                policy_kwargs={"cutoff_entry": cutoff_entry, "decay_rate": decay_rate_value},
                tensorboard_log=tensorboard_folder)

    elif algorithm == "A2C":
        n_steps *= 5  # Default n_steps for A2C
        m = CustomA2C(env=env, policy="CustomPolicy", gamma=1, learning_rate=learning_rate, seed=seed, n_steps=n_steps,
                 policy_kwargs={"cutoff_entry": cutoff_entry, "decay_rate": decay_rate_value},
                 tensorboard_log=tensorboard_folder)

    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    return m