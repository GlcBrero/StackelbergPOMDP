from stable_baselines3.common.policies import register_policy

from stackelberg_pomdp.gym_envs.envs.custom_envs import BaseMessageSPM, BertrandCompetitionEnv
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
        learning_rate = 7e-4 # SB3 A2C default, matches ai_collusion
        decay_rate_value = 0 # Default decay rate

    cutoff_entry = get_cutoff_entry(env)  # Determines which part of the observation is shared between actor and critic

    # For Bertrand, each step = one sub-episode, so n_steps = full episode length
    if isinstance(env.unwrapped, BertrandCompetitionEnv):
        n_steps = config_dict['tot_num_eq_episodes'] + config_dict['tot_num_reward_episodes']
        # Add openness evaluation steps when intervention penalty is active
        if config_dict.get('intervention_lambda', 0) > 0:
            m = config_dict.get('price_grid_length', 4)
            n_steps += m * m
    else:
        # Scale n_steps to match expected reward steps in default POMG
        n_steps = int(1 + config_dict['tot_num_eq_episodes'] * config_dict['response_phase_prob'] / config_dict['tot_num_reward_episodes'])
        n_steps = min(n_steps, 100) # Limit to factor 100 to avoid too long training times

        if algorithm == "PPO":
            n_steps *= 2048  # Default n_steps for PPO
        elif algorithm == "A2C":
            n_steps *= 5  # Default n_steps for A2C

    ent_coef = config_dict.get('ent_coef', 0.01)

    if algorithm == "PPO":
        # PPO needs multi-episode batches for stable learning.
        # batch_size = one episode, n_steps = multiple episodes, n_epochs > 1.
        ppo_episodes_per_batch = config_dict.get('ppo_episodes_per_batch', 16)
        ppo_n_epochs = config_dict.get('ppo_n_epochs', 4)
        ppo_n_steps = n_steps * ppo_episodes_per_batch
        m = CustomPPO(env=env, policy="CustomPolicy", gamma=1, learning_rate=learning_rate, seed=seed, n_steps=ppo_n_steps,
                ent_coef=ent_coef, batch_size=n_steps, n_epochs=ppo_n_epochs,
                policy_kwargs={"cutoff_entry": cutoff_entry, "decay_rate": decay_rate_value},
                tensorboard_log=tensorboard_folder)

    elif algorithm == "A2C":
        m = CustomA2C(env=env, policy="CustomPolicy", gamma=1, learning_rate=learning_rate, seed=seed, n_steps=n_steps,
                 ent_coef=ent_coef,
                 policy_kwargs={"cutoff_entry": cutoff_entry, "decay_rate": decay_rate_value},
                 tensorboard_log=tensorboard_folder)

    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    return m
