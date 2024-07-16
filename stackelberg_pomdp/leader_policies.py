import numpy as np
import hashlib
import gym
from stable_baselines3.common.utils import obs_as_tensor

class RandomPolicy:
    def __init__(self, env, seed):
        self.env = env
        self.seed = seed

    def get_action(self, observation):

        observation = observation["base_environment"]

        # Create a hash object
        hash_obj = hashlib.sha256()

        # Feed the state and seed into the hash function
        data = np.append(observation, self.seed)
        hash_obj.update(data.tobytes())

        # Use the hash to seed a random number generator
        hash_int = int(hash_obj.hexdigest(), 16) % (2 ** 32)
        random_state = np.random.RandomState(hash_int)

        # Generate a random action
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            return random_state.randint(self.env.action_space.n)
        elif isinstance(self.env.action_space, gym.spaces.Box):
            return random_state.uniform(self.env.action_space.low, self.env.action_space.high)
        else:
            raise NotImplementedError("This policy is not implemented for this type of action space.")


class BaselinePolicyWrapper:
    def __init__(self, baselines_policy, env):
        self.baselines_policy = baselines_policy
        self.env = env

    def get_action(self, observation):
        obs_full = self.env.observation_space.sample()
        obs_full['base_environment'] = observation['base_environment']
        return self.baselines_policy.predict(obs_full)[0]