import numpy as np
from copy import deepcopy
from itertools import product
from stable_baselines3.common.logger import HumanOutputFormat
import sys

def get_all_wrappers(env):
    """Returns all the wrappers of an environment up until it hits the Game class (not including the Game class).

    Args:
        env(MultiAgentAtariEnvWrapper): the environment for which the wrappers needs to be retrieved
    """
    from stackelberg_pomdp.gym_envs.envs.custom_envs import RLSupervisorQFollowersWrapper, RLSupervisorMWFollowersWrapper
    from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

    currentenv = env
    list_of_wrappers = [currentenv]
    while not ((type(currentenv) == RLSupervisorQFollowersWrapper) or (type(currentenv) == RLSupervisorMWFollowersWrapper)):
        if type(currentenv) == DummyVecEnv:
            currentenv = currentenv.envs[0]
        else:
            currentenv = currentenv.env
        list_of_wrappers.append(currentenv)

    return list_of_wrappers

def weights_to_action(weights, num_actions):
    probs = np.exp(weights)/sum(np.exp(weights))
    return np.random.choice([i for i in range(num_actions)], 1, p=probs)[0]

def weights_to_probs(weights, randomization_type="linear"):

    if randomization_type=="linear":
        if sum(weights)==0:
            return np.ones(len(weights))/sum(np.ones(len(weights)))
        else:
            return weights / sum(weights)

    elif randomization_type=="logit":
        return np.exp(weights)/sum(np.exp(weights))

    else:
        raise ValueError("Randomization type not supported")

def run_episode(env, policy):
    obs = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = policy.get_action(obs)
        full_obs, reward, done, _ = env.step(action)
        obs = full_obs
        total_reward += reward
    return total_reward

def check_for_profitable_deviations(env, leader_policy, followers_strategy):
        def compute_expected_payoff(policy, strategy_object, follower, observation):
            # We are assuming uniform distributions over types

            other_followers = [f for f in env.followers_list if f != follower]
            other_followers_spaces = [env.followers_observation_space[f] for f in other_followers]

            original_payoff = 0.0
            num_configurations = 1
            for space in other_followers_spaces:
                num_configurations *= len(space)

            for observations in product(*other_followers_spaces):
                # Build observation and action dictionaries
                observations_dict = {f: obs for f, obs in zip(other_followers, observations)}
                actions_dict = {f: strategy_object[f][obs] for f, obs in zip(other_followers, observations)}

                # Add the current follower's observation and action
                observations_dict[follower] = observation
                actions_dict[follower] = strategy_object[follower][observation]

                # Run episode with this configuration
                info_episode = env.run_episode(policy, observations_dict, actions_dict)
                config_payoff = info_episode['utilities'][follower]
                original_payoff += config_payoff # Accumulate payoff for the follower

            return original_payoff / num_configurations


        max_deviation = 0
        for follower in env.followers_list:
            current_strategy = followers_strategy[follower]
            for observation in env.followers_observation_space[follower]:

                original_payoff = compute_expected_payoff(leader_policy, followers_strategy, follower, observation)

                for action in env.followers_action_space[follower]:
                    if action != current_strategy[observation]:  # Skip current action

                        # Create a temporary modified strategy
                        modified_strategy = deepcopy(followers_strategy)
                        modified_strategy[follower][observation] = action
                        alt_payoff = compute_expected_payoff(leader_policy, modified_strategy, follower, observation)
                        if alt_payoff > original_payoff:
                            deviation = alt_payoff - original_payoff
                            if deviation > max_deviation:
                                max_deviation = deviation

        return max_deviation  # Return max deviation

def compute_welfare_loss(env, leader_policy, followers_strategy):
    welfare_loss = 0
    total_configurations = 1

    for observations in product(*[env.followers_observation_space[f] for f in env.followers_list]):
        observations_dict = {f: obs for f, obs in zip(env.followers_list, observations)}
        actions_dict = {f: followers_strategy[f][obs] for f, obs in zip(env.followers_list, observations)}
        info_episode = env.run_episode(leader_policy, observations_dict, actions_dict)
        welfare_loss += info_episode['utilities'][env.game.leader]
        total_configurations += 1

    return welfare_loss / total_configurations


class TemporaryMethod:
    def __init__(self, obj, method_name, new_method):
        self.obj = obj
        self.method_name = method_name
        self.new_method = new_method
        self.old_method = None

    def __enter__(self):
        self.old_method = getattr(self.obj, self.method_name)
        setattr(self.obj, self.method_name, self.new_method)

    def __exit__(self, exc_type, exc_val, exc_tb):
        setattr(self.obj, self.method_name, self.old_method)