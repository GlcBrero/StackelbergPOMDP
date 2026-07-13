import numpy as np
from itertools import product
import gym

def get_all_wrappers(env):
    """Returns all the wrappers of an environment, traversing down to the base env.

    Args:
        env: the environment for which the wrappers needs to be retrieved
    """
    from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

    currentenv = env
    list_of_wrappers = [currentenv]
    while hasattr(currentenv, 'env') or isinstance(currentenv, DummyVecEnv):
        if isinstance(currentenv, DummyVecEnv):
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


def space_values(space):
    """Enumerate values in finite Gym spaces used by follower observations/actions."""
    if isinstance(space, gym.spaces.Discrete):
        return range(space.n)
    if isinstance(space, gym.spaces.MultiDiscrete):
        return product(*[range(n) for n in space.nvec])
    raise TypeError(f"Cannot enumerate non-finite space {space!r}.")


def _empirical_strategy_info(env, leader_policy, followers, types, bids, payoff_cache):
    key = (
        tuple((follower, types[follower]) for follower in followers),
        tuple((follower, bids[follower]) for follower in followers),
    )
    if key not in payoff_cache:
        payoff_cache[key] = env.run_episode(leader_policy, dict(types), dict(bids))
    return payoff_cache[key]


def _empirical_strategy_payoff(env, leader_policy, followers, types, bids, payoff_cache):
    return _empirical_strategy_info(
        env,
        leader_policy,
        followers,
        types,
        bids,
        payoff_cache,
    )["utilities"]


def _type_profile_probability(env, types):
    game = env.unwrapped.game if hasattr(env, "unwrapped") else env.game
    if hasattr(game, "type_profile_probability"):
        return game.type_profile_probability(types)
    probability = 1.0
    for follower in env.followers_list:
        probability *= 1.0 / env.followers_observation_space[follower].n
    return probability


def _conditional_other_type_probability(env, other_followers, other_observations):
    if not other_followers:
        return 1.0
    game = env.unwrapped.game if hasattr(env, "unwrapped") else env.game
    probability = 1.0
    for follower, observation in zip(other_followers, other_observations):
        if hasattr(game, "type_probability"):
            probability *= game.type_probability(follower, observation)
        else:
            probability *= 1.0 / env.followers_observation_space[follower].n
    return probability


def check_empirical_bcce_gap(env, leader_policy, empirical_strategy):
    """Maximum interim coarse-deviation gain for an empirical MW strategy.

    The empirical strategy is a uniform distribution over mixed-strategy
    snapshots. A follower type may deviate ex ante to a fixed action, before
    seeing the sampled action recommendation.
    """
    followers = env.followers_list
    payoff_cache = {}
    max_gap = 0.0

    for follower in followers:
        other_followers = [f for f in followers if f != follower]
        other_type_profiles = list(product(*[
            list(space_values(env.followers_observation_space[f]))
            for f in other_followers
        ]))

        for observation in space_values(env.followers_observation_space[follower]):
            current_payoff = 0.0
            deviation_payoffs = {
                action: 0.0
                for action in space_values(env.followers_action_space[follower])
            }

            strategy_weight = 1.0 / len(empirical_strategy)
            for strategy_snapshot in empirical_strategy:
                for other_observations in other_type_profiles:
                    other_type_probability = _conditional_other_type_probability(
                        env,
                        other_followers,
                        other_observations,
                    )
                    observations_dict = {
                        f: obs for f, obs in zip(other_followers, other_observations)
                    }
                    observations_dict[follower] = observation

                    other_action_profiles = product(*[
                        list(space_values(env.followers_action_space[f]))
                        for f in other_followers
                    ])
                    for other_actions in other_action_profiles:
                        other_action_prob = 1.0
                        actions_dict = {}
                        for f, action in zip(other_followers, other_actions):
                            actions_dict[f] = action
                            other_action_prob *= strategy_snapshot[f][observations_dict[f]][action]

                        for own_action in space_values(env.followers_action_space[follower]):
                            actions_dict[follower] = own_action
                            own_action_prob = strategy_snapshot[follower][observation][own_action]
                            utilities = _empirical_strategy_payoff(
                                env,
                                leader_policy,
                                followers,
                                observations_dict,
                                actions_dict,
                                payoff_cache,
                            )
                            current_payoff += (
                                utilities[follower]
                                * other_action_prob
                                * own_action_prob
                                * other_type_probability
                                * strategy_weight
                            )

                        for deviation_action in space_values(env.followers_action_space[follower]):
                            actions_dict[follower] = deviation_action
                            utilities = _empirical_strategy_payoff(
                                env,
                                leader_policy,
                                followers,
                                observations_dict,
                                actions_dict,
                                payoff_cache,
                            )
                            deviation_payoffs[deviation_action] += (
                                utilities[follower]
                                * other_action_prob
                                * other_type_probability
                                * strategy_weight
                            )

            best_deviation_payoff = max(deviation_payoffs.values())
            max_gap = max(max_gap, best_deviation_payoff - current_payoff)

    return max(0.0, max_gap)


def compute_empirical_welfare(env, leader_policy, empirical_strategy):
    """Expected leader reward under an empirical MW mixed strategy."""
    followers = env.followers_list
    payoff_cache = {}
    welfare = 0.0
    observation_values = [
        list(space_values(env.followers_observation_space[f]))
        for f in followers
    ]
    strategy_weight = 1.0 / len(empirical_strategy)

    for strategy_snapshot in empirical_strategy:
        for observations in product(*observation_values):
            observations_dict = {f: obs for f, obs in zip(followers, observations)}
            type_probability = _type_profile_probability(env, observations_dict)
            action_values = [
                list(space_values(env.followers_action_space[f]))
                for f in followers
            ]
            for actions in product(*action_values):
                action_prob = 1.0
                actions_dict = {}
                for f, action in zip(followers, actions):
                    actions_dict[f] = action
                    action_prob *= strategy_snapshot[f][observations_dict[f]][action]

                utilities = _empirical_strategy_payoff(
                    env,
                    leader_policy,
                    followers,
                    observations_dict,
                    actions_dict,
                    payoff_cache,
                )
                welfare += (
                    utilities[env.game.leader]
                    * action_prob
                    * type_probability
                    * strategy_weight
                )

    return welfare


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
