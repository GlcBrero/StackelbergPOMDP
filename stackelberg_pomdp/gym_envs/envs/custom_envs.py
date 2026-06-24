from gym.spaces import Discrete, Box, MultiDiscrete, Dict
import numpy as np
import gym
import random, math, copy
import stackelberg_pomdp.utils as utils
from collections import OrderedDict
from stackelberg_pomdp.utils import compute_welfare_loss, check_for_profitable_deviations, TemporaryMethod, get_all_wrappers
from stackelberg_pomdp.leader_policies import BaselinePolicyWrapper


class BaseEnv(gym.Env):
    """Base class for all multi-agent environments.

    Provides: leader, followers_list, num_followers, _rng, logger.
    Subclasses implement step(all_actions_dict) and reset().
    """

    def __init__(self, game=None, leader=None, followers_list=None, logger=None, seed=None):
        super().__init__()
        self.logger = logger
        self._rng = random.Random()
        self._rng.seed(seed)

        if game is not None:
            self.game = game
            self.leader = game.leader
            self.followers_list = game.followers_list
            self.num_followers = len(self.followers_list)
            self.num_types = game.num_types
            self.game.set_seed(seed)
        else:
            self.leader = leader or "platform"
            self.followers_list = followers_list or []
            self.num_followers = len(self.followers_list)

    def run_episode(self, policy, types, bids):

        def new_sample_types():
            return types

        def new_get_followers_actions(observation):
            return bids

        with TemporaryMethod(self, 'sample_types', new_sample_types):
            with TemporaryMethod(self, 'get_followers_actions', new_get_followers_actions):
                observation = self.reset()
                done = False
                while not done:
                    action = policy.get_action(observation)
                    observation, reward, done, info = self.step(action)
        return info

    def sample_types(self):
        if not hasattr(self, "types") or not self.freeze_types:
            return self.game.sample_types()
        else:
            return self.types

    def log_info(self, info):
        return


"""Base environment for normal form games"""
class BaseEnvSimpleMatrixGame(BaseEnv):
    def __init__(
            self,
            game,
            logger=None,
            seed=None,
            randomized=False,
            randomization_type="linear",
    ):
        super().__init__(game, logger, seed)

        self.randomized = randomized
        self.randomization_type = randomization_type

        # Set dummy observation space for the leader
        self.observation_space = Dict({'base_environment': Discrete(1)})

        # This should matrix action
        if self.randomized == True:
            self.action_space = Box(
                low=np.array([0 for _ in range(self.game.action_space(self.leader))]),
                high=np.array([1 for _ in range(self.game.action_space(self.leader))]),
                dtype=np.float32,
            )
        else:
            self.action_space = Discrete(self.game.action_space(self.leader))

        self.followers_observation_space = {follower: Discrete(1) for follower in self.followers_list}
        self.followers_action_space = {follower: Discrete(self.game.action_space(follower)) for follower in self.followers_list}


    def reset(self):
        self.followers_obs = {follower: 0 for follower in self.followers_list}
        observation = {agent: 0 for agent in self.followers_list}
        observation[self.leader] = 0
        return observation

    def step(self, actions_dict):
        """Step the multi-agent game. actions_dict includes leader + all followers."""
        info = {}
        leader_action = actions_dict[self.leader]

        if self.randomized:
            action_probs = utils.weights_to_probs(leader_action)
            utilities = {agent: 0 for agent in self.game.list_of_agents}
            for curr_action in range(self.game.action_space(self.leader)):
                curr_actions = actions_dict.copy()
                curr_actions[self.leader] = curr_action
                curr_utilities = {agent: self.game.payoff(curr_actions, agent) for agent in self.game.list_of_agents}
                for agent in self.game.list_of_agents:
                    utilities[agent] += action_probs[curr_action] * curr_utilities[agent]
        else:
            utilities = {agent: self.game.payoff(actions_dict, agent) for agent in self.game.list_of_agents}

        observation = {agent: 0 for agent in self.followers_list}
        observation[self.leader] = 0
        reward = utilities[self.leader]

        info["reward_generated"] = True
        info["utilities"] = utilities
        info["surplus"] = reward

        return observation, reward, True, info


"""Base environment for matrix design games"""
class BaseEnvMatrixDesignGame(BaseEnv):
    def __init__(
            self,
            game,
            logger=None,
            seed=None,
    ):
        super().__init__(game, logger, seed)

        # Leader observes the game action taken by the followers
        self.observation_space = Dict({'base_environment': MultiDiscrete([self.game.follower_action_space(agent) for agent in self.followers_list])})

        # Leader sets a compensation for each follower for taking the action observed
        self.action_space = Box(low=0, high=10, shape=(game.num_agents,), dtype=np.float32)

        self.followers_observation_space = {follower: Discrete(1) for follower in self.followers_list}
        self.followers_action_space = {follower: Discrete(self.game.follower_action_space(follower)) for follower in self.followers_list}


    def reset(self):
        self.followers_obs = {follower: 0 for follower in self.followers_list}
        observation = {agent: 0 for agent in self.followers_list}
        observation[self.leader] = 0
        return observation

    def step(self, actions_dict):
        """Step the multi-agent game. actions_dict includes leader + all followers."""
        info = {}
        leader_action = actions_dict[self.leader]
        followers_actions = {a: actions_dict[a] for a in self.followers_list}

        game_utilities = {agent: self.game.follower_payoff(followers_actions, agent) for agent in self.followers_list}
        reward = sum(game_utilities.values())

        # Adjust follower utilities with leader's compensation
        adjusted_utilities = {
            follower: self.game.follower_payoff(actions_dict, follower) + leader_action[i]
            for i, follower in enumerate(self.followers_list)
        }
        adjusted_utilities[self.leader] = reward

        info["reward_generated"] = True
        info["utilities"] = adjusted_utilities
        info["surplus"] = reward

        observation = {agent: 0 for agent in self.followers_list}
        observation[self.leader] = 0
        return observation, reward, True, info


"""Base environment for simple allocation mechanisms"""
class BaseSimpleAllocation(BaseEnv):
    def __init__(
            self,
            game,
            logger=None,
            seed=None,
    ):
        # Initialize base environment
        super().__init__(game, logger, seed)
        self.num_messages = game.num_messages

        # Leader observes follower's message
        self.observation_space = Dict({'base_environment': Discrete(self.num_messages)})

        # We have as many items as types
        self.action_space = Discrete(self.num_types)

        self.followers_observation_space = {follower: Discrete(self.num_types) for follower in self.followers_list}
        self.followers_action_space = {follower: Discrete(self.num_messages) for follower in self.followers_list}
        self.freeze_types = False


    def reset(self):
        self.types = self.followers_obs = self.sample_types()
        observation = {agent: self.types[agent] for agent in self.followers_list}
        observation[self.leader] = 0
        return observation

    def step(self, actions_dict):
        """Step the multi-agent game. actions_dict includes leader + all followers."""
        leader_action = actions_dict[self.leader]
        utilities = {agent: 1 if leader_action == self.types[agent] else 0 for agent in self.followers_list}
        reward = utilities[self.followers_list[0]]
        utilities[self.leader] = reward

        observation = {agent: self.types[agent] for agent in self.followers_list}
        observation[self.leader] = 0

        info = {}
        info["reward_generated"] = True
        info["utilities"] = utilities
        info["surplus"] = reward

        return observation, reward, True, info

    def log_info(self, info):
        self.logger.record("sampled_type", list(self.types.values()))


"""Base environment for message SPM"""
class BaseMessageSPM(BaseEnv):
    def __init__(
            self,
            game,
            discrete_prices=False,
            logger=None,
            seed=None,
    ):
        # Initilize base environment
        super().__init__(game, logger, seed)

        self.num_messages = game.num_messages

        self.outcome = {}
        self.logger = logger
        self.discrete_prices = discrete_prices

        self.observation_space = Dict({
            'base_environment': Box(
                low = np.array([
                    0 for _ in range(self.num_followers +  # Agents left
                    len(self.game.units_per_item) +  # Items left
                    2 * self.game.num_diff_items * self.num_followers +  # Prices and Allocations (hence 2*)
                    self.num_followers  # Bids
                )]),
                high = np.array(
                    [1 for _ in range(self.num_followers)] +
                    self.game.units_per_item +
                    [1 for _ in range(2 * self.game.num_diff_items * self.num_followers)] +
                    [self.num_messages-1] * self.num_followers # Bids
                )
            )
        })

        if discrete_prices:
            self.discrete_price_vec = self.game.get_discrete_price_vec()
            self.action_space = Box(
                                    low=np.array([0 for i in range(
                                        self.num_followers +
                                        self.game.num_diff_items*len(self.discrete_price_vec))]),
                                    high=np.array([1 for i in range(
                                       self.num_followers +
                                       self.game.num_diff_items * len(self.discrete_price_vec))]),
                                    dtype=np.float32)
        else:
            self.action_space = Box(low=np.array([0 for i in range(self.num_followers + self.game.num_diff_items)]),
                                high=np.array([1 for i in range(self.num_followers + self.game.num_diff_items)]),
                                dtype=np.float32)

        self.followers_observation_space = {follower: Discrete(self.game.num_types) for follower in self.followers_list}
        self.followers_action_space = {follower: Discrete(self.num_messages) for follower in self.followers_list}
        self.freeze_types = False


    def reset(self):

        self.types = self.followers_obs = self.sample_types()
        self.valuations = self.game.get_vals_from_types(self.types)
        self.followers_actions = {}

        # Reset records
        self.overall_value = 0
        self.num_agents_left = self.num_followers
        self.num_items_left = np.sum(self.game.units_per_item)
        self.outcome = {'order': [], 'prices': [], 'mechanism_outcome': {}}
        self.utilities = {follower:0 for follower in self.followers_list}

        self.state = None
        observation = {agent: self.types[agent] for agent in self.followers_list}
        observation[self.leader] = 0
        return observation

    def step(self, actions_dict):
        """Step the sequential mechanism. Follower bids read from actions_dict on first call."""
        action = actions_dict[self.leader]

        # On first step after reset, read follower bids and build state
        if self.state is None:
            self.followers_actions = {a: actions_dict[a] for a in self.followers_list if a in actions_dict}
            self.state = np.concatenate((
                np.ones(self.num_followers),
                np.asarray(self.game.units_per_item),
                np.ones(self.game.num_diff_items * self.num_followers),
                np.zeros(self.game.num_diff_items * self.num_followers),
                [self.followers_actions[follower] for follower in self.followers_list]
            ))

        info = {}

        agent_scores = action[:self.num_followers]
        agent_scores = [agent_scores[i] if self.state[i] else -math.inf for i in range(self.num_followers)]

        prices = action[self.num_followers:]

        if self.discrete_prices:
            prices = np.array([self.discrete_price_vec[np.argmax(
                prices[i * len(self.discrete_price_vec):(i + 1) * len(self.discrete_price_vec)])] for i in
                               range(self.num_diff_items)])

        agent_idx = np.argmax(agent_scores)
        agent = self.followers_list[agent_idx]
        item_idx = self.buyer(agent, prices)

        self.outcome['order'].append(agent)
        self.outcome['prices'].append(prices.tolist())  # TODO: This doesn't work for multiple heterogeneous items

        # Remove agent from state
        self.state[agent_idx] = 0
        self.num_agents_left -= 1
        reward = 0

        # If agent buys, update allocation and items_left in state
        # TODO: This only works for 1 item!
        if item_idx != -1:
            self.num_items_left -= 1
            self.overall_value += self.valuations[agent][item_idx]

            # Remove item from state
            self.state[self.num_followers + item_idx] -= 1

            # Update allocation matrix
            self.state[self.num_followers + len(self.game.units_per_item) + self.game.num_diff_items * agent_idx + item_idx] -= 1

            # Add agent to outcome
            self.outcome['mechanism_outcome'][agent] = {'allocation': item_idx, 'payment': prices[item_idx]}


        # Update prices in state
        available_units_per_item = self.state[self.num_followers:self.num_followers + len(self.game.units_per_item)]
        prices_available = [0 if available_units_per_item[i] else 1 for i in range(self.game.num_diff_items)]
        self.state[self.num_followers + len(
            self.game.units_per_item) + self.game.num_diff_items * self.num_followers + self.game.num_diff_items * agent_idx: self.num_followers + len(
            self.game.units_per_item) + self.game.num_diff_items * self.num_followers + self.game.num_diff_items * (agent_idx + 1)] = prices_available


        observation = {agent: 0 for agent in self.followers_list}
        observation[self.leader] = copy.deepcopy(self.state)
        done = self.num_agents_left <= 0 or self.num_items_left <= 0

        if done:
            self.max_social_welfare = 0
            for j in range(self.game.num_diff_items):
                sorted_vals = sorted([value[j] for key, value in self.valuations.items()], reverse=True)
                self.max_social_welfare += sum(sorted_vals[:self.game.units_per_item[0]])

            if self.max_social_welfare == 0:
                self.allocative_efficiency = 1.0
            else:
                self.allocative_efficiency = self.overall_value / self.max_social_welfare

            reward = self.overall_value - self.max_social_welfare
            self.utilities[self.leader] = reward
            info["utilities"] = self.utilities
            info["surplus"] = reward
            info["reward_generated"] = done

        return observation, reward, done, info


    def policy_description(self, policy):
        types = {follower: 0 for follower in self.followers_list}
        outcome_strings = []  # Store outcomes for each combination

        for bid1 in range(self.num_messages):
            for bid2 in range(self.num_messages):
                # Create a copy of the environment for each combination of bid1 and bid2
                bids = {self.followers_list[0]: bid1, self.followers_list[1]: bid2}
                self.run_episode(policy, types, bids)
                outcome_strings.append(self.print_outcome())

        # Combine the outcome strings into a single representation
        final_outcome_string = "\n".join(outcome_strings)  # Example: Join with newlines

        return final_outcome_string

    def print_outcome(self):
        # Initialize an empty list to store the row of the table
        row = []

        # Add the bids of the agents
        for (follower, action) in self.followers_actions.items():
            row.append(f"B_{follower}: {action}")

        # Add the order of visiting the agents and the corresponding prices
        for i in range(len(self.outcome['order'])):
            row.extend(
                [f"O_{i + 1}: {self.outcome['order'][i]}", f"P_{i + 1}: {self.outcome['prices'][i][0]:.2f}"])

        table_md = "| " + " | ".join(row) + " |"

        return table_md

    def log_info(self, info):
        self.logger.record("efficiency", "%.5f" % self.allocative_efficiency)
        self.logger.record("overall_value", "%.5f" % self.overall_value)
        self.logger.record("opt", "%.5f" % self.max_social_welfare)

        for i in range(len(self.outcome['order'])):
            self.logger.record("order_%i" % i, self.outcome['order'][i])
            # The next line only works in settings with 1 item!
            self.logger.record("price_%i" % i, self.outcome['prices'][i][0])

        for follower in self.followers_list:
            self.logger.record("bids_"+follower, self.followers_actions[follower])
            # The next line only works in settings with 1 item!
            self.logger.record("value_"+follower, self.valuations[follower][0])

    def buyer(self, agent, prices):
        valuation = self.valuations[agent]
        available_units_per_item = self.state[self.num_followers:self.num_followers + len(self.game.units_per_item)]
        utility = [valuation[i] - prices[i] if available_units_per_item[i] else -math.inf for i in
                   range(self.game.num_diff_items)]
        choice = np.argmax(utility)
        self.utilities[agent] = max(utility[choice],0)
        return choice if utility[choice] >= 0 else -1



"""Wrapper for multiplicative weight followers"""
class MWFollowersWrapper(gym.Wrapper):
    """A wrapper for gym environments that applies the Multiplicative Weights method for followers."""

    CLIP_MIN = 0.001
    CLIP_ITERATIONS = 0
    DEFAULT_EPS = 0.01

    def __init__(
            self,
            env,
            epsilon=DEFAULT_EPS,
            clip_min=CLIP_MIN,
            clip_iterations=CLIP_ITERATIONS,
    ):

        super().__init__(env)

        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_iterations = clip_iterations

        self.step_counter = 0
        self.this_step_mode = "equilibrium"
        self._rng = self.env.unwrapped._rng
        self.weights = [np.ones((self.followers_observation_space[i].n, self.followers_action_space[i].n),
                                dtype=np.float64) for i in self.followers_list]

    def _to_leader_obs(self):
        return np.array([0])

    def reset(self):

        self.weights = [np.ones((self.followers_observation_space[follower].n, self.followers_action_space[follower].n),
                                dtype=np.float64) for follower in self.followers_list]
        self.utilities_table = [[0 for _ in range(self.followers_action_space[follower].n)] for follower in self.followers_list]

        # The following indexes keep track of the next counterfactual action to try
        self.follower_idx = 0
        self.action_idx = 0

        # Set this to true after testing all counterfactual actions. It will sample new types/action profile
        self.mw_iteration_done = True

        obs = self.env.reset()
        self.followers_obs = {a: obs[a] for a in self.followers_list}

        # If equilibrium step, we freeze types while we try counterfactual actions
        self.env.freeze_types = True if self.this_step_mode == "equilibrium" else False
        self.sub_env_done = False

        return OrderedDict({"base_environment": self._to_leader_obs()})

    def step(self, action):
        self.step_counter = self.step_counter+1

        if self.sub_env_done:
            obs = self.env.reset()
            self.followers_obs = {a: obs[a] for a in self.followers_list}
            self.sub_env_done = False

            # If equilibrium step, we freeze types while we try counterfactual actions
            self.env.freeze_types = True if self.this_step_mode == "equilibrium" else False
            return OrderedDict({"base_environment": self._to_leader_obs()}), 0, False, {}

        self.followers_actions = self.get_followers_actions(self.followers_obs)

        # Build full actions dict: leader + followers
        all_actions = dict(self.followers_actions)
        all_actions[self.env.leader] = action
        obs, reward, done, info = self.env.step(all_actions)

        if done:

            self.sub_env_done = True
            info["followers_actions"] = self.followers_actions

            # We save utility generated by counterfactual action to update weights later
            if self.this_step_mode != "reward":
                follower = self.followers_list[self.follower_idx]
                self.utilities_table[self.follower_idx][self.action_idx] = info['utilities'][follower]

            # Update counterfactual action idxs
            self.update_idxs()

            if self.mw_iteration_done and self.this_step_mode == "equilibrium":
                self.update_weights()
                self.utilities_table = [[0 for _ in range(self.followers_action_space[follower].n)] for follower in self.followers_list]

            # We want to sample new types next time we reset our subenv, even in equilibrium steps
            if self.mw_iteration_done or self.this_step_mode == "reward":
                self.env.freeze_types = False

        info["reward_generated"] = done
        reward = info.get('surplus', reward)
        return OrderedDict({"base_environment": self._to_leader_obs()}), reward, done, info

    def update_idxs(self):
        follower = self.followers_list[self.follower_idx]
        if self.action_idx < self.followers_action_space[follower].n:
            self.action_idx += 1
        if self.action_idx == self.followers_action_space[follower].n and self.follower_idx < self.num_followers - 1:
            self.follower_idx += 1
            self.action_idx = 0
        if self.action_idx == self.followers_action_space[follower].n and self.follower_idx == self.num_followers - 1:
            self.mw_iteration_done = True
            self.follower_idx = 0
            self.action_idx = 0

    def get_followers_actions(self, followers_observations):

        # Sample new actions if we are done with counterfactuals
        if self.this_step_mode == "reward":
            self.current_actions = {}
            for agent in range(len(self.followers_list)):
                self.current_actions[self.followers_list[agent]] = \
                    np.argmax(self.weights[agent][followers_observations[self.followers_list[agent]]])
            return self.current_actions

        # Sample new actions if we are done with counterfactuals
        if self.mw_iteration_done == True:
            self.current_actions = {}
            for agent in range(len(self.followers_list)):
                self.current_actions[self.followers_list[agent]] = \
                    self._rng.choices(range(self.followers_action_space[self.followers_list[agent]].n),
                                   self.weights[agent][followers_observations[self.followers_list[agent]]])[0]
            self.mw_iteration_done = False

        # If in the middle of a counterfactual iteration, use a counterfactual action for the current follower
        follower = self.followers_list[self.follower_idx]
        counterfactual_action = self.action_idx
        counterfactual_actions = self.current_actions.copy()
        counterfactual_actions[follower] = counterfactual_action

        return counterfactual_actions

    def weights_to_norm_vec(self):
        weights = np.empty((0))
        for follower in range(len(self.followers_list)):
            weights_vec = self.weights[follower].flatten()
            max_abs = max(abs(weights_vec))
            if max_abs > 0: weights_vec = weights_vec / max_abs
            weights = np.append(weights, weights_vec)
        return weights

    def update_weights(self):

        # Update weights
        for agent_idx, agent in enumerate(self.followers_list):
            for action in range(self.followers_action_space[agent].n):
                self.weights[agent_idx][self.followers_obs[agent]][action] *= \
                    (1 + self.epsilon) ** self.utilities_table[agent_idx][action]

        # Clip weights after self.clip_iterations iterations
        if self.clip_iterations>0 and ((self.step_counter - self.clip_iterations + 1) % self.clip_iterations == 0):
            for agent_idx, agent in enumerate(self.followers_list):
                    max_weight = max(self.weights[agent_idx][self.followers_obs[agent]])
                    for action in range(self.followers_action_space[agent].n):
                        self.weights[agent_idx][self.followers_obs[agent]][action] = max(
                            self.weights[agent_idx][self.followers_obs[agent]][action] / max_weight, self.clip_min)

    def log_info(self, info):
        self.logger.record("weights", str(self.weights))
        self.env.log_info(info)

    def weights_to_strat(self):
        result = {}
        for follower, weight in zip(self.followers_list, self.weights):
            inner_dict = {}
            for i, obs in enumerate(self.followers_observation_space[follower]):
                inner_dict[obs] = np.argmax(weight[i])
            result[follower] = inner_dict
        return result


"""Wrapper for Stackelberg POMDP"""
class StackPOMDPWrapper(gym.Wrapper):

    def __init__(
            self,
            env,
            tot_num_eq_episodes=1000,
            tot_num_reward_episodes=10,
            critic_obs="full",
            response_phase_prob=1,
    ):

        super(StackPOMDPWrapper, self).__init__(env)

        # This sets the total number of equilibrium and reward steps in StackMDP
        self.tot_num_eq_episodes = tot_num_eq_episodes
        self.tot_num_reward_episodes = tot_num_reward_episodes
        self.critic_obs = critic_obs

        self.tot_num_steps = 0
        self.response_phase_prob = response_phase_prob
        self._rng = self.env.unwrapped._rng


        # Set up observation space:
        #    entry 'base_environment' contains the part of observation for which action may be fixed during a StackMDP episode
        #    whatever starts with 'critic:' will only be seen by critic network
        # Detect which follower wrapper is in the chain
        self._follower_type = None
        if hasattr(self, "q_tables"):
            self._follower_type = "qlearning"
        elif hasattr(self, "weights"):
            self._follower_type = "mw"
        elif hasattr(self, "price_idx"):
            self._follower_type = "roundrobin"

        if self.critic_obs == "full":
            if self._follower_type == "qlearning":
                num_q_entries = 0
                for follower in self.q_tables.keys():
                    num_q_entries = num_q_entries + len(np.array(list(self.q_tables[follower].values())).flatten())
                self.observation_space = Dict({
                    'base_environment': self.env.observation_space['base_environment'],
                    'critic:is_reward_step': Discrete(2),
                    'critic:exploration_rates': Box(low=0, high=1.0, shape=(len(self.env.followers_list),)),
                    'critic:Q_matrices': Box(low=-1.0, high=1.0, shape=(num_q_entries,)),
                })
            elif self._follower_type == "mw":
                num_weights = 0
                for follower in range(len(self.env.followers_list)):
                    num_weights = num_weights + len(self.weights[follower].flatten())
                self.observation_space = Dict({
                    'base_environment': self.env.observation_space['base_environment'],
                    'critic:is_reward_step': Discrete(2),
                    'critic:weights': Box(low=-1.0, high=1.0, shape=(num_weights,)),
                })
            elif self._follower_type == "roundrobin":
                self.observation_space = Dict({
                    'base_environment': self.env.observation_space['base_environment'],
                    'critic:is_reward_step': Discrete(2),
                    'critic:strategy_idx': Discrete(self.env.n_actions + 1),
                    'critic:best_profit': Box(low=-10.0, high=10.0, shape=(1,)),
                })


    def reset(self):

        self.env.this_step_mode = "equilibrium"
        self.eq_episodes_counter = 0
        self.reward_episodes_counter = 0

        obs_sub_env = self.env.reset() # Restart sub_env

        full_observation = OrderedDict({"base_environment": obs_sub_env["base_environment"]})
        self.augment_observation(full_observation)
        return full_observation


    def augment_observation(self, observation, is_reward_step=0):
        if self.critic_obs == "flag" or self.critic_obs == "full":
            observation["critic:is_reward_step"] = is_reward_step
        if self.critic_obs == "full":
            if self._follower_type == "qlearning":
                observation["critic:Q_matrices"] = self.q_matrices_to_norm_vec()
                exp_rate = np.exp(-1 * self.env.beta * self.env.step_counter)
                observation['critic:exploration_rates'] = np.full(len(self.env.followers_list), exp_rate)
            elif self._follower_type == "mw":
                observation["critic:weights"] = self.weights_to_norm_vec()
            elif self._follower_type == "roundrobin":
                observation["critic:strategy_idx"] = min(self.price_idx, self.n_actions)
                observation["critic:best_profit"] = np.array([self.best_profit], dtype=np.float32)


    def step(self, action):

        if not hasattr(self, "is_eval"):
            self.tot_num_steps += 1

        if (self.tot_num_steps+1) % 100000 == 0:
            print("StackPOMDPWrapper steps completed: ", self.tot_num_steps+1)

        if self.eq_episodes_counter < self.tot_num_eq_episodes:

            if self.eq_episodes_counter == self.tot_num_eq_episodes:
                self.env.this_step_mode = "reward"
            else:
                self.env.this_step_mode = "equilibrium"

            obs, _, done, info = self.env.step(action)

            if info.__contains__("reward_generated") and info["reward_generated"]: self.eq_episodes_counter+=1

            info["exclude_from_buffer"] = False if self._rng.random() < self.response_phase_prob else True

            info["reward"] = 0
            info["is_reward_phase"] = False

            full_observation = OrderedDict({"base_environment": obs["base_environment"]})
            self.augment_observation(full_observation)

            return full_observation, 0, False, info

        elif self.reward_episodes_counter < self.tot_num_reward_episodes:

            self.env.this_step_mode = "reward"
            obs, reward, done, info = self.env.step(action)

            if info.get("reward_generated"): self.reward_episodes_counter+=1

            info["exclude_from_buffer"] = False
            info["leader_action"] = action
            info["is_reward_phase"] = True
            info["tot_num_reward_steps"] = self.tot_num_reward_episodes

            full_observation = OrderedDict({"base_environment": obs["base_environment"]})
            self.augment_observation(full_observation, is_reward_step=1)

            done = True if self.reward_episodes_counter == self.tot_num_reward_episodes else False

            # Add StackPOMDP info for top-level logging
            info["count_steps"] = self.tot_num_steps

            info["reward"] = reward

        return full_observation, reward, done, info


"""Base environment for Bertrand price competition (logit demand model)"""
class BertrandCompetitionEnv(BaseEnv):

    def __init__(
            self,
            num_agents=2,
            c_i=1,
            a=2,
            platform_intervention='pdp',
            a_0=0,
            mu=0.25,
            m=15,
            adv=0.3,
            k=1,
            price_min=1.05,
            price_max=1.7,
            seed=None,
            logger=None,
    ):
        agents = ['agent_' + str(i) for i in range(num_agents)]
        super().__init__(leader="platform", followers_list=agents, logger=logger, seed=seed)

        self.num_agents = num_agents
        self.agents = agents
        self.k = k
        self.c_i = c_i
        self.m = m
        self.adv = adv
        self.a = np.array([a] * num_agents)
        self.a_0 = a_0
        self.mu = mu

        # Price grid: [marginal_cost, 2.1] à la Johnson et al. (2023)
        p_N, p_M = self._compute_nash_monopoly_prices(c_i, a, a_0, mu, num_agents)
        self.p_nash = p_N
        self.p_monopoly = p_M
        self.action_price_space = np.linspace(price_min, price_max, m)
        self.platform_intervention = platform_intervention
        self._exp_a0_mu = np.exp(a_0 / mu)

        # Per-agent observation/action spaces (used by Q-learning wrapper)
        self.followers_observation_space = {}
        self.followers_action_space = {}
        for agent in self.agents:
            self.followers_observation_space[agent] = Box(
                np.array([0] * (k * num_agents)),
                np.array([m] * (k * num_agents)),
                dtype=int,
            )
            self.followers_action_space[agent] = Discrete(m)

        # Dummy leader spaces (overridden by QLearningFollowersWrapper)
        self.observation_space = Dict({'base_environment': Discrete(1)})
        self.action_space = Discrete(1)

    def reset(self):
        self.current_step = 0
        self.action_history = {}
        for agent in self.agents:
            self.action_history[agent] = [self._rng.randint(0, self.m - 1)]

        obs_agents = np.array([
            self.action_history[self.agents[i]][-self.k:]
            for i in range(self.num_agents)
        ], dtype=np.int64).flatten()
        observation = {agent: obs_agents for agent in self.agents}

        self.bbx_occupant = [0]
        self.platform_action = 0
        return observation

    @staticmethod
    def _compute_nash_monopoly_prices(c, a, a_0, mu, n):
        """Compute symmetric Nash and monopoly prices for logit demand.

        Nash FOC (unilateral):  p = c + mu / (1 - q(p))
        Monopoly FOC (joint):   maximize (p - c) * q(p)
        where q(p) = exp((a-p)/mu) / (n*exp((a-p)/mu) + exp(a_0/mu))
        """
        ps = np.linspace(c, c + 4 * mu * n, 10000)

        def q_sym(p):
            x = np.exp((a - p) / mu)
            y = np.exp(a_0 / mu)
            return x / (n * x + y)

        # Nash: minimize |p - c - mu/(1-q(p))|
        resid = np.abs(ps - c - mu / (1 - np.array([q_sym(p) for p in ps])))
        p_N = ps[np.argmin(resid)]

        # Monopoly: max (p-c)*q(p)
        profits = np.array([(p - c) * q_sym(p) for p in ps])
        p_M = ps[np.argmax(profits)]

        return p_N, p_M

    def step(self, actions_dict):
        """Step the multi-agent game. actions_dict includes leader + all followers."""
        info = {}
        self.current_step += 1

        # Extract leader action
        self.platform_action = actions_dict.get(self.leader, 0)

        # Extract follower actions
        follower_actions = {a: actions_dict[a] for a in self.agents if a in actions_dict}
        actions_idx = np.array([follower_actions[a] for a in self.agents]).flatten()
        for i in range(self.num_agents):
            self.action_history[self.agents[i]].append(actions_idx[i])

        obs_agents = np.array([
            self.action_history[self.agents[i]][-self.k:]
            for i in range(self.num_agents)
        ], dtype=np.int64).flatten()
        observation = {agent: obs_agents for agent in self.agents}

        self.prices_idx = [int(pr) for pr in actions_idx[:self.num_agents]]
        self.prices = self.action_price_space.take(self.prices_idx)

        bbx_idx = self.get_bbx_idx(self.prices, self.platform_action)
        info['bbx_idx'] = bbx_idx
        occupants = bbx_idx[0] if len(bbx_idx) == 1 else [bbx_idx[k] for k in range(len(bbx_idx))]
        self.bbx_occupant.append(occupants)

        if bbx_idx is None:
            exp_vals = np.exp((self.a - self.prices) / self.mu)
            denom = exp_vals.sum() + self._exp_a0_mu
            info['surplus'] = self.mu * np.log(denom)
            demands = exp_vals / denom
        elif len(bbx_idx) > 0:
            bb = np.array(bbx_idx)
            exp_vals = np.exp((self.a[bb] - self.prices[bb]) / self.mu)
            denom = exp_vals.sum() + self._exp_a0_mu
            info['surplus'] = self.mu * np.log(denom)
            demands = np.zeros(self.num_agents)
            demands[bb] = exp_vals / denom
        else:
            info['surplus'] = self.mu * np.log(self._exp_a0_mu)
            demands = np.zeros(self.num_agents)

        rewards = {}
        for i in range(self.num_agents):
            rewards[self.agents[i]] = (self.prices[i] - self.c_i) * demands[i]

        done = False  # Game continues — prices carry over between steps
        info["reward_generated"] = True  # Rewards are ready, count as one sub-episode
        return observation, rewards, done, info

    def _demand(self, a, p, mu, agent_idx, bb_idx):
        if bb_idx is None:
            return np.exp((a[agent_idx] - p[agent_idx]) / mu) / (
                np.sum(np.exp((a - p) / mu)) + np.exp(self.a_0 / mu)
            )
        if agent_idx not in bb_idx:
            return 0
        denom = np.sum([np.exp((a[idx] - p[idx]) / mu) for idx in bb_idx]) + np.exp(self.a_0 / mu)
        return np.exp((a[agent_idx] - p[agent_idx]) / mu) / denom

    def _compute_surplus(self, prices, bbx_idx):
        val = np.sum([np.exp((self.a[i] - float(prices[i])) / self.mu) for i in bbx_idx])
        val += np.exp(self.a_0 / self.mu)
        return self.mu * np.log(val)

    def get_bbx_idx(self, prices, supervisor_action):
        if self.platform_intervention == 'no_intervene':
            return list(range(self.num_agents))

        elif self.platform_intervention == 'pdp':
            return [int(np.argmin(prices))]

        elif self.platform_intervention == 'dpdp':
            prev_prices_idx = [self.action_history[self.agents[i]][-2] for i in range(self.num_agents)]
            prev_prices = self.action_price_space.take(prev_prices_idx)

            bbx_occ_idx = self.bbx_occupant[-1]
            occ_price = prices[bbx_occ_idx]
            occ_prev_price = prev_prices[bbx_occ_idx]
            non_bbx_idx = int(1 - bbx_occ_idx)

            undercut_diff = prices[bbx_occ_idx] - prices[non_bbx_idx]
            if undercut_diff < self.adv and occ_price <= occ_prev_price:
                return [bbx_occ_idx]
            else:
                return [int(np.argmin(prices))]

        elif self.platform_intervention == 'block_equal':
            # If all agents quote the same price, nobody gets displayed
            if len(set(prices)) == 1:
                return []
            return list(range(self.num_agents))

        elif self.platform_intervention == 'learn_threshold':
            price_thresh = self.action_price_space[supervisor_action]
            return [i for i in range(self.num_agents) if prices[i] <= price_thresh]

        elif self.platform_intervention == 'learn_binary_threshold':
            # Binary: action 0 = open (all included), action 1 = close (only lowest price included)
            if supervisor_action == 1:
                price_thresh = self.action_price_space[0]
            else:
                price_thresh = self.action_price_space[-1]
            return [i for i in range(self.num_agents) if prices[i] <= price_thresh]

    def compute_q_init_entry(self, delta):
        """Compute initial Q-values à la Calvano et al. (2019), eq. 8.

        Q_{i,0}(s, a_i) = sum_{a_{-i}} pi_i(a_i, a_{-i}) / ((1-delta) * |A|^{n-1})

        Assumes uniform opponent play and computes expected discounted profit
        for each action. When platform_intervention != 'no_intervene', uses
        buy-box demand (Johnson et al. 2021 heuristic).
        """
        n = self.m
        all_agents = list(range(self.num_agents))
        entry = np.empty(n)
        for i in range(n):
            avg_reward = 0
            for j in range(n):
                price_i = self.action_price_space[i]
                price_j = self.action_price_space[j]
                prices = np.array([price_i, price_j])

                if self.platform_intervention == 'no_intervene':
                    # Calvano: all agents always displayed
                    demand_i = self._demand(self.a, prices, self.mu, 0, all_agents)
                else:
                    # Johnson et al.: buy-box winner gets displayed
                    bbx_demand = self._demand(self.a, prices, self.mu, 0, [0])
                    non_bbx_demand = self._demand(self.a, prices, self.mu, 0, [1])

                    if price_i < price_j:
                        demand_i = bbx_demand
                    elif price_i > price_j:
                        demand_i = non_bbx_demand
                    else:
                        sigma = 0.01
                        numer = np.exp(-price_i / sigma)
                        denom = np.sum([np.exp(-p / sigma) for p in prices])
                        demand_i = (numer / denom) * bbx_demand + (1 - numer / denom) * non_bbx_demand

                avg_reward += (price_i - self.c_i) * demand_i / n
            entry[i] = avg_reward / (1 - delta)
        return entry

    def log_info(self, info):
        if self.logger is None:
            return
        self.logger.record("consumer_surplus", info.get("surplus", 0))
        self.logger.record("c_i", round(self.c_i, 2))
        for j in range(self.num_agents):
            if hasattr(self, 'prices'):
                self.logger.record("price_" + str(j),
                    np.where(self.action_price_space == self.prices[j])[0][0])


"""Q-learning wrapper for follower agents in repeated games.

Works with any base env that exposes:
  - followers_list, num_agents, followers_action_space (dict of Discrete)
  - leader, _rng, set_platform_action(action)
  - step(actions_dict) -> (obs_dict, rewards_dict, done_dict, info)
    where info contains 'surplus' (leader reward)
  - optionally compute_q_init_entry(delta) for smart Q-table initialization
"""
class QLearningFollowersWrapper(gym.Wrapper):

    def __init__(
            self,
            env,
            alpha=0.15,
            delta=0.95,
            beta=0.00001,
            leader_action_space=None,
            warm_start_q=False,
            q_tables_path=None,
    ):
        super().__init__(env)

        self.alpha = alpha
        self.delta = delta
        self.beta = beta
        self.warm_start_q = warm_start_q
        self._q_tables_path = q_tables_path
        self._rng = self.env.unwrapped._rng  # Share base env's RNG for reproducibility

        self.num_followers = len(self.env.followers_list)
        self.n_follower_actions = self.env.followers_action_space[self.env.followers_list[0]].n
        self.steps_since_restart = [0] * self.num_followers
        self.step_counter = 0
        self.this_step_mode = "equilibrium"

        self.q_tables = {}
        self._q_init()

        # Leader observation: null (no_observation). Subclass overrides for reactive.
        self.observation_space = Dict({'base_environment': MultiDiscrete([1])})

        # Leader action space
        if leader_action_space is not None:
            self.action_space = leader_action_space
        elif hasattr(self.env, 'platform_intervention') and self.env.platform_intervention == 'learn_binary_threshold':
            self.action_space = Discrete(2)
        elif hasattr(self.env, 'platform_intervention') and self.env.platform_intervention == 'learn_threshold':
            self.action_space = Discrete(self.n_follower_actions)
        else:
            self.action_space = Discrete(self.num_followers)

    def load_q_tables(self, path):
        """Load pre-converged Q-tables from a calvano_replication pkl file."""
        import pickle
        with open(path, 'rb') as f:
            state = pickle.load(f)
        for agent in self.env.followers_list:
            if agent in state['q_tables']:
                self.q_tables[agent] = state['q_tables'][agent]
        self._rebuild_flat_cache()

    def reset(self):
        if self.warm_start_q and self.q_tables:
            pass  # Keep existing Q-tables across episodes
        elif self._q_tables_path:
            self._q_init()
            self.load_q_tables(self._q_tables_path)
        else:
            self._q_init()
        self.step_counter = 0
        self.steps_since_restart = [0] * self.num_followers

        obs_sub_env = self.env.reset()
        # Handle both array and scalar observations from base env
        raw_obs = obs_sub_env[self.env.followers_list[0]]
        self.current_obs = np.atleast_1d(np.array(raw_obs, dtype=int))
        self.current_obs_key = tuple(self.current_obs)
        self.current_actions = self._get_follower_actions(self.current_obs_key, "standard")

        return OrderedDict({"base_environment": self._to_leader_obs()})

    def _to_leader_obs(self):
        return np.array([0])

    def _execute_step(self, action):
        """Execute one step: apply leader action, step base env with all actions, update Q-tables."""
        self.step_counter += 1
        for i in range(self.num_followers):
            self.steps_since_restart[i] += 1

        # Build full actions dict: leader + all followers
        all_actions = dict(self.current_actions)
        all_actions[self.env.leader] = action
        obs_sub_env, rewards, sub_done, info = self.env.step(all_actions)
        # rewards is a dict {agent: reward} for Bertrand, or scalar for old games
        if not isinstance(rewards, dict):
            rewards = info.get('utilities', {})

        raw_obs = obs_sub_env[self.env.followers_list[0]]
        new_obs = np.atleast_1d(np.array(raw_obs, dtype=int))[:self.num_followers]
        new_obs_key = tuple(new_obs)

        # Update Q-tables when sub-episode is done and in learning mode
        if info.get("reward_generated", sub_done) and self.this_step_mode in ('equilibrium', 'standard'):
            n_actions = self.n_follower_actions
            for agent_idx, agent in enumerate(self.env.followers_list):
                if new_obs_key not in self.q_tables[agent]:
                    self.q_tables[agent][new_obs_key] = self._make_q_entry()
                a = self.current_actions[agent]
                prev_q = self.q_tables[agent][self.current_obs_key]
                new_val = (
                    (1 - self.alpha) * prev_q[a]
                    + self.alpha * (rewards[agent] + self.delta * np.max(self.q_tables[agent][new_obs_key]))
                )
                prev_q[a] = new_val
                key_idx = self._key_to_idx.get(self.current_obs_key)
                if key_idx is not None:
                    flat_offset = agent_idx * self._q_flat_entries_per_agent + key_idx * n_actions + a
                    self._q_flat[flat_offset] = new_val
            self._q_cache_dirty = True

        self.current_obs = new_obs
        self.current_obs_key = new_obs_key

        # Pick next follower actions
        mode = "standard" if self.this_step_mode == 'equilibrium' else (
            "argmax" if self.this_step_mode == 'reward' else self.this_step_mode
        )
        self.current_actions = self._get_follower_actions(self.current_obs_key, mode)

        reward = info.get('surplus', 0)
        if "reward_generated" not in info:
            info["reward_generated"] = sub_done
        info["utilities"] = {self.env.leader: reward, **rewards}
        info["followers_actions"] = dict(self.current_actions)
        info["reward_pricing_agents"] = rewards

        return reward, info

    def step(self, action):
        """Basic (no_observation): leader acts simultaneously with followers."""
        reward, info = self._execute_step(action)
        return OrderedDict({"base_environment": self._to_leader_obs()}), reward, False, info

    def _get_follower_actions(self, observation, action_type):
        rng = self._rng
        actions = {}
        for idx, agent in enumerate(self.env.followers_list):
            q_vals = self.q_tables[agent][observation]
            if action_type == "random":
                actions[agent] = rng.randint(0, self.n_follower_actions - 1)
            elif action_type == "argmax":
                actions[agent] = int(np.argmax(q_vals))
            else:  # epsilon-greedy
                eps = np.exp(-self.beta * self.steps_since_restart[idx])
                actions[agent] = rng.randint(0, self.n_follower_actions - 1) if rng.random() < eps else int(np.argmax(q_vals))
        return actions

    def _q_init(self):
        self.q_tables = {agent: {} for agent in self.env.followers_list}
        # Generate all possible joint observation keys (cartesian product of action spaces)
        from itertools import product as iterproduct
        action_ranges = [range(self.env.followers_action_space[a].n) for a in self.env.followers_list]
        for combo in iterproduct(*action_ranges):
            key = tuple(combo)
            for agent in self.env.followers_list:
                self.q_tables[agent][key] = self._make_q_entry()
        self._q_cache_dirty = True
        self._q_norm_cache = None
        # Build flat array and key-to-offset mapping for fast cache rebuild
        self._q_keys_ordered = list(self.q_tables[self.env.followers_list[0]].keys())
        self._key_to_idx = {k: i for i, k in enumerate(self._q_keys_ordered)}
        n_keys = len(self._q_keys_ordered)
        n_actions = self.n_follower_actions
        self._q_flat = np.empty(len(self.env.followers_list) * n_keys * n_actions)
        offset = 0
        for agent in self.env.followers_list:
            for key in self._q_keys_ordered:
                self._q_flat[offset:offset + n_actions] = self.q_tables[agent][key]
                offset += n_actions
        self._q_flat_entries_per_agent = n_keys * n_actions

    def _rebuild_flat_cache(self):
        """Rebuild flat array from Q-tables (after loading from file)."""
        n_actions = self.n_follower_actions
        self._q_keys_ordered = list(self.q_tables[self.env.followers_list[0]].keys())
        self._key_to_idx = {k: i for i, k in enumerate(self._q_keys_ordered)}
        n_keys = len(self._q_keys_ordered)
        self._q_flat = np.empty(len(self.env.followers_list) * n_keys * n_actions)
        offset = 0
        for agent in self.env.followers_list:
            for key in self._q_keys_ordered:
                self._q_flat[offset:offset + n_actions] = self.q_tables[agent][key]
                offset += n_actions
        self._q_flat_entries_per_agent = n_keys * n_actions
        self._q_cache_dirty = True
        self._q_norm_cache = None

    def _make_q_entry(self):
        base = self.env.unwrapped
        if hasattr(base, 'compute_q_init_entry'):
            return base.compute_q_init_entry(self.delta)
        return np.zeros(self.n_follower_actions)

    def q_matrices_to_norm_vec(self):
        if not self._q_cache_dirty and self._q_norm_cache is not None:
            return self._q_norm_cache
        entries_per_agent = self._q_flat_entries_per_agent
        result = np.empty_like(self._q_flat)
        for i in range(len(self.env.followers_list)):
            start = i * entries_per_agent
            end = start + entries_per_agent
            chunk = self._q_flat[start:end]
            max_abs = np.max(np.abs(chunk))
            if max_abs > 0:
                result[start:end] = chunk / max_abs
            else:
                result[start:end] = chunk
        self._q_norm_cache = result
        self._q_cache_dirty = False
        return result

"""Round-robin (monopolist) follower wrapper.

Tries each price in sequence during equilibrium, picks the most profitable
for the reward phase. All followers play the same price at each step.

Eq phase = m steps (one per price). Reward phase = as configured.
Game-agnostic interface: step(leader_action) → (obs, reward, done, info).
"""
class RoundRobinFollowersWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.num_followers = len(env.followers_list)
        self.n_actions = env.followers_action_space[env.followers_list[0]].n
        self.this_step_mode = "equilibrium"
        self.price_idx = 0
        self.profits = np.zeros(self.n_actions)
        self.best_profit = 0.0
        self.best_price = 0

        # Leader obs = null (no_observation by default; use ReactiveLeaderWrapper for price_profile)
        self.observation_space = Dict({'base_environment': MultiDiscrete([1])})
        if hasattr(env, 'platform_intervention') and env.platform_intervention == 'learn_binary_threshold':
            self.action_space = Discrete(2)
        else:
            self.action_space = Discrete(self.n_actions)

    def _to_leader_obs(self):
        return np.array([0])

    def reset(self):
        self.env.reset()
        self.price_idx = 0
        self.profits = np.zeros(self.n_actions)
        self.best_profit = 0.0
        self.best_price = 0
        self.current_actions = {a: 0 for a in self.env.followers_list}
        return OrderedDict({"base_environment": self._to_leader_obs()})

    def step(self, action):
        if self.this_step_mode in ('equilibrium', 'standard') and self.price_idx < self.n_actions:
            # Test current price
            follower_actions = {a: self.price_idx for a in self.env.followers_list}
            all_actions = dict(follower_actions)
            all_actions[self.env.leader] = action
            obs, rewards, done, info = self.env.step(all_actions)

            if not isinstance(rewards, dict):
                rewards = info.get('utilities', {})

            self.profits[self.price_idx] = sum(r for a, r in rewards.items() if a != self.env.leader)
            self.price_idx += 1
            self.best_profit = float(np.max(self.profits[:self.price_idx]))

            if self.price_idx < self.n_actions:
                self.current_actions = {a: self.price_idx for a in self.env.followers_list}
            else:
                self.best_price = int(np.argmax(self.profits))
                self.current_actions = {a: self.best_price for a in self.env.followers_list}
        else:
            # Reward phase: play best price
            follower_actions = {a: self.best_price for a in self.env.followers_list}
            all_actions = dict(follower_actions)
            all_actions[self.env.leader] = action
            obs, rewards, done, info = self.env.step(all_actions)

            if not isinstance(rewards, dict):
                rewards = info.get('utilities', {})

        reward = info.get('surplus', 0)
        info["reward_generated"] = info.get("reward_generated", True)
        info["utilities"] = {self.env.leader: reward, **{a: rewards.get(a, 0) for a in self.env.followers_list}}
        info["followers_actions"] = dict(self.current_actions)

        return OrderedDict({"base_environment": self._to_leader_obs()}), reward, False, info


"""Reactive leader wrapper: leader observes follower actions before acting.

Wraps any follower wrapper. Overrides the observation so the leader sees
follower actions (e.g., price profile) instead of null. The leader reacts
to what followers just quoted. Game-agnostic — works with Q-learning, MW, etc.

Stack: StackPOMDP → ReactiveLeaderWrapper → FollowerWrapper → BaseEnv
"""
class ReactiveLeaderWrapper(gym.Wrapper):

    def __init__(self, env, leader_k=1, sort_obs=False):
        super().__init__(env)
        n_followers = len(env.env.followers_list)
        n_actions = env.env.followers_action_space[env.env.followers_list[0]].n
        self._followers_list = env.env.followers_list
        self._leader_k = leader_k
        self._sort_obs = sort_obs
        self.observation_space = Dict({
            'base_environment': MultiDiscrete([n_actions] * n_followers * leader_k)
        })

    def _to_leader_obs(self):
        if self._leader_k == 1:
            obs = np.array([self._last_follower_actions[a] for a in self._followers_list])
            if self._sort_obs:
                obs = np.sort(obs)
            return obs
        # k=2: concatenate [prev_actions, curr_actions]
        prev = np.array([self._prev_follower_actions[a] for a in self._followers_list])
        curr = np.array([self._last_follower_actions[a] for a in self._followers_list])
        if self._sort_obs:
            # Sort each time step independently to preserve temporal structure
            prev = np.sort(prev)
            curr = np.sort(curr)
        return np.concatenate([prev, curr])

    def reset(self):
        obs = self.env.reset()
        self._last_follower_actions = dict(self.env.current_actions)
        self._prev_follower_actions = dict(self._last_follower_actions)
        return OrderedDict({"base_environment": self._to_leader_obs()})

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._prev_follower_actions = dict(self._last_follower_actions)
        self._last_follower_actions = info.get("followers_actions", self._last_follower_actions)
        return OrderedDict({"base_environment": self._to_leader_obs()}), reward, done, info


"""Detects cycles in reward-phase states and normalizes reward.

During the reward phase, tracks base_environment observations. When a state
repeats (cycle detected), delivers the average per-step reward over one
complete cycle and zeros out all other reward steps. Keeps episode length
fixed for RL buffer alignment.

StackPOMDPWrapper divides each reward step by tot_num_reward_episodes. This
wrapper undoes that normalization when accumulating, computes the cycle
average on the raw rewards, then delivers it as the total episode reward on
the cycle-detection step. All other reward steps (before and after the cycle)
are zeroed out.

Works for any StackPOMDP setting with discrete/hashable observations. For
continuous observations, pass a custom state_key_fn that rounds appropriately.

If no cycle is detected within the reward phase, falls back to delivering the
average reward over all observed reward steps on the last step.
"""
class StationaryCycleRewardWrapper(gym.Wrapper):

    def __init__(self, env, state_key_fn=None):
        super().__init__(env)
        self._state_key_fn = state_key_fn or self._default_state_key

    @staticmethod
    def _default_state_key(obs):
        return tuple(obs['base_environment'].flatten())

    def reset(self):
        self._reward_states = []
        self._reward_accumulator = []  # Raw (un-normalized) rewards
        self._cycle_detected = False
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if info.get("is_reward_phase", False):
            state_key = self._state_key_fn(obs)

            if self._cycle_detected:
                reward = 0.0
            elif state_key in self._reward_states:
                # Cycle detected — deliver average reward over one cycle
                cycle_start = self._reward_states.index(state_key)
                cycle_rewards = self._reward_accumulator[cycle_start:]
                reward = sum(cycle_rewards) / len(cycle_rewards)
                info['consumer_surplus'] = reward
                self._cycle_detected = True
            else:
                # Accumulate, zero out
                self._reward_states.append(state_key)
                self._reward_accumulator.append(reward)
                reward = 0.0

            # Fallback: last step, no cycle detected
            if done and not self._cycle_detected and self._reward_accumulator:
                reward = sum(self._reward_accumulator) / len(self._reward_accumulator)
                info['consumer_surplus'] = reward

        return obs, reward, done, info


"""Openness evaluation wrapper.

After the normal episode ends, enters evaluation mode: cycles through all m^2
follower price pairs, presents each as an observation to the leader, records
the leader's threshold response, and computes the average exclusion rate.

The penalty (lambda * avg_exclusion) is subtracted from the episode reward.
This measures how restrictive the leader's POLICY is across all possible
price profiles, not just at the realized prices.

Sits above StationaryCycleRewardWrapper:
  OpennessEvaluationWrapper → StationaryCycleRewardWrapper → StackPOMDP → ...
"""
class OpennessEvaluationWrapper(gym.Wrapper):

    def __init__(self, env, intervention_lambda=0.0, m=4):
        super().__init__(env)
        self.intervention_lambda = intervention_lambda
        self.m = m
        self._eval_mode = False
        self._eval_idx = 0
        # Check if leader can see price profiles (obs size > 1)
        base_shape = env.observation_space['base_environment'].shape
        self._is_state_based = base_shape and base_shape[0] > 1
        # Detect leader_k from observation size: 2 entries = k=1, 4 entries = k=2
        self._leader_k = base_shape[0] // 2 if self._is_state_based else 1
        # Total eval pairs: m^2 for k=1, m^4 for k=2
        self._n_eval = m ** (2 * self._leader_k)
        # Detect binary action space
        self._binary_actions = (env.action_space.n == 2)

    def _make_eval_obs(self, idx):
        """Build evaluation observation for eval index."""
        obs = OrderedDict(self._held_obs)
        if self._is_state_based:
            if self._leader_k == 1:
                i = idx // self.m
                j = idx % self.m
                obs["base_environment"] = np.array([i, j])
            else:
                # k=2: decode idx into (p0_prev, p1_prev, p0_curr, p1_curr)
                p1c = idx % self.m
                p0c = (idx // self.m) % self.m
                p1p = (idx // (self.m ** 2)) % self.m
                p0p = (idx // (self.m ** 3)) % self.m
                obs["base_environment"] = np.array([p0p, p1p, p0c, p1c])
        return obs

    def reset(self):
        self._eval_mode = False
        self._eval_idx = 0
        self._held_reward = 0
        self._held_obs = None
        self._held_info = {}
        self._exclusions = []
        self._episode_reward = 0  # Accumulate total reward across episode
        return self.env.reset()

    def _current_prices_from_idx(self, idx):
        """Extract current price indices (i, j) from eval index."""
        if self._leader_k == 1:
            return idx // self.m, idx % self.m
        else:
            # k=2: current prices are the last two components
            p0c = (idx // self.m) % self.m
            p1c = idx % self.m
            return p0c, p1c

    def step(self, action):
        if self._eval_mode:
            # Evaluation step: action is the leader's threshold for this test observation
            i, j = self._current_prices_from_idx(self._eval_idx)
            if self._binary_actions:
                # Binary: action 0 = open (threshold=m-1), action 1 = close (threshold=0)
                effective_threshold = 0 if action == 1 else (self.m - 1)
            else:
                effective_threshold = action
            n_excluded = (1 if i > effective_threshold else 0) + (1 if j > effective_threshold else 0)
            self._exclusions.append(n_excluded / 2.0)
            self._eval_idx += 1

            if self._eval_idx >= self._n_eval:
                # All pairs evaluated — deliver penalized reward
                avg_exclusion = np.mean(self._exclusions)
                penalty = self.intervention_lambda * avg_exclusion
                self._eval_mode = False
                info = dict(self._held_info)
                info['avg_exclusion'] = avg_exclusion
                info['consumer_surplus'] = self._held_reward
                info['penalized_surplus'] = self._held_reward - penalty
                return self._held_obs, self._held_reward - penalty, True, info
            else:
                # Return next test observation
                return self._make_eval_obs(self._eval_idx), 0, False, {}

        # Normal step
        obs, reward, done, info = self.env.step(action)
        self._episode_reward += reward

        if done and self.intervention_lambda > 0:
            # Enter round-robin evaluation mode for both obs types
            self._eval_mode = True
            self._eval_idx = 0
            self._held_reward = self._episode_reward  # Total episode reward (pure CS)
            self._held_obs = obs
            self._held_info = info
            self._exclusions = []
            return self._make_eval_obs(0), 0, False, {}

        return obs, reward, done, info


"""Top-level logging wrapper.

Sits at the top of the wrapper stack. At done=True, reads the info dict
(populated by all layers below) and writes everything to the CSV logger.
No other layer should log — they only write to info.
"""
class LoggingWrapper(gym.Wrapper):

    def __init__(self, env, logger=None):
        super().__init__(env)
        self.logger = logger

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if done and self.logger:
            # Base env logging (domain-specific: prices, c_i, etc.)
            self.env.unwrapped.log_info(info)

            # Pure consumer surplus (set by StationaryCycleRewardWrapper or OpennessWrapper)
            cs = info.get("consumer_surplus", info.get("surplus", 0))
            self.logger.record("consumer_surplus", cs)

            # Leader reward = penalized surplus if available, else pure CS
            lr = info.get("penalized_surplus", cs)
            self.logger.record("leader_reward", lr)

            # Step count
            self.logger.record("count_steps", info.get("count_steps", 0))

            # Pure consumer surplus (from OpennessEvaluationWrapper or base env)
            cs = info.get("consumer_surplus", info.get("surplus", 0))
            self.logger.record("consumer_surplus", cs)

            # Penalized surplus and exclusion (from OpennessEvaluationWrapper)
            if "penalized_surplus" in info:
                self.logger.record("penalized_surplus", info["penalized_surplus"])
            if "avg_exclusion" in info:
                self.logger.record("avg_exclusion", info["avg_exclusion"])

            # Follower info
            if "utilities" in info:
                for agent, util in info["utilities"].items():
                    if agent != self.env.unwrapped.leader:
                        self.logger.record(f"{agent}_reward", util)
            if "followers_actions" in info:
                for agent, act in info["followers_actions"].items():
                    self.logger.record(f"{agent}_action", act)

            self.logger.dump(info.get("count_steps", 0))

        return obs, reward, done, info


class StopOnThresholdWrapper(gym.Wrapper):
    """
    A gym wrapper that stops the training when a certain reward threshold is reached.
    """
    def __init__(self, env, reward_threshold=0):
        super().__init__(env)
        self.reward_threshold = reward_threshold
        self.best_reward = float('-inf')
        self.zero_deviation_count = 0
        self.total_episodes = 0

    def reset(self):
        self.episode_reward = 0  # Reset episode reward at the start of each episode
        self.check_equilibrium = True
        self.response_is_not_equilibrium = False
        return self.env.reset()

    def step(self, action):

        if hasattr(self, "model"): self.policy = BaselinePolicyWrapper(self.model.policy, self)

        if not hasattr(self, "policy"):
            raise ValueError(
                "StopOnThresholdWrapper requires a policy object to check whether follower strategies are in equilibrium.")

        if self.eq_episodes_counter == self.tot_num_eq_episodes and self.check_equilibrium:
            self.followers_strategy = self.weights_to_strat()
            self.max_deviation = check_for_profitable_deviations(self.env, self.policy, self.followers_strategy)
            if self.max_deviation > 0: self.response_is_not_equilibrium = True
            self.check_equilibrium = False

        obs, reward, done, info = self.env.step(action)

        # Set reward to 0 unless subenv_done is True and response_is_not_equilibrium is True
        if self.response_is_not_equilibrium:
            if info.__contains__("reward_generated") and info["reward_generated"]:
                reward = -1/self.tot_num_reward_episodes
            else:
                reward = 0

        # Increment the episode reward
        self.episode_reward += reward

        if done:
            # Increment the total episodes counter
            self.total_episodes += 1

            # Increment the zero deviation counter if max_deviation is zero
            if self.max_deviation == 0:
                self.zero_deviation_count += 1

            # Update the best reward and log if the episode reward is greater and max_deviation is zero
            if self.episode_reward > self.best_reward and self.max_deviation == 0:
                self.best_reward = self.episode_reward
                welfare_loss = compute_welfare_loss(self.env, self.policy, self.followers_strategy)
                zero_deviation_fraction = self.zero_deviation_count / self.total_episodes

                self.logger.record("Welfare loss", f"{welfare_loss:.2f}")
                self.logger.record("Reward", f"{self.episode_reward:.2f}")
                self.logger.record("Fraction of eqs", f"{zero_deviation_fraction:.2f}")
                self.logger.record("Equilibrium", str(self.followers_strategy))
                self.logger.record("Outcome", self.env.policy_description(self.policy))
                self.logger.record("Optimal found", welfare_loss >= self.reward_threshold)
                self.logger.record("Timestep", self.env.tot_num_steps)
                self.logger.record("Episode count", self.total_episodes)
                self.logger.dump(self.env.tot_num_steps)

                if welfare_loss >= self.reward_threshold:
                    print("Optimal policy found!")
                    raise Exception("Policy found!")
        return obs, reward, done, info