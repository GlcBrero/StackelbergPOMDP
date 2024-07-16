from gym.spaces import Discrete, Box, MultiDiscrete, Dict
import numpy as np
import gym
import random, math, copy
import stackelberg_pomdp.utils as utils
from collections import OrderedDict
from stackelberg_pomdp.utils import compute_welfare_loss, check_for_profitable_deviations, TemporaryMethod
from stackelberg_pomdp.leader_policies import BaselinePolicyWrapper


class BaseEnv(gym.Env):

    def __init__(self, game, logger=None, seed=None):
        super().__init__()
        self.game = game
        self.leader = game.leader
        self.followers_list = game.followers_list
        self.num_followers = len(self.followers_list)
        self.num_types = game.num_types
        self.logger = logger

        self._rng = random.Random()  # Create a new random number generator
        self._rng.seed(seed)  # Set the seed
        self.game.set_seed(seed)  # Set the seed in the game

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

        # For now, treating followers separately instead of making this a MARL environment
        self.followers_observation_space = {follower: range(1) for follower in self.followers_list}
        self.followers_action_space = {follower: range(self.game.action_space(follower)) for follower in self.followers_list}


    def reset(self):
        observation = OrderedDict({"base_environment": 0})
        self.followers_obs = {follower: 0 for follower in self.followers_list}
        self.followers_actions = self.get_followers_actions(self.followers_obs)
        return observation


    def step(self, action):

        info = {}

        if self.randomized == True:
            utilities = self.randomized_step(action, self.followers_actions)
        else:
            utilities = self.deterministic_step(action, self.followers_actions)

        observation = OrderedDict({"base_environment": 0})
        reward = utilities[self.leader]
        done = True

        info["base_env_done"] = True
        info["utilities"] = utilities

        return observation, reward, done, info


    def randomized_step(self, leader_action, followers_actions):
        action_probs = utils.weights_to_probs(leader_action)
        utilities = {agent: 0 for agent in self.game.list_of_agents}
        for curr_action in range(self.game.action_space(self.leader)):
            curr_utilities = self.deterministic_step(curr_action, followers_actions)
            for agent in self.game.list_of_agents:
                utilities[agent] = utilities[agent] + action_probs[curr_action] * curr_utilities[agent]
        return utilities


    def deterministic_step(self, leader_action, followers_actions):
        action_dict = followers_actions.copy()
        action_dict[self.leader] = leader_action
        utilities = {agent: self.game.payoff(action_dict, agent) for agent in self.game.list_of_agents}
        return utilities


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

        # For now, treating followers separately instead of making this a MARL environment
        self.followers_observation_space = {follower: range(1) for follower in self.followers_list}
        self.followers_action_space = {follower: range(self.game.follower_action_space(follower)) for follower in self.followers_list}


    def reset(self):

        self.followers_obs = {follower: 0 for follower in self.followers_list}

        # Check if the get_followers_actions method is present in the environment
        if not hasattr(self, 'get_followers_actions') or not callable(getattr(self, 'get_followers_actions')):
            raise NotImplementedError("The 'get_followers_actions' method must be passed to the environment.")

        self.followers_actions = self.get_followers_actions(self.followers_obs)
        observation = OrderedDict({"base_environment": list(self.followers_actions.values())})

        return observation


    def step(self, action):

        info = {}

        game_utilities = {agent: self.game.follower_payoff(self.followers_actions, agent) for agent in self.followers_list}
        reward = sum(game_utilities.values())

        adjusted_utilities = self.adjust_utilities(action, self.followers_actions)
        adjusted_utilities[self.leader] = reward

        info["base_env_done"] = True
        info["utilities"] = adjusted_utilities

        observation = OrderedDict({"base_environment": list(self.followers_actions.values())})
        done = True

        return observation, reward, done, info


    def adjust_utilities(self, leader_action, followers_actions):
        action_dict = followers_actions.copy()
        action_dict[self.leader] = leader_action
        utilities = {follower: self.game.follower_payoff(action_dict, follower) + leader_action[i] for i, follower in
                     enumerate(self.followers_list)}
        return utilities


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

        # For now, treating followers separately instead of making this a MARL environment
        self.followers_observation_space = {follower: range(self.num_types) for follower in self.followers_list}
        self.followers_action_space = {follower: range(self.num_messages) for follower in self.followers_list}
        self.freeze_types = False


    def reset(self):

        self.types = self.followers_obs = self.sample_types()
        self.followers_actions = self.get_followers_actions(self.followers_obs)
        observation = OrderedDict({"base_environment": next(iter(self.followers_actions.values()))})
        return observation


    def step(self, action):


        utilities = {agent: 1 if action == self.types[agent] else 0 for agent in self.followers_list}
        observation = OrderedDict({"base_environment": next(iter(self.followers_actions.values()))})
        reward = utilities[self.followers_list[0]]
        utilities[self.leader] = reward

        info = {}
        info["base_env_done"] = True
        info["utilities"] = utilities

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

        self.followers_observation_space = {follower: range(self.game.num_types) for follower in self.followers_list}
        self.followers_action_space = {follower: range(self.num_messages) for follower in self.followers_list}
        self.freeze_types = False


    def reset(self):

        self.types = self.followers_obs = self.sample_types()
        self.valuations = self.game.get_vals_from_types(self.types)
        self.followers_actions = self.get_followers_actions(self.followers_obs)

        # Reset records
        self.overall_value = 0
        self.num_agents_left = self.num_followers
        self.num_items_left = np.sum(self.game.units_per_item)
        self.outcome = {'order': [], 'prices': [], 'mechanism_outcome': {}}
        self.utilities = {follower:0 for follower in self.followers_list}

        self.state = np.concatenate((
                                    np.ones(self.num_followers), # Agents left
                                    np.asarray(self.game.units_per_item), # Items left
                                    np.ones(self.game.num_diff_items * self.num_followers), # Allocation matrix (subtract when you allocate)
                                    np.zeros(self.game.num_diff_items * self.num_followers),
                                    [self.followers_actions[follower] for follower in self.followers_list]  # Bids
        )) # Bids and dummy observation

        observation_entry = copy.deepcopy(self.state)

        observation = OrderedDict({"base_environment": observation_entry})
        return observation

    def step(self, action):

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


        observation_entry = copy.deepcopy(self.state)
        observation = OrderedDict({"base_environment": observation_entry})
        done = self.num_agents_left <= 0 or self.num_items_left <= 0

        if done:
            # TODO: get max social welfare via setting-specific WD
            self.max_social_welfare = 0
            for j in range(self.game.num_diff_items):
                sorted_vals = sorted([value[j] for key, value in self.valuations.items()], reverse=True)
                self.max_social_welfare += sum(sorted_vals[:self.game.units_per_item[0]])

            if self.max_social_welfare == 0:
                self.allocative_efficiency = 1.0
            else:
                self.allocative_efficiency = self.overall_value / self.max_social_welfare

            if self.overall_value > self.max_social_welfare + 0.0000000001:
                print("Oops")
                pass
                pass

            reward = self.overall_value - self.max_social_welfare
            self.utilities[self.leader] = reward
            info["utilities"] = self.utilities
            info["base_env_done"] = done

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


"""Wrapper for Q learning followers"""
class RLSupervisorQFollowersWrapper(gym.Wrapper):

    def __init__(
            self,
            env,
            alpha=0.15,
            delta=0.95,
            beta=0.00001,
    ):
        super(RLSupervisorQFollowersWrapper, self).__init__(env)

        self.delta = delta
        self.beta = beta
        self.alpha = alpha
        self.this_step_mode = "equilibrium" # We set equilibrium mode as default

        self.q_init() # Needed to initialize observation space in StackPOMDP wrapper

        self.env.get_followers_actions = self.get_followers_actions

    def reset(self):
        self.q_init() # Initialize agents' q_table
        self.step_counter = 0
        obs = self.env.reset() # Restart sub_env
        self.sub_env_done = False
        return obs

    def step(self, action):

        self.step_counter = self.step_counter+1

        if self.sub_env_done:
            obs = self.env.reset()  # Restart sub_env
            self.sub_env_done = False
            return obs, 0, False, {}

        if self.env.get_followers_action: self.env.followers_actions = self.get_followers_actions(self.env.followers_obs, self.this_step_mode)

        obs, reward, done, info = self.env.step(action)

        if done:
            self.sub_env_done = True

            # Record for logging
            info["followers_actions"] = self.followers_actions
            info["sub_env_done"] = True

            # Update agents' tables
            if self.this_step_mode == 'equilibrium':
                self.update_q_tables("", self.followers_actions, info["utilities"], self.env.followers_obs)

        return obs, reward, False, info


    def q_matrices_to_norm_vec(self):
        q_matrices = np.empty((0))
        for follower in self.q_tables.keys():
            q_table_vec = np.array(list(self.q_tables[follower].values())).flatten()
            max_abs = max(abs(q_table_vec))
            if max_abs>0: q_table_vec = q_table_vec / max_abs
            q_matrices = np.append(q_matrices, q_table_vec)
        return q_matrices


    def q_init(self):
        self.q_tables = {follower : {} for follower in self.followers_list}
        for agent in self.followers_list:
            for observation in self.followers_observation_space[agent]:
                self.q_tables[agent][str(observation)] = [0] * len(self.followers_action_space[agent])


    def get_followers_actions(self, observation, action_type="equilibrium"):

        followers_actions = {}

        if action_type == "reward":
            for agent in self.followers_list:
                followers_actions[agent] = np.argmax(self.q_tables[agent][str(observation[agent])])

        if action_type == "equilibrium":
            for agent in self.followers_list:
                epsilon = np.exp(-1 * self.beta * self.step_counter)
                if self.env.unwrapped._rng.uniform(0, 1) < epsilon:
                    followers_actions[agent] = self.env.unwrapped._rng.randint(0, len(self.q_tables[agent][str(observation[agent])])-1)
                else:
                    followers_actions[agent] = np.argmax(self.q_tables[agent][str(observation[agent])])

        return followers_actions


    def update_q_tables(self, next_observation, actions_dict, reward, prev_observation):
        last_values = {agent: 0 for agent in self.followers_list}
        Q_maxes = {agent: 0 for agent in self.followers_list}

        for agent in self.followers_list:
            obs = str(prev_observation[agent])
            last_values[agent] = self.q_tables[agent][obs][actions_dict[agent]]
            # Q_maxes[agent] = np.max(self.q_tables[agent][next_observation])
            self.q_tables[agent][obs][actions_dict[agent]] = \
                ((1 - self.alpha) * last_values[agent]) + (self.alpha * (reward[agent] + self.delta * Q_maxes[agent]))


    def log_info(self, info):
        self.logger.record("q_tables", str(self.q_tables))
        self.env.log_info(info)


"""Wrapper for multiplicative weight followers"""
class RLSupervisorMWFollowersWrapper(gym.Wrapper):
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

        super(RLSupervisorMWFollowersWrapper, self).__init__(env)

        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_iterations = clip_iterations

        self.step_counter = 0
        self.this_step_mode = "equilibrium"
        self.weights = [np.ones((len(self.followers_observation_space[i]), len(self.followers_action_space[i])),
                                dtype=np.float64) for i in self.followers_list]
        self.env.get_followers_actions = self.get_followers_actions

    def reset(self):

        self.weights = [np.ones((len(self.followers_observation_space[follower]),len(self.followers_action_space[follower])),
                                dtype=np.float64) for follower in self.followers_list]
        self.utilities_table = [[0 for _ in self.followers_action_space[follower]] for follower in self.followers_list]

        # The following indexes keep track of the next counterfactual action to try
        self.follower_idx = 0
        self.action_idx = 0

        # Set this to true after testing all counterfactual actions. It will sample new types/action profile
        self.mw_iteration_done = True

        obs_sub_env = self.env.reset() # Restart sub_env

        # If equilibrium step, we freeze types while we try counterfactual actions
        self.env.freeze_types = True if self.this_step_mode == "equilibrium" else False
        self.sub_env_done = False

        return obs_sub_env

    def step(self, action):
        self.step_counter = self.step_counter+1

        if self.sub_env_done:
            obs = self.env.reset()  # Restart sub_env
            self.sub_env_done = False

            # If equilibrium step, we freeze types while we try counterfactual actions
            self.env.freeze_types = True if self.this_step_mode == "equilibrium" else False
            return obs, 0, False, {}

        obs, reward, done, info = self.env.step(action)

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
                self.utilities_table = [[0 for _ in self.followers_action_space[follower]] for follower in self.followers_list]

            # We want to sample new types next time we reset our subenv, even in equilibrium steps
            if self.mw_iteration_done or self.this_step_mode == "reward":
                self.env.freeze_types = False

        return obs, reward, done, info

    def update_idxs(self):
        follower = self.followers_list[self.follower_idx]
        if self.action_idx < len(self.followers_action_space[follower]):
            self.action_idx += 1
        if self.action_idx == len(self.followers_action_space[follower]) and self.follower_idx < self.num_followers - 1:
            self.follower_idx += 1
            self.action_idx = 0
        if self.action_idx == len(
                self.followers_action_space[follower]) and self.follower_idx == self.num_followers - 1:
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
                    self.env.unwrapped._rng.choices(self.followers_action_space[self.followers_list[agent]],
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
            for action in self.followers_action_space[agent]:
                self.weights[agent_idx][self.env.followers_obs[agent]][action] *= \
                    (1 + self.epsilon) ** self.utilities_table[agent_idx][action]

        # Clip weights after self.clip_iterations iterations
        if self.clip_iterations>0 and ((self.step_counter - self.clip_iterations + 1) % self.clip_iterations == 0):
            for agent_idx, agent in enumerate(self.followers_list):
                    max_weight = max(self.weights[agent_idx][self.env.followers_obs[agent]])
                    for action in self.followers_action_space[agent]:
                        self.weights[agent_idx][self.env.followers_obs[agent]][action] = max(
                            self.weights[agent_idx][self.env.followers_obs[agent]][action] / max_weight, self.clip_min)

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


        # Set up observation space:
        #    entry 'base_environment' contains the part of observation for which action may be fixed during a StackMDP episode
        #    whatever starts with 'critic:' will only be seen by critic network
        if self.critic_obs == "full":
            if hasattr(self, "q_tables"):
                num_q_entries = 0
                for follower in self.q_tables.keys():
                    num_q_entries = num_q_entries + len(np.array(list(self.q_tables[follower].values())).flatten())
                self.observation_space = Dict({
                    'base_environment': self.env.observation_space['base_environment'],
                    'critic:is_reward_step': Discrete(2),
                    'critic:exploration_rates': Box(low=0, high=1.0, shape=(len(self.env.followers_list),)),
                    'critic:Q_matrices': Box(low=-1.0, high=1.0, shape=(num_q_entries,)),
                })
            elif hasattr(self, "weights"):
                num_weights = 0
                for follower in range(len(self.env.followers_list)):
                    num_weights = num_weights + len(self.weights[follower].flatten())
                self.observation_space = Dict({
                    'base_environment': self.env.observation_space['base_environment'],
                    'critic:is_reward_step': Discrete(2),
                    'critic:weights': Box(low=-1.0, high=1.0, shape=(num_weights,)),
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
            if hasattr(self, "q_tables"):
                observation["critic:Q_matrices"] = self.q_matrices_to_norm_vec()
                observation['critic:exploration_rates'] = np.array([np.exp(-1 * self.env.beta * self.env.step_counter) for agent in range(len(self.env.followers_list))])
            elif hasattr(self, "weights"):
                observation["critic:weights"] = self.weights_to_norm_vec()


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

            if info.__contains__("base_env_done") and info["base_env_done"]: self.eq_episodes_counter+=1

            info["exclude_from_buffer"] = False if self.env.unwrapped._rng.random() < self.response_phase_prob else True

            info["reward"] = 0

            full_observation = OrderedDict({"base_environment": obs["base_environment"]})
            self.augment_observation(full_observation)

            return full_observation, 0, False, info

        elif self.reward_episodes_counter < self.tot_num_reward_episodes:

            self.env.this_step_mode = "reward"
            obs, reward, done, info = self.env.step(action)
            reward = reward/self.tot_num_reward_episodes # Normalize reward

            if info.__contains__("base_env_done") and info["base_env_done"]: self.reward_episodes_counter+=1

            info["exclude_from_buffer"] = False
            info["leader_action"] = action

            # We log only at the end of subepisodes
            if info.__contains__("base_env_done") and info["base_env_done"]:
                if hasattr(self, "is_eval"):
                    self.log_info(info)

            full_observation = OrderedDict({"base_environment": obs["base_environment"]})
            self.augment_observation(full_observation, is_reward_step=1)

            done = True if self.reward_episodes_counter == self.tot_num_reward_episodes else False

            info["reward"] = reward

        return full_observation, reward, done, info


    def log_info(self, info):

        self.env.log_info(info)
        self.logger.record("count_steps", self.tot_num_steps)

        if isinstance(self.unwrapped, BaseEnvSimpleMatrixGame):
            self.logger.record("leader_action", info["leader_action"])

        self.logger.record("leader_reward", info['utilities'][self.leader])
        for follower in self.followers_list:
            self.logger.record(follower+"_reward", info['utilities'][follower])
            self.logger.record(follower+"_action", info['followers_actions'][follower])
        self.logger.dump(self.tot_num_steps)


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
            if info.__contains__("base_env_done") and info["base_env_done"]:
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