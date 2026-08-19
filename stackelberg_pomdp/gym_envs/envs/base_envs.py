from collections import OrderedDict
import copy
from itertools import product
import random

import gym
from gym.spaces import Box, Dict, Discrete, MultiDiscrete
import numpy as np

import stackelberg_pomdp.utils as utils


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
        self.freeze_types = False

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
        missing = object()
        previous_freeze_types = self.freeze_types
        previous_types = getattr(self, "types", missing)
        previous_mechanism_episode = getattr(self, "mechanism_episode", missing)
        previous_max_social_welfare = getattr(self, "max_social_welfare", missing)
        previous_allocative_efficiency = getattr(self, "allocative_efficiency", missing)

        try:
            # Counterfactual diagnostics must evaluate the requested type
            # profile, regardless of whether the live response phase is
            # currently freezing types across MW deviations.
            self.types = dict(types)
            self.freeze_types = True
            self.reset()
            observation = self.reactive_leader_observation(bids)
            done = False
            while not done:
                action = policy.get_action(observation)
                actions = dict(bids)
                actions[self.leader] = action
                _, reward, done, info = self.step(actions)
                observation = self.reactive_leader_observation(bids)
        finally:
            self.freeze_types = previous_freeze_types
            for attribute, value in (
                    ("types", previous_types),
                    ("mechanism_episode", previous_mechanism_episode),
                    ("max_social_welfare", previous_max_social_welfare),
                    ("allocative_efficiency", previous_allocative_efficiency),
            ):
                if value is missing:
                    if hasattr(self, attribute):
                        delattr(self, attribute)
                else:
                    setattr(self, attribute, value)
        return info

    def leader_state_observation_space(self):
        return Discrete(1)

    def leader_state_observation(self):
        return 0

    def follower_action_observation_space(self):
        if len(self.followers_list) == 1:
            return self.followers_action_space[self.followers_list[0]]
        return MultiDiscrete([
            self.followers_action_space[follower].n
            for follower in self.followers_list
        ])

    def follower_action_observation(self, follower_actions):
        if follower_actions is None:
            return 0
        values = [follower_actions[follower] for follower in self.followers_list]
        if len(values) == 1:
            return values[0]
        return np.array(values, dtype=np.int64)

    def combine_leader_observation(self, leader_state, follower_action_obs):
        if leader_state is None:
            return follower_action_obs
        return np.concatenate((np.atleast_1d(leader_state), np.atleast_1d(follower_action_obs)))

    def reactive_observation_space(self):
        return Dict(OrderedDict({
            "base_environment": self.leader_state_observation_space(),
            "base:follower_actions": self.follower_action_observation_space(),
        }))

    def reactive_leader_observation(self, follower_actions):
        """Return the structured reactive leader observation.

        The actor sees both non-critic keys. Keeping follower actions in their
        own field makes the wrapper logic explicit while leaving domain-specific
        state construction in the base environment.
        """
        return OrderedDict({
            "base_environment": self.leader_state_observation(),
            "base:follower_actions": self.follower_action_observation(follower_actions),
        })

    def leader_observation(self, follower_actions=None):
        return 0

    def reset_types(self):
        """Sample or reuse private types for follower counterfactual evaluation."""
        if not self.freeze_types or not hasattr(self, "types"):
            self.types = self.game.sample_types()
        return self.types

    def reward_phase_length(self, default_length):
        """Return the reward-phase length for this environment.

        Most environments use the configured Monte Carlo length. Environments
        with a finite exact reward distribution can override this and enumerate
        the profiles directly.
        """
        return default_length

    def max_reward_phase_length(self, default_length):
        return self.reward_phase_length(default_length)

    def max_subepisode_transitions(self):
        """Maximum leader transitions in one generated response/reward game."""
        return 1

    def start_reward_phase(self):
        """Initialize any environment-specific exact reward-phase schedule."""
        return

    def end_reward_phase(self):
        """Clear any environment-specific reward schedule."""
        return

    def advance_reward_phase_profile(self):
        """Advance an exact reward-phase schedule, if one is active."""
        return False

    def current_reward_phase_profile(self):
        return None

    def log_info(self, info):
        return

    def threshold_domain_diagnostics(self, policy, followers_strategy):
        return OrderedDict()


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

        # Leader observes no state and chooses a row or mixed row.
        self.observation_space = Dict({'base_environment': Discrete(1)})
        if self.randomized == True:
            self.action_space = Box(
                low=np.array([0 for _ in range(self.game.action_space(self.leader))]),
                high=np.array([1 for _ in range(self.game.action_space(self.leader))]),
                dtype=np.float32,
            )
        else:
            self.action_space = Discrete(self.game.action_space(self.leader))

        # Follower observes no state and chooses a matrix column/action.
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
        follower_actions = {a: actions_dict[a] for a in self.followers_list}

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
        observation[self.leader] = self.leader_observation(follower_actions)
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
            discrete_prices=False,
    ):
        super().__init__(game, logger, seed)
        self.discrete_prices = discrete_prices
        self.discrete_price_vec = (
            self.game.discrete_price_vector()
            if self.discrete_prices
            else None
        )

        # Leader observes follower actions and chooses compensation transfers.
        self.observation_space = Dict({'base_environment': MultiDiscrete([self.game.follower_action_space(agent) for agent in self.followers_list])})
        self.action_space = Box(low=0, high=10, shape=(game.num_agents,), dtype=np.float32)

        # Followers observe no state and choose matrix actions.
        self.followers_observation_space = {follower: Discrete(1) for follower in self.followers_list}
        self.followers_action_space = {follower: Discrete(self.game.follower_action_space(follower)) for follower in self.followers_list}

    def leader_observation(self, follower_actions=None):
        if follower_actions is None:
            return np.zeros(len(self.followers_list), dtype=int)
        return np.array([follower_actions[follower] for follower in self.followers_list], dtype=int)


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
        observation[self.leader] = self.leader_observation(followers_actions)
        return observation, reward, True, info


"""Base environment for simple allocation mechanisms"""
class BaseSimpleAllocation(BaseEnv):
    def __init__(
            self,
            game,
            logger=None,
            seed=None,
    ):
        super().__init__(game, logger, seed)

        # Leader observes only the reported message and chooses an item/type.
        self.observation_space = Dict({'base_environment': Discrete(game.num_messages)})
        self.action_space = Discrete(self.num_types)

        # Follower observes its private type and reports one of the available messages.
        self.followers_observation_space = {follower: Discrete(self.num_types) for follower in self.followers_list}
        self.followers_action_space = {follower: Discrete(game.num_messages) for follower in self.followers_list}

    def leader_observation(self, follower_actions=None):
        if follower_actions is None:
            return 0
        return follower_actions[self.followers_list[0]]


    def reset(self):
        return {**self.reset_types(), self.leader: 0}

    def step(self, actions_dict):
        """Allocate the item selected by the leader after the follower message."""
        leader_action = actions_dict[self.leader]

        # Welfare is one iff the allocated item matches the follower's type.
        follower_rewards = {
            agent: int(leader_action == self.types[agent])
            for agent in self.followers_list
        }
        reward = follower_rewards[self.followers_list[0]]
        utilities = {self.leader: reward, **follower_rewards}

        info = {
            "reward_generated": True,
            "utilities": utilities,
            "surplus": reward,
        }

        follower_actions = {agent: actions_dict[agent] for agent in self.followers_list}
        return {**self.types, self.leader: self.leader_observation(follower_actions)}, reward, True, info

    def log_info(self, info):
        self.logger.record("sampled_type", list(self.types.values()))


"""Base environment for standard sequential posted-price mechanisms."""
class BaseSPM(BaseEnv):
    def __init__(
            self,
            game,
            logger=None,
            seed=None,
            discrete_prices=False,
    ):
        super().__init__(game, logger, seed)
        self.discrete_prices = discrete_prices
        self.discrete_price_vec = (
            self.game.discrete_price_vector()
            if self.discrete_prices
            else None
        )

        state_low, state_high = self.game.mechanism_state_bounds()
        self.observation_space = Dict({
            'base_environment': Box(
                low=state_low,
                high=state_high,
                dtype=np.float32,
            )
        })

        num_price_actions = (
            self.game.num_diff_items * len(self.discrete_price_vec)
            if self.discrete_prices
            else self.game.num_diff_items
        )
        self.action_space = Box(
            low=np.zeros(self.num_followers + num_price_actions, dtype=np.float32),
            high=np.ones(self.num_followers + num_price_actions, dtype=np.float32),
            dtype=np.float32,
        )

    def leader_action(self, action):
        """Translate PPO's SPM action into game-level scores and prices."""
        if not self.discrete_prices:
            return action

        agent_scores = np.asarray(action[:self.num_followers])
        price_logits = np.asarray(action[self.num_followers:])
        prices = []
        num_prices = len(self.discrete_price_vec)
        for item_idx in range(self.game.num_diff_items):
            start = item_idx * num_prices
            stop = start + num_prices
            price_idx = int(np.argmax(price_logits[start:stop]))
            prices.append(self.discrete_price_vec[price_idx])
        return np.concatenate((agent_scores, np.asarray(prices, dtype=np.float32)))

    def reset(self):
        self.types = self.reset_types()
        self.mechanism_episode = self.game.new_episode(self.types)
        return OrderedDict({
            "base_environment": self.game.mechanism_state_vector(
                self.mechanism_episode.mechanism_state
            )
        })

    def step(self, action):
        """Run one truthful-buyer step of the standard SPM.

        This is the Brero-style baseline: the leader policy directly controls
        the sequential posted-price mechanism, with no follower messages and no
        StackPOMDP response phase.
        """
        leader_action = self.leader_action(action)
        actions_dict = {
            self.leader: leader_action,
            **{follower: 0 for follower in self.followers_list},
        }
        result = self.game.step_episode(self.mechanism_episode, actions_dict)
        info = dict(result["info"])
        info["is_reward_phase"] = True
        info["leader_action"] = leader_action
        info["reward"] = result["reward"]
        info["reward_generated"] = result["done"]

        if result["done"]:
            self.max_social_welfare = info["max_social_welfare"]
            self.allocative_efficiency = info["efficiency"]

        return (
            OrderedDict({"base_environment": result["observation"]}),
            result["reward"],
            result["done"],
            info,
        )

    def max_episode_transitions(self):
        return self.max_subepisode_transitions()

    def max_subepisode_transitions(self):
        # A sequential mechanism can visit each buyer at most once. It may end
        # earlier when inventory is exhausted.
        return len(self.followers_list)

    def log_info(self, info):
        self.game.log_episode(
            self.mechanism_episode,
            self.logger,
            self.allocative_efficiency,
            self.max_social_welfare,
        )


class BaseMessageSPM(BaseSPM):
    def __init__(
            self,
            game,
            logger=None,
            seed=None,
    ):
        super().__init__(game, logger, seed)

        # Followers observe private types and submit messages/bids.
        self.followers_observation_space = {follower: Discrete(self.game.num_types) for follower in self.followers_list}
        self.followers_action_space = {follower: Discrete(self.game.num_messages) for follower in self.followers_list}
        self.reward_phase_profiles = []
        self.reward_phase_profile_idx = 0
        self.include_zero_weight_reward_profiles = True

    def _build_reward_phase_profiles(self):
        profiles = []
        for type_values in product(range(self.game.num_types), repeat=len(self.followers_list)):
            types = {
                follower: type_value
                for follower, type_value in zip(self.followers_list, type_values)
            }
            weight = self.game.type_profile_probability(types)
            if self.include_zero_weight_reward_profiles or weight > 0:
                profiles.append({"types": types, "weight": weight})
        return profiles

    def start_reward_phase(self):
        self.set_reward_phase_profiles(self._build_reward_phase_profiles())

    def set_reward_phase_profiles(self, profiles):
        self.reward_phase_profiles = list(profiles)
        self.reward_phase_profile_idx = 0
        if self.reward_phase_profiles:
            self.types = dict(self.reward_phase_profiles[0]["types"])

    def end_reward_phase(self):
        self.reward_phase_profiles = []
        self.reward_phase_profile_idx = 0

    def reward_phase_length(self, default_length):
        return (
            len(self.reward_phase_profiles)
            or len(self._build_reward_phase_profiles())
            or default_length
        )

    def max_reward_phase_length(self, default_length):
        return len(self._build_reward_phase_profiles()) or default_length

    def advance_reward_phase_profile(self):
        if not self.reward_phase_profiles:
            return False

        self.reward_phase_profile_idx += 1
        if self.reward_phase_profile_idx >= len(self.reward_phase_profiles):
            return False

        self.types = dict(self.reward_phase_profiles[self.reward_phase_profile_idx]["types"])
        return True

    def current_reward_phase_profile(self):
        if not self.reward_phase_profiles:
            return None
        if self.reward_phase_profile_idx >= len(self.reward_phase_profiles):
            return None
        return self.reward_phase_profiles[self.reward_phase_profile_idx]

    def reset(self):
        profile = self.current_reward_phase_profile()
        if profile is not None:
            self.types = dict(profile["types"])
        else:
            self.types = self.reset_types()
        self.mechanism_episode = self.game.new_episode(self.types)
        return {**self.types, self.leader: 0}

    def step(self, actions_dict):
        """Step the sequential mechanism. Follower bids read from actions_dict on first call."""
        actions_dict = dict(actions_dict)
        actions_dict[self.leader] = self.leader_action(actions_dict[self.leader])
        result = self.game.step_episode(self.mechanism_episode, actions_dict)
        observation = {agent: 0 for agent in self.followers_list}
        observation[self.leader] = result["observation"]

        if result["done"]:
            self.max_social_welfare = result["info"]["max_social_welfare"]
            self.allocative_efficiency = result["info"]["efficiency"]
            profile = self.current_reward_phase_profile()
            if profile is not None:
                weight = profile["weight"]
                result["info"]["unweighted_reward"] = result["reward"]
                result["info"]["exact_profile_weight"] = weight
                result["info"]["weighted_efficiency"] = result["info"]["efficiency"] * weight
                # Exact profiles contribute directly to the expected return.
                # Do not multiply by the number of profiles: doing so preserves
                # a reporting average but changes PPO's reward/entropy scale.
                result["reward"] = result["reward"] * weight
                result["info"]["reward"] = result["reward"]
                result["info"]["surplus"] = result["reward"]
                if "response_action_weight" in profile:
                    result["info"]["response_action_weight"] = profile[
                        "response_action_weight"
                    ]
                    result["info"]["response_candidate"] = profile.get(
                        "response_candidate"
                    )
            result["info"]["type_profile"] = dict(self.types)
            result["info"]["mechanism_outcome"] = copy.deepcopy(
                self.mechanism_episode.outcome
            )

        return observation, result["reward"], result["done"], result["info"]

    # Observation/state representation.
    def leader_observation(self, follower_actions=None):
        return self.leader_state_observation()

    def leader_state_observation_space(self):
        return self.observation_space["base_environment"]

    def leader_state_observation(self):
        return self.game.mechanism_state_vector(self.mechanism_episode.mechanism_state)

    def threshold_domain_diagnostics(self, policy, followers_strategy):
        return OrderedDict({
            "mechanism_outcome": self.game.describe_policy_outcomes(self, policy),
        })

    def log_info(self, info):
        self.game.log_episode(
            self.mechanism_episode,
            self.logger,
            self.allocative_efficiency,
            self.max_social_welfare,
        )



"""Wrapper for multiplicative weight followers"""
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
            leader_observation_space='no_observation',
            leader_k=1,
            sort_leader_observation=False,
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
        self.leader_observation_space = leader_observation_space
        self.leader_k = leader_k
        self.sort_leader_observation = sort_leader_observation
        self._exp_a0_mu = np.exp(a_0 / mu)

        if leader_observation_space == 'price_profile':
            self.observation_space = Dict({
                'base_environment': MultiDiscrete([m] * num_agents * leader_k)
            })
        else:
            self.observation_space = Dict({'base_environment': Discrete(1)})
        self.action_space = Discrete(1)

        # Followers observe recent price history and choose prices from the grid.
        self.followers_observation_space = {
            agent: Box(
                np.array([0] * (k * num_agents)),
                np.array([m] * (k * num_agents)),
                dtype=int,
            )
            for agent in self.agents
        }
        self.followers_action_space = {agent: Discrete(m) for agent in self.agents}

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

    def leader_observation(self, follower_actions=None):
        """Return the platform observation induced by current follower prices.

        For price-profile observations, the base state is the previous
        ``leader_k - 1`` executed price profiles and the reactive wrapper adds
        the current follower price profile. For no-observation experiments,
        return the null state.
        """
        if self.leader_observation_space != 'price_profile':
            return 0
        if follower_actions is None:
            follower_actions = {
                agent: self.action_history[agent][-1]
                for agent in self.agents
            }
        return self.combine_leader_observation(
            self.leader_state_observation(),
            self.follower_action_observation(follower_actions),
        )

    def leader_state_observation_space(self):
        if self.leader_observation_space != 'price_profile' or self.leader_k == 1:
            return Discrete(1)
        return MultiDiscrete([self.m] * self.num_agents * (self.leader_k - 1))

    def leader_state_observation(self):
        if self.leader_observation_space != 'price_profile' or self.leader_k == 1:
            return 0
        profiles = []
        for history_idx in range(self.leader_k - 1, 0, -1):
            profiles.append(self._price_profile_from_history(history_idx))
        return np.concatenate(profiles) if len(profiles) > 1 else profiles[0]

    def follower_action_observation_space(self):
        if self.leader_observation_space != 'price_profile':
            return Discrete(1)
        return MultiDiscrete([self.m] * self.num_agents)

    def follower_action_observation(self, follower_actions):
        if self.leader_observation_space != 'price_profile':
            return 0
        return self._current_price_profile(follower_actions)

    def _price_profile_from_history(self, history_idx):
        profile = np.array([
            self.action_history[agent][-min(history_idx, len(self.action_history[agent]))]
            for agent in self.agents
        ], dtype=np.int64)
        return np.sort(profile) if self.sort_leader_observation else profile

    def _current_price_profile(self, follower_actions):
        if follower_actions is None:
            return self._price_profile_from_history(1)
        profile = np.array([follower_actions[agent] for agent in self.agents], dtype=np.int64)
        return np.sort(profile) if self.sort_leader_observation else profile

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
