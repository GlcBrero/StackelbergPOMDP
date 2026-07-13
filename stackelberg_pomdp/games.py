from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
import random


class BaseStackelbergGame(ABC):
    def __init__(self, leader, followers_list, num_types=1):
        super().__init__()
        self.leader = leader
        self.followers_list = followers_list
        self.num_agents = len(followers_list) + 1
        self.num_types = num_types
        self._rng = random.Random()  # Create a new random number generator

    def set_seed(self, seed):
        self._rng.seed(seed)  # Seed the random number generator

    def sample_types(self):
        """Sample types for all agents. Returns a dictionary with agent names as keys and types as values.
        Sample uniformly at random by default."""
        return {follower: self._rng.choice(range(self.num_types)) for follower in self.followers_list}

    def type_probability(self, follower, type_value):
        """Marginal probability of a follower type.

        Settings with non-uniform type distributions override this. The default
        matches the uniform finite-type settings.
        """
        return 1.0 / self.num_types

    def type_profile_probability(self, types):
        probability = 1.0
        for follower in self.followers_list:
            probability *= self.type_probability(follower, types[follower])
        return probability


"""Abstraction of matrix game; used to pass/store all relevant information about the setting that we are working with. 
Not an RL environment."""
class MatrixGame(ABC):

    def __init__(self, matrix):
        self.payoff_matrix = matrix
        self.norm_factor = np.max(self.payoff_matrix)
        self.normalize_payoff_matrix = self.payoff_matrix / self.norm_factor

    def payoff(self, actions_tuple, agent_idx):
        return self.normalize_payoff_matrix[actions_tuple + (agent_idx,)]


""" The first two matrix games are from Bi-level Actor-Critic for Multi-agent Coordination (Zhang et al. 2020).
    The third one is from "On Stackelberg mixed strategies" (Conitzer 2017). """
class StackelbergMatrixGame(BaseStackelbergGame):

    def __init__(self, matrix):
        super().__init__("leader", ["follower_0"])
        self.matrix_game = MatrixGame(matrix)
        self.agent_to_number_map = self.create_agent_to_number_map()
        self.list_of_agents = [self.leader] + self.followers_list

    def create_agent_to_number_map(self):
        agent_to_number = {self.leader: 0}
        for i, follower in enumerate(self.followers_list, start=1):
            agent_to_number[follower] = i
        return agent_to_number

    def payoff(self, action_dict, agent):
        # Read agent_idx from the agent_to_number_map
        agent_idx = self.agent_to_number_map[agent]

        actions_tuple = tuple(
            action_dict[self.list_of_agents[i]] for i in range(self.num_agents)
        )
        return self.matrix_game.payoff(actions_tuple, agent_idx)

    def action_space(self, agent):
        agent_idx = self.agent_to_number_map[agent]
        return self.matrix_game.payoff_matrix.shape[agent_idx]


NORMAL_FORM_MATRICES = {
    "game_1": np.array(
        [
            [[15, 15], [10, 10], [0, 0]],
            [[10, 10], [10, 10], [0, 0]],
            [[0, 0], [0, 0], [30, 30]],
        ]
    ),
    "game_2": np.array(
        [
            [[20, 15], [0, 0], [0, 0]],
            [[30, 0], [10, 5], [0, 0]],
            [[0, 0], [0, 0], [5, 10]],
        ]
    ),
    "game_3": np.array([[[1, 1], [3, 0]], [[0, 0], [2, 1]]]),
    "game_4": np.array([[[3, 2], [1, 3]], [[2, 0], [0, 1]]]),
}


def get_normal_form_game(game_name):
    try:
        matrix = NORMAL_FORM_MATRICES[game_name]
    except KeyError:
        names = ", ".join(sorted(NORMAL_FORM_MATRICES))
        raise ValueError(f"Unknown normal-form game {game_name!r}. Available games: {names}")
    return StackelbergMatrixGame(matrix)


"""Matrix design game [Monderer and Tennenholtz, 2003]"""
class MatrixDesignGame(BaseStackelbergGame):

    def __init__(self):
        super().__init__("Game_designer", ["Agent_1", "Agent_2"])
        matrix = np.array(
                [
                    [[3, 3], [6, 4]],
                    [[4, 6], [2, 2]],
                ]
            )
        self.matrix_game = MatrixGame(matrix)
        self.agent_to_number_map = self.create_follower_to_number_map()

    def create_follower_to_number_map(self):
        agent_to_number = {}
        for i, follower in enumerate(self.followers_list):
            agent_to_number[follower] = i
        return agent_to_number

    def follower_payoff(self, action_dict, agent):
        # Read agent_idx from the agent_to_number_map
        agent_idx = self.agent_to_number_map[agent]

        actions_tuple = tuple(
            action_dict[follower] for follower in self.followers_list
        )
        return self.matrix_game.payoff(actions_tuple, agent_idx)

    def follower_action_space(self, agent):
        agent_idx = self.agent_to_number_map[agent]
        return self.matrix_game.payoff_matrix.shape[agent_idx]


class SimpleAllocationGame(BaseStackelbergGame):
    def __init__(self, num_messages):
        super().__init__("leader", ["follower_0"], num_types=3)
        self.num_messages = num_messages


@dataclass
class SPMEpisode:
    """Mutable state for one sequential posted-price mechanism episode."""

    types: dict
    valuations: dict
    mechanism_state: dict
    follower_actions: dict = None
    allocated_value: float = 0.0
    outcome: dict = field(default_factory=lambda: {
        "order": [],
        "price_vectors": [],
        "leader_actions": [],
        "mechanism_outcome": {},
    })
    utilities: dict = field(default_factory=dict)


class SequentialPostedPriceGame(BaseStackelbergGame):
    """Shared sequential posted-price mechanism core.

    This class is intentionally message-agnostic: it owns buyer behavior,
    mechanism state, posted-price history, welfare, and logging for both the
    standard SPM baseline and the message-SPM StackPOMDP environment.
    """

    def discrete_price_vector(self):
        """Candidate posted prices used by the discrete-price PPO baseline.

        Prices are midpoints between valuation support points, plus zero. This
        avoids asking PPO to learn knife-edge threshold prices exactly.
        """
        values = sorted({
            float(value)
            for follower_values in self.types_table.values()
            for type_values in follower_values
            for value in type_values
        })
        midpoints = [
            (values[i] + values[i + 1]) / 2
            for i in range(len(values) - 1)
        ]
        return np.asarray([0.0, *midpoints], dtype=np.float32)

    def new_episode(self, types):
        return SPMEpisode(
            types=dict(types),
            valuations=self.get_vals_from_types(types),
            mechanism_state=self.initial_mechanism_state(),
            utilities={follower: 0 for follower in self.followers_list},
        )

    def record_follower_actions(self, episode, actions_dict):
        if episode.follower_actions is None:
            episode.follower_actions = {
                follower: actions_dict[follower]
                for follower in self.followers_list
            }

    def parse_leader_action(self, action):
        return {
            "agent_scores": np.asarray(action[:len(self.followers_list)]),
            "prices": np.asarray(action[len(self.followers_list):]),
        }

    def step_episode(self, episode, actions_dict):
        leader_action = self.parse_leader_action(actions_dict[self.leader])
        self.record_follower_actions(episode, actions_dict)
        episode.outcome["leader_actions"].append(
            np.asarray(actions_dict[self.leader]).tolist()
        )

        agent_idx, agent = self.select_agent(
            leader_action["agent_scores"],
            episode.mechanism_state,
        )
        self.visit_agent(
            episode,
            agent_idx,
            agent,
            leader_action["prices"],
        )

        done = self.mechanism_done(episode.mechanism_state)
        reward = 0
        info = {}
        if done:
            result = self.allocation_result(episode.valuations, episode.allocated_value)
            reward = result["reward"]
            episode.utilities[self.leader] = reward
            info = {
                "utilities": episode.utilities,
                "surplus": reward,
                "efficiency": result["allocative_efficiency"],
                "max_social_welfare": result["efficient_value"],
                "reward_generated": True,
            }

        return {
            "observation": self.mechanism_state_vector(episode.mechanism_state),
            "reward": reward,
            "done": done,
            "info": info,
        }

    def describe_outcome(self, episode):
        row = []
        for follower, action in episode.follower_actions.items():
            row.append(f"B_{follower}: {action}")

        for i, agent in enumerate(episode.outcome["order"]):
            row.append(f"O_{i + 1}: {agent}")
            for item_idx, price in enumerate(episode.outcome["price_vectors"][i]):
                row.append(f"P_{i + 1},{item_idx}: {price:.2f}")

        return "| " + " | ".join(row) + " |"

    def describe_policy_outcomes(self, env, policy):
        types = {follower: 0 for follower in self.followers_list}
        outcome_strings = []

        for bid1 in range(self.num_messages):
            for bid2 in range(self.num_messages):
                bids = {self.followers_list[0]: bid1, self.followers_list[1]: bid2}
                env.run_episode(policy, types, bids)
                outcome_strings.append(self.describe_outcome(env.mechanism_episode))

        return "\n".join(outcome_strings)

    def log_episode(self, episode, logger, efficiency, max_social_welfare):
        logger.record("efficiency", "%.5f" % efficiency)
        logger.record("overall_value", "%.5f" % episode.allocated_value)
        logger.record("opt", "%.5f" % max_social_welfare)

        for i, agent in enumerate(episode.outcome["order"]):
            logger.record("order_%i" % i, agent)
            for item_idx, price in enumerate(episode.outcome["price_vectors"][i]):
                logger.record(f"price_{i}_{item_idx}", price)

        for follower in self.followers_list:
            logger.record("bids_" + follower, episode.follower_actions[follower])
            for item_idx, value in enumerate(episode.valuations[follower]):
                logger.record(f"value_{follower}_{item_idx}", value)

    def initial_mechanism_state(self):
        return {
            "agents_remaining": {follower: 1 for follower in self.followers_list},
            "items_remaining": np.asarray(self.units_per_item, dtype=np.float32),
            "allocation_available": np.ones(
                (len(self.followers_list), self.num_diff_items),
                dtype=np.float32,
            ),
            "posted_prices": np.zeros(
                (len(self.followers_list), self.num_diff_items),
                dtype=np.float32,
            ),
        }

    def mechanism_state_vector(self, mechanism_state):
        return np.concatenate((
            np.asarray(
                [mechanism_state["agents_remaining"][follower] for follower in self.followers_list],
                dtype=np.float32,
            ),
            mechanism_state["items_remaining"],
            mechanism_state["allocation_available"].reshape(-1),
            mechanism_state["posted_prices"].reshape(-1),
        ))

    def mechanism_state_bounds(self):
        low = np.zeros_like(self.mechanism_state_vector(self.initial_mechanism_state()))
        high = np.concatenate((
            np.ones(len(self.followers_list)),
            np.asarray(self.units_per_item),
            np.ones(2 * self.num_diff_items * len(self.followers_list)),
        )).astype(np.float32)
        return low, high

    def buyer_choice(self, valuation, prices, items_remaining):
        utility = np.asarray([
            valuation[i] - prices[i] if items_remaining[i] else -np.inf
            for i in range(self.num_diff_items)
        ])
        choice = int(np.argmax(utility))
        buyer_utility = max(float(utility[choice]), 0)
        item_idx = choice if utility[choice] > 0 else None
        return {
            "item": item_idx,
            "utility": buyer_utility,
        }

    def select_agent(self, agent_scores, mechanism_state):
        available_scores = [
            agent_scores[i]
            if mechanism_state["agents_remaining"][follower]
            else -np.inf
            for i, follower in enumerate(self.followers_list)
        ]
        agent_idx = int(np.argmax(available_scores))
        return agent_idx, self.followers_list[agent_idx]

    def visit_agent(self, episode, agent_idx, agent, prices):
        mechanism_state = episode.mechanism_state
        valuation = episode.valuations[agent]
        purchase = self.buyer_choice(valuation, prices, mechanism_state["items_remaining"])
        mechanism_state["agents_remaining"][agent] = 0
        episode.utilities[agent] = purchase["utility"]
        episode.outcome["order"].append(agent)
        episode.outcome["price_vectors"].append(prices.tolist())

        if purchase["item"] is not None:
            item_idx = purchase["item"]
            purchase["value"] = valuation[item_idx]
            purchase["payment"] = prices[item_idx]
            episode.allocated_value += purchase["value"]
            self.record_purchase(mechanism_state, agent_idx, item_idx)
            episode.outcome["mechanism_outcome"][agent] = {
                "allocation": item_idx,
                "payment": purchase["payment"],
            }

        self.record_posted_prices(mechanism_state, agent_idx, prices)
        return purchase

    def mechanism_done(self, mechanism_state):
        # Once inventory is exhausted the economic outcome is already fixed,
        # but ending then makes the horizon depend on the policy. Visiting any
        # remaining buyers with no inventory is outcome-neutral and gives PPO
        # the fixed horizon required for episode-aligned rollout buffers.
        return sum(mechanism_state["agents_remaining"].values()) <= 0

    def allocation_result(self, valuations, allocated_value):
        efficient_value = self.efficient_welfare(valuations)
        allocative_efficiency = 1.0 if efficient_value == 0 else allocated_value / efficient_value
        return {
            "efficient_value": efficient_value,
            "allocative_efficiency": allocative_efficiency,
            "reward": allocated_value - efficient_value,
        }

    def record_purchase(self, mechanism_state, agent_idx, item_idx):
        mechanism_state["items_remaining"][item_idx] -= 1
        mechanism_state["allocation_available"][agent_idx, item_idx] -= 1

    def record_posted_prices(self, mechanism_state, agent_idx, prices):
        """Store the price vector quoted to the visited agent.

        This matches the state information used in the original SPM code's
        continuous price-matrix observation: future leader decisions can
        condition on which prices have already been quoted.
        """
        mechanism_state["posted_prices"][agent_idx] = np.asarray(prices, dtype=np.float32)

    def efficient_welfare(self, valuations):
        """Efficient additive welfare benchmark for this MSPM setting.

        For the paper MSPMs, there is one item and this is simply the highest
        realized value. The per-item form also handles multiple independent
        item capacities without solving a combinatorial allocation problem.
        """
        welfare = 0.0
        for item_idx, units in enumerate(self.units_per_item):
            values = sorted(
                [value[item_idx] for value in valuations.values()],
                reverse=True,
            )
            welfare += sum(values[:int(units)])
        return welfare


# Backwards compatibility for older experiment scripts/imports.
MessageSPMSetting = SequentialPostedPriceGame


"""Two SPM/MSPM settings. The first was introduced by Agrawal et al. (2020); the second generalizes it."""
class PISetting(SequentialPostedPriceGame):
    def __init__(
            self,
            num_messages=2,
    ):
        super().__init__("spm_designer", ["A0", "A1"])
        self.EPSILON = 0.2
        self.types_table = {follower: values for follower, values in zip(self.followers_list,
                                                                         [[[0.5 / (1 / (2 * self.EPSILON))], [1]],
                                                                          [[0], [1 / (1 / (2 * self.EPSILON))]]])}
        self.num_diff_items = 1
        self.units_per_item = [1]
        self.num_messages = num_messages
        self.num_types = 2

    def sample_types(self):
        return {follower: self._rng.choices(range(len(self.types_table[follower])), [1 - self.EPSILON, self.EPSILON] if i == 0 else [0.5, 0.5])[0] for i, follower in enumerate(self.followers_list)}

    def type_probability(self, follower, type_value):
        if follower == self.followers_list[0]:
            return [1 - self.EPSILON, self.EPSILON][type_value]
        return [0.5, 0.5][type_value]

    def get_vals_from_types(self, types):
        return {follower: self.types_table[follower][type] for follower, type in types.items()}

class MSGSpaceSetting(SequentialPostedPriceGame):
    def __init__(
            self,
            num_types=5,
            num_messages=2,
    ):
        super().__init__("spm_designer", ["A0", "A1"])
        self.types_table = {follower: [[value] for value in np.linspace(0, 1, num_types).tolist()] for follower in
                            self.followers_list}
        self.num_diff_items = 1
        self.units_per_item = [1]
        self.num_messages = num_messages
        self.num_types = num_types

    def get_vals_from_types(self, types):
        return {follower: self.types_table[follower][type] for follower, type in types.items()}


def get_mspm_setting(setting_name, num_types, num_messages):
    if setting_name == "PI":
        if num_types != 2:
            import warnings

            warnings.warn("For 'PI' setting, num_types should be 2. Overriding num_types to 2.")
        return PISetting(num_messages=num_messages)
    if setting_name == "MSGSpace":
        return MSGSpaceSetting(num_types=num_types, num_messages=num_messages)
    raise ValueError("SPM setting not recognized. Available options: PI, MSGSpace")
