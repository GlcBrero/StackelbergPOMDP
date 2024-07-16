from abc import ABC, abstractmethod
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


"""Two mSPMs environments. The first one was originally introduced by  Agrawal et al. [2020]. The second is a generalization."""
class PISetting(BaseStackelbergGame):
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

    def get_discrete_price_vec(self):
        all_types_vec = sorted(set(value for value in self.types_table.values()))
        midpoints = [(all_types_vec[i] + all_types_vec[i + 1]) / 2 for i in range(len(all_types_vec) - 1)]
        return np.concatenate(([0], midpoints))

    def sample_types(self):
        return {follower: self._rng.choices(range(len(self.types_table[follower])), [1 - self.EPSILON, self.EPSILON] if i == 0 else [0.5, 0.5])[0] for i, follower in enumerate(self.followers_list)}

    def get_vals_from_types(self, types):
        return {follower: self.types_table[follower][type] for follower, type in types.items()}

class MSGSpaceSetting(BaseStackelbergGame):
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

    def get_discrete_price_vec(self):
        all_types_vec = sorted(set(value for value in self.types_table.values()))
        midpoints = [(all_types_vec[i] + all_types_vec[i + 1]) / 2 for i in range(len(all_types_vec) - 1)]
        return np.concatenate(([0], midpoints))

    def get_vals_from_types(self, types):
        return {follower: self.types_table[follower][type] for follower, type in types.items()}
