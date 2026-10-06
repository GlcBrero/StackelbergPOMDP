"""One-shot two-player coordination game; the response algorithm is separate from these game rules."""
from collections import OrderedDict
from gym import spaces
from stackelberg_pomdp import POMG
from experiments.ablations._support.game import LEADER, FOLLOWER, RepeatedMatrixGame, get_matrix_game


class Game(POMG):
    leader = LEADER

    def __init__(self, spec=None, *, phase_observable=None):
        self.game_spec = spec or get_matrix_game("coordination_zero_miscoordination")
        self.rules = RepeatedMatrixGame(self.game_spec)
        self.horizon = int(self.game_spec.episode_length)
        self.phase_observable = phase_observable
        self.action_spaces = {LEADER: spaces.Discrete(self.game_spec.num_leader_actions),
                              FOLLOWER: spaces.Discrete(self.game_spec.num_follower_actions)}
        self.observation_spaces = {LEADER: spaces.Discrete(self.game_spec.num_leader_states),
                                   FOLLOWER: spaces.Discrete(self.game_spec.num_follower_states)}
        leader_fields = OrderedDict(base_environment=self.observation_spaces[LEADER])
        if phase_observable is not None:
            leader_fields["base:is_reward_phase"] = spaces.Discrete(2)
        self.leader_observation_space = spaces.Dict(leader_fields)

    def reset(self):
        return self.rules.reset()

    def step(self, actions):
        if set(actions) != set(self.action_spaces):
            raise ValueError("Supply both leader and follower actions")
        return self.rules.step(actions[LEADER], actions[FOLLOWER])

    def leader_observation(self, history):
        return self.observation(history[-1], reward_phase=True)

    def observation(self, state, *, reward_phase=False):
        result = OrderedDict(base_environment=int(state))
        if self.phase_observable is not None:
            # This explicit information leak is the controlled ablation.
            result["base:is_reward_phase"] = int(bool(self.phase_observable) and reward_phase)
        return result
