"""Query the leader at each state, then return a conditioned follower strategy."""
import numpy as np
from stackelberg_pomdp import LeaderQuery, PolicyProfile, ResponsePhase, ResponseReady
from experiments.ablations._support.game import FOLLOWER
from ._support.follower import FollowerPolicy, load_follower


class MetaResponse(ResponsePhase):
    def __init__(self, spec, *, query_visibility="observed", seed=None, **checkpoint_options):
        if spec.episode_length <= 1 or spec.memory_mode == "none":
            raise ValueError("matrix E2 requires a repeated memory game")
        if query_visibility not in ("observed", "hidden"):
            raise ValueError("query_visibility must be 'observed' or 'hidden'")
        self.query_visibility = query_visibility
        self.model = load_follower(spec, **checkpoint_options)

    def run(self, game):
        game.reset()
        commitment = []
        for state in range(game.game_spec.num_leader_states):
            action = yield LeaderQuery(game.observation(state))
            action = int(np.asarray(action).reshape(-1)[0])
            if not game.action_space.contains(action):
                raise ValueError("Invalid leader action")
            commitment.append(action)
        strategy = FollowerPolicy(self.model, game.game_spec, tuple(commitment))
        return ResponseReady(PolicyProfile({FOLLOWER: strategy}))

    def max_episode_transitions(self, game):
        return game.horizon + (game.game_spec.num_leader_states if self.query_visibility == "observed" else 0)

    def max_executed_transitions(self, game):
        return game.horizon + game.game_spec.num_leader_states
