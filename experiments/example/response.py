"""Query once, then return a follower policy that plays the observed action."""
from stackelberg_pomdp import LeaderQuery, PolicyProfile, ResponsePhase, ResponseReady


class FollowerPolicy:
    """A strategy for this one-state game: always play the chosen action."""

    def __init__(self, action):
        self.action = int(action)

    def act(self, observation_history):
        return self.action


class CopyLeaderResponse(ResponsePhase):
    """Copy one leader sample; this is an exact best reply in lazy mode."""

    def run(self, game):
        observations = game.reset()
        leader_history = (observations[game.leader],)
        leader_action = yield LeaderQuery(game.leader_observation(leader_history))
        strategy = FollowerPolicy(leader_action)
        return ResponseReady(followers=PolicyProfile({"follower": strategy}))
