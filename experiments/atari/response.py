"""Five leader queries select a frozen, commitment-conditioned follower strategy."""
from stackelberg_pomdp import LeaderQuery, PolicyProfile, ResponsePhase, ResponseReady
from ._support.protocol import NUM_TRADE_EVENTS
from ._support.follower_policy import FollowerPolicy
from ._support.wrappers.meta_follower import AtariMetaFollowerWrapper, record_transition


class AtariResponse(ResponsePhase):
    """The pretrained response model chooses behavior; no emulator runs here."""

    def __init__(self, game, **checkpoint_options):
        self.runtime = AtariMetaFollowerWrapper(game, **checkpoint_options)

    def run(self, game):
        runtime = self.runtime
        values = runtime.reset()
        game.transition_records = []
        for event in range(NUM_TRADE_EVENTS):
            action = yield LeaderQuery(values, key=("trade", event))
            previous = values
            values, reward, done, info = runtime._query_step(action)
            record_transition(game, previous, action, reward, done, info, reward_phase=False)
        if not runtime.request_response_completion():
            raise RuntimeError("Frozen meta-response did not complete its five queries")
        strategy = FollowerPolicy(runtime.response.model,
                                  tuple(runtime.response.economic_commitment))
        followers = PolicyProfile({game.follower_role: strategy})
        runtime.followers = followers
        return ResponseReady(followers, info=runtime.response_phase_info())
