"""Evaluate consumer surplus on each frozen seller policy's recurrent cycle."""
import numpy as np
from stackelberg_pomdp import LeaderQuery, RewardPhase, Reward, RewardDone

from ._native import cycle


class PlatformReward(RewardPhase):
    """Cycle-average variant with a response-dependent starting price state.

    This preserves the paper experiment. It is distinct from independently
    initialized finite-horizon gameplay in the basic Stackelberg construction.
    """

    def __init__(self, response):
        self.response = response

    def run(self, game, response_result):
        scores = []
        policies = response_result.followers
        initial_prices = response_result.evaluation_context
        if initial_prices is None:
            raise ValueError("Platform cycle evaluation requires response terminal prices")
        first = response_result.info["response"]
        expected_scores = {result["cost"]: result["frozen_cs"]
                           for result in (first, first["paired_response"])}
        for cost, seller_policy in policies.tables.items():
            _, recurrent_states = cycle(
                seller_policy, initial_prices[cost]
            )
            cycle_surplus = []

            for state in recurrent_states:
                observations = game.reset(cost_index=game.costs.index(cost), price_state=state)
                # Frozen sellers choose prices from their own observed histories.
                actions = policies.act({seller: (observations[seller],) for seller in game.sellers})
                observations, _, _, _ = game.step({game.leader: 0, **actions})
                leader_view = game.leader_observation((observations[game.leader],))
                action = yield LeaderQuery(leader_view)
                price_pair = tuple(game.current_prices if game.ordered else sorted(game.current_prices))
                query_index = game.pairs.index(price_pair)
                self.response.record_answer(query_index, action)
                _, rewards, _, _ = game.step({game.leader: int(action), **actions})
                cycle_surplus.append(rewards[game.leader])

            score = float(np.mean(cycle_surplus))
            if expected_scores[cost] is not None and not np.isclose(score, expected_scores[cost], atol=1e-12, rtol=0):
                raise RuntimeError("Frozen cycle reward disagrees with seller response record")
            for result in (first, first["paired_response"]):
                if result["cost"] == cost:
                    result["frozen_cs"] = score
            scores.append(score)

        mean_surplus = float(np.mean(scores))
        yield Reward(mean_surplus)
        return RewardDone(game.observation(0), {
            "query_actions": self.response.state.leader_actions.tolist(),
            "frozen_cs": mean_surplus,
            "response": first,
        })
