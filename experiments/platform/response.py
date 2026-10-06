"""Learn seller strategies, querying the platform only at prices they encounter."""
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
from stackelberg_pomdp import LeaderQuery, ResponsePhase, ResponseReady
from ._support._critic import encode_critic_state
from ._support.q_learning import SellerQLearning, AdmissionRuleQLearning, make_seller_q_learning


@dataclass(frozen=True)
class SellerPolicies:
    """One frozen price strategy per seller and private cost condition."""
    tables: dict
    costs: tuple

    def act(self, histories):
        actions = {}
        for index, seller in enumerate(("seller_0", "seller_1")):
            observed = histories[seller][-1]
            table = self.tables[self.costs[int(observed["cost"])]]
            m = int(np.sqrt(table.shape[1]))
            first, second = map(int, observed["prices"])
            actions[seller] = int(table[index, first * m + second])
        return actions


class QLearningResponse(ResponsePhase):
    """Pause native seller learning whenever a new platform action is needed.

    Native Q-learning uses epsilon = exp(-beta * round), learning rate alpha,
    discount delta, and the configured round/stability stopping limits.
    The same seed is used for both private cost conditions.
    """

    def __init__(self, seller_learning, seeds, offset=0, seed=0):
        self.seller_learning, self.seeds, self.episode = seller_learning, seeds, offset
        self.seed(seed)
        self.state = SimpleNamespace(cost_index=0, query_index=0, seller_response=None,
            leader_actions=np.full(len(seller_learning.pairs), -1, dtype=np.int32))

    def seed(self, seed=None):
        from gym.utils import seeding
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def critic_observation(self, game, commitment, phase):
        return encode_critic_state(self, game, phase)

    def record_answer(self, index, action):
        previous = self.state.leader_actions[index]
        if previous >= 0 and previous != int(action):
            raise ValueError('A lazy episode cannot change its committed action')
        self.state.leader_actions[index] = int(action)
        self.state.query_index += int(previous < 0)

    def run(self, game):
        cost_index = int(self.np_random.randint(len(self.seller_learning.costs)))
        self.np_random.randint(1_000_000, 2_000_000_000)  # Preserve the recorded random stream.
        self.state = state = SimpleNamespace(cost_index=cost_index, query_index=0,
            seller_response=None, leader_actions=np.full(game.n_queries, -1, dtype=np.int32))
        seller_seed = self.seeds[self.episode]
        self.episode += 1
        responses = []
        for current_cost in (cost_index, 1 - cost_index):
            state.cost_index = current_cost
            with self.seller_learning.start_response(state.leader_actions, current_cost, seller_seed) as learning:
                index = learning.next_query()
                while index is not None:
                    action = yield LeaderQuery(game.observation(index))
                    self.record_answer(index, action)
                    index = learning.next_query()
                responses.append(learning.result())
        first, second = responses
        first["paired_response"] = second
        state.seller_response = first
        policies = SellerPolicies({result["cost"]: np.asarray(result["seller_policy"])
                                   for result in (first, second)}, game.costs)
        initial_prices = {result["cost"]: result["training_joint"] for result in (first, second)}
        return ResponseReady(policies, evaluation_context=initial_prices,
            info={"response_protocol": "on_demand_queries", "response": first})
