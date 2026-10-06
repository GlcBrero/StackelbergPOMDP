"""Query mechanism payoffs, then run the native certified MW response.

The initial sweep and certificate fill can request new leader actions, so
Python executes them through game.play_profile. Once the payoff table is
complete, mw.cpp performs the remaining updates and candidate certification.
Python supplies the same type draws as before; bcce.cpp retains the exact
certificate arithmetic. No new leader queries are needed by later sweeps.
"""
from itertools import product
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
from stackelberg_pomdp import ResponsePhase, ResponseReady
from ._support._native import NativeMW, bcce_gap
from ._support._critic import encode_critic_state


@dataclass(frozen=True)
class MessageProfile:
    """Exact joint message distribution conditional on the private type profile.

    The evaluator integrates this distribution without destroying correlation.
    It does not replace it by independently sampled buyer marginals.
    """
    joint: np.ndarray

    def probability(self, types, messages):
        return float(self.joint[(*types, *messages)])


def joint_strategy(weights):
    """Product strategies conditional on both private types."""
    return np.einsum("ac,bd->abcd", weights[0], weights[1])


class MSPMResponse(ResponsePhase):
    """MW with first certification of mixed, deterministic or empirical play."""

    def __init__(self, *, epsilon=.1, tolerance=.05, fixed_seed=0,
                 max_extra_updates=10000):
        if epsilon <= 0 or tolerance < 0 or max_extra_updates < 0:
            raise ValueError("Invalid MW update or certification parameters")
        self.epsilon, self.tolerance = float(epsilon), float(tolerance)
        self.fixed_seed = fixed_seed
        self.max_extra_updates = int(max_extra_updates)

    def critic_observation(self, game, commitment, phase):
        return encode_critic_state(self.state, game.types, commitment, phase)

    def run(self, game):
        t, m = game.num_types, game.num_messages
        if self.fixed_seed is not None:
            game.seed(self.fixed_seed)
        self.state = state = SimpleNamespace(num_messages=m)
        state.weights = np.full((2, t, m), 1 / m)
        state.empirical_joint = np.zeros((t, t, m, m))
        state.payoff_state = np.zeros((t, t, m, m, 4))
        state.progress = np.zeros(2)
        state.deviations = np.zeros((2, m))
        payoff_cache = {}
        simulated_steps = 0

        def outcome(types, messages):
            nonlocal simulated_steps
            key = (*types, *messages)
            if key not in payoff_cache:
                result = yield from game.play_profile(types, messages)
                simulated_steps += result.transitions
                payoff_cache[key] = result
                state.payoff_state[key] = np.r_[result.utilities, result.welfare, 1.]
            return payoff_cache[key]

        # Preserve the initial query order and the critic state at each query.
        types = game.sample_types()
        for profile_index, messages in enumerate(product(range(m), repeat=2)):
            state.progress[1] = profile_index
            result = yield from outcome(types, messages)
            a, b = messages
            state.deviations[0, a] += result.utilities[0] * state.weights[1, types[1], b]
            state.deviations[1, b] += result.utilities[1] * state.weights[0, types[0], a]
        for buyer in range(2):
            row = state.weights[buyer, types[buyer]]
            row *= (1 + self.epsilon) ** state.deviations[buyer]
            row /= row.sum()
        state.empirical_joint += joint_strategy(state.weights)
        state.progress[:] = (1, 0)

        # Exact certification requires every counterfactual payoff. Ask only
        # through the common commitment and retain the historical fill order.
        for first_type in range(t):
            for second_type in range(t):
                for second_message in range(m):
                    for first_message in range(m):
                        yield from outcome((first_type, second_type),
                                           (first_message, second_message))

        # All subsequent MW sweeps use this fixed table. Native code owns
        # weights, strategy history, empirical averages and certification.
        with NativeMW(
            state.weights, state.payoff_state[..., :2], game.type_probabilities,
            state.empirical_joint, state.deviations,
            epsilon=self.epsilon, tolerance=self.tolerance,
        ) as learner:
            learner.certify()
            while not learner.certified and learner.updates < self.max_extra_updates + 1:
                learner.advance(game.sample_types())
            result = learner.result()

        state.weights[:] = result["weights"]
        state.empirical_joint[:] = result["empirical_joint"]
        state.deviations[:] = result["deviations"]
        state.progress[:] = (result["updates"], 0)
        name, gaps = result["candidate"], result["gaps"]
        if name is None:
            raise RuntimeError(f"MW exhausted its certification budget; candidate gaps: {gaps}")
        gap = gaps[name]
        certificate = {
            "gap": gap, "tolerance": self.tolerance, "updates": result["updates"],
            "candidate": name, "candidate_gaps": gaps, "method": "exact",
        }
        return ResponseReady(MessageProfile(result["joint"]), certificate, {
            "response_candidate": name, "response_bcce_gap": gap,
            "response_bcce_certified": True, "response_updates": result["updates"],
            "response_simulated_steps": simulated_steps,
        })
