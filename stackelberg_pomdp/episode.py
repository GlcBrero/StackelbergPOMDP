"""Episode diagnostics and reward accumulation, separate from phase execution."""
from copy import deepcopy

import numpy as np


class EpisodeLogger:
    """Record internal events, including queries that never leave the executor.

    This recorder builds Gym's info dictionary; it does not write files, change
    actions or compute the reward returned to the learner.
    """

    def __init__(self):
        self.episode_count = 0
        self.last_episode = None
        self.reset()

    def reset(self):
        self.info = {}
        self.counts = {"policy_queries": 0, "cache_hits": 0,
                       "response_policy_queries": 0, "reward_policy_queries": 0,
                       "underlying_steps": 0}
        self.episode_return = 0.0

    def response(self, result):
        self.info.update(deepcopy(result.info))
        if result.certificate is not None:
            self.info["response_certificate"] = deepcopy(result.certificate)

    def query(self, phase, *, cached):
        if cached:
            self.counts["cache_hits"] += 1
        else:
            self.counts["policy_queries"] += 1
            self.counts[phase + "_policy_queries"] += 1

    def reward(self, event):
        self.episode_return += float(event.value)
        self.counts["underlying_steps"] += event.elapsed_steps
        self.info.update(deepcopy(event.info))

    def finish(self, info, *, commitment_entries, transition_info):
        self.info.update(deepcopy(info))
        self.episode_count += 1
        result = {**self.info, **self.counts,
                  "commitment_entries": commitment_entries,
                  "episode_return": self.episode_return,
                  **transition_info}
        self.last_episode = deepcopy(result)
        return result


class RewardAccumulator:
    """Rewards and elapsed time between consecutive external action requests.

    With gamma=1, used by the paper leaders, this just sums rewards. For a
    discounted extension, each Reward's duration advances time even when
    intervening queries are answered internally from the commitment.
    The executor validates gamma once when it is constructed.
    """

    def __init__(self, gamma):
        self.gamma = gamma
        self.total = 0.0
        self.discount = 1.0
        self.elapsed_steps = 0

    def add(self, event):
        if (not np.isfinite(event.value) or event.elapsed_steps < 0
                or int(event.elapsed_steps) != event.elapsed_steps):
            raise ValueError("Rewards must be finite and durations nonnegative integers")
        self.total += self.discount * float(event.value)
        self.discount *= self.gamma ** event.elapsed_steps
        self.elapsed_steps += event.elapsed_steps

    @property
    def info(self):
        """Continuation factors needed by the rollout collector."""
        return {"continuation_discount": self.discount,
                "elapsed_steps": self.elapsed_steps}
