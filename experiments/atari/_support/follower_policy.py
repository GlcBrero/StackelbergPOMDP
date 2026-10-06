"""A frozen follower strategy conditioned on the response's five answers."""
from copy import deepcopy
from dataclasses import dataclass
import numpy as np
from .protocol import ACTOR_STATE, OPPONENT_COMMITMENT_SLICE


@dataclass(frozen=True)
class FollowerPolicy:
    model: object
    commitment: tuple

    def act(self, history):
        values = deepcopy(history[-1])
        values[ACTOR_STATE][OPPONENT_COMMITMENT_SLICE] = self.commitment
        action, _ = self.model.predict(values, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (2,):
            raise RuntimeError("meta-follower must return two action coordinates")
        return action
