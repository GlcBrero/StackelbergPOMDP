"""Load the frozen meta-follower and bind a fixed leader commitment to it."""
from dataclasses import dataclass
import numpy as np
from experiments.ablations._support.game import (
    encode_meta_follower_observation, validate_response_checkpoint_contract, load_response_model,
)


def load_follower(spec, *, response_model=None, response_checkpoint=None,
                  response_algorithm="REINFORCE", response_contract_path=None, device="cpu"):
    if response_model is not None:
        return response_model
    if response_checkpoint is None:
        raise ValueError("response_checkpoint or response_model is required")
    validate_response_checkpoint_contract(spec, response_checkpoint,
        algorithm=response_algorithm, contract_path=response_contract_path)
    return load_response_model(response_checkpoint, response_algorithm, device=device)


@dataclass(frozen=True)
class FollowerPolicy:
    model: object
    spec: object
    commitment: tuple

    def act(self, history):
        encoded = encode_meta_follower_observation(self.spec, history[-1], self.commitment)
        action, _ = self.model.predict(encoded, deterministic=True)
        action = int(np.asarray(action).reshape(-1)[0])
        if not 0 <= action < self.spec.num_follower_actions:
            raise RuntimeError("frozen meta-follower returned an invalid action")
        return action
