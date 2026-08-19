"""Safe loading and deterministic control of frozen Atari policies."""

from pathlib import Path

import numpy as np


def load_frozen_atari_model(
        checkpoint,
        *,
        device="cpu",
        model_factory=None,
        description="Atari checkpoint",
):
    """Load one SB3 policy in deterministic, permanently frozen mode."""

    if model_factory is None:
        if checkpoint is None:
            raise ValueError(f"{description} is required")
        from stable_baselines3 import PPO

        path = Path(checkpoint).expanduser()
        if not path.is_file() and Path(f"{path}.zip").is_file():
            path = Path(f"{path}.zip")
        if not path.is_file():
            raise FileNotFoundError(f"{description} does not exist: {path}")
        model = PPO.load(str(path), device=device)
    else:
        model = model_factory(checkpoint, device=device)

    policy = getattr(model, "policy", None)
    if policy is None:
        raise TypeError(f"{description} does not contain an SB3 policy")
    policy.set_training_mode(False)
    for parameter in policy.parameters():
        parameter.requires_grad = False
    return model


class FrozenAtariPolicyController:
    """Use only the deterministic game action from a frozen Atari policy."""

    def __init__(self, checkpoint, *, device="cpu"):
        self.model = load_frozen_atari_model(
            checkpoint,
            device=device,
            description="Atari checkpoint",
        )

    def __call__(self, values):
        action, _ = self.model.predict(values, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (2,):
            raise RuntimeError("clean Atari controller must return two actions")
        return int(np.rint(action[0]))


__all__ = ["FrozenAtariPolicyController", "load_frozen_atari_model"]
