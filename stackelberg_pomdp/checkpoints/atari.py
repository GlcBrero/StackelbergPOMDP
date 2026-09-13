"""Loading and actor transfer for SB3 Atari checkpoints."""

import hashlib
from pathlib import Path

import numpy as np
import torch as th

from stackelberg_pomdp.policies.atari.meta_seller import (
    META_SELLER_TRANSFER_MODULES,
    is_meta_seller_architecture,
)


def resolve_atari_checkpoint(checkpoint, *, description="Atari checkpoint"):
    """Resolve an SB3 archive with or without its ``.zip`` suffix."""

    if checkpoint is None:
        raise ValueError(f"{description} is required")
    path = Path(checkpoint).expanduser()
    if not path.is_file() and Path(f"{path}.zip").is_file():
        path = Path(f"{path}.zip")
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    return path


def load_frozen_atari_model(
        checkpoint,
        *,
        device="cpu",
        model_factory=None,
        description="Atari checkpoint",
):
    """Load one SB3 policy in deterministic, permanently frozen mode."""

    if model_factory is None:
        from stable_baselines3 import PPO

        path = resolve_atari_checkpoint(
            checkpoint,
            description=description,
        )
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


def transfer_atari_actor(
        destination,
        checkpoint,
        *,
        include_economic=True,
        device="cpu",
):
    """Copy actor modules from one unified Atari policy, never its critic."""

    from stable_baselines3 import PPO
    from stackelberg_pomdp.policies.atari.policy import StackPOMDPAtariPolicy

    path = resolve_atari_checkpoint(checkpoint)
    source_model = PPO.load(str(path), device=device)
    source = source_model.policy
    if not isinstance(source, StackPOMDPAtariPolicy):
        raise TypeError(
            "clean stages require a StackPOMDPAtariPolicy checkpoint; "
            "legacy Atari checkpoints are intentionally incompatible"
        )

    modules = ["features_extractor", "game_action_net"]
    if include_economic:
        if source.economic_architecture != destination.economic_architecture:
            raise ValueError(
                "economic-head transfer requires matching seller "
                "economic parameterizations"
            )
        if is_meta_seller_architecture(destination.economic_architecture):
            modules.extend(META_SELLER_TRANSFER_MODULES)
        else:
            modules.append("economic_head")
    for name in modules:
        getattr(destination, name).load_state_dict(
            getattr(source, name).state_dict(),
            strict=True,
        )
    if include_economic and is_meta_seller_architecture(
            destination.economic_architecture
    ):
        with th.no_grad():
            destination.economic_current_slope.copy_(
                source.economic_current_slope
            )

    result = {
        "checkpoint": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "modules": tuple(modules),
        "critic_transferred": False,
        "source_economic_role": source.economic_role,
        "source_economic_input_mode": source.economic_input_mode,
        "source_pretrained_lr_scale": source.pretrained_lr_scale,
    }
    if source.economic_architecture is not None:
        result["source_economic_architecture"] = source.economic_architecture
    del source_model
    return result


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


def module_parameter_sha256(module):
    """Hash a module state without serialization or global RNG effects."""

    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def gameplay_actor_sha256(policy):
    """Hash each frozen gameplay component in stable module order."""

    names = (
        "features_extractor.visual",
        "features_extractor.state_encoder",
        "game_action_net",
    )
    return {
        name: module_parameter_sha256(module)
        for name, module in zip(names, policy.gameplay_actor_modules())
    }


__all__ = [
    "FrozenAtariPolicyController",
    "gameplay_actor_sha256",
    "load_frozen_atari_model",
    "module_parameter_sha256",
    "resolve_atari_checkpoint",
    "transfer_atari_actor",
]
