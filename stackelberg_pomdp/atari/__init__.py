"""Native Gym/SB3 components for the Atari replication.

The legacy RLlib replication remains under :mod:`stackelberg_pomdp.atari_models`
and :mod:`stackelberg_pomdp.gym_envs.envs.atari_envs`.  This package is the
framework-independent environment path used by the native SB3 experiments.
"""

from stackelberg_pomdp.atari.factory import AtariBuyerEnvConfig, make_atari_buyer_env

__all__ = ["AtariBuyerEnvConfig", "make_atari_buyer_env"]
