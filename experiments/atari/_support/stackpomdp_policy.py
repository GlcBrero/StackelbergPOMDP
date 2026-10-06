"""Compatibility import for checkpoints created before the package cleanup.

New code should import :mod:`experiments.atari._support.policy`.
Published SB3 archives retain this historical module path in their serialized
policy metadata, so removing the shim would make those artifacts unloadable.
"""

from experiments.atari._support.policy import *  # noqa: F401,F403
from experiments.atari._support.policy import __all__
