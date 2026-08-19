"""Compatibility import for checkpoints created before the package cleanup.

New code should import :mod:`stackelberg_pomdp.policies.atari.composite`.
Published SB3 archives retain this historical module path in their serialized
policy metadata, so removing the shim would make those artifacts unloadable.
"""

from stackelberg_pomdp.policies.atari.composite import *  # noqa: F401,F403
from stackelberg_pomdp.policies.atari.composite import __all__
