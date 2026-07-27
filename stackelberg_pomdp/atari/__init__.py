"""Clean Gym/SB3 implementation of the Atari Stackelberg curriculum.

The package deliberately has no eager imports.  Environment, protocol, and
policy classes live in focused modules so importing the raw ALE environment
does not also initialize Stable-Baselines3.
"""

__all__ = []
