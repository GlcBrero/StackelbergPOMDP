"""Compatibility imports for checkpoints using the historical module path.

Maintained code imports policies and algorithms from their dedicated packages.
"""

from stackelberg_pomdp.algorithms.on_policy import (
    CustomA2C,
    CustomOnPolicyAlgorithm,
    CustomPPO,
)
from stackelberg_pomdp.policies.generic import CustomMLPExtractor, CustomPolicy

__all__ = [
    "CustomA2C",
    "CustomMLPExtractor",
    "CustomOnPolicyAlgorithm",
    "CustomPPO",
    "CustomPolicy",
]
