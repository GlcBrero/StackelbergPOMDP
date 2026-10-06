"""Select the composite Atari actor/critic used by this experiment.

Its visual and economic neural heads are implemented in experiments/atari/_support/policy/.
Pretraining and leader training use that same implementation.
"""
from experiments.atari._support.policy import StackPOMDPAtariPolicy

POLICY = StackPOMDPAtariPolicy
