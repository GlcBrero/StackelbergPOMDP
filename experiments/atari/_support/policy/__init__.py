"""Public Atari policy interface.

The package exposes one learned composite actor--critic. Neural building
blocks remain private implementation details. The ``composite`` module alias
keeps checkpoints produced before the package cleanup loadable without
retaining a second implementation.
"""

import sys

from experiments.atari._support.policy import policy as _policy
from experiments.atari._support.policy.policy import (
    ATARI_POLICY_PROVENANCE_ALIASES,
    ATARI_POLICY_PROVENANCE_ID,
    StackPOMDPAtariPolicy,
    canonical_atari_policy_provenance_id,
)
from experiments.atari._support.policy.meta_seller import (
    BETA_PARAMETER_EPSILON,
    SELLER_SHARED_CONTEXT_BETA_V5,
    seller_shared_context_architecture_provenance,
)

# Old SB3 archives import this exact module name while unpickling. Point it to
# the maintained implementation instead of keeping a duplicate shim file.
sys.modules.setdefault(f"{__name__}.composite", _policy)
composite = _policy

__all__ = [
    "ATARI_POLICY_PROVENANCE_ALIASES",
    "ATARI_POLICY_PROVENANCE_ID",
    "BETA_PARAMETER_EPSILON",
    "SELLER_SHARED_CONTEXT_BETA_V5",
    "StackPOMDPAtariPolicy",
    "canonical_atari_policy_provenance_id",
    "seller_shared_context_architecture_provenance",
]
