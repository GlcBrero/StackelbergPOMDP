"""Specialized economic response head for the Atari meta-seller.

The buyer-leader experiment needs a seller follower that responds to the
buyer's five committed thresholds.  This module contains that role-specific
head; the common Atari policy remains responsible for gameplay, the critic,
action credit, and SB3 integration.

The learned modules are attached directly to the owning policy under their
historical attribute names.  That preserves the state-dict and optimizer
layout of the released SB3 checkpoint while keeping the specialized logic out
of the common policy implementation.
"""

from functools import partial
import math
from typing import NamedTuple

import numpy as np
import torch as th
from torch import nn

from stackelberg_pomdp.atari.protocol import (
    ACTOR_STATE_DIM,
    EVENT_SLICE,
    OPPONENT_COMMITMENT_SLICE,
)


SELLER_SHARED_CONTEXT_BETA_V5 = "seller_shared_context_beta_v5"
BETA_PARAMETER_EPSILON = 1.0e-4
UNIT_INTERVAL_EPSILON = 1.0e-4

_REMOVED_SELLER_FLAGS = (
    "economic_threshold_residual",
    "economic_threshold_residual_direct_input",
)


class MetaSellerModules(NamedTuple):
    """References to the meta-seller modules registered on the SB3 policy."""

    live_encoder: nn.Module
    live_output: nn.Module
    context_encoder: nn.Module
    context_output: nn.Module
    current_slope: nn.Parameter


def consume_removed_seller_checkpoint_kwargs(kwargs):
    """Accept inert keys serialized by one released meta-seller archive.

    The public release retains only the shared-context meta-seller.  False
    values are discarded solely so SB3 can reconstruct the released archive;
    true values would request removed experimental architectures and fail.
    """

    values = tuple(kwargs.pop(name, False) for name in _REMOVED_SELLER_FLAGS)
    if not all(isinstance(value, (bool, np.bool_)) for value in values):
        raise TypeError("legacy seller flags must be Boolean")
    if any(bool(value) for value in values):
        raise ValueError(
            "legacy threshold-residual seller architectures are not part "
            "of the release"
        )


def validate_meta_seller_configuration(architecture, *, role, input_mode):
    """Validate and normalize the optional specialized architecture name."""

    if architecture is None:
        return None
    architecture = str(architecture)
    if architecture != SELLER_SHARED_CONTEXT_BETA_V5:
        raise ValueError(f"unsupported economic architecture: {architecture!r}")
    if role != "seller" or input_mode != "full":
        raise ValueError(
            f"{architecture} is reserved for full-input meta-seller "
            "response policies"
        )
    return architecture


def is_meta_seller_architecture(architecture):
    """Return whether a policy uses the retained meta-seller response head."""

    return architecture == SELLER_SHARED_CONTEXT_BETA_V5


def attach_meta_seller_modules(policy):
    """Build the head while preserving released checkpoint parameter names."""

    live_features = OPPONENT_COMMITMENT_SLICE.start
    context_features = (
        OPPONENT_COMMITMENT_SLICE.stop - OPPONENT_COMMITMENT_SLICE.start
    )
    policy.economic_live_encoder = nn.Sequential(
        nn.Linear(live_features, 32),
        nn.Tanh(),
    )
    policy.economic_live_output = nn.Linear(32, 2)
    policy.economic_context_encoder = nn.Sequential(
        nn.Linear(2 * context_features, 32),
        nn.Tanh(),
    )
    policy.economic_context_output = nn.Linear(32, 1)
    policy.economic_current_slope = nn.Parameter(th.zeros(()))


def meta_seller_modules(policy):
    """Collect the historically named policy attributes as one component."""

    if not is_meta_seller_architecture(policy.economic_architecture):
        raise ValueError("meta-seller modules requested from another policy")
    return MetaSellerModules(
        live_encoder=policy.economic_live_encoder,
        live_output=policy.economic_live_output,
        context_encoder=policy.economic_context_encoder,
        context_output=policy.economic_context_output,
        current_slope=policy.economic_current_slope,
    )


def initialize_meta_seller(
        policy,
        *,
        mean,
        concentration,
        inverse_softplus,
):
    """Initialize a neutral response: mean 0.5 and no context dependence."""

    mean = float(mean)
    concentration = float(concentration)
    if not 0.0 < mean < 1.0:
        raise ValueError("Beta initialization mean must lie in (0, 1)")
    if concentration <= BETA_PARAMETER_EPSILON:
        raise ValueError(
            "meta-seller Beta initialization concentration must exceed "
            "epsilon"
        )

    modules = meta_seller_modules(policy)
    for module in (modules.live_encoder, modules.context_encoder):
        module.apply(partial(policy.init_weights, gain=1.0))
    nn.init.zeros_(modules.live_output.weight)
    nn.init.zeros_(modules.context_output.weight)
    with th.no_grad():
        modules.live_output.bias.copy_(th.tensor([
            math.log(mean / (1.0 - mean)),
            inverse_softplus(concentration - BETA_PARAMETER_EPSILON),
        ], dtype=modules.live_output.bias.dtype,
           device=modules.live_output.bias.device))
        modules.context_output.bias.zero_()
        modules.current_slope.zero_()


def meta_seller_beta_parameters(
        actor_state,
        modules,
        *,
        context_ablation=False,
):
    """Map live state and the five-threshold context to Beta parameters."""

    state = actor_state.float().reshape(-1, ACTOR_STATE_DIM)
    event = state[:, EVENT_SLICE]
    commitment = state[:, OPPONENT_COMMITMENT_SLICE]
    live = state[:, :OPPONENT_COMMITMENT_SLICE.start]

    base = modules.live_output(modules.live_encoder(live))
    centered_commitment = 2.0 * commitment - 1.0
    context_input = th.cat([centered_commitment, event], dim=1)
    context_residual = modules.context_output(
        modules.context_encoder(context_input)
    ).reshape(-1)
    current_threshold = th.sum(event * centered_commitment, dim=1)

    if context_ablation:
        context_residual = th.zeros_like(base[:, 0])
        current_skip = th.zeros_like(base[:, 0])
    else:
        current_skip = modules.current_slope * current_threshold

    mean = (
        UNIT_INTERVAL_EPSILON
        + (1.0 - 2.0 * UNIT_INTERVAL_EPSILON)
        * th.sigmoid(base[:, 0] + context_residual + current_skip)
    )
    concentration = (
        nn.functional.softplus(base[:, 1]) + BETA_PARAMETER_EPSILON
    )
    return mean * concentration, (1.0 - mean) * concentration


def meta_seller_optimizer_parameter_groups(policy, value_net, *, base_rate):
    """Return the three released learning-rate groups for response training."""

    modules = meta_seller_modules(policy)
    live = [
        parameter
        for module in (modules.live_encoder, modules.live_output)
        for parameter in module.parameters()
    ]
    context = [
        parameter
        for module in (modules.context_encoder, modules.context_output)
        for parameter in module.parameters()
    ] + [modules.current_slope]
    critic = list(value_net.parameters())

    partition = live + context + critic
    trainable = {
        id(parameter)
        for parameter in policy.parameters()
        if parameter.requires_grad
    }
    grouped = {id(parameter) for parameter in partition}
    if grouped != trainable or len(grouped) != len(partition):
        raise RuntimeError(
            "meta-seller optimizer groups do not partition trainable "
            "parameters exactly"
        )
    rate = float(base_rate)
    return [
        {
            "params": live,
            "lr": rate,
            "lr_scale": 1.0,
            "group_name": "seller_v5_live",
        },
        {
            "params": context,
            "lr": rate * 4.0,
            "lr_scale": 4.0,
            "group_name": "seller_v5_context",
        },
        {
            "params": critic,
            "lr": rate * 0.2,
            "lr_scale": 0.2,
            "group_name": "seller_v5_critic",
        },
    ]


META_SELLER_TRANSFER_MODULES = (
    "economic_live_encoder",
    "economic_live_output",
    "economic_context_encoder",
    "economic_context_output",
)


def seller_shared_context_architecture_provenance():
    """Return the JSON-safe contract for the retained seller architecture."""

    return {
        "schema": "stackpomdp.atari.economic_actor_architecture.v5",
        "parameterization": SELLER_SHARED_CONTEXT_BETA_V5,
        "economic_role": "seller",
        "economic_input_mode": "full",
        "gameplay_actor": {
            "source": "certified E0b actor",
            "modules": [
                "features_extractor.visual",
                "features_extractor.state_encoder",
                "game_action_net",
            ],
            "frozen_during_e1": True,
            "opponent_commitment_input_columns_zero": True,
            "rollout_action": "deterministic masked argmax",
            "economic_sampling_does_not_sample_game_action": True,
        },
        "live_base_branch": {
            "input": (
                "actor_state[ammo, projectile, normalized_time, trade_mode, "
                "event_one_hot]"
            ),
            "input_features": OPPONENT_COMMITMENT_SLICE.start,
            "commitment_masked": True,
            "architecture": [32, 2],
            "activation": "tanh",
            "outputs": ["base_mean_logit", "raw_concentration"],
        },
        "shared_context_branch": {
            "input_order": [
                "2 * opponent_commitment - 1",
                "event_one_hot",
            ],
            "input_features": 10,
            "architecture": [32, 1],
            "activation": "tanh",
            "output": "event-conditioned_context_residual_logit",
            "final_layer_initialization": "exact_zero",
        },
        "shared_current_threshold_skip": {
            "formula": (
                "slope * sum_i event_i * "
                "(2 * opponent_commitment_i - 1)"
            ),
            "trainable_shared_slopes": 1,
            "initialization": "exact_zero",
            "fixed_anchor": False,
        },
        "mean": {
            "formula": (
                "epsilon + (1 - 2 epsilon) * sigmoid("
                "base_mean_logit + shared_context_residual + "
                "shared_current_threshold_skip)"
            ),
            "epsilon": UNIT_INTERVAL_EPSILON,
        },
        "concentration": {
            "formula": "softplus(raw_concentration) + epsilon",
            "epsilon": BETA_PARAMETER_EPSILON,
            "context_conditioned": False,
        },
        "distribution": (
            "Beta(mean * concentration, (1 - mean) * concentration)"
        ),
        "optimizer": {
            "live_learning_rate": 5.0e-4,
            "context_learning_rate": 2.0e-3,
            "critic_learning_rate": 1.0e-4,
            "independent_group_gradient_clip_norm": 0.5,
            "gameplay_actor_in_optimizer": False,
        },
        "new_observation_fields": [],
        "fixed_threshold_anchor": False,
        "gameplay_action_selection": (
            "deterministic_argmax_during_training_and_evaluation"
        ),
    }


__all__ = [
    "BETA_PARAMETER_EPSILON",
    "SELLER_SHARED_CONTEXT_BETA_V5",
    "UNIT_INTERVAL_EPSILON",
    "seller_shared_context_architecture_provenance",
]
