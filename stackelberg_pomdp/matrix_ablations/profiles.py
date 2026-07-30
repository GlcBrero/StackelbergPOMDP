"""Immutable experiment profiles for the matrix-game replications.

The historical StackeRLberg checkout is provenance, not a runtime dependency.
These snapshots make every behaviorally important difference between the
current paper experiment and the historical runs explicit in maintained code.
"""

from dataclasses import asdict, dataclass
from typing import Mapping, Tuple


@dataclass(frozen=True)
class MatrixExperimentProfile:
    """Named state, query, reward, and reporting conventions."""

    profile_id: str
    memory_mode: str
    state_labels: Tuple[str, ...]
    state_index_formula: str
    query_state_order: Tuple[int, ...]
    context_encoding_version: str
    training_reward_offset: float
    reporting_transform: str
    interpretation: str

    def to_dict(self):
        payload = asdict(self)
        payload["state_labels"] = list(self.state_labels)
        payload["query_state_order"] = list(self.query_state_order)
        return payload


PAPER_JOINT_V1 = MatrixExperimentProfile(
    profile_id="paper_joint_v1",
    memory_mode="joint",
    state_labels=("Start", "CC", "CD", "DC", "DD"),
    state_index_formula="0=Start; 1+2*leader_action+follower_action",
    query_state_order=(0, 1, 2, 3, 4),
    context_encoding_version=(
        "mixed_radix_follower_state_then_ordered_commitment_v1"
    ),
    training_reward_offset=-4.0,
    reporting_transform="identity_on_centered_per_stage_reward",
    interpretation="current_paper_spec_qualitative_revalidation",
)


LEGACY_OPPONENT_V1 = MatrixExperimentProfile(
    profile_id="legacy_opponent_v1",
    memory_mode="opponent",
    state_labels=("Start", "opponent_action_0", "opponent_action_1"),
    state_index_formula=(
        "leader:0=Start,1+follower_action; "
        "follower:0=Start,1+leader_action"
    ),
    query_state_order=(0, 1, 2),
    context_encoding_version=(
        "mixed_radix_follower_state_then_ordered_commitment_v1"
    ),
    training_reward_offset=-2.5,
    reporting_transform="episode_return/5-1.5",
    interpretation="historical_stackerlberg_behavioral_profile",
)


PAPER_OPPONENT_SENSITIVITY_V1 = MatrixExperimentProfile(
    profile_id="paper_opponent_sensitivity_v1",
    memory_mode="opponent",
    state_labels=("Start", "opponent_action_0", "opponent_action_1"),
    state_index_formula=(
        "leader:0=Start,1+follower_action; "
        "follower:0=Start,1+leader_action"
    ),
    query_state_order=(0, 1, 2),
    context_encoding_version=(
        "mixed_radix_follower_state_then_ordered_commitment_v1"
    ),
    training_reward_offset=-4.0,
    reporting_transform="identity_on_centered_per_stage_reward",
    interpretation="paper_reward_scale_opponent_memory_sensitivity",
)


PROFILES: Mapping[str, MatrixExperimentProfile] = {
    profile.profile_id: profile
    for profile in (
        PAPER_JOINT_V1,
        LEGACY_OPPONENT_V1,
        PAPER_OPPONENT_SENSITIVITY_V1,
    )
}


def get_profile(profile_id):
    try:
        return PROFILES[str(profile_id)]
    except KeyError as exc:
        raise ValueError(
            "unknown matrix profile {!r}; available: {}".format(
                profile_id, ", ".join(sorted(PROFILES))
            )
        ) from exc


def profile_for_spec(spec):
    """Resolve an explicit profile without silently calling a sensitivity legacy."""

    if spec.memory_mode == "joint" and float(spec.reward_offset) == -4.0:
        return PAPER_JOINT_V1
    if spec.memory_mode == "opponent" and float(spec.reward_offset) == -2.5:
        return LEGACY_OPPONENT_V1
    if spec.memory_mode == "opponent" and float(spec.reward_offset) == -4.0:
        return PAPER_OPPONENT_SENSITIVITY_V1
    raise ValueError(
        "no immutable repeated-game profile for memory_mode={!r}, "
        "reward_offset={!r}".format(spec.memory_mode, spec.reward_offset)
    )


def profile_snapshot_for_spec(spec):
    """Return the immutable run profile for repeated or one-shot games."""

    if int(spec.episode_length) > 1:
        return profile_for_spec(spec).to_dict()
    return {
        "profile_id": "one_shot_{}_v1".format(spec.name),
        "memory_mode": "none",
        "state_labels": ["Start"],
        "state_index_formula": "0=Start",
        "query_state_order": [],
        "context_encoding_version": "one_shot_tabular_state_v1",
        "training_reward_offset": float(spec.reward_offset),
        "reporting_transform": "identity",
        "interpretation": "maintained_one_shot_qualitative_diagnostic",
    }
