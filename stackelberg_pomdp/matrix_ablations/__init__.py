"""Qualitative matrix-game replications for the StackPOMDP paper.

The package deliberately contains only matrix-game domain code.  Phase
management, actor action caching, and hidden-query buffer exclusion remain in
the shared StackPOMDP implementation.
"""

from .envs import (
    LegacyMatrixQLeaderEnv,
    MATRIX_GAMES,
    MatrixFixedCommitmentResponseEnv,
    MatrixGameSpec,
    MatrixMetaLeaderEnv,
    RepeatedMatrixGame,
    build_response_checkpoint_contract,
    decode_meta_follower_observation,
    encode_meta_follower_observation,
    get_matrix_game,
    load_response_model,
    make_meta_leader_env,
    make_tabular_q_leader_env,
    meta_follower_observation_space_n,
    response_game_contract,
    validate_response_checkpoint_contract,
    write_response_checkpoint_contract,
)
from .profiles import (
    LEGACY_OPPONENT_V1,
    PAPER_JOINT_V1,
    PAPER_OPPONENT_SENSITIVITY_V1,
    PROFILES,
    get_profile,
    profile_for_spec,
    profile_snapshot_for_spec,
)
from .pg import (
    LEGACY_LEADER_PG_LEARNING_RATE,
    LEGACY_LEADER_PG_OUTER_UPDATES,
    LeaderPGPolicy,
    LeaderPolicyGradient,
    leader_pg_policy_loss,
    leader_pg_rollout_geometry,
)
from .rllib_es import (
    EXPECTED_RAY_VERSION,
    RLLIB_ES_IMPLEMENTATION,
    RllibESSettings,
    require_rllib_es,
    train_rllib_es,
)

__all__ = [
    "MATRIX_GAMES",
    "LegacyMatrixQLeaderEnv",
    "MatrixFixedCommitmentResponseEnv",
    "MatrixGameSpec",
    "MatrixMetaLeaderEnv",
    "RepeatedMatrixGame",
    "build_response_checkpoint_contract",
    "decode_meta_follower_observation",
    "encode_meta_follower_observation",
    "get_matrix_game",
    "load_response_model",
    "make_meta_leader_env",
    "make_tabular_q_leader_env",
    "meta_follower_observation_space_n",
    "response_game_contract",
    "validate_response_checkpoint_contract",
    "write_response_checkpoint_contract",
    "LEGACY_OPPONENT_V1",
    "PAPER_JOINT_V1",
    "PAPER_OPPONENT_SENSITIVITY_V1",
    "PROFILES",
    "get_profile",
    "profile_for_spec",
    "profile_snapshot_for_spec",
    "LEGACY_LEADER_PG_LEARNING_RATE",
    "LEGACY_LEADER_PG_OUTER_UPDATES",
    "LeaderPGPolicy",
    "LeaderPolicyGradient",
    "leader_pg_policy_loss",
    "leader_pg_rollout_geometry",
    "EXPECTED_RAY_VERSION",
    "RLLIB_ES_IMPLEMENTATION",
    "RllibESSettings",
    "require_rllib_es",
    "train_rllib_es",
]
