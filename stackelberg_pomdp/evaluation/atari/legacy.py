"""Compatibility for helpers exposed by the former monolithic evaluator."""

from importlib import import_module


# Maintained code imports the owning modules directly. These names preserve
# older notebooks and trajectory fixtures without duplicating implementations.
_LEGACY_MODULE_EXPORTS = {
    "stackelberg_pomdp.atari.protocol": (
        "ACTION_CREDIT",
        "ACTOR_STATE",
        "canonical_leader_state",
    ),
    "stackelberg_pomdp.checkpoints.atari_evaluation": (
        "_load_model",
        "load_e1_response",
        "load_e2_checkpoint",
        "validate_candidate_provenance",
    ),
    "stackelberg_pomdp.checkpoints.files": (
        "_checkpoint_path",
        "atomic_copy_no_overwrite",
        "checkpoint_sha256",
    ),
    "stackelberg_pomdp.evaluation.atari.contracts": (
        "CANONICAL_EVENT_ACTION_MASK",
        "E1_BUYER_INIT_CONCENTRATION",
        "E1_BUYER_INIT_MEAN",
        "E1_SELLER_INIT_CONCENTRATION",
        "E1_SELLER_INIT_MEAN",
        "E2_INIT_CONCENTRATION",
        "E2_INIT_MEAN",
        "ECONOMIC_CONTROL_COMMITMENTS",
        "ECONOMIC_GATE_HYPOTHESIS",
        "ECONOMIC_GATE_MIN_FACTUAL_PAYOFF",
        "ECONOMIC_GATE_MIN_MEAN_ADVANTAGE",
        "ECONOMIC_GATE_MIN_MEDIAN_ADVANTAGE",
        "ECONOMIC_GATE_MIN_TOTAL_SHOTS",
        "ECONOMIC_GATE_MIN_WIN_RATE",
        "ECONOMIC_INTERVENTION_SCHEMA",
        "EVALUATOR_NAME",
        "PROTOCOL_ATOL",
        "REPOSITORY_ROOT",
        "SELECTION_RULE",
        "_action_list",
        "_canonical_json_bytes",
        "_close",
        "_jsonable",
        "_number",
        "actor_observation_sha256",
        "environment_config",
        "environment_config_sha256",
        "opposite_role",
    ),
    "stackelberg_pomdp.evaluation.atari.interventions": (
        "_canonical_economic_transition",
        "_normalized_commitment",
        "apply_economic_commitment_override",
        "economic_intervention_manifest",
    ),
    "stackelberg_pomdp.evaluation.atari.protocol_audit": (
        "_audit_episode",
        "_audit_transitions",
        "_violation",
        "audit_e2_protocol",
    ),
    "stackelberg_pomdp.evaluation.atari.reporting": (
        "_artifact_file_identity",
        "_condition_rows",
        "_counterfactual_results",
        "_csv_row",
        "_csv_value",
        "_event_rows",
        "_paired_gate_rows",
        "_publish_artifact_set",
        "_same_artifact_file",
        "_selection_artifact_tables",
        "default_run_name",
    ),
    "stackelberg_pomdp.evaluation.atari.rollouts": (
        "evaluate_checkpoint",
        "evaluate_e2_model",
        "evaluate_endpoint_controls",
        "leader_outcome_summary",
        "make_e2_env",
    ),
    "stackelberg_pomdp.evaluation.atari.selection": (
        "_gate_check",
        "_selection_key",
        "attach_economic_gate",
        "confirmation_matches_screen",
        "paired_economic_gate",
        "rank_candidates",
        "validate_common_screen",
    ),
    "stackelberg_pomdp.evaluation.atari.workflow": (
        "assert_evaluation_inputs_unchanged",
    ),
}


def __getattr__(name):
    for module, names in _LEGACY_MODULE_EXPORTS.items():
        if name in names:
            return getattr(import_module(module), name)
    raise AttributeError(f"legacy Atari evaluator has no attribute {name!r}")
