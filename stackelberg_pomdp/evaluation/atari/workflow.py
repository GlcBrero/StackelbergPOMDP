"""Screen all candidates, confirm the winner on fresh seeds, and publish its alias."""

from datetime import datetime, timezone

import numpy as np

from stackelberg_pomdp.atari.training import (
    ACTOR_LOSS_MODES,
    model_actor_loss_mode,
    model_economic_initialization,
)
from stackelberg_pomdp.checkpoints.atari_evaluation import (
    load_e1_response,
    load_e2_checkpoint,
)
from stackelberg_pomdp.checkpoints.files import (
    _checkpoint_path,
    atomic_copy_no_overwrite,
    checkpoint_sha256,
    rollback_new_selected_alias,
)
from stackelberg_pomdp.envs.atari.bilateral import BUYER
from stackelberg_pomdp.evaluation.atari.contracts import (
    CANONICAL_GAMEPLAY_HORIZON,
    CONFIRMATION_EPISODES,
    E1_BUYER_INIT_CONCENTRATION,
    E1_BUYER_INIT_MEAN,
    E1_SELLER_INIT_CONCENTRATION,
    E1_SELLER_INIT_MEAN,
    EVALUATOR_NAME,
    SCREEN_EPISODES,
    _ranges_overlap,
    environment_config,
    environment_config_sha256,
)
from stackelberg_pomdp.evaluation.atari.rollouts import (
    evaluate_checkpoint,
    evaluate_endpoint_controls,
)
from stackelberg_pomdp.evaluation.atari.selection import (
    attach_economic_gate,
    confirmation_matches_screen,
    rank_candidates,
    validate_common_screen,
)


def assert_evaluation_inputs_unchanged(*, response_path, response_hash, config):
    """Fail if the frozen response or ROM changed during E2 evaluation."""

    if checkpoint_sha256(response_path) != response_hash:
        raise RuntimeError("frozen E1 response bytes changed during evaluation")
    if checkpoint_sha256(config["rom_path"]) != config["rom_sha256"]:
        raise RuntimeError("Space Invaders ROM bytes changed during evaluation")


def run_selection(args):
    """Screen, confirm only its top candidate, then alias only after a pass."""

    if int(args.screen_episodes) != SCREEN_EPISODES:
        raise ValueError(
            f"clean E2 selection requires exactly {SCREEN_EPISODES} episodes"
        )
    if int(args.confirmation_episodes) != CONFIRMATION_EPISODES:
        raise ValueError(
            "clean E2 confirmation requires exactly "
            f"{CONFIRMATION_EPISODES} episodes"
        )
    if _ranges_overlap(
            args.screen_seed_start,
            args.screen_episodes,
            args.confirmation_seed_start,
            args.confirmation_episodes,
    ):
        raise ValueError("screen and confirmation seed ranges must be disjoint")
    if int(args.gameplay_horizon) != CANONICAL_GAMEPLAY_HORIZON:
        raise ValueError(
            "clean E2 selection requires the canonical 200-step gameplay "
            "horizon"
        )

    checkpoints = tuple(
        _checkpoint_path(path, label="E2 candidate") for path in args.checkpoint
    )
    if len(set(checkpoints)) != len(checkpoints):
        raise ValueError("E2 candidate paths must be unique")
    hashes = [checkpoint_sha256(path) for path in checkpoints]
    if len(set(hashes)) != len(hashes):
        raise ValueError("E2 candidates must contain distinct checkpoint bytes")
    response_path = _checkpoint_path(
        args.response_checkpoint, label="frozen E1 response"
    )
    response_hash = checkpoint_sha256(response_path)
    args.response_checkpoint = str(response_path)
    config = environment_config(args)
    config_hash = environment_config_sha256(config)
    response_model = load_e1_response(
        response_path,
        leader_role=args.leader_role,
        device=args.device,
        expected_sha256=response_hash,
    )
    response_is_buyer = response_model.policy.economic_role == BUYER
    response_actor_loss_mode = model_actor_loss_mode(response_model)
    if response_actor_loss_mode not in ACTOR_LOSS_MODES:
        raise ValueError("frozen E1 response has an unknown actor loss mode")
    response_target_kl = getattr(response_model, "target_kl", None)
    if response_target_kl is not None:
        response_target_kl = float(response_target_kl)
        if not np.isfinite(response_target_kl) or response_target_kl <= 0.0:
            raise ValueError("frozen E1 response has an invalid target KL")
    response_initialization = model_economic_initialization(
        response_model,
        default_mean=(
            E1_BUYER_INIT_MEAN if response_is_buyer else E1_SELLER_INIT_MEAN
        ),
        default_concentration=(
            E1_BUYER_INIT_CONCENTRATION
            if response_is_buyer
            else E1_SELLER_INIT_CONCENTRATION
        ),
    )
    response_metadata = {
        "checkpoint_path": str(response_path),
        "checkpoint_sha256": response_hash,
        "economic_role": response_model.policy.economic_role,
        "economic_input_mode": response_model.policy.economic_input_mode,
        "training_total_timesteps": int(
            getattr(response_model, "num_timesteps", 0)
        ),
        "actor_loss_mode": response_actor_loss_mode,
        "target_kl": response_target_kl,
        "economic_head_initialization": response_initialization,
        "frozen": True,
        "deterministic": True,
    }

    screen_results = []
    for checkpoint, expected_hash in zip(checkpoints, hashes):
        model = load_e2_checkpoint(
            checkpoint,
            leader_role=args.leader_role,
            device=args.device,
            expected_sha256=expected_hash,
        )
        try:
            factual = evaluate_checkpoint(
                model,
                checkpoint,
                args=args,
                response_model=response_model,
                response_hash=response_hash,
                config=config,
                config_hash=config_hash,
                episodes=args.screen_episodes,
                seed_start=args.screen_seed_start,
                phase="screen",
                expected_checkpoint_sha256=expected_hash,
            )
            controls = (
                evaluate_endpoint_controls(
                    model,
                    checkpoint,
                    args=args,
                    response_model=response_model,
                    response_hash=response_hash,
                    config=config,
                    config_hash=config_hash,
                    episodes=args.screen_episodes,
                    seed_start=args.screen_seed_start,
                    phase="screen",
                    expected_checkpoint_sha256=expected_hash,
                )
                if factual["protocol"]["passed"]
                else []
            )
            screen_results.append(attach_economic_gate(
                factual, controls, required_episodes=args.screen_episodes
            ))
        finally:
            del model
    common_screen = validate_common_screen(screen_results)
    ranking = rank_candidates(screen_results)
    screen_selected_hash = ranking["selected_checkpoint_sha256"]
    ranking = {
        **ranking,
        "screen_selected_checkpoint_sha256": screen_selected_hash,
    }
    if screen_selected_hash is None:
        assert_evaluation_inputs_unchanged(
            response_path=response_path,
            response_hash=response_hash,
            config=config,
        )
        return {
            "schema_version": 2,
            "evaluator": EVALUATOR_NAME,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "environment_config": config,
            "environment_config_sha256": config_hash,
            "response_checkpoint": str(response_path),
            "response_checkpoint_sha256": response_hash,
            "response_metadata": response_metadata,
            "screen": {
                "episodes_per_checkpoint": int(args.screen_episodes),
                "seed_start": int(args.screen_seed_start),
                "seed_end": int(args.screen_seed_start) + args.screen_episodes - 1,
                "common_seed_schedule_check": common_screen,
                "checkpoint_results": screen_results,
            },
            "selection": ranking,
            "selected_alias": None,
            "confirmation": None,
            "confirmation_attempts": [],
            "passed": False,
        }

    selected = next(
        result for result in screen_results
        if result["checkpoint_sha256"] == screen_selected_hash
    )
    selected_alias = None
    confirmation_model = load_e2_checkpoint(
        selected["checkpoint_path"],
        leader_role=args.leader_role,
        device=args.device,
        expected_sha256=screen_selected_hash,
    )
    try:
        factual = evaluate_checkpoint(
            confirmation_model,
            selected["checkpoint_path"],
            args=args,
            response_model=response_model,
            response_hash=response_hash,
            config=config,
            config_hash=config_hash,
            episodes=args.confirmation_episodes,
            seed_start=args.confirmation_seed_start,
            phase="confirmation",
            expected_checkpoint_sha256=screen_selected_hash,
        )
        controls = (
            evaluate_endpoint_controls(
                confirmation_model,
                selected["checkpoint_path"],
                args=args,
                response_model=response_model,
                response_hash=response_hash,
                config=config,
                config_hash=config_hash,
                episodes=args.confirmation_episodes,
                seed_start=args.confirmation_seed_start,
                phase="confirmation",
                expected_checkpoint_sha256=screen_selected_hash,
            )
            if factual["protocol"]["passed"]
            else []
        )
        confirmed = attach_economic_gate(
            factual,
            controls,
            required_episodes=args.confirmation_episodes,
        )
    finally:
        del confirmation_model
    confirmation_check = confirmation_matches_screen(selected, confirmed)
    confirmation = {
        "screen_rank": 1,
        "episodes": int(args.confirmation_episodes),
        "seed_start": int(args.confirmation_seed_start),
        "seed_end": (
            int(args.confirmation_seed_start) + args.confirmation_episodes - 1
        ),
        "disjoint_from_screen": True,
        "result": confirmed,
        "checks": confirmation_check,
    }
    selected_hash = None
    if confirmation_check["passed"]:
        selected_alias = atomic_copy_no_overwrite(
            selected["checkpoint_path"], args.selected_checkpoint
        )
        if selected_alias["checkpoint_sha256"] != screen_selected_hash:
            rollback_new_selected_alias(
                {"selected_alias": selected_alias},
                expected_path=args.selected_checkpoint,
            )
            raise RuntimeError(
                "selected E2 checkpoint bytes changed after confirmation"
            )
        selected_hash = screen_selected_hash
    ranking = {
        **ranking,
        "selected_checkpoint_sha256": selected_hash,
    }
    assert_evaluation_inputs_unchanged(
        response_path=response_path,
        response_hash=response_hash,
        config=config,
    )
    return {
        "schema_version": 2,
        "evaluator": EVALUATOR_NAME,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment_config": config,
        "environment_config_sha256": config_hash,
        "response_checkpoint": str(response_path),
        "response_checkpoint_sha256": response_hash,
        "response_metadata": response_metadata,
        "screen": {
            "episodes_per_checkpoint": int(args.screen_episodes),
            "seed_start": int(args.screen_seed_start),
            "seed_end": int(args.screen_seed_start) + args.screen_episodes - 1,
            "common_seed_schedule_check": common_screen,
            "checkpoint_results": screen_results,
        },
        "selection": ranking,
        "selected_alias": selected_alias,
        "confirmation": confirmation,
        "confirmation_attempts": [confirmation],
        "passed": bool(confirmation_check["passed"]),
    }
