"""Deterministically screen, select, and confirm clean Atari E1 responses."""

import argparse
from copy import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np

from replication.atari import train_atari_meta_response_sb3 as trainer
from replication.atari.sb3_common import (
    ACTOR_LOSS_MODES,
    ScaledLearningRatePPO,
    model_actor_loss_mode,
    model_economic_initialization,
    write_csv,
    write_json,
)
from stackelberg_pomdp.atari.core import default_rom_path
from stackelberg_pomdp.atari.protocol import (
    ACTOR_STATE,
    ACTOR_STATE_DIM,
    EVENT_SLICE,
    NUM_TRADE_EVENTS,
    TRADE_MODE_INDEX,
)
from stackelberg_pomdp.atari.stackpomdp_env import BUYER, SELLER
from stackelberg_pomdp.atari.stackpomdp_policy import StackPOMDPAtariPolicy


SCREEN_EPISODES = 20
CONFIRMATION_EPISODES = 100
FIXED_EPISODES = 20
TIMING_EPISODES = 20
PROTOCOL_ATOL = 1.0e-6
EVALUATOR_NAME = "clean_atari_e1_selector_v2"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "replication/atari/results/e1_selections"
CANONICAL_FIXED_VALUES = tuple(value / 10.0 for value in range(11))
CANONICAL_GRID_EVENT_STEPS = (20, 50, 80, 110, 140)
CANONICAL_TIMING_FIRST_FOUR = (20, 50, 80, 110)
CANONICAL_TIMING_FIFTH_STEPS = {"early": 140, "late": 195}
CANONICAL_TIMING_PRICES = (0.5, 0.75, 0.9)
CANONICAL_TIMING_CALIBRATION_PRICE = 0.75
TIMING_MIN_CALIBRATION_ADVANTAGE = 0.15
TIMING_MIN_EARLY_ACCEPTANCE = 0.75
TIMING_MAX_LATE_ACCEPTANCE = 0.25
TIMING_MIN_ACCEPTANCE_DROP = 0.5
TIMING_MAX_PAYOFF_REGRET = 0.15


def checkpoint_path(raw, *, label="checkpoint"):
    path = Path(raw).expanduser()
    if path.suffix != ".zip":
        path = path.with_suffix(".zip")
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pin_file(source, destination):
    """Copy one source into immutable run-private storage and verify both ends."""

    source = Path(source).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"artifact does not exist: {source}")
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite pinned artifact: {destination}")
    before = checkpoint_sha256(source)
    shutil.copyfile(source, destination)
    after = checkpoint_sha256(source)
    pinned = checkpoint_sha256(destination)
    if not (before == after == pinned):
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"artifact changed while being pinned: {source}")
    return {
        "source_path": str(source),
        "pinned_path": str(destination),
        "sha256": pinned,
    }


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _load_model(path, *, device, expected_sha256=None):
    before = checkpoint_sha256(path)
    if expected_sha256 is not None and before != expected_sha256:
        raise RuntimeError(f"checkpoint changed before loading: {path}")
    model = ScaledLearningRatePPO.load(str(path), device=device)
    if checkpoint_sha256(path) != before:
        raise RuntimeError(f"checkpoint changed while loading: {path}")
    return model


def validate_e0b(path, *, device="cpu"):
    """Validate and identify the exact gameplay source supplied to E1."""

    path = checkpoint_path(path, label="E0b checkpoint")
    digest = checkpoint_sha256(path)
    model = _load_model(path, device=device, expected_sha256=digest)
    policy = model.policy
    if not isinstance(policy, StackPOMDPAtariPolicy):
        raise TypeError("E0b is not a clean composite Atari checkpoint")
    if policy.economic_role != "gameplay" or policy.economic_input_mode != "full":
        raise ValueError("E0b must be a full-input gameplay checkpoint")
    if not np.isclose(policy.pretrained_lr_scale, 0.1, atol=0, rtol=0):
        raise ValueError("E0b must have pretrained_lr_scale=0.1")
    result = {
        "path": str(path),
        "sha256": digest,
        "policy_class": f"{type(policy).__module__}.{type(policy).__qualname__}",
        "economic_role": policy.economic_role,
        "economic_input_mode": policy.economic_input_mode,
        "pretrained_lr_scale": float(policy.pretrained_lr_scale),
    }
    del model
    return result


def _canonical_sampler_provenance(raw, *, label):
    """Validate one exact, canonical E1 training-sampler contract."""

    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be a mapping")
    value = _jsonable(raw)
    mode = value.get("mode")
    if mode not in trainer.E1_SAMPLER_MODES:
        raise ValueError(f"{label} has an unknown sampler mode: {mode!r}")
    expected = trainer.e1_sampler_provenance(
        mode, gameplay_horizon=200, event_tail_steps=0
    )
    if value != expected:
        raise ValueError(f"{label} does not match its canonical provenance")
    return value


def _canonical_sampler_history(raw, *, current, training_timesteps):
    """Validate and retain every sampler stage and byte-bound resume parent."""

    if not isinstance(raw, list) or not raw:
        raise ValueError("E1 sampler history must be a nonempty list")
    history = _jsonable(raw)
    previous_start = -1
    previous_resume = -1
    for index, stage in enumerate(history):
        label = f"E1 sampler history stage {index}"
        if not isinstance(stage, dict):
            raise ValueError(f"{label} must be a mapping")
        start = stage.get("start_total_timesteps")
        if type(start) is not int or start < 0:
            raise ValueError(f"{label} has invalid start_total_timesteps")
        if index == 0 and start != 0:
            raise ValueError("E1 sampler history must start at timestep zero")
        if start <= previous_start or start > int(training_timesteps):
            raise ValueError(f"{label} has a nonmonotone or future start")
        previous_start = start
        stage["sampler"] = _canonical_sampler_provenance(
            stage.get("sampler"), label=f"{label} sampler"
        )
        inferred = stage.get("inferred_for_legacy_checkpoint")
        if type(inferred) is not bool:
            raise ValueError(f"{label} has no Boolean legacy-inference flag")
        if inferred and (
                index != 0
                or stage["sampler"]["mode"] != trainer.UNIFORM_E1_SAMPLER
        ):
            raise ValueError(
                "only the initial uniform sampler stage may be legacy-inferred"
            )
        resume_sources = stage.get("resume_sources")
        if not isinstance(resume_sources, list):
            raise ValueError(f"{label} resume_sources must be a list")
        for source_index, source in enumerate(resume_sources):
            source_label = f"{label} resume source {source_index}"
            if not isinstance(source, dict):
                raise ValueError(f"{source_label} must be a mapping")
            path = source.get("path")
            digest = source.get("sha256")
            source_step = source.get("training_total_timesteps")
            resume_step = source.get("resume_total_timesteps")
            if not isinstance(path, str) or not path:
                raise ValueError(f"{source_label} has no source path")
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"{source_label} has no SHA-256 digest")
            try:
                int(digest, 16)
            except ValueError as error:
                raise ValueError(
                    f"{source_label} has an invalid SHA-256 digest"
                ) from error
            if (
                    type(source_step) is not int
                    or type(resume_step) is not int
                    or source_step != resume_step
                    or resume_step < start
                    or resume_step < previous_resume
                    or resume_step > int(training_timesteps)
            ):
                raise ValueError(
                    f"{source_label} has invalid resume timesteps"
                )
            previous_resume = resume_step
    if history[-1]["sampler"] != current:
        raise ValueError(
            "E1 current sampler provenance does not match its final history stage"
        )
    return history


def candidate_sampler_contract(model):
    """Return validated sampler provenance/history, inferring legacy uniform."""

    current = getattr(model, "atari_e1_sampler_provenance", None)
    history = getattr(model, "atari_e1_sampler_history", None)
    if current is None and history is None:
        current = trainer.e1_sampler_provenance(
            trainer.UNIFORM_E1_SAMPLER,
            gameplay_horizon=200,
            event_tail_steps=0,
        )
        history = [{
            "start_total_timesteps": 0,
            "sampler": current,
            "inferred_for_legacy_checkpoint": True,
            "resume_sources": [],
        }]
        inferred = True
    elif current is None or history is None:
        raise ValueError(
            "E1 candidate must store sampler provenance and history together"
        )
    else:
        inferred = False
    current = _canonical_sampler_provenance(
        current, label="E1 current sampler provenance"
    )
    history = _canonical_sampler_history(
        history,
        current=current,
        training_timesteps=int(model.num_timesteps),
    )
    return {
        "atari_e1_sampler_provenance": current,
        "atari_e1_sampler_history": history,
        "sampler_contract_inferred_for_legacy_checkpoint": inferred,
    }


def load_candidate(
        path,
        *,
        role,
        e0b_sha256,
        device="cpu",
        display_path=None,
):
    """Load one E1 candidate and bind it to the supplied E0b bytes."""

    path = checkpoint_path(path)
    digest = checkpoint_sha256(path)
    model = _load_model(path, device=device, expected_sha256=digest)
    policy = model.policy
    if not isinstance(policy, StackPOMDPAtariPolicy):
        raise TypeError("candidate is not a clean composite Atari checkpoint")
    if policy.economic_role != role:
        raise ValueError(
            f"candidate role {policy.economic_role!r} does not match {role!r}"
        )
    if policy.economic_input_mode != "full":
        raise ValueError("E1 candidate must use economic_input_mode='full'")
    provenance = getattr(model, "atari_e1_source_provenance", None)
    if not isinstance(provenance, dict):
        raise ValueError("E1 candidate lacks E0b source provenance")
    if provenance.get("sha256") != e0b_sha256:
        raise ValueError("E1 candidate was initialized from different E0b bytes")
    trainer._validate_e0b_source(provenance)
    if provenance.get("zero_initialized_actor_state_indices") != [9, 10, 11, 12, 13]:
        raise ValueError("E1 candidate lacks the clean commitment-column reset")
    if int(model.n_steps) != 205 or not np.isclose(model.gamma, 1.0):
        raise ValueError("E1 candidate does not use the canonical 205-step return")
    if not np.isclose(model.gae_lambda, 1.0):
        raise ValueError("E1 candidate must have gae_lambda=1")
    actor_loss_mode = model_actor_loss_mode(model)
    if actor_loss_mode not in ACTOR_LOSS_MODES:
        raise ValueError(
            "E1 candidate has an unknown actor-loss mode: "
            f"{actor_loss_mode!r}"
        )
    economic_initialization = model_economic_initialization(
        model,
        default_mean=(
            trainer.BUYER_INIT_MEAN if role == BUYER else trainer.SELLER_INIT_MEAN
        ),
        default_concentration=(
            trainer.BUYER_INIT_CONCENTRATION
            if role == BUYER
            else trainer.SELLER_INIT_CONCENTRATION
        ),
    )
    target_kl = getattr(model, "target_kl", None)
    if target_kl is not None:
        target_kl = float(target_kl)
        if not np.isfinite(target_kl) or target_kl <= 0.0:
            raise ValueError("E1 candidate has an invalid target KL")
    reported_path = (
        Path(display_path).expanduser().resolve()
        if display_path is not None
        else path
    )
    named_step = re.search(r"_step(\d+)$", reported_path.stem)
    if (
            named_step is not None
            and int(named_step.group(1)) != int(model.num_timesteps)
    ):
        raise ValueError(
            "step-checkpoint filename does not match saved training timesteps"
        )
    sampler_contract = candidate_sampler_contract(model)
    policy.set_training_mode(False)
    metadata = {
        "path": str(reported_path),
        "sha256": digest,
        "training_timesteps": int(model.num_timesteps),
        "role": role,
        "economic_input_mode": policy.economic_input_mode,
        "training_config": {
            "algorithm": "PPO",
            "actor_loss_mode": actor_loss_mode,
            "economic_head_initialization": economic_initialization,
            "target_kl": target_kl,
            "seed": int(model.seed),
            "learning_rate": float(model.learning_rate),
            "n_steps": int(model.n_steps),
            "batch_size": int(model.batch_size),
            "n_epochs": int(model.n_epochs),
            "gamma": float(model.gamma),
            "gae_lambda": float(model.gae_lambda),
            "clip_range_at_start": float(model.clip_range(1.0)),
            "entropy_coefficient": float(model.ent_coef),
            "value_coefficient": float(model.vf_coef),
            "max_grad_norm": float(model.max_grad_norm),
            "policy_class": (
                f"{type(policy).__module__}.{type(policy).__qualname__}"
            ),
            "visual_features": int(policy.visual_features),
            "state_features": int(policy.state_features),
            "economic_hidden": int(policy.economic_hidden),
            "critic_hidden": int(policy.critic_hidden),
            "pretrained_lr_scale": float(policy.pretrained_lr_scale),
        },
        "e0b_source_provenance": _jsonable(provenance),
        **sampler_contract,
    }
    return model, metadata


def common_training_family(results):
    """Require one PPO configuration and one complete sampler lineage."""

    metadata = [
        result["metadata"] for result in results if result["episode_rows"]
    ]
    configs = [item["training_config"] for item in metadata]
    config_encodings = {
        json.dumps(config, sort_keys=True, separators=(",", ":"))
        for config in configs
    }
    if len(config_encodings) > 1:
        raise ValueError(
            "candidate checkpoints do not share one PPO seed/configuration"
        )
    sampler_contracts = [{
        "atari_e1_sampler_provenance": item[
            "atari_e1_sampler_provenance"
        ],
        "atari_e1_sampler_history": item["atari_e1_sampler_history"],
        "sampler_contract_inferred_for_legacy_checkpoint": item[
            "sampler_contract_inferred_for_legacy_checkpoint"
        ],
    } for item in metadata]
    sampler_encodings = {
        json.dumps(contract, sort_keys=True, separators=(",", ":"))
        for contract in sampler_contracts
    }
    if len(sampler_encodings) > 1:
        raise ValueError(
            "candidate checkpoints do not share one complete E1 sampler "
            "provenance/history"
        )
    common_sampler = sampler_contracts[0] if sampler_contracts else None
    return {
        "common_training_config": configs[0] if configs else None,
        "common_sampler_provenance": (
            None if common_sampler is None
            else common_sampler["atari_e1_sampler_provenance"]
        ),
        "common_sampler_history": (
            None if common_sampler is None
            else common_sampler["atari_e1_sampler_history"]
        ),
        "sampler_contract_inferred_for_legacy_checkpoint": (
            None if common_sampler is None
            else common_sampler[
                "sampler_contract_inferred_for_legacy_checkpoint"
            ]
        ),
    }


def random_context(seed):
    """Generate a context independently of the Atari/environment RNG."""

    return np.random.default_rng(int(seed) + 8_731_019).uniform(
        0.0, 1.0, size=NUM_TRADE_EVENTS
    ).astype(np.float32)


def checkpoint_family(path):
    """Return the retained-checkpoint family independent of ``_stepN``."""

    path = checkpoint_path(path)
    stem = re.sub(r"_step\d+$", "", path.stem)
    return str(path.parent), stem


def environment_config(args):
    """Record the exact bilateral environment used by the selector."""

    config = trainer.bilateral_config(args, seed=0).resolved()
    rom = Path(config.rom_path or default_rom_path()).expanduser().resolve()
    return {
        "gameplay_horizon": int(config.gameplay_horizon),
        "event_tail_steps": int(config.event_tail_steps),
        "random_event_steps": None,
        "fixed_grid_event_steps": list(args.grid_event_steps),
        "paired_timing_first_four_event_steps": list(
            CANONICAL_TIMING_FIRST_FOUR
        ),
        "paired_timing_fifth_event_steps": dict(
            CANONICAL_TIMING_FIFTH_STEPS
        ),
        "paired_timing_fifth_prices": list(CANONICAL_TIMING_PRICES),
        "paired_timing_calibration_price": (
            CANONICAL_TIMING_CALIBRATION_PRICE
        ),
        "seller_game_reward_scale": float(config.seller_game_reward_scale),
        "buyer_game_reward_scale": float(config.buyer_game_reward_scale),
        "noop_max": int(config.noop_max),
        "frame_skip": int(config.frame_skip),
        "frame_stack": int(config.frame_stack),
        "episodic_life": bool(config.episodic_life),
        "clip_game_rewards": bool(config.clip_game_rewards),
        "max_frames": int(config.max_frames),
        "rom_path": str(rom),
        "rom_sha256": checkpoint_sha256(rom),
    }


def apply_fifth_economic_override(observation, action, value):
    """Override only the economic coordinate at the fifth paused trade.

    The helper reads only the canonical actor state.  In particular, the
    deterministic Atari action returned by the model is copied unchanged.
    """

    if value is None:
        return np.array(action, dtype=np.float32, copy=True), False
    override = float(value)
    if not np.isfinite(override) or not 0.0 <= override <= 1.0:
        raise ValueError("fifth economic override must lie in [0, 1]")
    state = np.asarray(observation[ACTOR_STATE], dtype=np.float32).reshape(-1)
    if state.shape != (ACTOR_STATE_DIM,):
        raise ValueError(
            "canonical Atari actor state must contain "
            f"{ACTOR_STATE_DIM} scalars"
        )
    event = state[EVENT_SLICE]
    at_fifth_trade = bool(
        state[TRADE_MODE_INDEX] > 0.5
        and np.isclose(event[-1], 1.0, atol=0.0, rtol=0.0)
        and np.isclose(np.sum(event), 1.0, atol=0.0, rtol=0.0)
    )
    result = np.asarray(action, dtype=np.float32)
    original_shape = result.shape
    result = np.array(result, copy=True).reshape(-1)
    if result.shape != (2,):
        raise ValueError("Atari action must contain game and economic coordinates")
    if at_fifth_trade:
        result[1] = override
    return result.reshape(original_shape), at_fifth_trade


def _episode(
        model,
        args,
        *,
        seed,
        context,
        checkpoint_metadata,
        phase,
        fifth_economic_override=None,
):
    local = copy(args)
    local.fixed_event_steps = args.fixed_event_steps
    env = trainer.make_env(
        local,
        seed=int(seed),
        context_sampler=lambda rng, values=np.asarray(context): values.copy(),
    )
    try:
        observation = env.reset()
        done = False
        total = 0.0
        steps = 0
        info = {}
        override_applications = 0
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            action, applied = apply_fifth_economic_override(
                observation, action, fifth_economic_override
            )
            override_applications += int(applied)
            observation, reward, done, info = env.step(action)
            total += float(reward)
            steps += 1
    finally:
        env.close()
    row = {
        "phase": phase,
        "checkpoint_path": checkpoint_metadata["path"],
        "checkpoint_sha256": checkpoint_metadata["sha256"],
        "training_timesteps": checkpoint_metadata["training_timesteps"],
        "evaluation_seed": int(seed),
        "opponent_commitment": [float(value) for value in context],
        "fifth_economic_override": (
            None
            if fifth_economic_override is None
            else float(fifth_economic_override)
        ),
        "fifth_economic_override_applied": int(override_applications),
        "evaluation_return": float(total),
        "evaluation_steps": int(steps),
        **_jsonable(dict(info)),
    }
    row.pop("episode", None)
    return row


def _violation(row, field, expected, actual):
    return {
        "evaluation_seed": int(row.get("evaluation_seed", -1)),
        "field": field,
        "expected": _jsonable(expected),
        "actual": _jsonable(actual),
    }


def _finite_number(row, field, violations):
    value = row.get(field)
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = np.nan
    if not np.isfinite(number):
        violations.append(_violation(row, field, "finite numeric", value))
    return number


def audit_episode(row, *, role):
    """Audit the complete terminal record of one 200+5 E1 episode."""

    violations = []
    exact = {
        "evaluation_steps": 205,
        "outer_transition_count": 205,
        "reward_transition_count": 205,
        "gameplay_transitions": 200,
        "trade_transitions": 5,
        "seller_emulator_step_calls": 200,
        "buyer_emulator_step_calls": 200,
        "bullets_arrived": 5,
        "controlled_role": role,
    }
    for field, expected in exact.items():
        if row.get(field) != expected:
            violations.append(_violation(row, field, expected, row.get(field)))
    numeric_fields = (
        "seller_bullet_error",
        "buyer_bullet_error",
        "seller_payoff_error",
        "buyer_payoff_error",
        "purchases",
        "payments",
        "seller_game_reward",
        "buyer_game_reward",
        "seller_reward",
        "buyer_reward",
        "seller_shots_fired",
        "buyer_shots_fired",
        "seller_final_ammo",
        "buyer_final_ammo",
        "evaluation_return",
    )
    numeric = {
        field: _finite_number(row, field, violations)
        for field in numeric_fields
    }
    for field in (
            "seller_bullet_error",
            "buyer_bullet_error",
            "seller_payoff_error",
            "buyer_payoff_error",
    ):
        if np.isfinite(numeric[field]) and abs(numeric[field]) > PROTOCOL_ATOL:
            violations.append(_violation(row, field, 0.0, row.get(field)))

    events = list(row.get("events", ()))
    context = list(row.get("opponent_commitment", ()))
    event_steps = list(row.get("event_steps", ()))
    if len(events) != 5:
        violations.append(_violation(row, "events", 5, len(events)))
    if len(context) != 5:
        violations.append(_violation(row, "opponent_commitment", 5, len(context)))
    if (
            len(event_steps) != 5
            or event_steps != sorted(set(event_steps))
            or (event_steps and (event_steps[0] < 0 or event_steps[-1] >= 200))
    ):
        violations.append(_violation(
            row, "event_steps", "five distinct sorted indices in [0,199]", event_steps
        ))
    accepted = 0
    expected_payments = 0.0
    for index, event in enumerate(events):
        if int(event.get("event_index", -1)) != index:
            violations.append(_violation(row, f"event_{index}_index", index, event.get("event_index")))
        if index < len(event_steps) and int(event.get("game_step", -1)) != event_steps[index]:
            violations.append(_violation(row, f"event_{index}_game_step", event_steps[index], event.get("game_step")))
        price = float(event.get("price", np.nan))
        threshold = float(event.get("threshold", np.nan))
        if not np.isfinite(price) or not 0.0 <= price <= 1.0:
            violations.append(_violation(
                row, f"event_{index}_price", "finite value in [0,1]", price
            ))
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            violations.append(_violation(
                row,
                f"event_{index}_threshold",
                "finite value in [0,1]",
                threshold,
            ))
        actual_acceptance = bool(event.get("accepted", False))
        expected_acceptance = bool(price <= threshold)
        if actual_acceptance != expected_acceptance:
            violations.append(_violation(row, f"event_{index}_accepted", expected_acceptance, actual_acceptance))
        if index < len(context):
            observed = price if role == BUYER else threshold
            if not np.isclose(observed, context[index], atol=PROTOCOL_ATOL, rtol=0):
                violations.append(_violation(row, f"event_{index}_opponent_action", context[index], observed))
        seller_before = int(event.get("seller_ammo_before", -999))
        seller_after = int(event.get("seller_ammo_after", -999))
        buyer_before = int(event.get("buyer_ammo_before", -999))
        buyer_after = int(event.get("buyer_ammo_after", -999))
        expected_seller = seller_before - int(expected_acceptance)
        expected_buyer = buyer_before + int(expected_acceptance)
        if seller_after != expected_seller:
            violations.append(_violation(row, f"event_{index}_seller_inventory", expected_seller, seller_after))
        if buyer_after != expected_buyer:
            violations.append(_violation(row, f"event_{index}_buyer_inventory", expected_buyer, buyer_after))
        accepted += int(actual_acceptance)
        expected_payments += price * int(actual_acceptance)

    fifth_override = row.get("fifth_economic_override")
    override_applications = row.get("fifth_economic_override_applied")
    if fifth_override is not None:
        override = _finite_number(
            row, "fifth_economic_override", violations
        )
        if not 0.0 <= override <= 1.0:
            violations.append(_violation(
                row,
                "fifth_economic_override",
                "finite value in [0,1]",
                fifth_override,
            ))
        if override_applications != 1:
            violations.append(_violation(
                row,
                "fifth_economic_override_applied",
                1,
                override_applications,
            ))
        if len(events) == NUM_TRADE_EVENTS and not np.isclose(
                float(events[-1].get("threshold", np.nan)),
                override,
                atol=PROTOCOL_ATOL,
                rtol=0.0,
        ):
            violations.append(_violation(
                row,
                "event_5_forced_threshold",
                override,
                events[-1].get("threshold"),
            ))
    elif override_applications not in (None, 0):
        violations.append(_violation(
            row,
            "fifth_economic_override_applied",
            0,
            override_applications,
        ))

    if not np.isclose(numeric["purchases"], accepted, atol=0.0, rtol=0.0):
        violations.append(_violation(row, "purchases", accepted, row.get("purchases")))
    if not np.isclose(
            numeric["payments"], expected_payments,
            atol=PROTOCOL_ATOL, rtol=0,
    ):
        violations.append(_violation(row, "payments", expected_payments, row.get("payments")))
    seller_bullet_error = (
        5.0
        - numeric["seller_shots_fired"]
        - numeric["purchases"]
        - numeric["seller_final_ammo"]
    )
    buyer_bullet_error = (
        numeric["purchases"]
        - numeric["buyer_shots_fired"]
        - numeric["buyer_final_ammo"]
    )
    if not np.isclose(seller_bullet_error, 0.0, atol=0.0, rtol=0.0):
        violations.append(_violation(
            row, "independent_seller_bullet_conservation",
            0.0, seller_bullet_error,
        ))
    if not np.isclose(buyer_bullet_error, 0.0, atol=0.0, rtol=0.0):
        violations.append(_violation(
            row, "independent_buyer_bullet_conservation",
            0.0, buyer_bullet_error,
        ))
    seller_expected = 0.1 * numeric["seller_game_reward"] + expected_payments
    buyer_expected = numeric["buyer_game_reward"] - expected_payments
    if not np.isclose(numeric["seller_reward"], seller_expected, atol=PROTOCOL_ATOL, rtol=0):
        violations.append(_violation(row, "seller_reward", seller_expected, row.get("seller_reward")))
    if not np.isclose(numeric["buyer_reward"], buyer_expected, atol=PROTOCOL_ATOL, rtol=0):
        violations.append(_violation(row, "buyer_reward", buyer_expected, row.get("buyer_reward")))
    controlled = numeric[f"{role}_reward"]
    if not np.isclose(numeric["evaluation_return"], controlled, atol=PROTOCOL_ATOL, rtol=0):
        violations.append(_violation(row, "evaluation_return", controlled, row.get("evaluation_return")))
    return violations


def _summary(rows, *, role):
    def mean(field):
        return float(np.mean([float(row[field]) for row in rows]))

    payoffs = np.asarray([float(row[f"{role}_reward"]) for row in rows])
    result = {
        "episodes": len(rows),
        "mean_controlled_payoff": float(np.mean(payoffs)),
        "median_controlled_payoff": float(np.median(payoffs)),
        "minimum_controlled_payoff": float(np.min(payoffs)),
        "std_controlled_payoff": float(np.std(payoffs)),
        "mean_purchases": mean("purchases"),
        "mean_payments": mean("payments"),
        "mean_seller_game_reward": mean("seller_game_reward"),
        "mean_buyer_game_reward": mean("buyer_game_reward"),
        "mean_seller_shots_fired": mean("seller_shots_fired"),
        "mean_buyer_shots_fired": mean("buyer_shots_fired"),
        "mean_seller_final_ammo": mean("seller_final_ammo"),
        "mean_buyer_final_ammo": mean("buyer_final_ammo"),
    }
    events = [event for row in rows for event in row["events"]]
    result["mean_price"] = float(np.mean([event["price"] for event in events]))
    result["mean_threshold"] = float(np.mean([event["threshold"] for event in events]))
    result["acceptance_rate"] = float(np.mean([event["accepted"] for event in events]))
    result.update(trainer._trade_diagnostics(rows, gameplay_horizon=200))
    return result


def evaluate_rows(
        model,
        args,
        metadata,
        *,
        seeds,
        contexts,
        phase,
        fifth_economic_override=None,
):
    rows = [
        _episode(
            model,
            args,
            seed=seed,
            context=context,
            checkpoint_metadata=metadata,
            phase=phase,
            fifth_economic_override=fifth_economic_override,
        )
        for seed, context in zip(seeds, contexts)
    ]
    violations = [
        item for row in rows for item in audit_episode(row, role=args.role)
    ]
    event_rows = []
    for row in rows:
        for event in row["events"]:
            event_rows.append({
                "phase": phase,
                "checkpoint_path": metadata["path"],
                "checkpoint_sha256": metadata["sha256"],
                "training_timesteps": metadata["training_timesteps"],
                "evaluation_seed": row["evaluation_seed"],
                "fifth_economic_override": row.get(
                    "fifth_economic_override"
                ),
                **event,
            })
    return {
        "summary": _summary(rows, role=args.role),
        "protocol": {"passed": not violations, "violations": violations},
        "episode_rows": rows,
        "event_rows": event_rows,
    }


def screen_candidate(
        path,
        args,
        *,
        e0b_sha256,
        seeds,
        contexts,
        display_path=None,
):
    try:
        model, metadata = load_candidate(
            path,
            role=args.role,
            e0b_sha256=e0b_sha256,
            device=args.device,
            display_path=display_path,
        )
        result = evaluate_rows(
            model, args, metadata, seeds=seeds, contexts=contexts, phase="screen"
        )
        if checkpoint_sha256(path) != metadata["sha256"]:
            raise RuntimeError("candidate changed during screening")
        return {"metadata": metadata, **result}
    except Exception as error:
        path = checkpoint_path(path)
        reported = (
            Path(display_path).expanduser().resolve()
            if display_path is not None
            else path
        )
        return {
            "metadata": {
                "path": str(reported),
                "sha256": checkpoint_sha256(path),
                "training_timesteps": None,
            },
            "summary": None,
            "protocol": {
                "passed": False,
                "violations": [{"field": "candidate_load", "actual": repr(error)}],
            },
            "episode_rows": [],
            "event_rows": [],
        }


def validate_common_screen(results, *, seeds, contexts):
    expected = {
        int(seed): tuple(float(value) for value in context)
        for seed, context in zip(seeds, contexts)
    }
    schedules = None
    checked = 0
    for result in results:
        rows = result["episode_rows"]
        if not rows:
            continue
        actual = {
            int(row["evaluation_seed"]): tuple(row["opponent_commitment"])
            for row in rows
        }
        if actual != expected:
            raise RuntimeError("candidate screens did not use identical seeds/contexts")
        current = {
            int(row["evaluation_seed"]): tuple(row["event_steps"])
            for row in rows
        }
        if schedules is None:
            schedules = current
        elif current != schedules:
            raise RuntimeError("candidate screens did not use identical event schedules")
        checked += 1
    return {
        "passed": True,
        "candidates_checked": checked,
        "seed_context_pairs": [
            {"evaluation_seed": seed, "opponent_commitment": list(expected[seed]), "event_steps": None if schedules is None else list(schedules[seed])}
            for seed in seeds
        ],
    }


def rank_candidates(results):
    valid = [result for result in results if result["protocol"]["passed"]]
    valid.sort(key=lambda result: (
        -result["summary"]["mean_controlled_payoff"],
        -result["summary"]["median_controlled_payoff"],
        -result["summary"]["minimum_controlled_payoff"],
        result["summary"]["std_controlled_payoff"],
        result["metadata"]["training_timesteps"],
        result["metadata"]["sha256"],
    ))
    ranks = {result["metadata"]["sha256"]: index + 1 for index, result in enumerate(valid)}
    rows = []
    for result in results:
        metadata = result["metadata"]
        summary = result["summary"] or {}
        rows.append({
            "rank": ranks.get(metadata["sha256"]),
            "mechanically_valid": bool(result["protocol"]["passed"]),
            "checkpoint_path": metadata["path"],
            "checkpoint_sha256": metadata["sha256"],
            "training_timesteps": metadata.get("training_timesteps"),
            **summary,
        })
    rows.sort(key=lambda row: (row["rank"] is None, row["rank"] or 10**9, row["checkpoint_sha256"]))
    return valid, rows


def fixed_grid(model, args, metadata, *, seeds):
    evaluations = []
    original_steps = args.fixed_event_steps
    try:
        args.fixed_event_steps = args.grid_event_steps
        for value in args.fixed_eval_values:
            context = np.full(NUM_TRADE_EVENTS, float(value), dtype=np.float32)
            result = evaluate_rows(
                model,
                args,
                metadata,
                seeds=seeds,
                contexts=[context] * len(seeds),
                phase=f"fixed_{float(value):.2f}",
            )
            evaluations.append({"opponent_value": float(value), **result})
    finally:
        args.fixed_event_steps = original_steps
    return evaluations


def paired_timing_confirmation(model, args, metadata, *, seeds):
    """Evaluate actual and calibrated fifth-trade behavior on paired seeds."""

    evaluations = []
    original_steps = args.fixed_event_steps
    try:
        for timing, fifth_step in CANONICAL_TIMING_FIFTH_STEPS.items():
            event_steps = (*CANONICAL_TIMING_FIRST_FOUR, int(fifth_step))
            args.fixed_event_steps = event_steps
            for fifth_price in CANONICAL_TIMING_PRICES:
                context = np.array(
                    [0.0, 0.0, 0.0, 0.0, float(fifth_price)],
                    dtype=np.float32,
                )
                modes = [("actual", None)]
                if np.isclose(
                        fifth_price,
                        CANONICAL_TIMING_CALIBRATION_PRICE,
                        atol=0.0,
                        rtol=0.0,
                ):
                    modes.extend((
                        ("forced_buy", 1.0),
                        ("forced_reject", 0.0),
                    ))
                for policy_mode, override in modes:
                    phase = (
                        f"timing_{timing}_p{fifth_price:.2f}_{policy_mode}"
                    )
                    result = evaluate_rows(
                        model,
                        args,
                        metadata,
                        seeds=seeds,
                        contexts=[context] * len(seeds),
                        phase=phase,
                        fifth_economic_override=override,
                    )
                    evaluations.append({
                        "timing": timing,
                        "fifth_event_step": int(fifth_step),
                        "fifth_price": float(fifth_price),
                        "policy_mode": policy_mode,
                        "fifth_economic_override": override,
                        "event_steps": list(event_steps),
                        "opponent_commitment": context.tolist(),
                        **result,
                    })
    finally:
        args.fixed_event_steps = original_steps
    return evaluations


def _check(name, actual, relation, target):
    passed = {
        ">=": actual >= target,
        "<=": actual <= target,
        ">": actual > target,
    }[relation]
    return {"name": name, "actual": float(actual), "relation": relation, "target": float(target), "passed": bool(passed)}


def behavioral_gate(
        *,
        role,
        random_result,
        fixed_results,
        timing_results=None,
):
    by_value = {round(row["opponent_value"], 6): row["summary"] for row in fixed_results}
    required = (0.0, 0.5, 1.0)
    missing = [value for value in required if value not in by_value]
    if missing:
        return {"passed": False, "checks": [], "error": f"fixed grid missing {missing}"}
    zero, mid, one = (by_value[value] for value in required)
    ordered_values = sorted(by_value)
    curves = [by_value[value] for value in ordered_values]
    checks = []
    calibration_checks = []
    timing_behavior_checks = []
    timing_results = [] if timing_results is None else list(timing_results)
    if role == BUYER:
        timing_by_key = {}
        for result in timing_results:
            key = (
                result.get("timing"),
                round(float(result.get("fifth_price", np.nan)), 6),
                result.get("policy_mode"),
            )
            if key in timing_by_key:
                return {
                    "passed": False,
                    "mechanics_passed": False,
                    "data_calibration_passed": False,
                    "timing_behavior_passed": False,
                    "checks": [],
                    "error": f"duplicate paired timing condition {key}",
                }
            timing_by_key[key] = result
        expected_timing = {
            (timing, round(price, 6), "actual")
            for timing in CANONICAL_TIMING_FIFTH_STEPS
            for price in CANONICAL_TIMING_PRICES
        } | {
            (timing, CANONICAL_TIMING_CALIBRATION_PRICE, mode)
            for timing in CANONICAL_TIMING_FIFTH_STEPS
            for mode in ("forced_buy", "forced_reject")
        }
        missing_timing = sorted(expected_timing - set(timing_by_key))
        if missing_timing:
            return {
                "passed": False,
                "mechanics_passed": False,
                "data_calibration_passed": False,
                "timing_behavior_passed": False,
                "checks": [],
                "error": (
                    "paired timing confirmation missing conditions "
                    f"{missing_timing}"
                ),
            }
        low = [by_value[value] for value in sorted(by_value) if value <= 0.5]
        purchases = float(np.mean([row["mean_purchases"] for row in low]))
        shots = float(np.mean([row["mean_buyer_shots_fired"] for row in low]))
        max_up = max(
            later["mean_purchases"] - earlier["mean_purchases"]
            for earlier, later in zip(curves, curves[1:])
        )
        checks.extend([
            _check("random buyer net payoff", random_result["summary"]["mean_controlled_payoff"], ">", 0.0),
            _check("low/mid-price mean purchases", purchases, ">=", 4.5),
            _check("low/mid-price mean shots", shots, ">=", 4.25),
            _check("price 0.5 purchases", mid["mean_purchases"], ">=", 4.5),
            _check(
                "price 0.5 shots",
                mid["mean_buyer_shots_fired"],
                ">=",
                4.25,
            ),
            _check(
                "price 0.5 purchased bullets used",
                mid["mean_purchases"] - mid["mean_buyer_shots_fired"],
                "<=",
                0.5,
            ),
            _check("price 0.5 buyer net payoff", mid["mean_controlled_payoff"], ">", 0.0),
            _check("price 1 demand", one["mean_purchases"], "<=", 0.5),
            _check("low-to-high demand drop", zero["mean_purchases"] - one["mean_purchases"], ">=", 4.0),
            _check("largest adjacent demand reversal", max_up, "<=", 0.5),
        ])
        checks.extend(
            _check(
                f"price {value:.1f} buyer net payoff",
                by_value[value]["mean_controlled_payoff"],
                ">",
                0.0,
            )
            for value in ordered_values
            if value <= 0.5
        )
        price = CANONICAL_TIMING_CALIBRATION_PRICE
        early_actual = timing_by_key[("early", price, "actual")]["summary"]
        late_actual = timing_by_key[("late", price, "actual")]["summary"]
        early_buy = timing_by_key[("early", price, "forced_buy")]["summary"]
        early_reject = timing_by_key[
            ("early", price, "forced_reject")
        ]["summary"]
        late_buy = timing_by_key[("late", price, "forced_buy")]["summary"]
        late_reject = timing_by_key[
            ("late", price, "forced_reject")
        ]["summary"]
        calibration_checks.extend([
            _check(
                "paired timing early forced-buy payoff advantage",
                early_buy["mean_controlled_payoff"]
                - early_reject["mean_controlled_payoff"],
                ">=",
                TIMING_MIN_CALIBRATION_ADVANTAGE,
            ),
            _check(
                "paired timing late forced-reject payoff advantage",
                late_reject["mean_controlled_payoff"]
                - late_buy["mean_controlled_payoff"],
                ">=",
                TIMING_MIN_CALIBRATION_ADVANTAGE,
            ),
        ])
        early_acceptance = early_actual["event_5_acceptance_rate"]
        late_acceptance = late_actual["event_5_acceptance_rate"]
        early_best = max(
            early_buy["mean_controlled_payoff"],
            early_reject["mean_controlled_payoff"],
        )
        late_best = max(
            late_buy["mean_controlled_payoff"],
            late_reject["mean_controlled_payoff"],
        )
        timing_behavior_checks.extend([
            _check(
                "paired timing early price 0.75 acceptance",
                early_acceptance,
                ">=",
                TIMING_MIN_EARLY_ACCEPTANCE,
            ),
            _check(
                "paired timing late price 0.75 acceptance",
                late_acceptance,
                "<=",
                TIMING_MAX_LATE_ACCEPTANCE,
            ),
            _check(
                "paired timing price 0.75 acceptance drop",
                early_acceptance - late_acceptance,
                ">=",
                TIMING_MIN_ACCEPTANCE_DROP,
            ),
            _check(
                "paired timing early policy regret",
                early_best - early_actual["mean_controlled_payoff"],
                "<=",
                TIMING_MAX_PAYOFF_REGRET,
            ),
            _check(
                "paired timing late policy regret",
                late_best - late_actual["mean_controlled_payoff"],
                "<=",
                TIMING_MAX_PAYOFF_REGRET,
            ),
        ])
        checks.extend(calibration_checks)
        checks.extend(timing_behavior_checks)
    else:
        prices = [row["mean_price"] for row in curves]
        max_down = max(earlier - later for earlier, later in zip(prices, prices[1:]))
        high_values = [value for value in ordered_values if value >= 0.5]
        high = [by_value[value] for value in high_values]
        minimum_high_sales = min(row["mean_purchases"] for row in high)
        maximum_high_price_gap = max(
            abs(value - by_value[value]["mean_price"])
            for value in high_values
        )
        checks.extend([
            _check("random seller payoff", random_result["summary"]["mean_controlled_payoff"], ">", 0.5),
            _check("threshold 0 purchases", zero["mean_purchases"], "<=", 0.5),
            _check("threshold 0 retained-bullet shots", zero["mean_seller_shots_fired"], ">=", 4.0),
            _check("threshold 0.5 purchases", mid["mean_purchases"], ">=", 4.0),
            _check("threshold 0.5 mean price", mid["mean_price"], ">=", 0.35),
            _check("threshold 1 purchases", one["mean_purchases"], ">=", 4.5),
            _check("threshold 1 mean price", one["mean_price"], ">=", 0.75),
            _check("threshold 1 seller payoff", one["mean_controlled_payoff"], ">=", 3.5),
            _check("low-to-high price response", one["mean_price"] - zero["mean_price"], ">=", 0.5),
            _check("largest adjacent price reversal", max_down, "<=", 0.15),
            _check(
                "minimum purchases for thresholds at least 0.5",
                minimum_high_sales,
                ">=",
                4.0,
            ),
            _check(
                "largest high-threshold price gap",
                maximum_high_price_gap,
                "<=",
                0.2,
            ),
        ])
    mechanics = random_result["protocol"]["passed"] and all(
        result["protocol"]["passed"] for result in fixed_results
    )
    if role == BUYER:
        mechanics = mechanics and all(
            result["protocol"]["passed"] for result in timing_results
        )
    calibration_passed = (
        None
        if role != BUYER
        else all(row["passed"] for row in calibration_checks)
    )
    timing_behavior_passed = (
        None
        if role != BUYER
        else all(row["passed"] for row in timing_behavior_checks)
    )
    return {
        "passed": bool(mechanics and all(row["passed"] for row in checks)),
        "mechanics_passed": bool(mechanics),
        "data_calibration_passed": calibration_passed,
        "timing_behavior_passed": timing_behavior_passed,
        "checks": checks,
    }


def atomic_copy_no_overwrite(source, destination):
    source = checkpoint_path(source)
    destination = Path(destination).expanduser().resolve()
    if destination.suffix != ".zip":
        destination = destination.with_suffix(".zip")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite selected alias: {destination}")
    temporary = destination.with_name(f".{destination.name}.{checkpoint_sha256(source)[:12]}.tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary alias path already exists: {temporary}")
    shutil.copyfile(source, temporary)
    try:
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    if checkpoint_sha256(destination) != checkpoint_sha256(source):
        raise RuntimeError("selected alias copy failed SHA-256 verification")
    return {"path": str(destination), "sha256": checkpoint_sha256(destination)}


def run_selection(args):
    source_e0b = checkpoint_path(args.e0b_checkpoint, label="E0b checkpoint")
    source_candidates = [checkpoint_path(path) for path in args.checkpoint]
    source_hashes = [checkpoint_sha256(path) for path in source_candidates]
    if len(source_hashes) != len(set(source_hashes)):
        raise ValueError("candidate checkpoints must have distinct SHA-256 hashes")
    families = {checkpoint_family(path) for path in source_candidates}
    if len(families) != 1:
        raise ValueError(
            "candidate checkpoints must be retained from one training family"
        )
    training_directory, training_stem = next(iter(families))

    source_rom = Path(args.rom_path or default_rom_path()).expanduser().resolve()
    canonical_rom = Path(default_rom_path()).expanduser().resolve()
    canonical_rom_sha256 = checkpoint_sha256(canonical_rom)
    if checkpoint_sha256(source_rom) != canonical_rom_sha256:
        raise ValueError(
            "official E1 selection requires the canonical Space Invaders ROM"
        )

    with tempfile.TemporaryDirectory(prefix="stackpomdp-e1-selector-") as raw:
        pin_root = Path(raw)
        e0b_pin = pin_file(source_e0b, pin_root / "e0b.zip")
        rom_pin = pin_file(source_rom, pin_root / "space_invaders.bin")
        candidate_pins = [
            pin_file(path, pin_root / f"candidate_{index}.zip")
            for index, path in enumerate(source_candidates)
        ]
        if [item["sha256"] for item in candidate_pins] != source_hashes:
            raise RuntimeError("candidate bytes changed before immutable pinning")
        if rom_pin["sha256"] != canonical_rom_sha256:
            raise RuntimeError("ROM bytes changed before immutable pinning")
        pinned_by_sha = {
            item["sha256"]: item for item in candidate_pins
        }

        local = copy(args)
        local.e0b_checkpoint = e0b_pin["pinned_path"]
        local.rom_path = rom_pin["pinned_path"]
        e0b = validate_e0b(local.e0b_checkpoint, device=local.device)
        e0b["path"] = e0b_pin["source_path"]

        screen_seeds = list(range(
            local.screen_seed_start,
            local.screen_seed_start + SCREEN_EPISODES,
        ))
        screen_contexts = [random_context(seed) for seed in screen_seeds]
        screen = [
            screen_candidate(
                item["pinned_path"],
                local,
                e0b_sha256=e0b["sha256"],
                seeds=screen_seeds,
                contexts=screen_contexts,
                display_path=item["source_path"],
            )
            for item in candidate_pins
        ]
        common = validate_common_screen(
            screen, seeds=screen_seeds, contexts=screen_contexts
        )
        training_family = common_training_family(screen)
        ranked, ranking_rows = rank_candidates(screen)

        confirmation_seeds = list(range(
            local.confirmation_seed_start,
            local.confirmation_seed_start + CONFIRMATION_EPISODES,
        ))
        confirmation_contexts = [
            random_context(seed) for seed in confirmation_seeds
        ]
        fixed_seeds = list(range(
            local.fixed_seed_start,
            local.fixed_seed_start + FIXED_EPISODES,
        ))
        timing_seeds = list(range(
            local.timing_seed_start,
            local.timing_seed_start + TIMING_EPISODES,
        ))
        attempts = []
        selected = None
        selected_pin = None
        # Screening chooses the candidate.  Confirmation estimates its
        # out-of-sample behavior; it must never search down the ranking after
        # observing a failed fresh-seed gate.
        for candidate in ranked[:1]:
            screened_sha256 = candidate["metadata"]["sha256"]
            pinned = pinned_by_sha[screened_sha256]
            try:
                model, metadata = load_candidate(
                    pinned["pinned_path"],
                    role=local.role,
                    e0b_sha256=e0b["sha256"],
                    device=local.device,
                    display_path=pinned["source_path"],
                )
                if metadata["sha256"] != screened_sha256:
                    raise RuntimeError("candidate bytes changed after screening")
                random_result = evaluate_rows(
                    model,
                    local,
                    metadata,
                    seeds=confirmation_seeds,
                    contexts=confirmation_contexts,
                    phase="confirmation_random",
                )
                fixed_results = fixed_grid(
                    model, local, metadata, seeds=fixed_seeds
                )
                timing_results = (
                    paired_timing_confirmation(
                        model, local, metadata, seeds=timing_seeds
                    )
                    if local.role == BUYER
                    else []
                )
                if (
                        checkpoint_sha256(pinned["pinned_path"])
                        != screened_sha256
                ):
                    raise RuntimeError(
                        "candidate bytes changed during confirmation"
                    )
                gate = behavioral_gate(
                    role=local.role,
                    random_result=random_result,
                    fixed_results=fixed_results,
                    timing_results=timing_results,
                )
                attempt = {
                    "metadata": metadata,
                    "random": random_result,
                    "fixed_contexts": fixed_results,
                    "paired_timing": timing_results,
                    "behavioral_gate": gate,
                }
            except Exception as error:
                metadata = candidate["metadata"]
                attempt = {
                    "metadata": metadata,
                    "random": {
                        "summary": None,
                        "protocol": {
                            "passed": False,
                            "violations": [{
                                "field": "confirmation",
                                "actual": repr(error),
                            }],
                        },
                        "episode_rows": [],
                        "event_rows": [],
                    },
                    "fixed_contexts": [],
                    "paired_timing": [],
                    "behavioral_gate": {
                        "passed": False,
                        "mechanics_passed": False,
                        "data_calibration_passed": False,
                        "timing_behavior_passed": False,
                        "error": repr(error),
                        "checks": [],
                    },
                }
            attempts.append(attempt)
            if attempt["behavioral_gate"]["passed"]:
                selected_pin = pinned
                break

        environment = environment_config(local)
        if checkpoint_sha256(e0b_pin["pinned_path"]) != e0b_pin["sha256"]:
            raise RuntimeError("pinned E0b bytes changed during evaluation")
        if checkpoint_sha256(rom_pin["pinned_path"]) != rom_pin["sha256"]:
            raise RuntimeError("pinned ROM bytes changed during evaluation")
        environment["rom_path"] = rom_pin["source_path"]
        report = {
            "evaluator": EVALUATOR_NAME,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "role": local.role,
            "training_family": {
                "directory": training_directory,
                "stem": training_stem,
                **training_family,
            },
            "e0b_source": e0b,
            "environment": environment,
            "immutable_evaluation": {
                "e0b_sha256": e0b_pin["sha256"],
                "rom_sha256": rom_pin["sha256"],
                "candidate_sha256": source_hashes,
            },
            "protocol": {
                "gameplay_transitions": 200,
                "trade_transitions": 5,
                "outer_transitions": 205,
                "deterministic_evaluation": (
                    "masked Atari argmax and Beta mean"
                ),
                "screen_episodes": SCREEN_EPISODES,
                "screen_seed_start": int(local.screen_seed_start),
                "confirmation_episodes": CONFIRMATION_EPISODES,
                "confirmation_seed_start": int(
                    local.confirmation_seed_start
                ),
                "confirmation_policy": "screen_winner_only_no_fallback",
                "fixed_context_episodes": FIXED_EPISODES,
                "fixed_context_seed_start": int(local.fixed_seed_start),
                "fixed_context_values": list(local.fixed_eval_values),
                "fixed_context_event_steps": list(local.grid_event_steps),
                "paired_timing_required_for_role": BUYER,
                "paired_timing_run": bool(local.role == BUYER),
                "paired_timing_episodes_per_condition": TIMING_EPISODES,
                "paired_timing_actual_conditions": 6,
                "paired_timing_forced_conditions": 4,
                "paired_timing_seed_start": int(local.timing_seed_start),
                "paired_timing_shared_seeds": True,
                "paired_timing_first_four_event_steps": list(
                    CANONICAL_TIMING_FIRST_FOUR
                ),
                "paired_timing_fifth_event_steps": dict(
                    CANONICAL_TIMING_FIFTH_STEPS
                ),
                "paired_timing_fifth_prices": list(
                    CANONICAL_TIMING_PRICES
                ),
                "paired_timing_calibration": {
                    "fifth_price": CANONICAL_TIMING_CALIBRATION_PRICE,
                    "forced_fifth_buy_threshold": 1.0,
                    "forced_fifth_reject_threshold": 0.0,
                },
                "paired_timing_gate": {
                    "minimum_early_forced_buy_advantage": (
                        TIMING_MIN_CALIBRATION_ADVANTAGE
                    ),
                    "minimum_late_forced_reject_advantage": (
                        TIMING_MIN_CALIBRATION_ADVANTAGE
                    ),
                    "minimum_actual_early_acceptance": (
                        TIMING_MIN_EARLY_ACCEPTANCE
                    ),
                    "maximum_actual_late_acceptance": (
                        TIMING_MAX_LATE_ACCEPTANCE
                    ),
                    "minimum_acceptance_drop": TIMING_MIN_ACCEPTANCE_DROP,
                    "maximum_actual_payoff_regret": TIMING_MAX_PAYOFF_REGRET,
                },
            },
            "screen": {"common_pairing": common, "results": screen},
            "ranking": ranking_rows,
            "confirmation_attempts": attempts,
            "selection": {
                "fallback_allowed": False,
                "screen_selected_checkpoint_sha256": (
                    None
                    if not ranked
                    else ranked[0]["metadata"]["sha256"]
                ),
                "selected_checkpoint_sha256": (
                    None
                    if selected_pin is None
                    else selected_pin["sha256"]
                ),
            },
            "selected_alias": selected,
            "passed": selected_pin is not None,
        }
        if selected_pin is not None:
            selected = atomic_copy_no_overwrite(
                selected_pin["pinned_path"], local.selected_checkpoint
            )
            selected["source_path"] = selected_pin["source_path"]
            report["selected_alias"] = selected
        return report


def artifact_paths(args):
    output = Path(args.output_dir).expanduser().resolve()
    paths = {
        "json": output / f"{args.run_name}.json",
        "ranking": output / f"{args.run_name}.ranking.csv",
        "screen_episodes": output / f"{args.run_name}.screen.episodes.csv",
        "screen_events": output / f"{args.run_name}.screen.events.csv",
        "confirmation_episodes": output / f"{args.run_name}.confirmation.episodes.csv",
        "confirmation_events": output / f"{args.run_name}.confirmation.events.csv",
        "fixed_contexts": output / f"{args.run_name}.fixed_contexts.csv",
    }
    if args.role == BUYER:
        paths.update({
            "paired_timing_conditions": (
                output / f"{args.run_name}.paired_timing.conditions.csv"
            ),
            "paired_timing_episodes": (
                output / f"{args.run_name}.paired_timing.episodes.csv"
            ),
            "paired_timing_events": (
                output / f"{args.run_name}.paired_timing.events.csv"
            ),
        })
    return paths


def persist_report(report, args):
    paths = artifact_paths(args)
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f"refusing to overwrite E1 selection artifact: {path}")
    report["artifacts"] = {name: str(path) for name, path in paths.items()}
    screen = report["screen"]["results"]
    attempts = report["confirmation_attempts"]
    write_json(paths["json"], report)
    write_csv(paths["ranking"], report["ranking"])
    write_csv(paths["screen_episodes"], [row for result in screen for row in result["episode_rows"]])
    write_csv(paths["screen_events"], [row for result in screen for row in result["event_rows"]])
    write_csv(paths["confirmation_episodes"], [row for attempt in attempts for row in attempt["random"]["episode_rows"]])
    write_csv(paths["confirmation_events"], [row for attempt in attempts for row in attempt["random"]["event_rows"]])
    write_csv(paths["fixed_contexts"], [
        {
            "checkpoint_path": attempt["metadata"]["path"],
            "checkpoint_sha256": attempt["metadata"]["sha256"],
            "opponent_value": result["opponent_value"],
            "protocol_passed": result["protocol"]["passed"],
            **result["summary"],
        }
        for attempt in attempts for result in attempt["fixed_contexts"]
    ])
    if args.role == BUYER:
        timing = [
            (attempt, result)
            for attempt in attempts
            for result in attempt["paired_timing"]
        ]
        write_csv(paths["paired_timing_conditions"], [
            {
                "checkpoint_path": attempt["metadata"]["path"],
                "checkpoint_sha256": attempt["metadata"]["sha256"],
                "timing": result["timing"],
                "fifth_event_step": result["fifth_event_step"],
                "fifth_price": result["fifth_price"],
                "policy_mode": result["policy_mode"],
                "fifth_economic_override": result[
                    "fifth_economic_override"
                ],
                "event_steps": result["event_steps"],
                "opponent_commitment": result["opponent_commitment"],
                "protocol_passed": result["protocol"]["passed"],
                **result["summary"],
            }
            for attempt, result in timing
        ])
        write_csv(paths["paired_timing_episodes"], [
            {
                "timing": result["timing"],
                "fifth_event_step": result["fifth_event_step"],
                "fifth_price": result["fifth_price"],
                "policy_mode": result["policy_mode"],
                **row,
            }
            for _, result in timing
            for row in result["episode_rows"]
        ])
        write_csv(paths["paired_timing_events"], [
            {
                "timing": result["timing"],
                "fifth_event_step": result["fifth_event_step"],
                "fifth_price": result["fifth_price"],
                "policy_mode": result["policy_mode"],
                **row,
            }
            for _, result in timing
            for row in result["event_rows"]
        ])
    return paths


def _parse_floats(raw):
    values = tuple(float(value.strip()) for value in raw.split(","))
    if any(value < 0 or value > 1 for value in values):
        raise ValueError("fixed evaluation values must lie in [0,1]")
    if not all(any(np.isclose(value, required) for value in values) for required in (0, 0.5, 1)):
        raise ValueError("fixed evaluation values must include 0, 0.5, and 1")
    return values


def _parse_steps(raw):
    values = tuple(int(value.strip()) for value in raw.split(","))
    if len(values) != 5 or sorted(set(values)) != list(values) or values[0] < 0 or values[-1] >= 200:
        raise ValueError("grid event steps must be five distinct sorted values in [0,199]")
    return values


def _overlap(start_a, size_a, start_b, size_b):
    return max(start_a, start_b) < min(start_a + size_a, start_b + size_b)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=(BUYER, SELLER), required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--e0b-checkpoint", required=True)
    parser.add_argument("--selected-checkpoint", required=True)
    parser.add_argument("--screen-episodes", type=int, default=SCREEN_EPISODES)
    parser.add_argument("--screen-seed-start", type=int, default=3_500_001)
    parser.add_argument("--confirmation-episodes", type=int, default=CONFIRMATION_EPISODES)
    parser.add_argument("--confirmation-seed-start", type=int, default=3_600_001)
    parser.add_argument("--fixed-episodes", type=int, default=FIXED_EPISODES)
    parser.add_argument("--fixed-seed-start", type=int, default=3_700_001)
    parser.add_argument("--timing-episodes", type=int, default=TIMING_EPISODES)
    parser.add_argument("--timing-seed-start", type=int, default=3_800_001)
    parser.add_argument("--fixed-eval-values", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1")
    parser.add_argument("--grid-event-steps", default="20,50,80,110,140")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-name")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--noop-max", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=100_000)
    parser.add_argument("--rom-path")
    args = parser.parse_args(argv)
    if args.screen_episodes != SCREEN_EPISODES:
        parser.error(f"screen requires exactly {SCREEN_EPISODES} episodes")
    if args.confirmation_episodes != CONFIRMATION_EPISODES:
        parser.error(f"confirmation requires exactly {CONFIRMATION_EPISODES} episodes")
    if args.fixed_episodes != FIXED_EPISODES:
        parser.error(f"fixed grid requires exactly {FIXED_EPISODES} episodes per value")
    if args.timing_episodes != TIMING_EPISODES:
        parser.error(
            "paired timing confirmation requires exactly "
            f"{TIMING_EPISODES} episodes per condition"
        )
    try:
        args.fixed_eval_values = _parse_floats(args.fixed_eval_values)
        args.grid_event_steps = _parse_steps(args.grid_event_steps)
    except ValueError as error:
        parser.error(str(error))
    if tuple(args.fixed_eval_values) != CANONICAL_FIXED_VALUES:
        parser.error(
            "official selection requires fixed values 0,0.1,...,0.9,1"
        )
    if tuple(args.grid_event_steps) != CANONICAL_GRID_EVENT_STEPS:
        parser.error(
            "official selection requires grid event steps 20,50,80,110,140"
        )
    if args.noop_max != 30 or args.max_frames != 100_000:
        parser.error(
            "official selection requires noop_max=30 and max_frames=100000"
        )
    ranges = (
        (args.screen_seed_start, SCREEN_EPISODES),
        (args.confirmation_seed_start, CONFIRMATION_EPISODES),
        (args.fixed_seed_start, FIXED_EPISODES),
        (args.timing_seed_start, TIMING_EPISODES),
    )
    if any(start < 0 for start, _ in ranges):
        parser.error("evaluation seeds must be nonnegative")
    if any(_overlap(*first, *second) for index, first in enumerate(ranges) for second in ranges[index + 1:]):
        parser.error(
            "screen, confirmation, fixed, and timing seed ranges must be disjoint"
        )
    args.seed = 0
    args.gameplay_horizon = 200
    args.event_tail_steps = 0
    args.fixed_event_steps = None
    args.start_method = "spawn"
    if args.run_name is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.run_name = f"e1_{args.role}_selection_{stamp}"
    return args


def main(argv=None):
    args = parse_args(argv)
    for path in artifact_paths(args).values():
        if path.exists():
            raise FileExistsError(f"refusing to overwrite E1 selection artifact: {path}")
    selected = Path(args.selected_checkpoint).expanduser()
    if selected.suffix != ".zip":
        selected = selected.with_suffix(".zip")
    args.selected_checkpoint = str(selected.resolve())
    if selected.exists():
        raise FileExistsError(f"refusing to overwrite selected alias: {selected}")
    report = run_selection(args)
    try:
        paths = persist_report(report, args)
    except Exception:
        selected_info = report.get("selected_alias")
        if selected_info is not None:
            selected_path = Path(selected_info["path"])
            if (
                    selected_path.is_file()
                    and checkpoint_sha256(selected_path)
                    == selected_info["sha256"]
            ):
                selected_path.unlink()
        for path in artifact_paths(args).values():
            path.unlink(missing_ok=True)
        raise
    print({"passed": report["passed"], "selected": report["selected_alias"], "report": str(paths["json"])}, flush=True)
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
