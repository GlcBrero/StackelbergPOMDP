"""Deterministically screen, select, and confirm clean Atari E1 responses."""

import argparse
from copy import copy
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import re
import shutil

import numpy as np

from replication.atari import train_atari_meta_response_sb3 as trainer
from replication.atari.sb3_common import (
    ScaledLearningRatePPO,
    write_csv,
    write_json,
)
from stackelberg_pomdp.atari.core import default_rom_path
from stackelberg_pomdp.atari.protocol import NUM_TRADE_EVENTS
from stackelberg_pomdp.atari.stackpomdp_env import BUYER, SELLER
from stackelberg_pomdp.atari.stackpomdp_policy import StackPOMDPAtariPolicy


SCREEN_EPISODES = 20
CONFIRMATION_EPISODES = 100
FIXED_EPISODES = 20
PROTOCOL_ATOL = 1.0e-6
EVALUATOR_NAME = "clean_atari_e1_selector_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "replication/atari/results/e1_selections"


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


def load_candidate(path, *, role, e0b_sha256, device="cpu"):
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
    policy.set_training_mode(False)
    metadata = {
        "path": str(path),
        "sha256": digest,
        "training_timesteps": int(model.num_timesteps),
        "role": role,
        "economic_input_mode": policy.economic_input_mode,
        "e0b_source_provenance": _jsonable(provenance),
    }
    return model, metadata


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


def _episode(model, args, *, seed, context, checkpoint_metadata, phase):
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
        while not done:
            action, _ = model.predict(observation, deterministic=True)
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
    for field in (
        "seller_bullet_error",
        "buyer_bullet_error",
        "seller_payoff_error",
        "buyer_payoff_error",
    ):
        value = row.get(field)
        if not isinstance(value, (int, float)) or abs(float(value)) > PROTOCOL_ATOL:
            violations.append(_violation(row, field, 0.0, value))

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

    if row.get("purchases") != accepted:
        violations.append(_violation(row, "purchases", accepted, row.get("purchases")))
    if not np.isclose(float(row.get("payments", np.nan)), expected_payments, atol=PROTOCOL_ATOL, rtol=0):
        violations.append(_violation(row, "payments", expected_payments, row.get("payments")))
    seller_expected = 0.1 * float(row.get("seller_game_reward", np.nan)) + expected_payments
    buyer_expected = float(row.get("buyer_game_reward", np.nan)) - expected_payments
    if not np.isclose(float(row.get("seller_reward", np.nan)), seller_expected, atol=PROTOCOL_ATOL, rtol=0):
        violations.append(_violation(row, "seller_reward", seller_expected, row.get("seller_reward")))
    if not np.isclose(float(row.get("buyer_reward", np.nan)), buyer_expected, atol=PROTOCOL_ATOL, rtol=0):
        violations.append(_violation(row, "buyer_reward", buyer_expected, row.get("buyer_reward")))
    controlled = float(row.get(f"{role}_reward", np.nan))
    if not np.isclose(float(row.get("evaluation_return", np.nan)), controlled, atol=PROTOCOL_ATOL, rtol=0):
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


def evaluate_rows(model, args, metadata, *, seeds, contexts, phase):
    rows = [
        _episode(
            model,
            args,
            seed=seed,
            context=context,
            checkpoint_metadata=metadata,
            phase=phase,
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
                **event,
            })
    return {
        "summary": _summary(rows, role=args.role),
        "protocol": {"passed": not violations, "violations": violations},
        "episode_rows": rows,
        "event_rows": event_rows,
    }


def screen_candidate(path, args, *, e0b_sha256, seeds, contexts):
    try:
        model, metadata = load_candidate(
            path, role=args.role, e0b_sha256=e0b_sha256, device=args.device
        )
        result = evaluate_rows(
            model, args, metadata, seeds=seeds, contexts=contexts, phase="screen"
        )
        if checkpoint_sha256(metadata["path"]) != metadata["sha256"]:
            raise RuntimeError("candidate changed during screening")
        return {"metadata": metadata, **result}
    except Exception as error:
        path = checkpoint_path(path)
        return {
            "metadata": {
                "path": str(path),
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


def _check(name, actual, relation, target):
    passed = {
        ">=": actual >= target,
        "<=": actual <= target,
        ">": actual > target,
    }[relation]
    return {"name": name, "actual": float(actual), "relation": relation, "target": float(target), "passed": bool(passed)}


def behavioral_gate(*, role, random_result, fixed_results):
    by_value = {round(row["opponent_value"], 6): row["summary"] for row in fixed_results}
    required = (0.0, 0.5, 1.0)
    missing = [value for value in required if value not in by_value]
    if missing:
        return {"passed": False, "checks": [], "error": f"fixed grid missing {missing}"}
    zero, mid, one = (by_value[value] for value in required)
    curves = [by_value[round(value, 6)] for value in sorted(by_value)]
    checks = []
    if role == BUYER:
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
            _check("price 0.5 buyer net payoff", mid["mean_controlled_payoff"], ">", 0.0),
            _check("price 1 demand", one["mean_purchases"], "<=", 0.5),
            _check("low-to-high demand drop", zero["mean_purchases"] - one["mean_purchases"], ">=", 4.0),
            _check("largest adjacent demand reversal", max_up, "<=", 0.5),
        ])
    else:
        prices = [row["mean_price"] for row in curves]
        max_down = max(earlier - later for earlier, later in zip(prices, prices[1:]))
        checks.extend([
            _check("random seller payoff", random_result["summary"]["mean_controlled_payoff"], ">", 0.5),
            _check("threshold 0 purchases", zero["mean_purchases"], "<=", 0.5),
            _check("threshold 0 retained-bullet shots", zero["mean_seller_shots_fired"], ">=", 4.0),
            _check("threshold 1 purchases", one["mean_purchases"], ">=", 4.5),
            _check("threshold 1 mean price", one["mean_price"], ">=", 0.75),
            _check("threshold 1 seller payoff", one["mean_controlled_payoff"], ">=", 3.5),
            _check("low-to-high price response", one["mean_price"] - zero["mean_price"], ">=", 0.5),
            _check("largest adjacent price reversal", max_down, "<=", 0.15),
        ])
    mechanics = random_result["protocol"]["passed"] and all(
        result["protocol"]["passed"] for result in fixed_results
    )
    return {"passed": bool(mechanics and all(row["passed"] for row in checks)), "mechanics_passed": bool(mechanics), "checks": checks}


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
    e0b = validate_e0b(args.e0b_checkpoint, device=args.device)
    candidates = [checkpoint_path(path) for path in args.checkpoint]
    hashes = [checkpoint_sha256(path) for path in candidates]
    if len(hashes) != len(set(hashes)):
        raise ValueError("candidate checkpoints must have distinct SHA-256 hashes")
    families = {checkpoint_family(path) for path in candidates}
    if len(families) != 1:
        raise ValueError(
            "candidate checkpoints must be retained from one training family"
        )
    training_directory, training_stem = next(iter(families))
    screen_seeds = list(range(args.screen_seed_start, args.screen_seed_start + SCREEN_EPISODES))
    screen_contexts = [random_context(seed) for seed in screen_seeds]
    screen = [
        screen_candidate(path, args, e0b_sha256=e0b["sha256"], seeds=screen_seeds, contexts=screen_contexts)
        for path in candidates
    ]
    common = validate_common_screen(screen, seeds=screen_seeds, contexts=screen_contexts)
    ranked, ranking_rows = rank_candidates(screen)
    confirmation_seeds = list(range(args.confirmation_seed_start, args.confirmation_seed_start + CONFIRMATION_EPISODES))
    confirmation_contexts = [random_context(seed) for seed in confirmation_seeds]
    fixed_seeds = list(range(args.fixed_seed_start, args.fixed_seed_start + FIXED_EPISODES))
    attempts = []
    selected = None
    for candidate in ranked:
        try:
            screened_sha256 = candidate["metadata"]["sha256"]
            model, metadata = load_candidate(
                candidate["metadata"]["path"],
                role=args.role,
                e0b_sha256=e0b["sha256"],
                device=args.device,
            )
            if metadata["sha256"] != screened_sha256:
                raise RuntimeError("candidate bytes changed after screening")
            random_result = evaluate_rows(
                model,
                args,
                metadata,
                seeds=confirmation_seeds,
                contexts=confirmation_contexts,
                phase="confirmation_random",
            )
            fixed_results = fixed_grid(model, args, metadata, seeds=fixed_seeds)
            if checkpoint_sha256(metadata["path"]) != screened_sha256:
                raise RuntimeError("candidate bytes changed during confirmation")
            gate = behavioral_gate(role=args.role, random_result=random_result, fixed_results=fixed_results)
            attempt = {"metadata": metadata, "random": random_result, "fixed_contexts": fixed_results, "behavioral_gate": gate}
        except Exception as error:
            metadata = candidate["metadata"]
            attempt = {
                "metadata": metadata,
                "random": {"summary": None, "protocol": {"passed": False, "violations": [{"field": "confirmation", "actual": repr(error)}]}, "episode_rows": [], "event_rows": []},
                "fixed_contexts": [],
                "behavioral_gate": {"passed": False, "mechanics_passed": False, "error": repr(error), "checks": []},
            }
        attempts.append(attempt)
        if attempt["behavioral_gate"]["passed"]:
            selected = atomic_copy_no_overwrite(metadata["path"], args.selected_checkpoint)
            selected["source_path"] = metadata["path"]
            break
    return {
        "evaluator": EVALUATOR_NAME,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "role": args.role,
        "training_family": {
            "directory": training_directory,
            "stem": training_stem,
        },
        "e0b_source": e0b,
        "environment": environment_config(args),
        "protocol": {
            "gameplay_transitions": 200,
            "trade_transitions": 5,
            "outer_transitions": 205,
            "deterministic_evaluation": "masked Atari argmax and Beta mean",
            "screen_episodes": SCREEN_EPISODES,
            "confirmation_episodes": CONFIRMATION_EPISODES,
            "fixed_context_episodes": FIXED_EPISODES,
            "fixed_context_values": list(args.fixed_eval_values),
            "fixed_context_event_steps": list(args.grid_event_steps),
        },
        "screen": {"common_pairing": common, "results": screen},
        "ranking": ranking_rows,
        "confirmation_attempts": attempts,
        "selected_alias": selected,
        "passed": selected is not None,
    }


def artifact_paths(args):
    output = Path(args.output_dir).expanduser().resolve()
    return {
        "json": output / f"{args.run_name}.json",
        "ranking": output / f"{args.run_name}.ranking.csv",
        "screen_episodes": output / f"{args.run_name}.screen.episodes.csv",
        "screen_events": output / f"{args.run_name}.screen.events.csv",
        "confirmation_episodes": output / f"{args.run_name}.confirmation.episodes.csv",
        "confirmation_events": output / f"{args.run_name}.confirmation.events.csv",
        "fixed_contexts": output / f"{args.run_name}.fixed_contexts.csv",
    }


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
    try:
        args.fixed_eval_values = _parse_floats(args.fixed_eval_values)
        args.grid_event_steps = _parse_steps(args.grid_event_steps)
    except ValueError as error:
        parser.error(str(error))
    ranges = (
        (args.screen_seed_start, SCREEN_EPISODES),
        (args.confirmation_seed_start, CONFIRMATION_EPISODES),
        (args.fixed_seed_start, FIXED_EPISODES),
    )
    if any(start < 0 for start, _ in ranges):
        parser.error("evaluation seeds must be nonnegative")
    if any(_overlap(*first, *second) for index, first in enumerate(ranges) for second in ranges[index + 1:]):
        parser.error("screen, confirmation, and fixed seed ranges must be disjoint")
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
    paths = persist_report(report, args)
    print({"passed": report["passed"], "selected": report["selected_alias"], "report": str(paths["json"])}, flush=True)
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
