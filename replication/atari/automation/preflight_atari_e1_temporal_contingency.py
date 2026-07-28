"""Run the no-save, no-W&B real-ALE preflight for E1 temporal training."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import math
import os
from pathlib import Path

import numpy as np

from replication.atari import train_atari_meta_response_sb3 as trainer
from replication.atari.automation import (
    validate_atari_e1_temporal_contingency as validator,
)
from replication.atari.sb3_common import make_vec_env
from stackelberg_pomdp.atari.e1_sampling import (
    CONTEXT_STRATUM_WEIGHTS,
    EARLY_FIFTH_INTERVAL,
    LATE_FIFTH_INTERVAL,
    LOW_PREFIX_PRICE_HIGH,
    SCHEDULE_STRATUM_WEIGHTS,
    TEMPORAL_MIX_E1_SAMPLER,
    TemporalMarginalE1Sampler,
    UNIFORM_E1_SAMPLER,
    e1_sampler_provenance,
)


PREFLIGHT_KIND = "atari_e1_buyer_temporal_contingency_preflight"
SAMPLER_DRAWS = 20_000
FREQUENCY_TOLERANCE = 0.02
INDEPENDENCE_TOLERANCE = 0.02


def audit_sampler(*, seed=97_531, draws=SAMPLER_DRAWS):
    sampler = TemporalMarginalE1Sampler(
        seed=seed,
        gameplay_horizon=200,
        event_tail_steps=0,
    )
    schedules = Counter()
    contexts = Counter()
    joint = Counter()
    low_prefix_checked = 0
    for _ in range(int(draws)):
        draw = sampler.sample()
        schedules[draw.schedule_stratum] += 1
        contexts[draw.context_stratum] += 1
        joint[(draw.schedule_stratum, draw.context_stratum)] += 1
        steps = tuple(int(value) for value in draw.event_steps)
        if len(steps) != 5 or tuple(sorted(set(steps))) != steps:
            raise RuntimeError("temporal sampler produced a malformed schedule")
        if steps[0] < 0 or steps[-1] >= 200:
            raise RuntimeError("temporal sampler produced an out-of-range schedule")
        if draw.schedule_stratum == "early_fifth" and not (
                EARLY_FIFTH_INTERVAL[0] <= steps[-1] < EARLY_FIFTH_INTERVAL[1]
        ):
            raise RuntimeError("early-fifth stratum violated its interval")
        if draw.schedule_stratum == "late_fifth" and not (
                LATE_FIFTH_INTERVAL[0] <= steps[-1] < LATE_FIFTH_INTERVAL[1]
        ):
            raise RuntimeError("late-fifth stratum violated its interval")
        values = np.asarray(draw.opponent_commitment, dtype=np.float64)
        if values.shape != (5,) or np.any(values < 0.0) or np.any(values > 1.0):
            raise RuntimeError("temporal sampler produced an invalid context")
        if draw.context_stratum == "low_prefix":
            low_prefix_checked += 1
            if np.any(values[:4] > LOW_PREFIX_PRICE_HIGH):
                raise RuntimeError("low-prefix stratum violated its support")

    schedule_frequencies = {
        name: schedules[name] / float(draws) for name in SCHEDULE_STRATUM_WEIGHTS
    }
    context_frequencies = {
        name: contexts[name] / float(draws) for name in CONTEXT_STRATUM_WEIGHTS
    }
    for name, expected in SCHEDULE_STRATUM_WEIGHTS.items():
        if abs(schedule_frequencies[name] - expected) > FREQUENCY_TOLERANCE:
            raise RuntimeError(f"schedule stratum {name!r} frequency is out of tolerance")
    for name, expected in CONTEXT_STRATUM_WEIGHTS.items():
        if abs(context_frequencies[name] - expected) > FREQUENCY_TOLERANCE:
            raise RuntimeError(f"context stratum {name!r} frequency is out of tolerance")
    maximum_independence_error = max(
        abs(
            joint[(schedule_name, context_name)] / float(draws)
            - schedule_frequencies[schedule_name]
            * context_frequencies[context_name]
        )
        for schedule_name in SCHEDULE_STRATUM_WEIGHTS
        for context_name in CONTEXT_STRATUM_WEIGHTS
    )
    if maximum_independence_error > INDEPENDENCE_TOLERANCE:
        raise RuntimeError("temporal schedule/context strata are not independent")
    return {
        "seed": int(seed),
        "draws": int(draws),
        "schedule_frequencies": schedule_frequencies,
        "context_frequencies": context_frequencies,
        "maximum_independence_error": float(maximum_independence_error),
        "frequency_tolerance": FREQUENCY_TOLERANCE,
        "independence_tolerance": INDEPENDENCE_TOLERANCE,
        "low_prefix_draws_checked": int(low_prefix_checked),
        "passed": True,
    }


def _training_args(activation, *, device):
    temporary_checkpoint = (
        Path("/private/tmp")
        / f"stackpomdp-e1-temporal-preflight-{os.getpid()}-must-not-exist.zip"
    )
    if temporary_checkpoint.exists():
        raise FileExistsError(f"preflight no-save sentinel already exists: {temporary_checkpoint}")
    args = trainer.parse_args([
        "--role", "buyer",
        "--seed", "1",
        "--timesteps", str(validator.ADDITIONAL_TIMESTEPS),
        "--gameplay-horizon", "200",
        "--event-tail-steps", "0",
        "--e1-sampler-mode", TEMPORAL_MIX_E1_SAMPLER,
        "--num-envs", "1",
        "--n-steps", "205",
        "--batch-size", "205",
        "--n-epochs", "4",
        "--learning-rate", "0.0001",
        "--pretrained-lr-scale", "0.1",
        "--actor-loss-mode", "balanced",
        "--entropy-coeff", "0.01",
        "--clip-range", "0.1",
        "--value-coefficient", "0.5",
        "--max-grad-norm", "0.5",
        "--noop-max", "30",
        "--max-frames", "100000",
        "--rom-path", activation["rom"]["path"],
        "--e0b-checkpoint", activation["e0b_source"]["path"],
        "--resume", activation["resume_source"]["path"],
        "--checkpoint", str(temporary_checkpoint),
        "--checkpoint-every", str(validator.CHECKPOINT_INTERVAL),
        "--eval-episodes", "1",
        "--fixed-eval-episodes", "1",
        "--device", device,
        "--no-wandb",
    ])
    return args, temporary_checkpoint


def _sampler_plumbing(model, args, activation):
    if model.atari_e1_sampler_provenance != activation["protocol"]["training_sampler"]:
        raise RuntimeError("resume did not attach the temporal sampler provenance")
    history = model.atari_e1_sampler_history
    if len(history) != 2:
        raise RuntimeError("resume did not retain exactly uniform and temporal stages")
    if history[0]["sampler"]["mode"] != UNIFORM_E1_SAMPLER:
        raise RuntimeError("resume sampler history does not start with uniform training")
    if history[1]["start_total_timesteps"] != activation["resume_source"]["training_total_timesteps"]:
        raise RuntimeError("temporal sampler stage starts at the wrong timestep")
    sources = history[1].get("resume_sources", [])
    if len(sources) != 1 or sources[0]["sha256"] != activation["resume_source"]["sha256"]:
        raise RuntimeError("temporal sampler stage is not byte-bound to rank one")
    evaluation_args = trainer.canonical_evaluation_args(args)
    expected_evaluation = e1_sampler_provenance(
        UNIFORM_E1_SAMPLER, gameplay_horizon=200, event_tail_steps=0
    )
    if evaluation_args.e1_sampler_mode != UNIFORM_E1_SAMPLER:
        raise RuntimeError("canonical evaluator inherited the treatment sampler")
    if trainer._sampler_provenance(evaluation_args) != expected_evaluation:
        raise RuntimeError("canonical evaluator provenance is not uniform")
    if model.atari_e1_sampler_provenance["mode"] != TEMPORAL_MIX_E1_SAMPLER:
        raise RuntimeError("evaluation argument conversion mutated checkpoint provenance")
    return {
        "training_sampler": model.atari_e1_sampler_provenance,
        "training_sampler_history": history,
        "evaluation_sampler": expected_evaluation,
        "passed": True,
    }


def _run_one_ale_episode(model, vec_env):
    observation = vec_env.reset()
    terminal_info = None
    for transition in range(1, 207):
        action, _ = model.predict(observation, deterministic=True)
        observation, rewards, dones, infos = vec_env.step(action)
        if not np.all(np.isfinite(rewards)):
            raise RuntimeError("real-ALE preflight produced a nonfinite reward")
        if bool(dones[0]):
            terminal_info = dict(infos[0])
            break
    if terminal_info is None or transition != 205:
        raise RuntimeError(f"real-ALE E1 episode had {transition} transitions, expected 205")
    required = {
        "gameplay_transitions": 200,
        "trade_transitions": 5,
        "reward_transition_count": 205,
        "outer_transition_count": 205,
        "e1_sampler_mode": TEMPORAL_MIX_E1_SAMPLER,
    }
    for key, expected in required.items():
        if terminal_info.get(key) != expected:
            raise RuntimeError(f"real-ALE terminal {key}={terminal_info.get(key)!r}, expected {expected!r}")
    if len(terminal_info.get("events", [])) != 5:
        raise RuntimeError("real-ALE preflight did not execute exactly five trades")
    if terminal_info.get("e1_schedule_stratum") not in SCHEDULE_STRATUM_WEIGHTS:
        raise RuntimeError("real-ALE preflight did not expose its schedule stratum")
    if terminal_info.get("e1_context_stratum") not in CONTEXT_STRATUM_WEIGHTS:
        raise RuntimeError("real-ALE preflight did not expose its context stratum")
    purchases = int(terminal_info["purchases"])
    bullets_arrived = int(terminal_info["bullets_arrived"])
    payments = float(terminal_info["payments"])
    if not (0 <= purchases == bullets_arrived <= 5):
        raise RuntimeError("real-ALE bullet-transfer accounting failed")
    if not math.isfinite(payments) or payments < 0.0 or payments > purchases + 1e-6:
        raise RuntimeError("real-ALE payment accounting failed")
    return {
        key: terminal_info[key]
        for key in (
            "gameplay_transitions", "trade_transitions",
            "reward_transition_count", "outer_transition_count",
            "e1_sampler_mode", "e1_schedule_stratum",
            "e1_context_stratum", "event_steps", "opponent_commitment",
            "purchases", "bullets_arrived", "payments",
            "buyer_game_reward", "buyer_shots_fired", "buyer_final_ammo",
        )
    }


def validate_preflight(path, *, activation_path):
    value = validator.load_json(path)
    if value.get("kind") != PREFLIGHT_KIND or value.get("passed") is not True:
        raise ValueError("unknown or failed temporal contingency preflight")
    if value.get("activation_sha256") != validator.sha256_file(activation_path):
        raise ValueError("preflight belongs to another activation manifest")
    if value.get("real_ale_episodes") != 1:
        raise ValueError("preflight did not run exactly one real-ALE episode")
    if value.get("wandb_initialized") is not False or value.get("checkpoint_saved") is not False:
        raise ValueError("preflight performed a forbidden external write")
    return value


def run_preflight(args):
    activation_path = Path(args.activation).expanduser().resolve()
    activation = validator.validate_activation(activation_path)
    if validator._git_revision(args.code_root) != activation["code_revision"]:
        raise RuntimeError("preflight code revision differs from activation")
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        return validate_preflight(output, activation_path=activation_path)
    sampler = audit_sampler(seed=args.seed, draws=args.sampler_draws)
    training_args, no_save_path = _training_args(activation, device=args.device)
    vec_env = make_vec_env(
        lambda rank: trainer.make_env(
            training_args, seed=training_args.seed + 10_000 * rank
        ),
        num_envs=1,
        start_method="spawn",
    )
    try:
        model = trainer.build_model(training_args, vec_env)
        plumbing = _sampler_plumbing(model, training_args, activation)
        episode = _run_one_ale_episode(model, vec_env)
    finally:
        vec_env.close()
    if no_save_path.exists():
        raise RuntimeError("preflight unexpectedly saved a treatment checkpoint")
    result = {
        "schema_version": 1,
        "kind": PREFLIGHT_KIND,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "activation": str(activation_path),
        "activation_sha256": validator.sha256_file(activation_path),
        "code_revision": activation["code_revision"],
        "sampler_audit": sampler,
        "sampler_plumbing": plumbing,
        "real_ale_episode": episode,
        "real_ale_episodes": 1,
        "wandb_initialized": False,
        "checkpoint_saved": False,
        "passed": True,
    }
    validator.atomic_write_json(output, result)
    return validate_preflight(output, activation_path=activation_path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--seed", type=int, default=97_531)
    parser.add_argument("--sampler-draws", type=int, default=SAMPLER_DRAWS)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.sampler_draws < 10_000:
        parser.error("preflight requires at least 10000 sampler draws")
    return args


def main(argv=None):
    result = run_preflight(parse_args(argv))
    print({
        "passed": result["passed"],
        "real_ale_episodes": result["real_ale_episodes"],
        "checkpoint_saved": result["checkpoint_saved"],
        "wandb_initialized": result["wandb_initialized"],
    }, flush=True)


if __name__ == "__main__":
    main()
