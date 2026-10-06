"""Execute deterministic Atari leader evaluations and retain their complete trajectories."""

from pathlib import Path

import numpy as np

from experiments.atari._support.protocol import (
    ACTION_CREDIT,
    ACTION_MASK,
    ACTOR_STATE,
    CACHED_TRADE_REPLAY,
    IMAGE,
    LEADER_QUERY,
    NUM_TRADE_EVENTS,
)
from experiments.atari._support.training import (
    model_actor_loss_mode,
    model_economic_initialization,
)
from experiments.atari._support.checkpoints.atari_evaluation import validate_candidate_provenance
from experiments.atari._support.checkpoints.files import checkpoint_sha256
from experiments.atari.game import BilateralAtariConfig
from experiments.atari._support.evaluation.contracts import (
    E2_INIT_CONCENTRATION,
    E2_INIT_MEAN,
    ECONOMIC_CONTROL_COMMITMENTS,
    _action_list,
    _jsonable,
    actor_observation_sha256,
)
from experiments.atari._support.evaluation.interventions import (
    apply_economic_commitment_override,
    economic_intervention_manifest,
)
from experiments.atari._support.evaluation.protocol_audit import audit_e2_protocol
from experiments.atari._support.wrappers.meta_follower import make_stackpomdp_atari_leader_env


def make_e2_env(args, *, seed, response_model):
    config = BilateralAtariConfig(
        seed=int(seed),
        gameplay_horizon=int(args.gameplay_horizon),
        event_tail_steps=int(args.event_tail_steps),
        seller_game_reward_scale=0.1,
        buyer_game_reward_scale=1.0,
        noop_max=int(args.noop_max),
        frame_skip=4,
        frame_stack=4,
        episodic_life=True,
        clip_game_rewards=True,
        max_frames=int(args.max_frames),
        rom_path=args.rom_path,
        fixed_event_steps=args.fixed_event_steps,
    )
    return make_stackpomdp_atari_leader_env(
        leader_role=args.leader_role,
        response_checkpoint=args.response_checkpoint,
        config=config,
        response_model_factory=lambda path, device: response_model,
        device=args.device,
    )


def evaluate_e2_model(
        model,
        env_factory,
        *,
        episodes,
        seed_start,
        checkpoint_path,
        checkpoint_hash,
        response_hash,
        config_hash,
        provenance_fingerprint,
        phase,
        economic_intervention,
):
    """Run deterministic episodes while retaining complete protocol rows."""

    episode_rows, transition_rows, decision_rows = [], [], []
    for episode in range(int(episodes)):
        env = env_factory(episode)
        env.game.capture_transitions = True
        metadata = dict(phase=phase, checkpoint_path=str(checkpoint_path),
            checkpoint_sha256=checkpoint_hash, response_sha256=response_hash,
            environment_config_sha256=config_hash, e2_provenance_fingerprint=provenance_fingerprint,
            evaluation_episode=episode, evaluation_seed=int(seed_start) + episode,
            economic_intervention_id=economic_intervention["intervention_id"],
            economic_intervention_sha256=economic_intervention["manifest_sha256"],
            economic_commitment_override=economic_intervention["economic_commitment"])
        try:
            observation, total_reward, done, trace_position = env.reset(), 0., False, 0
            override_counts = {kind: np.zeros(NUM_TRADE_EVENTS, dtype=np.int64)
                               for kind in (LEADER_QUERY, CACHED_TRADE_REPLAY)}
            original_queries = {}
            while not done:
                action, _ = model.predict(observation, deterministic=True)
                policy_action = _action_list(action)
                requested, overridden = apply_economic_commitment_override(
                    observation, policy_action, economic_intervention["economic_commitment"])
                if overridden is not None:
                    original_queries[overridden[1]] = policy_action
                observation, _, done, terminal_info = env.step(np.asarray(requested, dtype=np.float32))
                for record in env.game.transition_records[trace_position:]:
                    substep = record["substep_type"]
                    event = record.get("query_index") if substep == LEADER_QUERY else record.get("event_index")
                    economic = substep in (LEADER_QUERY, CACHED_TRADE_REPLAY)
                    applied = economic and economic_intervention["economic_commitment"] is not None
                    if applied:
                        override_counts[substep][event] += 1
                    total_reward += record["reward"]
                    row = dict(metadata, **record, transition_index=trace_position,
                        cumulative_return=total_reward, economic_override_applied=applied,
                        economic_override_kind=substep if applied else None,
                        economic_override_event=event if applied else None,
                        policy_action_before_intervention=(original_queries[event] if applied
                            else record["requested_action"]),
                        )
                    row["cache_hit"] = substep == CACHED_TRADE_REPLAY
                    transition_rows.append(row)
                    if economic:
                        decision_rows.append(dict(row))
                    trace_position += 1
            terminal = _jsonable(dict(terminal_info))
            episode_rows.append(dict(metadata, evaluation_return=total_reward,
                evaluation_steps=trace_position,
                economic_override_query_applications=int(sum(override_counts[LEADER_QUERY])),
                economic_override_replay_applications=int(sum(override_counts[CACHED_TRADE_REPLAY])),
                economic_override_query_event_counts=override_counts[LEADER_QUERY].tolist(),
                economic_override_replay_event_counts=override_counts[CACHED_TRADE_REPLAY].tolist(),
                terminal_episode_summary=terminal.get("episode"),
                **{key: value for key, value in terminal.items() if key != "episode"}))
        finally:
            env.close()
    return dict(episode_rows=episode_rows, transition_rows=transition_rows, decision_rows=decision_rows)


def leader_outcome_summary(rows):
    rows = list(rows)
    if not rows:
        return {"episodes": 0, "mean_leader_payoff": None}
    payoff = np.asarray([float(row["leader_reward"]) for row in rows])
    purchases = np.asarray([float(row["purchases"]) for row in rows])
    payments = np.asarray([float(row["payments"]) for row in rows])
    seller_shots = np.asarray([
        float(row["seller_shots_fired"]) for row in rows
    ])
    buyer_shots = np.asarray([
        float(row["buyer_shots_fired"]) for row in rows
    ])
    retained_bullets = NUM_TRADE_EVENTS - purchases
    return {
        "episodes": len(rows),
        "mean_leader_payoff": float(np.mean(payoff)),
        "median_leader_payoff": float(np.median(payoff)),
        "std_leader_payoff": float(np.std(payoff)),
        "min_leader_payoff": float(np.min(payoff)),
        "max_leader_payoff": float(np.max(payoff)),
        "mean_seller_payoff": float(np.mean([
            float(row["seller_reward"]) for row in rows
        ])),
        "mean_buyer_payoff": float(np.mean([
            float(row["buyer_reward"]) for row in rows
        ])),
        "mean_payments": float(np.mean(payments)),
        "mean_purchases": float(np.mean(purchases)),
        "mean_purchase_rate": float(
            np.mean(purchases) / NUM_TRADE_EVENTS
        ),
        "mean_accepted_price": float(
            np.sum(payments) / np.sum(purchases)
        ) if float(np.sum(purchases)) > 0.0 else 0.0,
        "mean_seller_shots_fired": float(np.mean(seller_shots)),
        "mean_buyer_shots_fired": float(np.mean(buyer_shots)),
        "mean_total_shots_fired": float(np.mean(seller_shots + buyer_shots)),
        "buyer_purchased_bullet_utilization": float(
            np.sum(buyer_shots) / np.sum(purchases)
        ) if float(np.sum(purchases)) > 0.0 else 0.0,
        "mean_seller_retained_bullets": float(np.mean(retained_bullets)),
        "seller_retained_bullet_utilization": float(
            np.sum(seller_shots) / np.sum(retained_bullets)
        ) if float(np.sum(retained_bullets)) > 0.0 else 0.0,
        "mean_seller_final_ammo": float(np.mean([
            float(row["seller_final_ammo"]) for row in rows
        ])),
        "mean_buyer_final_ammo": float(np.mean([
            float(row["buyer_final_ammo"]) for row in rows
        ])),
        "mean_seller_game_reward": float(np.mean([
            float(row["seller_game_reward"]) for row in rows
        ])),
        "mean_buyer_game_reward": float(np.mean([
            float(row["buyer_game_reward"]) for row in rows
        ])),
    }


def evaluate_checkpoint(
        model,
        checkpoint,
        *,
        args,
        response_model,
        response_hash,
        config,
        config_hash,
        episodes,
        seed_start,
        phase,
        intervention_id="factual",
        economic_commitment_override=None,
        expected_checkpoint_sha256=None,
):
    checkpoint = Path(checkpoint).resolve()
    digest = checkpoint_sha256(checkpoint)
    if (
            expected_checkpoint_sha256 is not None
            and digest != expected_checkpoint_sha256
    ):
        raise RuntimeError("E2 candidate bytes changed before evaluation")
    loaded = getattr(model, "e2_evaluation_loaded_checkpoint", None)
    if not isinstance(loaded, dict) or loaded != {
            "path": str(checkpoint), "sha256": digest,
    }:
        raise RuntimeError(
            "E2 in-memory model is not bound to the reported checkpoint bytes"
        )
    provenance = validate_candidate_provenance(
        model, response_hash=response_hash, config=config
    )
    actor_loss_mode = model_actor_loss_mode(model)
    economic_initialization = model_economic_initialization(
        model,
        default_mean=E2_INIT_MEAN,
        default_concentration=E2_INIT_CONCENTRATION,
    )
    intervention = economic_intervention_manifest(
        intervention_id=intervention_id,
        commitment=economic_commitment_override,
        checkpoint_hash=digest,
        response_hash=response_hash,
        config_hash=config_hash,
        provenance_fingerprint=provenance["fingerprint_sha256"],
    )
    evaluation = evaluate_e2_model(
        model,
        lambda episode: make_e2_env(
            args, seed=int(seed_start) + int(episode),
            response_model=response_model,
        ),
        episodes=episodes,
        seed_start=seed_start,
        checkpoint_path=checkpoint,
        checkpoint_hash=digest,
        response_hash=response_hash,
        config_hash=config_hash,
        provenance_fingerprint=provenance["fingerprint_sha256"],
        phase=phase,
        economic_intervention=intervention,
    )
    protocol = audit_e2_protocol(
        evaluation,
        required_episodes=episodes,
        leader_role=args.leader_role,
        gameplay_horizon=config["gameplay_horizon"],
        event_tail_steps=config["event_tail_steps"],
        fixed_event_steps=config["fixed_event_steps"],
        seller_game_reward_scale=config["seller_game_reward_scale"],
        buyer_game_reward_scale=config["buyer_game_reward_scale"],
    )
    if checkpoint_sha256(checkpoint) != digest:
        raise RuntimeError("E2 candidate bytes changed during evaluation")
    return {
        "checkpoint_id": checkpoint.stem,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": digest,
        "response_checkpoint_sha256": response_hash,
        "environment_config_sha256": config_hash,
        "e2_provenance_fingerprint": provenance["fingerprint_sha256"],
        "e2_provenance_manifest": provenance,
        "training_total_timesteps": int(getattr(model, "num_timesteps", 0)),
        "economic_role": model.policy.economic_role,
        "economic_input_mode": model.policy.economic_input_mode,
        "actor_loss_mode": actor_loss_mode,
        "target_kl": getattr(model, "target_kl", None),
        "economic_head_initialization": economic_initialization,
        "economic_intervention": intervention,
        "phase": phase,
        "seed_start": int(seed_start),
        "seed_end": int(seed_start) + int(episodes) - 1,
        "summary": leader_outcome_summary(evaluation["episode_rows"]),
        "protocol": protocol,
        **evaluation,
    }


def evaluate_endpoint_controls(
        model,
        checkpoint,
        *,
        args,
        response_model,
        response_hash,
        config,
        config_hash,
        episodes,
        seed_start,
        phase,
        expected_checkpoint_sha256,
):
    """Evaluate both canonical endpoint commitments on fresh matched episodes."""

    return [
        evaluate_checkpoint(
            model,
            checkpoint,
            args=args,
            response_model=response_model,
            response_hash=response_hash,
            config=config,
            config_hash=config_hash,
            episodes=episodes,
            seed_start=seed_start,
            phase=phase,
            intervention_id=intervention_id,
            economic_commitment_override=commitment,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
        )
        for intervention_id, commitment in ECONOMIC_CONTROL_COMMITMENTS.items()
    ]
