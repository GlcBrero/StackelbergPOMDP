"""Train the native SB3 Atari buyer through the staged curriculum.

The same hybrid policy class is used in every stage.  Stage gates decide which
actor loss is active:

* gameplay: five initial bullets, gameplay loss only;
* free_trade: five zero-price offers, gameplay loss only;
* priced: frozen deterministic gameplay, threshold loss at offers only;
* joint: gameplay and threshold losses together.
"""

import argparse
from collections import defaultdict, deque
import csv
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch as th


# The project environment deliberately ignores broken user-site packages.  Set
# this in the shell as well; retaining the assignment here documents the
# requirement in checkpoints launched outside conda activate.
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "stackpomdp-matplotlib")
)
os.environ.setdefault("WANDB_START_METHOD", "thread")

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from stackelberg_pomdp.atari import AtariBuyerEnvConfig, make_atari_buyer_env
from stackelberg_pomdp.atari.evaluation import (
    DEFAULT_FIXED_PRICES,
    evaluate_economics,
    evaluate_gameplay as evaluate_gameplay_policy,
)
from stackelberg_pomdp.atari.policy import PriceAwareAtariPolicy
from replication.atari.evaluate_price_aware_atari_sb3 import (
    write_trade_events_csv,
)


WANDB_PROJECT = "StackPOMDP"
WANDB_GROUP = "atari_price_aware_sb3"
JOB_TYPES = {
    "gameplay": "atari_sb3_e0_pretraining",
    "free_trade": "atari_sb3_free_trade_adaptation",
    "priced": "atari_sb3_e1_threshold",
    "joint": "atari_sb3_joint_finetuning",
}
STOCHASTIC_E1_JOB_TYPE = "atari_sb3_e1_stochastic_timing"


def checkpoint_with_zip(path):
    path = Path(path).expanduser().resolve()
    return path if path.suffix == ".zip" else path.with_suffix(".zip")


def environment_stage(policy_stage):
    return "priced" if policy_stage in {"priced", "joint"} else policy_stage


def env_config(args, *, seed, stage=None, fixed_price=None):
    resolved_stage = stage or environment_stage(args.stage)
    return AtariBuyerEnvConfig(
        stage=resolved_stage,
        seed=int(seed),
        initial_bullets=5 if resolved_stage == "gameplay" else 0,
        bullet_capacity=5,
        offer_chances=5,
        max_purchases=5,
        price_min=args.price_min,
        price_max=args.price_max,
        fixed_price=(args.fixed_price if fixed_price is None else fixed_price),
        offer_timing=args.offer_timing,
        offer_probability=args.offer_probability,
        actor_economic_context=args.actor_economic_context,
        noop_max=30,
        frame_skip=4,
        frame_stack=4,
        episodic_life=True,
        clip_game_rewards=True,
        max_steps=args.max_steps,
    )


def make_training_vec_env(args):
    constructors = []
    for rank in range(args.num_envs):
        seed = args.seed + 10_000 * rank

        def constructor(seed=seed):
            return make_atari_buyer_env(env_config(args, seed=seed))

        constructors.append(constructor)
    if args.num_envs == 1:
        return DummyVecEnv(constructors)
    return SubprocVecEnv(constructors, start_method=args.start_method)


def evaluate_gameplay(model, args, *, episodes=None):
    return evaluate_gameplay_policy(
        model,
        env_config(
            args,
            seed=args.eval_seed,
            stage=environment_stage(args.stage),
        ),
        episodes=int(episodes or args.eval_episodes),
        eval_seed=args.eval_seed,
    )


def evaluate_stage(model, args):
    if args.stage in {"gameplay", "free_trade"}:
        return evaluate_gameplay(model, args)
    return evaluate_economics(
        model,
        env_config(args, seed=args.eval_seed, stage="priced"),
        fixed_prices=args.fixed_eval_prices,
        episodes_per_price=args.eval_episodes_per_price,
        random_episodes=args.random_eval_episodes,
        eval_seed=args.eval_seed,
    )


def evaluation_metrics(result, *, total_timesteps):
    metrics = {
        f"evaluation/{key}": value
        for key, value in result["summary"].items()
        if isinstance(value, (int, float, bool))
    }
    for row in result.get("fixed_price_table", []):
        label = f"p{int(round(100 * float(row['price']))):03d}"
        for field in (
            "mean_net_reward",
            "mean_game_reward",
            "mean_payments",
            "mean_purchases",
            "mean_trade_opportunities",
            "aggregate_acceptance_rate",
            "mean_shots_fired",
            "aggregate_fired_fraction_of_purchases",
            "mean_threshold",
            "profitable_price_passed",
            "mean_offer_step",
            "mean_purchase_step",
            "mean_offer_normalized_timestep",
            "mean_purchase_normalized_timestep",
            "late_rejection_rate",
        ):
            if field in row:
                metrics[f"evaluation/fixed_{label}/{field}"] = row[field]
        for time_bin in ("early", "middle", "late"):
            for field in (
                    "offer_count",
                    "purchase_count",
                    "rejection_count",
                    "acceptance_rate",
                    "mean_threshold",
            ):
                key = f"{field}_{time_bin}"
                value = row.get(key)
                if isinstance(value, (int, float, bool)):
                    metrics[f"evaluation/fixed_{label}/{key}"] = value
    metrics["total_timesteps"] = int(total_timesteps)
    return metrics


def evaluation_key(result, stage, *, contextual_timing=False):
    summary = result["summary"]
    if stage == "gameplay":
        return (
            float(summary["five_point_gate_passed"]),
            float(summary["score_five_rate"]),
            float(summary["mean_game_reward"]),
            float(summary["fired_all_five_rate"]),
        )
    if stage == "free_trade":
        return (
            float(summary["free_trade_gate_passed"]),
            float(summary["score_five_rate"]),
            float(summary["mean_game_reward"]),
            float(summary["aggregate_fired_fraction_of_purchases"]),
        )
    if contextual_timing:
        # This treatment explicitly permits rational late-offer rejection.
        # Select the policy on its actual training objective rather than the
        # immediate-offer control's discontinuous buy-almost-everything gate.
        return (
            float(summary["random_mean_net_reward"]),
            float(summary["random_positive_net_reward_rate"]),
            float(summary["random_fired_fraction_of_purchases"]),
            float(summary["zero_price_acceptance_rate"]),
        )
    profitable_rows = [
        row
        for row in result["fixed_price_table"]
        if row["buying_profitable"]
    ]
    minimum_profitable_acceptance = min(
        (row["aggregate_acceptance_rate"] for row in profitable_rows),
        default=0.0,
    )
    return (
        float(summary["pass_condition"]),
        float(summary["random_mean_net_reward"]),
        float(minimum_profitable_acceptance),
        float(summary["random_fired_fraction_of_purchases"]),
    )


def write_evaluation_result(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_trade_events_csv(path.with_suffix(".trade_events.csv"), result)
    rows = result.get("fixed_price_table")
    if not rows:
        return None
    csv_path = path.with_suffix(".fixed_prices.csv")
    scalar_rows = [
        {
            key: value
            for key, value in row.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }
        for row in rows
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scalar_rows[0]))
        writer.writeheader()
        writer.writerows(scalar_rows)
    return csv_path


class AtariTrainingCallback(BaseCallback):
    def __init__(self, args, checkpoint_path, wandb_run=None):
        super().__init__(verbose=0)
        self.args = args
        self.checkpoint_path = checkpoint_path
        self.best_path = checkpoint_path.with_name(
            f"{checkpoint_path.stem}_best{checkpoint_path.suffix}"
        )
        self.wandb_run = wandb_run
        self.next_log = args.log_every
        self.next_checkpoint = args.checkpoint_every
        self.next_evaluation = args.eval_every
        self.recent = defaultdict(lambda: deque(maxlen=100))
        self.best_key = None
        self.started = time.time()
        self.starting_timesteps = None

    def _on_training_start(self):
        self.starting_timesteps = int(self.num_timesteps)
        for target_name, interval in (
                ("next_log", self.args.log_every),
                ("next_checkpoint", self.args.checkpoint_every),
                ("next_evaluation", self.args.eval_every),
        ):
            setattr(
                self,
                target_name,
                (
                    int(self.num_timesteps) // int(interval) + 1
                ) * int(interval),
            )

    def _record_episodes(self):
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if not episode:
                continue
            mappings = {
                "episode_reward": "r",
                "episode_game_reward": "game_reward",
                "episode_payments": "payments",
                "episode_length": "l",
                "shots_fired": "shots_fired",
                "purchases": "purchases",
                "trade_opportunities": "trade_opportunities",
                "final_ammo": "final_ammo",
                "reward_per_bullet": "reward_per_bullet",
                "acceptance_rate": "acceptance_rate",
                "fired_fraction_of_purchases": "fired_fraction_of_purchases",
                "unused_purchased_bullets": "unused_purchased_bullets",
                "offer_step_sum": "offer_step_sum",
                "offer_normalized_timestep_sum": (
                    "offer_normalized_timestep_sum"
                ),
                "purchase_step_sum": "purchase_step_sum",
                "purchase_normalized_timestep_sum": (
                    "purchase_normalized_timestep_sum"
                ),
                "last_offer_accepted": "last_offer_accepted",
            }
            for time_bin in ("early", "middle", "late"):
                mappings[f"offer_count_{time_bin}"] = (
                    f"offer_count_{time_bin}"
                )
                mappings[f"purchase_count_{time_bin}"] = (
                    f"purchase_count_{time_bin}"
                )
            for output, source in mappings.items():
                self.recent[output].append(float(episode.get(source, 0.0)))
            for count in range(6):
                name = f"opportunity_count_is_{count}"
                self.recent[name].append(float(episode.get(name, 0.0)))

    def _metrics(self):
        metrics = {
            f"train/{name}": float(np.mean(values))
            for name, values in self.recent.items()
            if values
        }
        total_offers = float(sum(self.recent["trade_opportunities"]))
        total_purchases = float(sum(self.recent["purchases"]))
        if total_offers:
            metrics["train/acceptance_rate_aggregate"] = (
                total_purchases / total_offers
            )
            metrics["train/mean_offer_step"] = float(
                sum(self.recent["offer_step_sum"]) / total_offers
            )
            metrics["train/mean_offer_normalized_timestep"] = float(
                sum(self.recent["offer_normalized_timestep_sum"])
                / total_offers
            )
        if total_purchases:
            metrics["train/mean_purchase_step"] = float(
                sum(self.recent["purchase_step_sum"]) / total_purchases
            )
            metrics["train/mean_purchase_normalized_timestep"] = float(
                sum(self.recent["purchase_normalized_timestep_sum"])
                / total_purchases
            )
        for time_bin in ("early", "middle", "late"):
            offers = float(sum(self.recent[f"offer_count_{time_bin}"]))
            purchases = float(
                sum(self.recent[f"purchase_count_{time_bin}"])
            )
            if offers:
                metrics[f"train/acceptance_rate_{time_bin}"] = (
                    purchases / offers
                )
                metrics[f"train/rejection_rate_{time_bin}"] = (
                    (offers - purchases) / offers
                )
        elapsed = max(time.time() - self.started, 1.0e-9)
        starting_timesteps = (
            int(self.starting_timesteps)
            if self.starting_timesteps is not None
            else 0
        )
        metrics.update({
            "total_timesteps": int(self.num_timesteps),
            "train/steps_per_second": float(
                max(int(self.num_timesteps) - starting_timesteps, 0) / elapsed
            ),
            "train/learning_rate": float(
                self.model.policy.optimizer.param_groups[0]["lr"]
            ),
        })
        return metrics

    def _log(self, metrics):
        print(json.dumps(metrics, sort_keys=True), flush=True)
        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=int(self.num_timesteps))

    def _save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        print(f"checkpoint={path}", flush=True)

    def _evaluate(self):
        step_path = self.checkpoint_path.with_name(
            f"{self.checkpoint_path.stem}_step{int(self.num_timesteps)}"
            f"{self.checkpoint_path.suffix}"
        )
        self._save(step_path)
        result = evaluate_stage(self.model, self.args)
        write_evaluation_result(
            step_path.with_suffix(".evaluation.json"), result
        )
        summary = result["summary"]
        self._log(
            evaluation_metrics(result, total_timesteps=self.num_timesteps)
        )

        contextual_timing = bool(
            self.args.actor_economic_context
            and self.args.offer_timing == "bernoulli"
        )
        key = evaluation_key(
            result,
            self.args.stage,
            contextual_timing=contextual_timing,
        )
        if self.best_key is None or key > self.best_key:
            self.best_key = key
            self._save(self.best_path)
            best_result_path = self.best_path.with_suffix(".evaluation.json")
            best_csv_path = write_evaluation_result(best_result_path, result)
            if self.wandb_run is not None:
                self.wandb_run.summary.update({
                    f"best/{name}": value for name, value in summary.items()
                })
                self.wandb_run.summary["best/checkpoint"] = str(self.best_path)
                self.wandb_run.summary["best/evaluation"] = str(best_result_path)
                if best_csv_path is not None:
                    self.wandb_run.summary["best/fixed_price_csv"] = str(
                        best_csv_path
                    )
                self.wandb_run.summary["best/trade_events_csv"] = str(
                    best_result_path.with_suffix(".trade_events.csv")
                )
        return result

    def _on_step(self):
        self._record_episodes()
        if self.num_timesteps >= self.next_log:
            self._log(self._metrics())
            while self.next_log <= self.num_timesteps:
                self.next_log += self.args.log_every
        if self.num_timesteps >= self.next_checkpoint:
            self._save(self.checkpoint_path)
            while self.next_checkpoint <= self.num_timesteps:
                self.next_checkpoint += self.args.checkpoint_every
        if self.num_timesteps >= self.next_evaluation:
            self._evaluate()
            while self.next_evaluation <= self.num_timesteps:
                self.next_evaluation += self.args.eval_every
        return True

    def _on_training_end(self):
        self._save(self.checkpoint_path)
        result = self._evaluate()
        result_path = self.checkpoint_path.with_suffix(".evaluation.json")
        csv_path = write_evaluation_result(result_path, result)
        if self.wandb_run is not None:
            self.wandb_run.summary["checkpoint_path"] = str(self.checkpoint_path)
            self.wandb_run.summary["evaluation_path"] = str(result_path)
            if csv_path is not None:
                self.wandb_run.summary["fixed_price_csv"] = str(csv_path)
            self.wandb_run.summary["trade_events_csv"] = str(
                result_path.with_suffix(".trade_events.csv")
            )


def init_wandb(args, checkpoint_path):
    if not args.wandb:
        return None
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        job_type=(
            args.wandb_job_type
            or (
                STOCHASTIC_E1_JOB_TYPE
                if args.actor_economic_context
                and args.offer_timing == "bernoulli"
                else JOB_TYPES[args.stage]
            )
        ),
        name=args.wandb_name,
        config={
            **vars(args),
            "algorithm": "SB3-PPO",
            "architecture": (
                "native_sb3_contextual_atari_v2"
                if args.actor_economic_context
                else "native_sb3_price_aware_atari_v1"
            ),
            "checkpoint_path": str(checkpoint_path),
            "actor_observes_price": args.actor_economic_context,
            "actor_observes_normalized_timestep": (
                args.actor_economic_context
            ),
            "critic_observes_price": True,
            "initial_bullets": 5 if args.stage == "gameplay" else 0,
            "no_replenishment": True,
            "offer_chances": 0 if args.stage == "gameplay" else 5,
            "max_purchases": 0 if args.stage == "gameplay" else 5,
            "clip_game_rewards": True,
            "noop_reset": True,
            "episodic_life": True,
            "frame_skip": 4,
            "frame_stack": 4,
            "fire_mask": True,
            "evaluation_policy": "deterministic_argmax",
            "offer_timing": args.offer_timing,
            "offer_probability": args.offer_probability,
            "one_shot_offers": True,
        },
    )
    run.define_metric("total_timesteps")
    run.define_metric("train/*", step_metric="total_timesteps")
    run.define_metric("evaluation/*", step_metric="total_timesteps")
    print(f"wandb_url={run.url}", flush=True)
    return run


def _new_model(args, vec_env):
    return PPO(
        PriceAwareAtariPolicy,
        vec_env,
        policy_kwargs={
            "stage": args.stage,
            "visual_features": 512,
            "ammo_features": 32,
            "market_features": 16,
            "threshold_hidden": 64,
            "actor_economic_context": bool(
                getattr(args, "actor_economic_context", False)
            ),
        },
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.entropy_coeff,
        vf_coef=args.value_coefficient,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        device=args.device,
        verbose=1,
    )


GAMEPLAY_STATE_PREFIXES = (
    "features_extractor.",
    "pi_features_extractor.",
    "vf_features_extractor.",
    "game_action_net.",
)


def copy_pretrained_gameplay(source_policy, target_policy):
    """Copy and verify only the protected E0 gameplay computation."""

    source_state = source_policy.state_dict()
    target_state = target_policy.state_dict()
    expected = {
        name
        for name in target_state
        if name.startswith(GAMEPLAY_STATE_PREFIXES)
    }
    copied = set()
    for name in sorted(expected):
        source_tensor = source_state.get(name)
        if source_tensor is None:
            raise RuntimeError(f"pretrained gameplay tensor missing: {name}")
        if source_tensor.shape != target_state[name].shape:
            raise RuntimeError(
                "pretrained gameplay tensor shape changed for "
                f"{name}: {tuple(source_tensor.shape)} != "
                f"{tuple(target_state[name].shape)}"
            )
        target_state[name] = source_tensor.detach().clone()
        copied.add(name)
    if not expected or copied != expected:
        raise RuntimeError("failed to identify every pretrained gameplay tensor")
    target_policy.load_state_dict(target_state, strict=True)
    migrated_state = target_policy.state_dict()
    mismatched = [
        name
        for name in sorted(copied)
        if not th.equal(migrated_state[name], source_state[name])
    ]
    if mismatched:
        raise RuntimeError(
            f"pretrained gameplay verification failed: {mismatched[:3]}"
        )
    return tuple(sorted(copied))


def _load_existing_model(args, vec_env):
    model = PPO.load(
        args.resume,
        env=vec_env,
        device=args.device,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.entropy_coeff,
        clip_range=args.clip_range,
        vf_coef=args.value_coefficient,
        max_grad_norm=args.max_grad_norm,
    )
    stage_changed = model.policy.stage != args.stage
    model.policy.set_stage(args.stage, rebuild_optimizer=stage_changed)
    model.policy_kwargs = dict(model.policy_kwargs)
    model.policy_kwargs["stage"] = args.stage
    for parameter_group in model.policy.optimizer.param_groups:
        parameter_group["lr"] = args.learning_rate
    return model


def build_or_load_model(args, vec_env):
    if not args.resume:
        return _new_model(args, vec_env)

    target_context = bool(getattr(args, "actor_economic_context", False))
    if not target_context:
        return _load_existing_model(args, vec_env)

    source = PPO.load(args.resume, device=args.device)
    source_context = bool(
        getattr(source.policy, "actor_economic_context", False)
    )
    if source_context:
        del source
        return _load_existing_model(args, vec_env)

    model = _new_model(args, vec_env)
    copied = copy_pretrained_gameplay(source.policy, model.policy)
    model.policy.set_stage(args.stage, rebuild_optimizer=True)
    model.policy_kwargs = dict(model.policy_kwargs)
    model.policy_kwargs["stage"] = args.stage
    model.policy_kwargs["actor_economic_context"] = True
    print(json.dumps({
        "migration": "e0_gameplay_to_contextual_e1",
        "source_checkpoint": str(Path(args.resume).expanduser().resolve()),
        "copied_gameplay_tensors": len(copied),
        "economic_head_initialized_fresh": True,
    }, sort_keys=True), flush=True)
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["gameplay", "free_trade", "priced", "joint"],
        default="gameplay",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument(
        "--timesteps-are-target",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "interpret --timesteps as the desired final model timestep when "
            "resuming, rather than as additional training"
        ),
    )
    parser.add_argument("--resume")
    parser.add_argument("--checkpoint")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--start-method", choices=["spawn", "forkserver", "fork"], default="spawn")
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--gae-lambda", type=float)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--clip-range", type=float, default=0.1)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--offer-timing",
        choices=["immediate", "bernoulli"],
        default="immediate",
    )
    parser.add_argument("--offer-probability", type=float, default=0.04)
    parser.add_argument(
        "--actor-economic-context",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--price-min", type=float, default=0.0)
    parser.add_argument("--price-max", type=float, default=1.0)
    parser.add_argument("--fixed-price", type=float, default=0.0)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument(
        "--fixed-eval-prices",
        default=",".join(str(price) for price in DEFAULT_FIXED_PRICES),
    )
    parser.add_argument("--eval-episodes-per-price", type=int, default=10)
    parser.add_argument("--random-eval-episodes", type=int, default=50)
    parser.add_argument("--eval-seed", type=int, default=100_001)
    parser.add_argument("--eval-every", type=int, default=100_000)
    parser.add_argument("--checkpoint-every", type=int, default=100_000)
    parser.add_argument("--log-every", type=int, default=10_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", default=WANDB_PROJECT)
    parser.add_argument("--wandb-entity", default="glcbrero")
    parser.add_argument("--wandb-group", default=WANDB_GROUP)
    parser.add_argument("--wandb-job-type")
    parser.add_argument("--wandb-name")
    return parser.parse_args()


def validate_args(args):
    if args.timesteps <= 0:
        raise ValueError("timesteps must be positive")
    if args.num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if not 0.0 <= args.offer_probability <= 1.0:
        raise ValueError("offer_probability must lie in [0, 1]")
    if args.actor_economic_context and args.max_steps is None:
        args.max_steps = 125
    if args.actor_economic_context and args.stage not in {"priced", "joint"}:
        raise ValueError(
            "actor economic context is intended for priced or joint training"
        )
    if args.n_steps <= 0 or args.batch_size <= 1 or args.n_epochs <= 0:
        raise ValueError("invalid PPO batch geometry")
    if (
            args.eval_episodes <= 0
            or args.eval_episodes_per_price <= 0
            or args.random_eval_episodes <= 0
    ):
        raise ValueError("evaluation episode counts must be positive")
    args.fixed_eval_prices = tuple(
        float(value.strip())
        for value in str(args.fixed_eval_prices).split(",")
        if value.strip()
    )
    if (
            not args.fixed_eval_prices
            or not any(np.isclose(price, 0.0) for price in args.fixed_eval_prices)
            or any(price < 0.0 or price > 1.0 for price in args.fixed_eval_prices)
    ):
        raise ValueError(
            "fixed evaluation prices must lie in [0, 1] and include zero"
        )
    rollout_size = args.n_steps * args.num_envs
    if rollout_size % args.batch_size:
        raise ValueError(
            "n_steps * num_envs must be divisible by batch_size; "
            f"got {rollout_size} and {args.batch_size}"
        )
    if args.stage in {"priced", "joint"}:
        args.gamma = 1.0 if args.gamma is None else args.gamma
        args.gae_lambda = 1.0 if args.gae_lambda is None else args.gae_lambda
    else:
        args.gamma = 0.99 if args.gamma is None else args.gamma
        args.gae_lambda = 0.95 if args.gae_lambda is None else args.gae_lambda
    if args.stage != "gameplay" and not args.resume:
        raise ValueError(f"stage {args.stage!r} requires --resume")


def main():
    args = parse_args()
    validate_args(args)
    if args.wandb_name is None:
        variant = (
            "_stochastic_context"
            if args.actor_economic_context
            and args.offer_timing == "bernoulli"
            else ""
        )
        args.wandb_name = (
            f"sb3_{args.stage}_five_bullet_ppo{variant}_seed{args.seed}_"
            f"{args.timesteps // 1_000_000}m_local"
        )
    checkpoint_path = checkpoint_with_zip(
        args.checkpoint
        or (
            "replication/atari/checkpoints/sb3/"
            f"space_invaders_{args.stage}_ppo_seed{args.seed}_{args.timesteps}.zip"
        )
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    vec_env = make_training_vec_env(args)
    model = None
    run = None
    try:
        model = build_or_load_model(args, vec_env)
        run = init_wandb(args, checkpoint_path)
        callback = AtariTrainingCallback(args, checkpoint_path, run)
        learn_timesteps = int(args.timesteps)
        if args.timesteps_are_target:
            learn_timesteps = max(
                int(args.timesteps) - int(model.num_timesteps), 0
            )
            if learn_timesteps == 0:
                raise ValueError(
                    "checkpoint has already reached --timesteps target"
                )
        model.learn(
            total_timesteps=learn_timesteps,
            callback=callback,
            reset_num_timesteps=args.resume is None,
        )
        if run is not None:
            artifact_variant = (
                "-stochastic-context"
                if args.actor_economic_context
                and args.offer_timing == "bernoulli"
                else ""
            )
            artifact = __import__("wandb").Artifact(
                f"sb3-atari-{args.stage}{artifact_variant}-seed{args.seed}",
                type="model",
                metadata={
                    "stage": args.stage,
                    "seed": args.seed,
                    "timesteps": int(model.num_timesteps),
                    "offer_timing": args.offer_timing,
                    "actor_economic_context": args.actor_economic_context,
                },
            )
            artifact.add_file(str(checkpoint_path))
            evaluation_path = checkpoint_path.with_suffix(".evaluation.json")
            if evaluation_path.exists():
                artifact.add_file(str(evaluation_path))
            fixed_price_path = evaluation_path.with_suffix(
                ".fixed_prices.csv"
            )
            if fixed_price_path.exists():
                artifact.add_file(str(fixed_price_path))
            trade_events_path = evaluation_path.with_suffix(
                ".trade_events.csv"
            )
            if trade_events_path.exists():
                artifact.add_file(str(trade_events_path))
            run.log_artifact(artifact)
    finally:
        if run is not None:
            run.finish()
        if model is not None and model.get_env() is not None:
            model.get_env().close()
        else:
            vec_env.close()


if __name__ == "__main__":
    main()
