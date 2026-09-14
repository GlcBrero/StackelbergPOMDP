"""Held-out reward-play evaluation with independently initialized responses."""

from pathlib import Path
import time
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from stackelberg_pomdp.matrix_ablations.artifacts import append_jsonl, mean_summary


def evaluate_leader_policy(model, env_factory, episodes, warmup, seed_start):
    """Evaluate reward play using a fresh response and seed for each episode."""
    env = env_factory(seed_start)
    rows = []
    try:
        if hasattr(model.policy, "fix_policy_actions"):
            model.policy.fix_policy_actions()
        for sequence_index in range(int(warmup) + int(episodes)):
            if hasattr(env.unwrapped, "seed"):
                env.unwrapped.seed(int(seed_start) + sequence_index)
            if hasattr(model.policy, "clear_obs_action_map"):
                model.policy.clear_obs_action_map()
            observation = env.reset()
            total = 0.0
            reward_phase_total = 0.0
            reward_phase_steps = 0
            executed_steps = 0
            commitment_consistent = True
            done = False
            while not done:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, done, info = env.step(action)
                total += float(reward)
                executed_steps += 1
                if info.get("is_reward_phase", False):
                    reward_phase_total += float(reward)
                    reward_phase_steps += 1
                    commitment_consistent = (
                        commitment_consistent
                        and info.get("commitment_consistent", True)
                    )
            if sequence_index >= warmup:
                rows.append({
                    "episode": int(sequence_index - warmup),
                    "seed": int(seed_start) + sequence_index,
                    "outer_return": total,
                    "reward_phase_return": reward_phase_total,
                    "reward_phase_steps": reward_phase_steps,
                    "leader_reward_per_stage": (
                        reward_phase_total / reward_phase_steps
                    ),
                    "executed_steps": executed_steps,
                    "commitment_consistent": bool(commitment_consistent),
                })
    finally:
        env.close()
        if hasattr(model.policy, "clear_obs_action_map"):
            model.policy.clear_obs_action_map()
    return {
        "response_state_source": "fresh_initialization",
        "per_stage_summary": mean_summary([
            row["leader_reward_per_stage"] for row in rows
        ]),
        "return_summary": mean_summary([
            row["reward_phase_return"] for row in rows
        ]),
        "episode_rows": rows,
    }


class JsonlEvaluationCallback(BaseCallback):
    """Evaluate at completed outer episodes on a fixed target-step grid."""

    def __init__(
            self,
            eval_factory,
            output_path,
            eval_freq,
            eval_episodes,
            eval_warmup,
            eval_seed_start,
            metadata,
            wandb_run=None,
    ):
        super().__init__()
        self.eval_factory = eval_factory
        self.output_path = Path(output_path)
        self.eval_freq = int(eval_freq)
        self.eval_episodes = int(eval_episodes)
        self.eval_warmup = int(eval_warmup)
        self.eval_seed_start = int(eval_seed_start)
        self.metadata = dict(metadata)
        self.wandb_run = wandb_run
        self.next_target = self.eval_freq

    def _on_step(self):
        dones = np.asarray(self.locals.get("dones", [False]), dtype=bool)
        if not np.any(dones) or self.num_timesteps < self.next_target:
            return True
        self.record(self.model, self.next_target)
        while self.next_target <= self.num_timesteps:
            self.next_target += self.eval_freq
        return True

    def record(self, model, target):
        """Write an evaluation; PG/SimpleQ call this after an optimizer update."""
        result = evaluate_leader_policy(
            model,
            self.eval_factory,
            self.eval_episodes,
            self.eval_warmup,
            self.eval_seed_start,
        )
        summary = result["per_stage_summary"]
        row = {
            **self.metadata,
            "evaluation_target_step": int(target),
            "global_step": int(model.num_timesteps),
            "policy_updates": int(model._n_updates),
            "wall_time": time.time(),
            "evaluation_mean": summary["mean"],
            "evaluation_std": summary["std"],
            "evaluation_sem": summary["sem"],
            "evaluation_episodes": summary["n"],
            "response_state_source": result["response_state_source"],
        }
        append_jsonl(self.output_path, row)
        if self.wandb_run is not None:
            self.wandb_run.log({
                "global_step": row["global_step"],
                "evaluation/leader_reward_mean": row["evaluation_mean"],
                "evaluation/leader_reward_sem": row["evaluation_sem"],
            })
