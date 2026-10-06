"""MSPM training, evaluation and command-line handling.

The game and response algorithm are defined in experiments/mspm/.
"""
import numpy as np
from experiments.mspm.experiment import build

def load_plan():
    import json
    from pathlib import Path
    return json.loads((Path(__file__).with_name("paper.json")).read_text())


def training_config(num_types, seed, *, episodes_per_rollout=40,
                    learning_rate=3e-4, entropy=0.):
    return dict(num_types=num_types, num_messages=2, seed=seed,
                mw_epsilon=.1, mw_fixed_seed=0, response_bcce_threshold=.05,
                response_bcce_max_extra_updates=10000,
                episodes_per_rollout=episodes_per_rollout,
                learning_rate=learning_rate, ent_coef=entropy)


def learning_rate_schedule(total_episodes, initial_rate, hold_episodes):
    def schedule(progress_remaining):
        completed = total_episodes * (1 - float(progress_remaining))
        if completed <= hold_episodes:
            return float(initial_rate)
        return initial_rate * max(0., (total_episodes - completed) /
                                  (total_episodes - hold_episodes))
    return schedule


def evaluate_commitments(model, config, repetitions, deterministic, eval_seed):
    """Isolated environment and RNG: evaluation cannot extend training's commitment."""
    import copy
    import random
    import torch
    state = copy.deepcopy((random.getstate(), np.random.get_state(), torch.get_rng_state()))
    env = build(config)
    try:
        random.seed(eval_seed)
        np.random.seed(eval_seed)
        torch.manual_seed(eval_seed)
        rows = []
        for repetition in range(repetitions):
            observation, done, total = env.reset(), False, 0.
            while not done:
                action, _ = model.predict(observation, deterministic=deterministic)
                observation, reward, done, info = env.step(action)
                total += reward
            certificate = info["response_certificate"]
            rows.append(dict(repetition=repetition, reward=total, raw_reward=info["unconstrained_reward"],
                bcce_gap=certificate["gap"], queries=info["policy_queries"],
                underlying_transitions=info["underlying_steps"]))
        return dict(deterministic=deterministic, repetitions=len(rows),
                    mean_reward=float(np.mean([row["reward"] for row in rows])),
                    mean_raw_reward=float(np.mean([row["raw_reward"] for row in rows])), rows=rows)
    finally:
        env.close()
        random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])


def construct_model(config, folder, total_episodes, initial_std=.5,
                    learning_rate=3e-4, hold_episodes=3000, entropy=0.):
    import math
    import torch
    from stackelberg_pomdp.policy import DefaultPolicy
    from support.episode_weighting import EpisodeWeightedPPO
    from experiments.mspm import experiment as mspm
    from stable_baselines3.common.logger import configure
    torch.set_num_threads(1)
    env = build(config)
    episodes = config["episodes_per_rollout"]
    policy = getattr(mspm, "POLICY", DefaultPolicy)
    model = EpisodeWeightedPPO(policy, env, n_steps=episodes, episodes_per_rollout=episodes,
        batch_size=64, n_epochs=10,
        gamma=1., gae_lambda=1., ent_coef=entropy, normalize_advantage=True,
        learning_rate=learning_rate_schedule(total_episodes, learning_rate, hold_episodes),
        seed=config["seed"], device="cpu")
    with torch.no_grad():
        model.policy.log_std.fill_(math.log(initial_std))
    model.set_logger(configure(str(folder / "training"), format_strings=[]))
    return model, config


def train_run(spec, folder, *, total_episodes=None, eval_every=None, sampled_repetitions=None):
    import json
    import time
    from support.common import fresh_output, record_run, write_json
    from stable_baselines3.common.callbacks import BaseCallback
    plan = load_plan()
    params = plan["parameters"]
    total_episodes = int(total_episodes or plan["total_episodes"])
    eval_every = int(eval_every or plan["eval_every_episodes"])
    sampled_repetitions = int(sampled_repetitions or plan["sampled_evaluation_repetitions"])
    config = training_config(spec["num_types"], spec["seed"],
        episodes_per_rollout=spec.get("episodes_per_rollout", params["ppo_episodes_per_batch"]))
    folder = fresh_output(folder)
    record_run(folder, "mspm", dict(config, total_episodes=total_episodes,
        initial_action_std=.5, critic="response state and private commitment", gamma=1., gae_lambda=1.))
    write_json(folder / "plan.json", plan)
    write_json(folder / "config.json", config)
    state = dict(spec, status="running", episodes=0, total_episodes=total_episodes)
    write_json(folder / "run.json", state)
    started = time.monotonic()
    model = None
    def append(path, row):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    class Callback(BaseCallback):
        def _on_step(self):
            return True
    try:
        model, _ = construct_model(config, folder, total_episodes,
            learning_rate=params["learning_rate"], hold_episodes=params["learning_rate_hold_episodes"])
        model.save(folder / "initial.zip")
        _, callback = model._setup_learn(total_episodes, Callback(), True, spec["key"], False)
        callback.on_training_start({}, {})
        def evaluate():
            result = evaluate_commitments(model, config, 1, True, 8_000_000 + spec["seed"])
            append(folder / "evaluation/curve.jsonl", dict(episodes=model.num_episodes,
                raw_reward=result["mean_raw_reward"], bcce_certified=1., **result))
        evaluate()
        while model.num_episodes < total_episodes:
            model.episodes_per_rollout = min(config["episodes_per_rollout"], total_episodes - model.num_episodes)
            if not model.collect_rollouts(model.env, callback, model.rollout_buffer, model.n_steps):
                raise RuntimeError("Training callback stopped collection")
            model._update_current_progress_remaining(model.num_episodes, total_episodes)
            model.train()
            buffer = model.rollout_buffer
            starts = np.flatnonzero(buffer.episode_starts[:, 0])
            rewards = buffer.returns[starts, 0]
            append(folder / "updates.jsonl", dict(episodes=model.num_episodes,
                completed_episodes=model.last_rollout_completed_episodes,
                stored_query_rows=buffer.buffer_size, credited_draws=int(buffer.actor_credit.sum()),
                training_reward_mean=float(rewards.mean()),
                policy_std=model.policy.log_std.detach().exp().tolist()))
            if model.num_episodes % eval_every == 0 or model.num_episodes == total_episodes:
                evaluate()
                model.save(folder / "latest.zip")
        callback.on_training_end()
        model.save(folder / "final.zip")
        state.update(status="completed", episodes=model.num_episodes,
            final_greedy=evaluate_commitments(model, config, 1, True, 8_000_000 + spec["seed"]),
            final_sampled=evaluate_commitments(model, config, sampled_repetitions, False, 9_000_000 + spec["seed"]),
            elapsed_seconds=time.monotonic() - started)
        write_json(folder / "run.json", state)
        return state
    except BaseException as error:
        state.update(status="failed", error=repr(error), elapsed_seconds=time.monotonic() - started)
        write_json(folder / "run.json", state)
        raise
    finally:
        if model is not None:
            model.env.close()


def main():
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Certified MSPM Stackelberg POMDP")
    parser.add_argument("--index", type=int)
    parser.add_argument("--types", type=int, choices=range(2, 7))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--sampled-repetitions", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key")
    args = parser.parse_args()
    if args.index is not None:
        if args.types is not None or args.seed is not None or not 0 <= args.index < 125:
            parser.error("Use --index 0–124 or --types and --seed")
        spec = load_plan()["runs"][args.index]
    else:
        if args.types is None or args.seed is None:
            parser.error("Specify --types and --seed")
        spec = dict(num_types=args.types, seed=args.seed, key=args.key or f"t{args.types}.s{args.seed}")
    state = train_run(spec, args.output, total_episodes=args.episodes,
                      eval_every=args.eval_every, sampled_repetitions=args.sampled_repetitions)
    print(json.dumps(state, sort_keys=True))


if __name__ == "__main__":
    main()
