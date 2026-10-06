"""Run a new experiment module using the shared default PPO trainer."""
import argparse
import importlib
import json
from pathlib import Path

from support.common import fresh_output, record_run, write_json


def main(module_name, arguments):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1024,
                        help="Minimum policy decisions; finish the current rollout.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=Path, help="JSON object passed to build().")
    parser.add_argument("--check", action="store_true",
                        help="Check one episode with sampled actions, without training.")
    parser.add_argument("--output", type=Path, help="Fresh directory for training outputs.")
    args = parser.parse_args(arguments)
    if args.steps < 1:
        parser.error("--steps must be positive")
    if not args.check and args.output is None:
        parser.error("Training requires --output; use --check for an episode check.")
    config = {} if args.config is None else json.loads(args.config.read_text())
    if not isinstance(config, dict):
        parser.error("--config must contain a JSON object")
    config.setdefault("seed", args.seed)

    import torch
    from stackelberg_pomdp.policy import DefaultPolicy
    from stackelberg_pomdp.training import EpisodePPO

    folder = Path(__file__).resolve().parents[1] / "experiments" / module_name
    import_name = "experiments." + module_name
    if (folder / "experiment.py").is_file():
        import_name += ".experiment"
    module = importlib.import_module(import_name)
    if not callable(getattr(module, "build", None)):
        parser.error("The experiment must define build(seed=0, ...).")
    torch.set_num_threads(1)
    env = module.build(**config)
    model = None
    try:
        if args.check:
            print(json.dumps(check_episode(env, args.seed), sort_keys=True))
            return
        output = fresh_output(args.output)
        record_run(output, module_name, {"game": config, "steps": args.steps,
                                        "training_seed": args.seed})
        model = EpisodePPO(
            getattr(module, "POLICY", DefaultPolicy), env,
            policy_kwargs=getattr(module, "POLICY_KWARGS", None),
            n_steps=64, batch_size=64, episodes_per_rollout=16,
            gamma=1.0, gae_lambda=1.0, seed=args.seed, device="cpu",
        )
        model.learn(total_timesteps=args.steps)
        model.save(output / "model.zip")
        summary = {"experiment": module_name, "policy_decisions": model.num_timesteps,
                   "completed_episodes": model.num_episodes}
        write_json(output / "run.json", summary)
        print(json.dumps(summary, sort_keys=True))
    finally:
        if model is not None:
            model.env.close()
        else:
            env.close()


def check_episode(env, seed=0, max_decisions=10000):
    """Check spaces and episode completion without knowing the domain rules."""
    env.action_space.seed(seed)
    observation = env.reset()
    total_reward = 0.0
    for decision in range(1, max_decisions + 1):
        if not env.observation_space.contains(observation):
            raise ValueError("An observation does not match the declared observation_space")
        observation, reward, done, info = env.step(env.action_space.sample())
        total_reward += float(reward)
        if done:
            if not env.observation_space.contains(observation):
                raise ValueError("The terminal observation does not match observation_space")
            return {
                "check": "passed", "policy_decisions": decision,
                "episode_return": total_reward,
                "response_queries": info.get("response_policy_queries"),
                "reward_queries": info.get("reward_policy_queries"),
                "cached_reuses": info.get("cache_hits"),
            }
    raise RuntimeError(f"Episode did not finish within {max_decisions} policy decisions")
