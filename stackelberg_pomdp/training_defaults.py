"""Library optimizer defaults and the episode-length adjustment we require.

Read defaults from the installed SB3 version instead of copying a second set
of PPO/A2C/DQN defaults into each experiment. Scientific protocol exceptions
are documented in replication/PARAMETERS.md.
"""

from functools import lru_cache
from inspect import signature


@lru_cache(None)
def algorithm_defaults(algorithm):
    from stable_baselines3 import A2C, DQN, PPO

    return {
        name: parameter.default
        for name, parameter in signature(
            {"PPO": PPO, "A2C": A2C, "DQN": DQN}[algorithm].__init__
        ).parameters.items()
        if parameter.default is not parameter.empty
    }


def complete_episode_count(algorithm, episode_steps):
    """Round the library rollout budget up to complete episode budgets."""
    if episode_steps < 1:
        raise ValueError("episode_steps must be positive")
    steps = algorithm_defaults(algorithm)["n_steps"]
    return max(1, (steps + episode_steps - 1) // episode_steps)


def resolve_common_optimizer_defaults(config):
    """Resolve omitted CLI values before naming or recording a training run."""
    defaults = algorithm_defaults(config["algorithm"])
    if config.get("learning_rate") is None:
        config["learning_rate"] = defaults["learning_rate"]
    if config["algorithm"] == "PPO":
        for key, parameter in (("ppo_batch_size", "batch_size"),
                               ("ppo_n_epochs", "n_epochs")):
            if config.get(key) is None:
                config[key] = defaults[parameter]
    return config
