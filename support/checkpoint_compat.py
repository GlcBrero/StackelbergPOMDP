"""Load historical checkpoints after moving their class definitions.

Use load_checkpoint(Algorithm, path, ...) at checkpoint-loading boundaries.
Importing the core Stackelberg POMDP package installs no import redirects.

These aliases resolve to the one active implementation. They are not a second
library and do not restore the retired policy-side action cache.
"""
import importlib
import importlib.abc
import importlib.util
import sys

ALIASES = {
    "experiments.atari._support.game.bilateral": "experiments.atari.game",
    "support.atari.game.bilateral": "experiments.atari.game",
    "stackelberg_pomdp.algorithms": "stackelberg_pomdp.training",
    "stackelberg_pomdp.algorithms.credit": "support.checkpoint_compat",
    "stackelberg_pomdp.training.credit": "support.checkpoint_compat",
    "stackelberg_pomdp.compat": "support.checkpoint_compat",
    "stackelberg_pomdp.algorithms.on_policy": "stackelberg_pomdp.training.on_policy",
    "stackelberg_pomdp.atari": "experiments.atari._support",
    "stackelberg_pomdp.atari.policies": "experiments.atari._support.policy",
    "stackelberg_pomdp.atari.policies.composite": "experiments.atari._support.policy.policy",
    "stackelberg_pomdp.atari.stackpomdp_policy": "experiments.atari._support.stackpomdp_policy",
    "stackelberg_pomdp.matrix_ablations": "experiments.ablations._support",
    "stackelberg_pomdp.matrix_ablations.reinforce": "experiments.ablations._support.reinforce",
    "stackelberg_pomdp.matrix_ablations.simple_q": "experiments.ablations._support.simple_q",
    "stackelberg_pomdp.policies": "experiments.atari._support.policy",
    "stackelberg_pomdp.policies.atari": "experiments.atari._support.policy",
    "stackelberg_pomdp.policies.atari.components": "experiments.atari._support.policy.components",
    "stackelberg_pomdp.policies.atari.composite": "experiments.atari._support.policy.policy",
    "stackelberg_pomdp.policies.atari.meta_seller": "experiments.atari._support.policy.meta_seller",
    "stackelberg_pomdp.policies.atari.policy": "experiments.atari._support.policy.policy",
    "stackelberg_pomdp.policies.generic": "stackelberg_pomdp.policy"
}

_MOVED_PACKAGES = {
    "pretraining": "pretrained",
    "support.normal_form": "experiments.ablations._support",
    "experiments.commitment_consistency": "experiments.ablations.commitment_consistency",
    "experiments.reward_timing": "experiments.ablations.reward_timing",
    "experiments.response_credit": "experiments.ablations.response_credit",
    "support.atari": "experiments.atari._support",
}


def _target(fullname):
    if fullname in ALIASES:
        return ALIASES[fullname]
    for old, new in _MOVED_PACKAGES.items():
        if fullname == old or fullname.startswith(old + "."):
            return new + fullname[len(old):]
    return None


class _CheckpointImports(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if _target(fullname) is not None:
            return importlib.util.spec_from_loader(fullname, self)
    def create_module(self, spec):
        return importlib.import_module(_target(spec.name))
    def exec_module(self, module):
        pass


def install():
    if not any(isinstance(finder, _CheckpointImports) for finder in sys.meta_path):
        sys.meta_path.insert(0, _CheckpointImports())


def load_checkpoint(algorithm, *args, **kwargs):
    """Load a saved model, resolving class paths used before the reorganization."""
    install()
    return algorithm.load(*args, **kwargs)


def __getattr__(name):
    """Resolve old credit-module symbols without retaining them in the core."""
    buffer_names = {
        "CreditRolloutSamples": "EpisodeRolloutSamples",
        "CreditRolloutBuffer": "EpisodeRolloutBuffer",
        "CreditDictRolloutBuffer": "EpisodeDictRolloutBuffer",
    }
    if name in buffer_names:
        module = importlib.import_module("stackelberg_pomdp.training.rollout_buffer")
        return getattr(module, buffer_names[name])
    if name in {"credit_mean", "masked_mean", "normalize_advantages"}:
        module = importlib.import_module("experiments.atari._support._ppo")
        return getattr(module, name)
    raise AttributeError(name)
