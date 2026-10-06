"""Evaluation tables, repeated-rule caching and compact paper-run records."""
import numpy as np


def actor_table(model, env):
    """Read distributions without changing caches, response state, or RNG."""
    import torch

    observations = [env.observation(i) for i in range(env.n_queries)]
    batch = {key: np.stack([obs[key] for obs in observations]) for key in observations[0]}
    tensor, _ = model.policy.obs_to_tensor(batch)
    with torch.no_grad():
        probabilities = model.policy.get_distribution(tensor).distribution.probs.cpu().numpy()
    return env.seller_learning.table_from_actions(probabilities.argmax(axis=1)), probabilities


class Evaluator:
    def __init__(self,seller_learning,seeds):
        self.seller_learning=seller_learning;self.seeds=tuple(seeds);self.cache={};self.calls=0
    def evaluate(self,rule):
        key=tuple(np.asarray(rule).ravel())
        if key not in self.cache:
            self.cache[key]=[self.seller_learning.evaluate_rule(rule,c,s)
                             for c in self.seller_learning.costs for s in self.seeds]
            self.calls+=len(self.cache[key])
        return self.cache[key]
    def mean(self,rule):return float(np.mean([x['frozen_cs'] for x in self.evaluate(rule)]))


def response_record(result):
    return {k:result[k] for k in ('cost','seed','frozen_cs','seller_policy','cycle_actions',
        'training_joint','training_tail_cs','stable_policy_rounds','final_distributions',
        'response_steps')}
