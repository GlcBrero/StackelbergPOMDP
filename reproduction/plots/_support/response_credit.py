"""Scalar response-credit figure: exact gradients and recorded batch-count checks."""

import math

import numpy as np

METHODS = ("full", "gameplay_only")

LABELS = ("Stackelberg POMDP", "Basic POMDP (gameplay only)")

FIELDS = ("seed", "method", "update", "transitions", "logit", "exact_value",
          "gradient", "n00", "n01", "n10", "n11")

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def event_gradient(p, x, a, method):
    """One complete episode's logit score times its terminal reward."""
    reward = 2 * x - a
    score = a - p
    if method == "full":
        score = score + x - p
    elif method != "gameplay_only":
        raise ValueError(method)
    return reward * score

def exact_moments(p, method):
    """Independent enumeration, including response and gameplay randomness."""
    value = gradient = 0.0
    for x in (0, 1):
        for a in (0, 1):
            weight = (p if x else 1-p) * (p if a else 1-p)
            value += weight * (2*x-a)
            gradient += weight * event_gradient(p, x, a, method)
    return value, gradient

def validate(records, config):
    """Reconstruct every update from raw event counts and the preceding policy."""
    by_key = {}
    for row in records:
        key = (int(row["seed"]), row["method"], int(row["update"]))
        if key in by_key:
            raise ValueError(f"Duplicate record: {key}")
        by_key[key] = row
    expected = {(seed, method, step) for seed in config["seeds"]
                for method in METHODS for step in range(config["updates"]+1)}
    if set(by_key) != expected:
        raise ValueError("Incomplete or unexpected seed/method/update grid")
    values = np.empty((len(config["seeds"]), 2, config["updates"]+1))
    batch = config["episodes_per_update"]
    for i, seed in enumerate(config["seeds"]):
        for j, method in enumerate(METHODS):
            z = config["initial_logit"]
            for step in range(config["updates"]+1):
                row = by_key[seed, method, step]
                counts = [int(row[f"n{x}{a}"]) for x in (0, 1) for a in (0, 1)]
                if any(n < 0 for n in counts) or sum(counts) != (batch if step else 0):
                    raise ValueError(f"Invalid event counts: {seed}, {method}, {step}")
                gradient = 0.0
                if step:
                    p = float(sigmoid(z))
                    gradient = sum(n*event_gradient(p, x, a, method)
                        for n, (x, a) in zip(counts, ((0,0),(0,1),(1,0),(1,1)))) / batch
                    z += config["learning_rate"]*gradient
                value = float(sigmoid(z))
                for field, expected_value in (("gradient", gradient), ("logit", z),
                                               ("exact_value", value)):
                    if not math.isclose(float(row[field]), expected_value, abs_tol=1e-11,
                                        rel_tol=1e-11):
                        raise ValueError(f"Inconsistent {field}: {seed}, {method}, {step}")
                if int(row["transitions"]) != step*batch*config["transitions_per_episode"]:
                    raise ValueError("The arms must have identical actual-transition budgets")
                values[i,j,step] = value
    return values

def stats(values):
    return {"mean": float(np.mean(values)),
            "sem": float(np.std(values, ddof=1)/np.sqrt(len(values))),
            "min": float(np.min(values)), "max": float(np.max(values))}

def summarize(values, config):
    summary = {"study": config["study"], "seeds": config["seeds"],
               "transitions_per_arm": config["updates"]*config["episodes_per_update"]*2,
               "primary": {}, "secondary": {}}
    final = values[:,:,-1]
    average = values[:,:,1:].mean(axis=2)
    for j, method in enumerate(METHODS):
        summary["primary"][method] = stats(final[:,j])
        summary["secondary"][method] = stats(average[:,j])
    summary["primary"]["paired_full_minus_gameplay"] = stats(final[:,0]-final[:,1])
    summary["secondary"]["paired_full_minus_gameplay"] = stats(average[:,0]-average[:,1])
    return summary

CONFIG = {'study': 'one-query-response-credit-20260921', 'status_at_creation': 'fixed before training; no tuning runs', 'purpose': "Isolate omission of the response-query score in the paper's one-query example.", 'leader_reward': 'R = 2 X - A', 'follower_response': 'One independent query X ~ Bernoulli(p); freeze b = X for gameplay. Reset every episode.', 'gameplay': 'A ~ Bernoulli(p), freshly sampled independently of X conditional on the current policy.', 'policy': 'One scalar logit z, p = sigmoid(z), identical observation in both phases.', 'initial_logit': 0.0, 'methods': ['full', 'gameplay_only'], 'gradient_full': 'R * ((X-p) + (A-p))', 'gradient_gameplay_only': 'R * (A-p)', 'normalization': 'Mean over complete outer episodes in both arms; never over stored transitions.', 'optimizer': 'plain stochastic gradient ascent', 'learning_rate': 0.1, 'updates': 1000, 'episodes_per_update': 64, 'transitions_per_episode': 2, 'seeds': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25], 'randomization': "NumPy PCG64, one independent stream per seed. Within a seed, common independent query/gameplay uniform draws are thresholded at each arm's current p.", 'evaluation': 'Exact expected value J(p) = p at every update, including initialization; no evaluation sampling or greedy conversion.', 'primary_endpoint': 'Exact expected value after 1000 updates (128000 actual transitions per arm).', 'secondary_endpoint': 'Mean exact expected value after updates 1 through 1000.', 'uncertainty': 'Sample standard deviation divided by sqrt(25) across independent seeds; paired differences use within-seed contrasts.', 'selection': 'All seeds, fixed horizon, no checkpoint selection or outcome-dependent stopping.', 'not_used': ['critic', 'baseline', 'entropy regularization', 'advantage normalization', 'learning-rate decay', 'action caching', 'pretrained follower', 'PPO clipping'], 'scope': 'Controlled empirical illustration of the specified REINFORCE estimator. No general impossibility, convergence, cached-action, or privileged-critic claim.'}
