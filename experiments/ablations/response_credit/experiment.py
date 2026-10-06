#!/usr/bin/env python3
"""Train or reproduce the controlled one-query response-credit ablation.

Execute the frozen protocol, saving raw event counts and provenance.
Paper plotting lives in the companion reproducibility package.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parents[2]
PROTOCOL = HERE / "_support" / "paper.json"
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


def simulate(config):
    """Return all update records; the same episode denominator is used in both arms."""
    records = []
    batch = config["episodes_per_update"]
    for seed in config["seeds"]:
        rng = np.random.Generator(np.random.PCG64(seed))
        logits = np.full(2, config["initial_logit"], dtype=float)
        for method, z in zip(METHODS, logits):
            records.append(dict(zip(FIELDS, (seed, method, 0, 0, float(z),
                           float(sigmoid(z)), 0.0, 0, 0, 0, 0))))
        for update in range(1, config["updates"]+1):
            # Separate columns are independent fresh draws within an episode.
            # Sharing uniforms between methods pairs the experiment, not X and A.
            uniforms = rng.random((batch, 2))
            for i, method in enumerate(METHODS):
                p = sigmoid(logits[i])
                x = (uniforms[:, 0] < p).astype(int)
                a = (uniforms[:, 1] < p).astype(int)
                gradient = float(np.mean(event_gradient(p, x, a, method)))
                counts = np.bincount(2*x+a, minlength=4)
                logits[i] += config["learning_rate"] * gradient
                records.append(dict(zip(FIELDS, (seed, method, update,
                    update*batch*config["transitions_per_episode"], float(logits[i]),
                    float(sigmoid(logits[i])), gradient, *map(int, counts)))))
    return records


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--data-dir", type=Path, help="Validate an existing run instead of simulating.")
    parser.add_argument("--output-dir", type=Path, default=BUNDLE/"results/response-credit")
    args = parser.parse_args()
    config = json.loads(args.protocol.read_text())
    if config["methods"] != list(METHODS) or config["transitions_per_episode"] != 2:
        raise ValueError("Unexpected protocol")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.simulate or args.data_dir is None:
        records = simulate(config)
        raw = args.output_dir/"learning_curves.csv"
        with raw.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(records)
        metadata = {"protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
                    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "learning_curves_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                    "python": sys.version, "numpy": np.__version__, "platform": platform.platform()}
        (args.output_dir/"run_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    else:
        with (args.data_dir/"learning_curves.csv").open(newline="") as handle:
            records = list(csv.DictReader(handle))
    values = validate(records, config)
    summary = summarize(values, config)
    (args.output_dir/"response_credit_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
