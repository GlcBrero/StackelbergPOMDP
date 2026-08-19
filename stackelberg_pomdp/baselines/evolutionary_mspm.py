"""Evolutionary mechanism-search baseline for MSPM.

This is a non-StackPOMDP baseline: the outer loop directly searches over a
tabular leader mechanism, while each candidate is evaluated by running MW
followers against that fixed mechanism.
"""

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import json

import numpy as np

from stackelberg_pomdp.games import get_mspm_setting
from stackelberg_pomdp.envs.base import BaseMessageSPM
from stackelberg_pomdp.wrappers.core import (
    MWFollowersWrapper,
    ReactiveLeaderWrapper,
)
from stackelberg_pomdp.utils import (
    check_empirical_bcce_gap,
    compute_empirical_welfare,
)


@dataclass
class EvaluationResult:
    score: float
    welfare: float
    bcce_gap: float
    records: int
    response_steps: int
    stop_reason: str


class TabularMSPMPolicy:
    """Tabular leader mechanism for two-follower one-item MSPMs.

    The table is keyed by the remaining-agent mask and the current follower
    message profile. Each entry stores two agent scores and one posted price.
    """

    def __init__(self, table, followers_list, num_messages, num_items=1):
        self.table = np.asarray(table, dtype=np.float64)
        self.followers_list = list(followers_list)
        self.num_followers = len(self.followers_list)
        self.num_messages = int(num_messages)
        self.num_items = int(num_items)
        self.masks = self._masks()
        self.mask_to_idx = {mask: i for i, mask in enumerate(self.masks)}

    @classmethod
    def random(cls, rng, followers_list, num_messages, num_items=1):
        shape = (
            (2 ** len(followers_list)) - 1,
            *(num_messages for _ in followers_list),
            len(followers_list) + num_items,
        )
        return cls(rng.random(shape), followers_list, num_messages, num_items)

    def copy(self):
        return TabularMSPMPolicy(
            self.table.copy(),
            self.followers_list,
            self.num_messages,
            self.num_items,
        )

    def mutated(self, rng, mutation_std, mutation_prob):
        child = self.copy()
        mask = rng.random(child.table.shape) < mutation_prob
        noise = rng.normal(0.0, mutation_std, size=child.table.shape)
        child.table = np.clip(child.table + mask * noise, 0.0, 1.0)
        return child

    def crossover(self, rng, other):
        child = self.copy()
        take_other = rng.random(child.table.shape) < 0.5
        child.table[take_other] = other.table[take_other]
        return child

    def get_action(self, observation):
        state = np.asarray(observation["base_environment"])
        bids = np.asarray(observation["base:follower_actions"], dtype=int)

        remaining = tuple(int(round(x)) for x in state[: self.num_followers])
        if sum(remaining) <= 0:
            remaining = tuple(1 for _ in range(self.num_followers))
        table_idx = (self.mask_to_idx[remaining], *bids.tolist())
        action = self.table[table_idx].copy()

        # Do not let unavailable agents win the score comparison.
        for idx, is_remaining in enumerate(remaining):
            if not is_remaining:
                action[idx] = 0.0
        return action.astype(np.float32)

    def to_json_dict(self):
        return {
            "followers_list": self.followers_list,
            "num_messages": self.num_messages,
            "num_items": self.num_items,
            "masks": [list(mask) for mask in self.masks],
            "table": self.table.tolist(),
        }

    def _masks(self):
        masks = [
            tuple(mask)
            for mask in product((0, 1), repeat=self.num_followers)
            if sum(mask) > 0
        ]
        # Put the full mechanism state first for readability.
        return sorted(masks, key=lambda mask: (-sum(mask), mask))


def _make_response_env(setting_name, num_types, num_messages, seed, mw_epsilon):
    game = get_mspm_setting(setting_name, num_types, num_messages)
    base_env = BaseMessageSPM(game=game, seed=seed)
    mw_env = MWFollowersWrapper(
        base_env,
        epsilon=mw_epsilon,
        reset_weights_each_episode=True,
    )
    reactive_env = ReactiveLeaderWrapper(mw_env)
    return base_env, mw_env, reactive_env


def evaluate_policy(
        policy,
        *,
        setting_name,
        num_types,
        num_messages,
        seed,
        mw_epsilon,
        response_records,
        bcce_threshold,
        bcce_min_records,
        bcce_check_freq,
        bcce_penalty,
):
    base_env, mw_env, reactive_env = _make_response_env(
        setting_name,
        num_types,
        num_messages,
        seed,
        mw_epsilon,
    )
    obs = reactive_env.reset()
    records = 0
    bcce_gap = float("inf")
    stop_reason = "max_response_records"

    while records < response_records:
        action = policy.get_action(obs)
        obs, _, _, info = reactive_env.step(action)
        if not info.get("response_record_generated", False):
            continue

        response_strategy = mw_env.response_strategy()
        records = len(response_strategy)
        if (
                records >= bcce_min_records
                and records % bcce_check_freq == 0
        ):
            bcce_gap = check_empirical_bcce_gap(
                base_env,
                policy,
                response_strategy,
            )
            if bcce_gap <= bcce_threshold:
                stop_reason = "bcce_threshold"
                break

    response_strategy = mw_env.response_strategy()
    if np.isinf(bcce_gap) and response_strategy:
        bcce_gap = check_empirical_bcce_gap(base_env, policy, response_strategy)
    if np.isinf(bcce_gap):
        bcce_gap = 1.0

    welfare = compute_empirical_welfare(base_env, policy, response_strategy)
    excess_gap = max(0.0, bcce_gap - bcce_threshold)
    score = welfare - bcce_penalty * excess_gap
    return EvaluationResult(
        score=float(score),
        welfare=float(welfare),
        bcce_gap=float(bcce_gap),
        records=len(response_strategy),
        response_steps=int(mw_env.step_counter),
        stop_reason=stop_reason,
    )


def evolutionary_search(
        *,
        setting_name,
        num_types,
        num_messages,
        seed,
        generations,
        population_size,
        elite_size,
        mutation_std,
        mutation_prob,
        mw_epsilon,
        response_records,
        bcce_threshold,
        bcce_min_records,
        bcce_check_freq,
        bcce_penalty,
        output_dir,
        metrics_callback=None,
):
    rng = np.random.default_rng(seed)
    followers_list = ["A0", "A1"]
    population = [
        TabularMSPMPolicy.random(rng, followers_list, num_messages)
        for _ in range(population_size)
    ]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_policy = None
    best_result = None
    history = []

    for generation in range(generations):
        evaluated = []
        for idx, policy in enumerate(population):
            result = evaluate_policy(
                policy,
                setting_name=setting_name,
                num_types=num_types,
                num_messages=num_messages,
                seed=seed + generation * population_size + idx,
                mw_epsilon=mw_epsilon,
                response_records=response_records,
                bcce_threshold=bcce_threshold,
                bcce_min_records=bcce_min_records,
                bcce_check_freq=bcce_check_freq,
                bcce_penalty=bcce_penalty,
            )
            evaluated.append((result, policy))

        evaluated.sort(key=lambda item: item[0].score, reverse=True)
        generation_best, generation_policy = evaluated[0]
        if best_result is None or generation_best.score > best_result.score:
            best_result = generation_best
            best_policy = generation_policy.copy()

        row = {
            "generation": generation,
            "best_score": generation_best.score,
            "best_welfare": generation_best.welfare,
            "best_bcce_gap": generation_best.bcce_gap,
            "best_records": generation_best.records,
            "best_response_steps": generation_best.response_steps,
            "best_stop_reason": generation_best.stop_reason,
            "overall_best_score": best_result.score,
            "overall_best_welfare": best_result.welfare,
            "overall_best_bcce_gap": best_result.bcce_gap,
        }
        history.append(row)
        if metrics_callback is not None:
            metrics_callback(row)
        print(
            "[evolution] "
            f"generation={generation} "
            f"best_score={generation_best.score:.6g} "
            f"best_welfare={generation_best.welfare:.6g} "
            f"best_bcce_gap={generation_best.bcce_gap:.6g} "
            f"records={generation_best.records} "
            f"stop_reason={generation_best.stop_reason} "
            f"overall_best_welfare={best_result.welfare:.6g}",
            flush=True,
        )

        elites = [policy.copy() for _, policy in evaluated[:elite_size]]
        next_population = elites.copy()
        while len(next_population) < population_size:
            parent_a = elites[rng.integers(len(elites))]
            parent_b = elites[rng.integers(len(elites))]
            child = parent_a.crossover(rng, parent_b)
            child = child.mutated(rng, mutation_std, mutation_prob)
            next_population.append(child)
        population = next_population

    (output_dir / "history.jsonl").write_text(
        "\n".join(json.dumps(row) for row in history) + "\n",
        encoding="utf-8",
    )
    (output_dir / "best_policy.json").write_text(
        json.dumps(best_policy.to_json_dict(), indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps({
            "best_score": best_result.score,
            "best_welfare": best_result.welfare,
            "best_bcce_gap": best_result.bcce_gap,
            "best_records": best_result.records,
            "best_response_steps": best_result.response_steps,
            "best_stop_reason": best_result.stop_reason,
        }, indent=2),
        encoding="utf-8",
    )
    return best_result, best_policy
