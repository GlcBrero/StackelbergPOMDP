"""Paper PPO training and held-out selection for the platform experiment."""

import copy
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
from .evaluation import Evaluator, response_record
from experiments.platform.response import make_seller_q_learning
from .evaluation import actor_table
from support.common import write_json

def hold_then_decay(initial, hold_fraction=.25):
    def schedule(progress_remaining):
        return initial * min(1., max(0., progress_remaining) / (1. - hold_fraction))
    return schedule

def verify_checkpoint_policy(model, checkpoint, probabilities, expected):
    """Require exact saved weights/actions, allowing only float32 output roundoff."""
    import torch
    from stable_baselines3.common.save_util import load_from_zip_file
    _, saved, _ = load_from_zip_file(checkpoint, device='cpu', load_data=False)
    actual = model.policy.state_dict()
    assert actual.keys() == saved['policy'].keys()
    for name, value in saved['policy'].items():
        current = actual[name].detach().cpu()
        assert value.dtype == current.dtype and torch.equal(value, current), name
    probabilities, expected = np.asarray(probabilities), np.asarray(expected)
    # Identical weights can produce slightly different outputs on different CPUs.
    # This does not permit changed greedy commitments or materially changed logits.
    np.testing.assert_allclose(probabilities, expected, rtol=0, atol=1e-6, equal_nan=False)
    np.testing.assert_array_equal(probabilities.argmax(axis=1), expected.argmax(axis=1))
    return {'checkpoint_sha256': hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        'parameter_tensors_bitwise_equal': len(actual), 'all_greedy_actions_equal': True,
        'probability_absolute_tolerance': 1e-6,
        'max_absolute_probability_error': float(np.abs(probabilities - expected).max())}

def response_seeds(seed, count):
    rng = np.random.RandomState(seed + 710000)
    values, seen = [], set()
    while len(values) < count:
        value = int(rng.randint(2000000000, 3000000000))
        if value not in seen:
            values.append(value)
            seen.add(value)
    return values

class Archive:
    def __init__(self, rows=()):
        self.data = {tuple(r['actions']): [r['total'], r['count']] for r in rows}

    def add(self, actions, score):
        key = tuple(map(int, actions))
        total, count = self.data.get(key, (0., 0))
        self.data[key] = [total + float(score), count + 1]

    def incumbent(self):
        return max(self.data, key=lambda k: self.data[k][0] / self.data[k][1])

    def rows(self):
        return [dict(actions=list(k), total=v[0], count=v[1]) for k, v in self.data.items()]

def restore_evaluator(seller_learning, seeds, path=None):
    evaluator = Evaluator(seller_learning, seeds)
    if path is not None:
        for row in json.loads(Path(path).read_text()):
            evaluator.cache[tuple(np.asarray(row['rule']).ravel())] = row['responses']
        evaluator.calls = sum(len(v) for v in evaluator.cache.values())
    return evaluator

def save_evaluator(evaluator, path, m):
    write_json(path, [{'rule': np.asarray(key).reshape(
        (m, m, 2) if len(key) == 2 * m * m else (m, m)).tolist(),
        'responses': [response_record(x) for x in values]} for key, values in evaluator.cache.items()])

class Returns:
    """Matched primary selection: archive incumbent only; union is secondary."""
    def __init__(self, run, seller_learning, evaluator, p, history=()):
        self.run, self.seller_learning, self.evaluator, self.p = Path(run), seller_learning, evaluator, p
        self.history = copy.deepcopy(list(history))
        self.best, self.union = None, None
        for entry in self.history:
            self.update(entry)
        if self.history:
            self.save()

    def update(self, entry):
        if self.best is None or entry['archive']['validation_cs'] > self.best['validation_cs'] + 1e-10:
            self.best = copy.deepcopy(entry['archive'])
        for name in ('archive', 'mode'):
            if self.union is None or entry[name]['validation_cs'] > self.union['validation_cs'] + 1e-10:
                self.union = copy.deepcopy(entry[name])

    def evaluate(self, archive, probabilities, calls, **extra):
        assert not self.history or calls > self.history[-1]['responses']
        incumbent = archive.incumbent()
        entry = {'responses': int(calls), **extra}
        for kind, actions in [('archive', incumbent), ('mode', probabilities.argmax(axis=1))]:
            rule = self.seller_learning.table_from_actions(actions)
            entry[kind] = {'responses': int(calls), 'kind': kind,
                'rule': rule.tolist(), 'validation_cs': self.evaluator.mean(rule)}
        entry['probabilities'] = probabilities.tolist()
        entry['training_incumbent_mean'] = archive.data[incumbent][0] / archive.data[incumbent][1]
        entry['validation_calls'] = self.evaluator.calls
        self.history.append(entry)
        self.update(entry)
        self.save()
        if calls in self.p['report_budgets']:
            dest = self.run / f'budget_{calls:05d}'
            dest.mkdir(exist_ok=False)
            for name in ('best.json', 'best_union.json', 'final.json', 'mode.json'):
                shutil.copy2(self.run / name, dest / name)
        print(json.dumps({'run': self.run.name, 'responses': calls,
            'validation_archive': entry['archive']['validation_cs'],
            'validation_mode': entry['mode']['validation_cs'],
            'best_archive': self.best['validation_cs'], 'validation_calls': self.evaluator.calls}), flush=True)

    def save(self):
        write_json(self.run / 'history.json', self.history)
        write_json(self.run / 'best.json', self.best)
        write_json(self.run / 'best_union.json', self.union)
        write_json(self.run / 'final.json', self.history[-1]['archive'])
        write_json(self.run / 'mode.json', self.history[-1]['mode'])
        write_json(self.run / 'latest.json', self.history[-1])

def rl_settings(p, method):
    return p["rl_methods"][method]


def model_class(p, method):
    from support.episode_weighting import EpisodeWeightedPPO
    if rl_settings(p, method)["algorithm"] != "ppo":
        raise ValueError("The paper platform uses PPO")
    return EpisodeWeightedPPO


def make_env(output, p, seed, offset=0):
    from experiments.platform.experiment import build
    seller_learning = make_seller_q_learning(output, p)
    schedule = response_seeds(seed, p["episodes"] // 2 + 2)
    return build(seller_learning, schedule, seed, offset)


def make_model(env, p, seed, method="ppo_class_mean40"):
    import torch
    from stackelberg_pomdp.policy import DefaultPolicy
    from experiments.platform import experiment as platform
    torch.set_num_threads(1)
    settings = rl_settings(p, method)
    lr = settings["learning_rate"]
    if settings.get("learning_rate_schedule") == "hold25_then_linear_to_zero":
        lr = hold_then_decay(lr)
    policy = getattr(platform, "POLICY", DefaultPolicy)
    model = model_class(p, method)(policy, env,
        n_steps=env.n_queries * settings["episodes_per_rollout"],
        episodes_per_rollout=settings["episodes_per_rollout"],
        actor_episode_denominator=env.n_queries + 1, gamma=1., gae_lambda=1.,
        learning_rate=lr, ent_coef=settings["ent_coef"],
        normalize_advantage=settings.get("normalize_advantage", True), seed=seed, device="cpu",
        **{key: settings[key] for key in ("batch_size", "n_epochs", "clip_range")})
    return model


def train(output, p, method, seed, initial_policy_state=None):
    from stable_baselines3.common.callbacks import BaseCallback
    run = Path(output) / "runs" / f"{method}_seed{seed}"
    run.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    seller_learning = make_seller_q_learning(output, p)
    evaluator = restore_evaluator(seller_learning, p["validation_seeds"])
    archive = Archive()
    selection = Returns(run, seller_learning, evaluator, p)
    env = make_env(output, p, seed)
    model = make_model(env, p, seed, method)
    if initial_policy_state is not None:
        model.policy.load_state_dict(initial_policy_state, strict=True)
    model.save(run / "initial_model.zip")
    write_json(run / "config.json", dict(protocol=p, method=method, seed=seed,
        response_protocol="on_demand_queries", n_steps=model.n_steps,
        archive_unqueried_completion="current_greedy_action_for_held_out_candidate_only"))
    settings = rl_settings(p, method)
    if p["episodes"] % (2 * settings["episodes_per_rollout"]):
        raise ValueError("Response budget must contain complete PPO batches")
    with (run / "episodes.jsonl").open("x", buffering=1) as log:
        class Progress(BaseCallback):
            def __init__(self):
                super().__init__()
                self.last_evaluation = 0
            def evaluate(self):
                if env.episodes == self.last_evaluation:
                    return
                _, probabilities = actor_table(self.model, env)
                selection.evaluate(archive, probabilities, 2 * env.episodes,
                                   optimizer_updates=self.model._n_updates)
                self.model.save(run / "latest_model")
                if 2 * env.episodes in p["report_budgets"]:
                    self.model.save(run / f"budget_{2 * env.episodes:05d}" / "final_model")
                self.last_evaluation = env.episodes
            def _on_rollout_start(self):
                if 2 * env.episodes % p["evaluation_every"] == 0:
                    self.evaluate()
            def _on_step(self):
                if not self.locals["dones"][0]:
                    return True
                info = self.locals["infos"][0]
                first = info["response"]
                second = first["paired_response"]
                score = (first["frozen_cs"] + second["frozen_cs"]) / 2
                np.testing.assert_allclose(self.locals["rewards"][0], score, atol=3e-7, rtol=0)
                archive_actions = np.asarray(info["query_actions"], dtype=int).copy()
                missing = archive_actions < 0
                if np.any(missing):
                    # This completes only the held-out archive candidate, not the
                    # episode commitment or training buffer. No extra query occurs.
                    _, probabilities = actor_table(self.model, env)
                    archive_actions[missing] = probabilities.argmax(axis=1)[missing]
                for number, result in ((2 * env.episodes - 1, first), (2 * env.episodes, second)):
                    log.write(json.dumps(dict(episode=number, role="actor_component", actor_reward=score,
                        query_actions=info["query_actions"],
                        policy_queries=info["policy_queries"],
                        response_policy_queries=info["response_policy_queries"],
                        reward_policy_queries=info["reward_policy_queries"],
                        archive_completed_entries=int(missing.sum()),
                        **response_record(result))) + "\n")
                archive.add(archive_actions, score)
                return True
        progress = Progress()
        total_episodes = p["episodes"] // 2
        _, callback = model._setup_learn(total_episodes, progress, True, method, False)
        callback.on_training_start({}, {})
        # Query counts can vary; budgets and the LR schedule count outer episodes.
        while env.episodes < total_episodes:
            model.episodes_per_rollout = min(settings["episodes_per_rollout"],
                                             total_episodes - env.episodes)
            if not model.collect_rollouts(model.env, callback, model.rollout_buffer, model.n_steps):
                raise RuntimeError("Training callback stopped collection")
            model._update_current_progress_remaining(env.episodes, total_episodes)
            model.train()
        callback.on_training_end()
        progress.evaluate()
    model.save(run / "final_model")
    assert env.episodes * 2 == p["episodes"]
    write_json(run / "archive.json", archive.rows())
    save_evaluator(evaluator, run / "validation_responses.json", p["m"])
    result = dict(status="complete", method=method, seed=seed, responses=2 * env.episodes,
                  validation_calls=evaluator.calls, updates=model._n_updates,
                  elapsed_seconds=time.perf_counter() - start)
    write_json(run / "completion.json", result)
    env.close()
    return result
