"""Native seller Q-learning, evaluation and paper configuration helpers."""
import itertools
from pathlib import Path
import numpy as np
from .market import (admission_game, admission_payoff_tables, initial_q, payoff_tables,
                     table_from_actions, admission_table_from_actions)
from ._native import (compile_q_kernel, learn as run_native_q_learning,
                      frozen_policy, cycle, QLearningSession)

PRICES = np.linspace(1.05, 1.7, 4)


def summarize_response(result, cost, seed, surplus=None):
    """Freeze seller strategies; gameplay supplies surplus for a partial rule."""
    policy = frozen_policy(result['q'][0])
    _, recurrent = cycle(policy, result['joint'][0])
    actions = policy[:, recurrent].T
    m = int(np.sqrt(policy.shape[1]))
    marginals = np.array([np.bincount(actions[:, i], minlength=m) / len(actions)
                          for i in range(2)])
    a, b = actions.T
    return {
        'cost': float(cost), 'seed': int(seed),
        'frozen_cs': None if surplus is None else float(surplus[a, b].mean()),
        'final_distributions': marginals.tolist(), 'seller_policy': policy.tolist(),
        'cycle_actions': actions.tolist(), 'training_joint': int(result['joint'][0]),
        'training_tail_cs': float(result['stats'][0, 1]),
        'stable_policy_rounds': int(result['stats'][0, 4]), 'q': result['q'][0].tolist(),
        'response_steps': result['actual_steps'],
    }


class OnDemandSellerResponse:
    """Translate native payoff requests into entries of the leader commitment."""

    def __init__(self, learner, actions, cost_index, seed):
        self.learner, self.actions, self.seed = learner, actions, seed
        self.cost = learner.costs[cost_index]
        self.complete = False
        self.session = QLearningSession(learner.kernel, learner.q_init[self.cost],
            seed, learner.steps, alpha=learner.alpha, delta=learner.delta,
            beta=learner.beta, stability_required=learner.stability_required)

    def next_query(self):
        learner = self.learner
        # Unanswered entries are masked out: their placeholder payoffs cannot run.
        rule = learner.table_from_actions(np.maximum(self.actions, 0))
        utility, surplus = admission_game(learner.admission_tables[self.cost], rule)
        known = np.zeros((learner.m, learner.m), dtype=np.uint8)
        for index, (first, second) in enumerate(learner.pairs):
            if self.actions[index] >= 0:
                known[first, second] = 1
                if not learner.ordered:
                    known[second, first] = 1
        pair = self.session.advance(utility, surplus, known)
        self.complete = pair is None
        if pair is None:
            return None
        return learner.pairs.index(pair if learner.ordered else tuple(sorted(pair)))

    def result(self):
        if not self.complete:
            raise RuntimeError('Answer the outstanding query before freezing the sellers')
        return summarize_response(self.session.result(), self.cost, self.seed)

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.session.close()


class SellerQLearning:
    """Stateful Q response and exact joint recurrent-cycle surplus."""

    def __init__(self, build_dir, costs=(1., 1.4), steps=5000000, beta=4e-6,
                 ordered=False, prices=PRICES, *, alpha=.25, delta=.95,
                 stability_required=0, q_initialization="buybox"):
        self.costs = tuple(map(float, costs))
        self.prices = np.asarray(prices, dtype=float)
        self.m = len(self.prices)
        if self.m < 2 or not np.all(np.diff(self.prices) > 0):
            raise ValueError("Need at least two increasing prices")
        self.steps, self.beta = int(steps), float(beta)
        self.alpha, self.delta = float(alpha), float(delta)
        self.stability_required = int(stability_required)
        self.q_initialization = q_initialization
        self.ordered = bool(ordered)
        self.pairs = (list(itertools.product(range(self.m), repeat=2)) if ordered else
                      list(itertools.combinations_with_replacement(range(self.m), 2)))
        self.kernel = compile_q_kernel(build_dir)
        self.admission_tables = {cost: admission_payoff_tables(cost, self.prices)
                                 for cost in self.costs}
        self._threshold_tables = None
        self.q_init = {cost: initial_q(cost, True, self.delta, self.prices,
                                     initialization=q_initialization)
                       for cost in self.costs}

    def table_from_actions(self, actions):
        return table_from_actions(actions, self.m, self.ordered)

    @property
    def tables(self):
        """Legacy threshold tensors, built only for callers that request them."""
        if self._threshold_tables is None:
            self._threshold_tables = {cost: payoff_tables(cost, self.prices)
                                      for cost in self.costs}
        return self._threshold_tables

    def evaluate_rule(self, rule, cost, seed, steps=None, beta=None):
        utilities, surplus = admission_game(self.admission_tables[cost], rule)
        result = run_native_q_learning(
            self.kernel, utilities, surplus, self.q_init[cost], seed,
            [self.steps if steps is None else steps], True,
            alpha=self.alpha, delta=self.delta, beta=self.beta if beta is None else beta,
            stability_required=self.stability_required,
        )
        return summarize_response(result, cost, seed, surplus)

    def start_response(self, actions, cost_index, seed):
        return OnDemandSellerResponse(self, actions, cost_index, seed)

    def run_seller_q_learning(self, actions, cost_index, seed):
        rule = self.table_from_actions(actions)
        result = self.evaluate_rule(rule, self.costs[cost_index], seed)
        result['rule'] = rule.tolist()
        return result

    __call__ = run_seller_q_learning


class AdmissionRuleQLearning:
    """Evaluate direct neither/lowest-price/both seller admission decisions."""

    n_actions = 3

    def __init__(self, response):
        self.response = response

    def __getattr__(self, name):
        return getattr(self.response, name)

    @staticmethod
    def valid_action_mask(first, second):
        low, high = sorted((int(first), int(second)))
        return np.asarray([low > 0, low < high, True], dtype=bool)

    def table_from_actions(self, actions):
        return admission_table_from_actions(actions, self.m, self.ordered)

    def evaluate_rule(self, *args, **kwargs):
        return self.response.evaluate_rule(*args, **kwargs)

    def start_response(self, actions, cost_index, seed):
        return OnDemandSellerResponse(self, actions, cost_index, seed)

    def run_seller_q_learning(self, actions, cost_index, seed):
        rule = self.table_from_actions(actions)
        result = self.evaluate_rule(rule, self.costs[cost_index], seed)
        result.update(rule=rule.tolist(), class_actions=list(map(int, actions)),
                      rule_representation='seller_admissions')
        return result

    __call__ = run_seller_q_learning


def make_seller_q_learning(output, parameters, costs=None):
    """Construct the paper's seller learner and chosen display-rule encoding."""
    learner = SellerQLearning(
        Path(output) / 'build',
        parameters['costs'] if costs is None else costs,
        steps=parameters['response_steps'],
        beta=parameters['beta'],
        prices=parameters['prices'],
        alpha=parameters['alpha'],
        delta=parameters['delta'],
        stability_required=parameters.get('stability_required', 0),
        q_initialization=parameters.get('q_initialization', 'buybox'),
    )
    representation = parameters.get('leader_action_representation', 'threshold')
    if representation == 'threshold':
        return learner
    if representation == 'admission_classes':
        return AdmissionRuleQLearning(learner)
    raise ValueError('Unknown leader action representation: ' + representation)
