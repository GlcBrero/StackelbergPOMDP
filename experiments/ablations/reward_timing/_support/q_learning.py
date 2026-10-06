"""One-shot follower Q-learning, preserving the paper's initialization and exploration."""
import numpy as np


class TabularQLearner:
    def __init__(self, num_actions, *, q_alpha=.1, q_epsilon=1., exploration="epsilon_greedy",
                 q_init="small_normal", q_init_std=.01, seed=None):
        if not 0 <= q_alpha <= 1 or q_epsilon < 0:
            raise ValueError("Invalid Q-learning rate or exploration scale")
        if exploration not in ("epsilon_greedy", "parameter_noise"):
            raise ValueError("unsupported Q exploration mode")
        if q_init not in ("small_normal", "zero"):
            raise ValueError("q_init must be 'small_normal' or 'zero'")
        self.num_actions, self.alpha, self.epsilon = num_actions, q_alpha, q_epsilon
        self.exploration, self.q_init, self.q_init_std = exploration, q_init, q_init_std
        self.seed(seed)

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.q = (np.zeros(self.num_actions, dtype=np.float64) if self.q_init == "zero" else
                  self.rng.normal(0., self.q_init_std, size=self.num_actions))

    def action(self):
        if self.exploration == "parameter_noise":
            return int(np.argmax(self.q + self.rng.normal(0., self.epsilon, size=self.num_actions)))
        if self.rng.random() < min(self.epsilon, 1.):
            return int(self.rng.integers(self.num_actions))
        return int(np.argmax(self.q))
