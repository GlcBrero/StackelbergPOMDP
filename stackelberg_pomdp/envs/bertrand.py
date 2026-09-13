"""Bertrand competition environment."""

from gym.spaces import Box, Dict, Discrete, MultiDiscrete
import numpy as np

from .base import BaseEnv


class BertrandCompetitionEnv(BaseEnv):

    def __init__(
            self,
            num_agents=2,
            c_i=1,
            a=2,
            platform_intervention='pdp',
            a_0=0,
            mu=0.25,
            m=15,
            adv=0.3,
            k=1,
            price_min=1.05,
            price_max=1.7,
            leader_observation_space='no_observation',
            leader_k=1,
            sort_leader_observation=False,
            seed=None,
            logger=None,
    ):
        agents = ['agent_' + str(i) for i in range(num_agents)]
        super().__init__(leader="platform", followers_list=agents, logger=logger, seed=seed)

        self.num_agents = num_agents
        self.agents = agents
        self.k = k
        self.c_i = c_i
        self.m = m
        self.adv = adv
        self.a = np.array([a] * num_agents)
        self.a_0 = a_0
        self.mu = mu

        # Price grid: [marginal_cost, 2.1] à la Johnson et al. (2023)
        p_N, p_M = self._compute_nash_monopoly_prices(c_i, a, a_0, mu, num_agents)
        self.p_nash = p_N
        self.p_monopoly = p_M
        self.action_price_space = np.linspace(price_min, price_max, m)
        self.platform_intervention = platform_intervention
        self.leader_observation_space = leader_observation_space
        self.leader_k = leader_k
        self.sort_leader_observation = sort_leader_observation
        self._exp_a0_mu = np.exp(a_0 / mu)

        if leader_observation_space == 'price_profile':
            self.observation_space = Dict({
                'base_environment': MultiDiscrete([m] * num_agents * leader_k)
            })
        else:
            self.observation_space = Dict({'base_environment': Discrete(1)})
        self.action_space = Discrete(1)

        # Followers observe recent price history and choose prices from the grid.
        self.followers_observation_space = {
            agent: Box(
                np.array([0] * (k * num_agents)),
                np.array([m] * (k * num_agents)),
                dtype=int,
            )
            for agent in self.agents
        }
        self.followers_action_space = {agent: Discrete(m) for agent in self.agents}

    def reset(self):
        self.current_step = 0
        self.action_history = {}
        for agent in self.agents:
            self.action_history[agent] = [self._rng.randint(0, self.m - 1)]

        obs_agents = np.array([
            self.action_history[self.agents[i]][-self.k:]
            for i in range(self.num_agents)
        ], dtype=np.int64).flatten()
        observation = {agent: obs_agents for agent in self.agents}

        self.bbx_occupant = [0]
        self.platform_action = 0
        return observation

    def leader_observation(self, follower_actions=None):
        """Return the platform observation induced by current follower prices.

        For price-profile observations, the base state is the previous
        ``leader_k - 1`` executed price profiles and the reactive wrapper adds
        the current follower price profile. For no-observation experiments,
        return the null state.
        """
        if self.leader_observation_space != 'price_profile':
            return 0
        if follower_actions is None:
            follower_actions = {
                agent: self.action_history[agent][-1]
                for agent in self.agents
            }
        return self.combine_leader_observation(
            self.leader_state_observation(),
            self.follower_action_observation(follower_actions),
        )

    def leader_state_observation_space(self):
        if self.leader_observation_space != 'price_profile' or self.leader_k == 1:
            return Discrete(1)
        return MultiDiscrete([self.m] * self.num_agents * (self.leader_k - 1))

    def leader_state_observation(self):
        if self.leader_observation_space != 'price_profile' or self.leader_k == 1:
            return 0
        profiles = []
        for history_idx in range(self.leader_k - 1, 0, -1):
            profiles.append(self._price_profile_from_history(history_idx))
        return np.concatenate(profiles) if len(profiles) > 1 else profiles[0]

    def follower_action_observation_space(self):
        if self.leader_observation_space != 'price_profile':
            return Discrete(1)
        return MultiDiscrete([self.m] * self.num_agents)

    def follower_action_observation(self, follower_actions):
        if self.leader_observation_space != 'price_profile':
            return 0
        return self._current_price_profile(follower_actions)

    def _price_profile_from_history(self, history_idx):
        profile = np.array([
            self.action_history[agent][-min(history_idx, len(self.action_history[agent]))]
            for agent in self.agents
        ], dtype=np.int64)
        return np.sort(profile) if self.sort_leader_observation else profile

    def _current_price_profile(self, follower_actions):
        if follower_actions is None:
            return self._price_profile_from_history(1)
        profile = np.array([follower_actions[agent] for agent in self.agents], dtype=np.int64)
        return np.sort(profile) if self.sort_leader_observation else profile

    @staticmethod
    def _compute_nash_monopoly_prices(c, a, a_0, mu, n):
        """Compute symmetric Nash and monopoly prices for logit demand.

        Nash FOC (unilateral):  p = c + mu / (1 - q(p))
        Monopoly FOC (joint):   maximize (p - c) * q(p)
        where q(p) = exp((a-p)/mu) / (n*exp((a-p)/mu) + exp(a_0/mu))
        """
        ps = np.linspace(c, c + 4 * mu * n, 10000)

        def q_sym(p):
            x = np.exp((a - p) / mu)
            y = np.exp(a_0 / mu)
            return x / (n * x + y)

        # Nash: minimize |p - c - mu/(1-q(p))|
        resid = np.abs(ps - c - mu / (1 - np.array([q_sym(p) for p in ps])))
        p_N = ps[np.argmin(resid)]

        # Monopoly: max (p-c)*q(p)
        profits = np.array([(p - c) * q_sym(p) for p in ps])
        p_M = ps[np.argmax(profits)]

        return p_N, p_M

    def step(self, actions_dict):
        """Step the multi-agent game. actions_dict includes leader + all followers."""
        info = {}
        self.current_step += 1

        # Extract leader action
        self.platform_action = actions_dict.get(self.leader, 0)

        # Extract follower actions
        follower_actions = {a: actions_dict[a] for a in self.agents if a in actions_dict}
        actions_idx = np.array([follower_actions[a] for a in self.agents]).flatten()
        for i in range(self.num_agents):
            self.action_history[self.agents[i]].append(actions_idx[i])

        obs_agents = np.array([
            self.action_history[self.agents[i]][-self.k:]
            for i in range(self.num_agents)
        ], dtype=np.int64).flatten()
        observation = {agent: obs_agents for agent in self.agents}

        self.prices_idx = [int(pr) for pr in actions_idx[:self.num_agents]]
        self.prices = self.action_price_space.take(self.prices_idx)

        bbx_idx = self.get_bbx_idx(self.prices, self.platform_action)
        info['bbx_idx'] = bbx_idx
        occupants = bbx_idx[0] if len(bbx_idx) == 1 else [bbx_idx[k] for k in range(len(bbx_idx))]
        self.bbx_occupant.append(occupants)

        if bbx_idx is None:
            exp_vals = np.exp((self.a - self.prices) / self.mu)
            denom = exp_vals.sum() + self._exp_a0_mu
            info['surplus'] = self.mu * np.log(denom)
            demands = exp_vals / denom
        elif len(bbx_idx) > 0:
            bb = np.array(bbx_idx)
            exp_vals = np.exp((self.a[bb] - self.prices[bb]) / self.mu)
            denom = exp_vals.sum() + self._exp_a0_mu
            info['surplus'] = self.mu * np.log(denom)
            demands = np.zeros(self.num_agents)
            demands[bb] = exp_vals / denom
        else:
            info['surplus'] = self.mu * np.log(self._exp_a0_mu)
            demands = np.zeros(self.num_agents)

        rewards = {}
        for i in range(self.num_agents):
            rewards[self.agents[i]] = (self.prices[i] - self.c_i) * demands[i]

        done = False  # Game continues — prices carry over between steps
        info["reward_generated"] = True  # Rewards are ready, count as one sub-episode
        return observation, rewards, done, info

    def _demand(self, a, p, mu, agent_idx, bb_idx):
        if bb_idx is None:
            return np.exp((a[agent_idx] - p[agent_idx]) / mu) / (
                np.sum(np.exp((a - p) / mu)) + np.exp(self.a_0 / mu)
            )
        if agent_idx not in bb_idx:
            return 0
        denom = np.sum([np.exp((a[idx] - p[idx]) / mu) for idx in bb_idx]) + np.exp(self.a_0 / mu)
        return np.exp((a[agent_idx] - p[agent_idx]) / mu) / denom

    def _compute_surplus(self, prices, bbx_idx):
        val = np.sum([np.exp((self.a[i] - float(prices[i])) / self.mu) for i in bbx_idx])
        val += np.exp(self.a_0 / self.mu)
        return self.mu * np.log(val)

    def get_bbx_idx(self, prices, supervisor_action):
        if self.platform_intervention == 'no_intervene':
            return list(range(self.num_agents))

        elif self.platform_intervention == 'pdp':
            return [int(np.argmin(prices))]

        elif self.platform_intervention == 'dpdp':
            prev_prices_idx = [self.action_history[self.agents[i]][-2] for i in range(self.num_agents)]
            prev_prices = self.action_price_space.take(prev_prices_idx)

            bbx_occ_idx = self.bbx_occupant[-1]
            occ_price = prices[bbx_occ_idx]
            occ_prev_price = prev_prices[bbx_occ_idx]
            non_bbx_idx = int(1 - bbx_occ_idx)

            undercut_diff = prices[bbx_occ_idx] - prices[non_bbx_idx]
            if undercut_diff < self.adv and occ_price <= occ_prev_price:
                return [bbx_occ_idx]
            else:
                return [int(np.argmin(prices))]

        elif self.platform_intervention == 'block_equal':
            # If all agents quote the same price, nobody gets displayed
            if len(set(prices)) == 1:
                return []
            return list(range(self.num_agents))

        elif self.platform_intervention == 'learn_threshold':
            price_thresh = self.action_price_space[supervisor_action]
            return [i for i in range(self.num_agents) if prices[i] <= price_thresh]

        elif self.platform_intervention == 'learn_binary_threshold':
            # Binary: action 0 = open (all included), action 1 = close (only lowest price included)
            if supervisor_action == 1:
                price_thresh = self.action_price_space[0]
            else:
                price_thresh = self.action_price_space[-1]
            return [i for i in range(self.num_agents) if prices[i] <= price_thresh]

    def compute_q_init_entry(self, delta):
        """Compute initial Q-values à la Calvano et al. (2019), eq. 8.

        Q_{i,0}(s, a_i) = sum_{a_{-i}} pi_i(a_i, a_{-i}) / ((1-delta) * |A|^{n-1})

        Assumes uniform opponent play and computes expected discounted profit
        for each action. When platform_intervention != 'no_intervene', uses
        buy-box demand (Johnson et al. 2021 heuristic).
        """
        n = self.m
        all_agents = list(range(self.num_agents))
        entry = np.empty(n)
        for i in range(n):
            avg_reward = 0
            for j in range(n):
                price_i = self.action_price_space[i]
                price_j = self.action_price_space[j]
                prices = np.array([price_i, price_j])

                if self.platform_intervention == 'no_intervene':
                    # Calvano: all agents always displayed
                    demand_i = self._demand(self.a, prices, self.mu, 0, all_agents)
                else:
                    # Johnson et al.: buy-box winner gets displayed
                    bbx_demand = self._demand(self.a, prices, self.mu, 0, [0])
                    non_bbx_demand = self._demand(self.a, prices, self.mu, 0, [1])

                    if price_i < price_j:
                        demand_i = bbx_demand
                    elif price_i > price_j:
                        demand_i = non_bbx_demand
                    else:
                        sigma = 0.01
                        numer = np.exp(-price_i / sigma)
                        denom = np.sum([np.exp(-p / sigma) for p in prices])
                        demand_i = (numer / denom) * bbx_demand + (1 - numer / denom) * non_bbx_demand

                avg_reward += (price_i - self.c_i) * demand_i / n
            entry[i] = avg_reward / (1 - delta)
        return entry

    def log_info(self, info):
        if self.logger is None:
            return
        self.logger.record("consumer_surplus", info.get("surplus", 0))
        self.logger.record("c_i", round(self.c_i, 2))
        for j in range(self.num_agents):
            if hasattr(self, 'prices'):
                self.logger.record("price_" + str(j),
                    np.where(self.action_price_space == self.prices[j])[0][0])
