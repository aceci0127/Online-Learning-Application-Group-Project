import numpy as np
from abc import ABC, abstractmethod
from scipy.optimize import linprog
from collections import deque, Counter


class Agent(ABC):
    """Abstract base class for all agents"""

    @abstractmethod
    def pull_arm(self):
        """Select an action/arm"""
        pass

    @abstractmethod
    def update(self, feedback):
        """Update the agent with the received feedback"""
        pass


class UCB1PricingAgent(Agent):
    """UCB1 agent for simple pricing"""

    def __init__(self, K, T, range=1):
        self.K = K
        self.T = T
        self.range = range
        self.a_t = None
        self.average_rewards = np.zeros(K)
        self.N_pulls = np.zeros(K)
        self.t = 0

    def pull_arm(self):
        if self.t < self.K:
            self.a_t = self.t
        else:
            ucbs = self.average_rewards + self.range * \
                np.sqrt(2*np.log(self.t)/self.N_pulls)
            self.a_t = np.argmax(ucbs)
        return self.a_t

    def update(self, r_t):
        self.N_pulls[self.a_t] += 1
        self.average_rewards[self.a_t] += (
            r_t - self.average_rewards[self.a_t])/self.N_pulls[self.a_t]
        self.t += 1


class ConstrainedUCBPricingAgent(Agent):
    """Constrained UCB agent for pricing with budget"""

    def __init__(self, K, B, T, range=1):
        self.K = K
        self.T = T
        self.range = range
        self.a_t = None
        self.avg_f = np.zeros(K)
        self.avg_c = np.zeros(K)
        self.N_pulls = np.zeros(K)
        self.rem_budget = B
        self.budget = B
        self.rho = B/T
        self.t = 0

    def pull_arm(self):
        # 1) Stop if out of budget
        if self.rem_budget < 1:
            self.a_t = None
            return None

        # 2) Pure exploration: each real arm exactly once
        if self.t < self.K:
            self.a_t = self.t
            return self.a_t

        # 3) Build UCB/LCB for all arms
        f_ucbs = self.avg_f + self.range*np.sqrt(2*np.log(self.t)/self.N_pulls)
        c_lcbs = self.avg_c - self.range*np.sqrt(2*np.log(self.t)/self.N_pulls)
        c_lcbs = np.clip(c_lcbs, 0.0, 1.0)

        # 4) Set the last arm to 0 for the dummy arm
        f_ucbs[-1] = 0.0
        c_lcbs[-1] = 0.0

        # 5) Solve the LP
        gamma_t, expected_t = self.compute_opt(f_ucbs, c_lcbs)
        self.a_t = np.random.choice(self.K, p=gamma_t)
        return self.a_t

    def compute_opt(self, f_ucbs, c_lcbs):
        c = -f_ucbs
        A_ub = [c_lcbs]
        b_ub = [self.rho]
        A_eq = [np.ones(self.K)]
        b_eq = [1]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq,
                      b_eq=b_eq, bounds=(0, 1))
        gamma = res.x
        expected = -res.fun

        if not np.all(gamma >= 0):
            raise ValueError(
                "Invalid distribution: negative probabilities found in gamma.")
        if not np.isclose(np.sum(gamma), 1.0, atol=1e-6):
            raise ValueError(
                f"Invalid distribution: probabilities do not sum to 1. Found {np.sum(gamma)}.")

        return gamma, expected

    def update(self, f_t, c_t):
        self.N_pulls[self.a_t] += 1
        self.avg_f[self.a_t] += (f_t - self.avg_f[self.a_t]
                                 )/self.N_pulls[self.a_t]
        self.avg_c[self.a_t] += (c_t - self.avg_c[self.a_t]
                                 )/self.N_pulls[self.a_t]
        self.rem_budget -= c_t
        self.t += 1


class ConstrainedCombinatorialUCBAgent(Agent):
    """Constrained Combinatorial UCB agent for multi-product"""

    def __init__(self, price_grid, B, T, alpha=2):
        self.price_grid = price_grid
        self.N = len(price_grid)
        self.Ks = [len(pg) for pg in price_grid]
        self.B_rem = B
        self.B = B
        self.T = T
        self.t = 0
        self.alpha = alpha
        self.rng = np.random.default_rng()

        self.N_pulls = [np.zeros(K) for K in self.Ks]
        self.avg_f = [np.zeros(K) for K in self.Ks]
        self.avg_c = [np.zeros(K) for K in self.Ks]
        self.current_choice = None

    def _solve_marginal_lp(self, f_ucb, c_lcb, rho):
        """
        Solve the per-round LP:
          max_x sum_j f_ucb[j]·x_j
          s.t. ∑_k x_{j,k} = 1  ∀j,
               ∑_{j,k} c_lcb[j][k]·x_{j,k} ≤ rho,
               x ≥ 0.
        Returns a list of marginals [x_0, x_1, …, x_{N-1}].
        """
        f_flat = np.concatenate(f_ucb)
        c_flat = np.concatenate(c_lcb)
        num_vars = len(f_flat)

        c_obj = -f_flat

        A_eq = np.zeros((self.N, num_vars))
        b_eq = np.ones(self.N)
        offset = 0
        for j, K in enumerate(self.Ks):
            A_eq[j, offset:offset+K] = 1
            offset += K

        A_ub = c_flat.reshape(1, -1)
        b_ub = np.array([rho])

        bounds = [(0, None)] * num_vars

        res = linprog(c=c_obj,
                      A_ub=A_ub, b_ub=b_ub,
                      A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds,
                      method='highs')

        if not res.success:
            raise RuntimeError("LP failed: " + res.message)

        x_flat = res.x
        marginals = []
        offset = 0
        for K in self.Ks:
            marginals.append(x_flat[offset:offset+K])
            offset += K

        return marginals

    def pull_arm(self):
        if self.B_rem < 1 or self.t >= self.T:
            return None
        '''
        # Pure exploration: choose price index self.t for each product if possible.
        if all(self.t < K for K in self.Ks):
            choice = tuple(self.t for _ in range(self.N))
            self.current_choice = choice
            self.t += 1  # Advance time after exploration
            return choice
        '''
        # TODO: capire quale rho è meglio usare
        rho = self.B / self.T

        f_ucb = []
        c_lcb = []
        for j in range(self.N):
            n_j = self.N_pulls[j]
            f_j = self.avg_f[j]
            c_j = self.avg_c[j]

            bonus = np.sqrt(
                self.alpha * np.log(np.maximum(self.t, 1)) / np.maximum(1, n_j))
            bonus[n_j == 0] = self.T
            #bonus[-1] = 0.0

            f_ucb.append(f_j + bonus)
            c_lcb.append(np.clip(c_j - bonus, 0.0, 1.0))

        marginals = self._solve_marginal_lp(f_ucb, c_lcb, rho)

        choice = tuple(
            self.rng.choice(self.Ks[j], p=marginals[j])
            for j in range(self.N)
        )

        self.current_choice = choice
        return choice

    def update(self, rewards, costs):
        for j, idx in enumerate(self.current_choice):
            self.N_pulls[j][idx] += 1
            n = self.N_pulls[j][idx]
            self.avg_f[j][idx] += (rewards[j] - self.avg_f[j][idx]) / n
            self.avg_c[j][idx] += (costs[j] - self.avg_c[j][idx]) / n
        self.B_rem -= costs.sum()
        self.t += 1


class HedgeAgent:
    """Hedge agent for online learning"""

    def __init__(self, K, learning_rate, rng):
        self.K = K
        self.learning_rate = learning_rate
        self.weights = np.ones(K)
        self.x_t = np.ones(K)/K
        self.a_t = None
        self.rng = rng if rng is not None else np.random
        self.t = 0

    def pull_arm(self):
        self.x_t = self.weights/sum(self.weights)
        self.a_t = self.rng.choice(np.arange(self.K), p=self.x_t)
        return self.a_t

    def update(self, l_t):
        self.weights *= np.exp(-self.learning_rate*l_t)
        self.t += 1


class FFPrimalDualPricingAgent(Agent):
    """Primal-Dual agent with Full-Feedback for non-stationary pricing"""

    def __init__(self, prices, T, B, rng, eta):
        self.prices = np.array(prices)
        self.K = len(prices)
        self.T = T
        self.inventory = B
        self.eta = eta
        self.rng = rng if rng is not None else np.random

        self.hedge = HedgeAgent(self.K, np.sqrt(
            np.log(self.K) / T), rng=self.rng)
        self.rho = B / T
        self.lmbd = 1.0
        self.t = 0
        self.pull_counts = np.zeros(self.K, int)
        self.last_arm = None

    def pull_arm(self):
        if self.inventory < 1:
            self.last_arm = None
            return self.last_arm
        self.last_arm = self.hedge.pull_arm()
        return self.last_arm

    def update(self, v_t):
        sale_mask = (self.prices <= v_t).astype(float)
        f_full = self.prices * sale_mask
        c_full = sale_mask

        L = f_full - self.lmbd * (c_full - self.rho)

        f_max = self.prices.max()
        L_up = f_max - self.lmbd * (0 - self.rho)
        L_low = 0.0 - self.lmbd * (1.0 - self.rho)

        rescaled = (L - L_low) / (L_up - L_low + 1e-12)
        losses = 1.0 - rescaled

        self.hedge.update(losses)

        if self.last_arm is not None:
            c_t = 1 if self.prices[self.last_arm] <= v_t else 0
            f_t = self.prices[self.last_arm] * c_t

            self.inventory -= c_t
            self.pull_counts[self.last_arm] += 1
        else:
            c_t = 0
            f_t = 0

        self.lmbd = np.clip(self.lmbd - self.eta * (self.rho - c_t),
                            a_min=0.0,
                            a_max=1.0/self.rho)

        self.t += 1
        return f_t, c_t


class MultiProductFFPrimalDualPricingAgent(Agent):
    """Primal-Dual agent with Full-Feedback for multi-product"""

    def __init__(self, prices, T, B, m, rng, eta):
        self.prices = np.array(prices)
        self.K = len(prices)
        self.T = T
        self.m = m
        self.B = B
        self.rho = B / (m * T)
        self.eta = eta

        self.rng = rng
        self.hedges = [HedgeAgent(self.K, np.sqrt(np.log(self.K)/T), self.rng)
                       for _ in range(m)]

        self.lmbd = 1.0
        self.inventory = B

        self.debug_chosen_prices = [[] for _ in range(m)]
        self.debug_sold_prices = [[] for _ in range(m)]

    def pull_arm(self):
        if self.inventory < 1:
            return [None] * self.m
        arms = [hedge.pull_arm() for hedge in self.hedges]
        return arms

    def update(self, v_t):
        arms = self.pull_arm()
        total_revenue = 0.0
        total_units_sold = 0
        losses = []

        p_max = self.prices.max()
        L_up = p_max - self.lmbd * (0 - self.rho)
        L_low = 0.0 - self.lmbd * (1 - self.rho)
        norm_factor = L_up - L_low + 1e-12

        for j in range(self.m):
            arm = arms[j]
            if arm is None:
                self.debug_chosen_prices[j].append(None)
                self.debug_sold_prices[j].append(None)
                losses.append(np.zeros(self.K))
                continue

            p_chosen = self.prices[arm]
            self.debug_chosen_prices[j].append(p_chosen)
            sale = 1 if p_chosen <= v_t[j] else 0
            self.debug_sold_prices[j].append(p_chosen if sale == 1 else None)

            f = p_chosen * sale
            total_revenue += f
            total_units_sold += sale

            would_sell = (self.prices <= v_t[j]).astype(float)
            f_vec = self.prices * would_sell
            L_vec = f_vec - self.lmbd * (would_sell - self.rho)
            loss_vec = 1.0 - (L_vec - L_low) / norm_factor

            loss_vec[-1] = 1.0
            losses.append(loss_vec)

        for j in range(self.m):
            self.hedges[j].update(losses[j])

        self.inventory -= total_units_sold
        self.lmbd = np.clip(self.lmbd - self.eta * (self.rho * self.m - total_units_sold),
                            a_min=0.0, a_max=1 / self.rho)
        return total_revenue, total_units_sold


class SlidingWindowConstrainedCombinatorialUCBAgent(ConstrainedCombinatorialUCBAgent):
    """Constrained Combinatorial UCB agent with Sliding Window for non-stationarity"""

    def __init__(self, price_grid, B, T, alpha=1, window_size=1000):
        super().__init__(price_grid, B, T, alpha)
        self.window_size = window_size

        self.samples = [
            [deque() for _ in range(K)]
            for K in self.Ks
        ]

    def pull_arm(self):
        if self.B_rem < 1 or self.t >= self.T:
            return None

        rho = self.B_rem / (self.T - self.t)
        f_ucb = []
        c_lcb = []

        for j in range(self.N):
            K = self.Ks[j]
            f_j_vals = []
            c_j_vals = []
            n_j_vals = []

            for k in range(K):
                while self.samples[j][k] and self.samples[j][k][0][0] < self.t - self.window_size:
                    self.samples[j][k].popleft()

                samples = self.samples[j][k]
                n = len(samples)
                avg_reward = np.mean([item[1]
                                     for item in samples]) if n > 0 else 0.0
                avg_cost = np.mean([item[2]
                                   for item in samples]) if n > 0 else 0.0
                n_j_vals.append(n)
                f_j_vals.append(avg_reward)
                c_j_vals.append(avg_cost)

            n_j = np.array(n_j_vals)
            f_j = np.array(f_j_vals)
            c_j = np.array(c_j_vals)

            bonus = np.sqrt(
                self.alpha * np.log(max(self.t, 1)) / np.maximum(1, n_j))
            bonus[n_j == 0] = self.T
            bonus[-1] = 0.0

            f_ucb.append(f_j + bonus)
            c_lcb.append(np.clip(c_j + bonus, 0.0, 1.0))

        marginals = self._solve_marginal_lp(f_ucb, c_lcb, self.B / self.T)
        choice = tuple(
            self.rng.choice(self.Ks[j], p=marginals[j])
            for j in range(self.N)
        )
        self.current_choice = choice
        return choice

    def update(self, rewards, costs):
        for j, idx in enumerate(self.current_choice):
            self.samples[j][idx].append((self.t, rewards[j], costs[j]))
        self.B_rem -= costs.sum()
        self.t += 1
