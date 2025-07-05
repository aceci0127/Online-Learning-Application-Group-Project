from typing import List, Optional, Tuple
from abc import ABC, abstractmethod
from scipy.optimize import linprog
from collections import deque
import math as math
from typing import List, Tuple, Optional, Union
import numpy as np


class Agent(ABC):
    """Abstract base class for all agents"""

    @abstractmethod
    def pull_arm(self) -> Union[int, List[Optional[int]], Tuple[int, ...]]:
        """Select an action/arm"""
        pass

    @abstractmethod
    def update(self, feedback: object) -> None:
        """Update the agent with the received feedback"""
        pass


class UCB1PricingAgent(Agent):
    """UCB1 agent for simple pricing"""

    def __init__(self, K: int, T: int, range: float = 1) -> None:
        self.K: int = K
        self.T: int = T
        self.range: float = range
        self.a_t: Optional[int] = None
        self.average_rewards: np.ndarray = np.zeros(K)
        self.N_pulls: np.ndarray = np.zeros(K)
        self.t: int = 0

    def pull_arm(self) -> int:
        if self.t < self.K:
            self.a_t = self.t
        else:
            ucbs = self.average_rewards + self.range * \
                np.sqrt(2 * np.log(self.t) / self.N_pulls)
            self.a_t = int(np.argmax(ucbs))
        return int(self.a_t)

    def update(self, r_t: float) -> None:
        if self.a_t is None:
            return
        self.N_pulls[self.a_t] += 1
        self.average_rewards[self.a_t] += (
            r_t - self.average_rewards[self.a_t]) / self.N_pulls[self.a_t]
        self.t += 1


class ConstrainedUCBPricingAgent(Agent):
    """Constrained UCB agent for pricing with budget"""

    def __init__(self, K: int, B: float, T: int, alpha: float = 2) -> None:
        self.K: int = K
        self.T: int = T
        self.alpha: float = alpha
        self.a_t: Optional[int] = None
        self.avg_f: np.ndarray = np.zeros(K)
        self.avg_c: np.ndarray = np.zeros(K)
        self.N_pulls: np.ndarray = np.zeros(K)
        self.rng = np.random.default_rng()
        self.rem_budget: float = B
        self.budget: float = B
        self.rho: float = B / T
        self.t: int = 0

    def pull_arm(self) -> Optional[int]:
        if self.rem_budget < 1:
            self.a_t = None
            return None

        if self.t < self.K:
            self.a_t = self.t
            return self.a_t

        f_ucbs = self.avg_f + \
            np.sqrt(self.alpha * np.log(self.t) / self.N_pulls)
        c_lcbs = self.avg_c - \
            np.sqrt(self.alpha * np.log(self.t) / self.N_pulls)
        c_lcbs = np.clip(c_lcbs, 0.0, 1.0)
        f_ucbs[-1] = 0.0
        c_lcbs[-1] = 0.0

        gamma_t = self.compute_opt(f_ucbs, c_lcbs)
        self.a_t = int(self.rng.choice(self.K, p=gamma_t))
        return self.a_t

    def compute_opt(self, f_ucbs: np.ndarray, c_lcbs: np.ndarray) -> np.ndarray:
        c = -f_ucbs
        A_ub = [c_lcbs]
        b_ub = [self.rho]
        A_eq = [np.ones(self.K)]
        b_eq = [1]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq,
                      b_eq=b_eq, bounds=(0, 1))
        gamma: np.ndarray = res.x
        if not np.all(gamma >= 0):
            raise ValueError(
                "Invalid distribution: negative probabilities found in gamma.")
        if not np.isclose(np.sum(gamma), 1.0, atol=1e-6):
            raise ValueError(
                f"Invalid distribution: probabilities do not sum to 1. Found {np.sum(gamma)}.")
        return gamma

    def update(self, f_t: float, c_t: float) -> None:
        if self.a_t is None:
            return
        self.N_pulls[self.a_t] += 1
        self.avg_f[self.a_t] += (f_t - self.avg_f[self.a_t]
                                 ) / self.N_pulls[self.a_t]
        self.avg_c[self.a_t] += (c_t - self.avg_c[self.a_t]
                                 ) / self.N_pulls[self.a_t]
        self.rem_budget -= c_t
        self.t += 1


class ConstrainedCombinatorialUCBAgent(Agent):
    """Constrained Combinatorial UCB agent for multi-product"""

    def __init__(self, price_grid: List[np.ndarray], B: float, T: int, alpha: float = 2) -> None:
        self.price_grid: List[np.ndarray] = price_grid
        self.N: int = len(price_grid)
        self.Ks: List[int] = [len(pg) for pg in price_grid]
        self.B_rem: float = B
        self.B: float = B
        self.T: int = T
        self.rho: float = B / T
        self.t: int = 0
        self.alpha: float = alpha
        self.rng = np.random.default_rng()
        self.N_pulls: List[np.ndarray] = [np.zeros(K) for K in self.Ks]
        self.avg_f: List[np.ndarray] = [np.zeros(K) for K in self.Ks]
        self.avg_c: List[np.ndarray] = [np.zeros(K) for K in self.Ks]
        self.current_choice: Optional[Tuple[int, ...]] = None

    def _solve_marginal_lp(self, f_ucb: List[np.ndarray], c_lcb: List[np.ndarray], rho: float) -> List[np.ndarray]:
        f_flat: np.ndarray = np.concatenate(f_ucb)
        c_flat: np.ndarray = np.concatenate(c_lcb)
        num_vars: int = len(f_flat)
        c_obj: np.ndarray = -f_flat

        A_eq = np.zeros((self.N, num_vars))
        b_eq = np.ones(self.N)
        offset = 0
        for j, K in enumerate(self.Ks):
            A_eq[j, offset:offset + K] = 1
            offset += K

        A_ub = c_flat.reshape(1, -1)
        b_ub = np.array([rho])
        bounds = [(0, 1)] * num_vars

        res = linprog(c=c_obj,
                      A_ub=A_ub, b_ub=b_ub,
                      A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds,
                      method='highs')
        if not res.success:
            raise RuntimeError("LP failed: " + res.message)

        x_flat: np.ndarray = res.x
        marginals: List[np.ndarray] = []
        offset = 0
        for K in self.Ks:
            marginals.append(x_flat[offset:offset + K])
            offset += K
        return marginals

    def pull_arm(self) -> Optional[Tuple[int, ...]]:
        if self.B_rem < 1 or self.t >= self.T:
            return None

        f_ucb: List[np.ndarray] = []
        c_lcb: List[np.ndarray] = []
        for j in range(self.N):
            n_j: np.ndarray = self.N_pulls[j]
            f_j: np.ndarray = self.avg_f[j]
            c_j: np.ndarray = self.avg_c[j]

            bonus: np.ndarray = np.sqrt(
                self.alpha * np.log(max(self.t, 1)) / np.maximum(1, n_j))
            bonus[n_j == 0] = self.T
            bonus[-1] = 0.0

            f_ucb.append(f_j + bonus)
            c_lcb.append(np.clip(c_j - bonus, 0.0, 1.0))

        marginals: List[np.ndarray] = self._solve_marginal_lp(
            f_ucb, c_lcb, self.rho)
        choice: Tuple[int, ...] = tuple(
            int(self.rng.choice(self.Ks[j], p=marginals[j])) for j in range(self.N))
        self.current_choice = choice
        return choice

    def update(self, rewards: np.ndarray, costs: np.ndarray) -> None:
        for j, idx in enumerate(self.current_choice):
            self.N_pulls[j][idx] += 1
            n: float = self.N_pulls[j][idx]
            self.avg_f[j][idx] += (rewards[j] - self.avg_f[j][idx]) / n
            self.avg_c[j][idx] += (costs[j] - self.avg_c[j][idx]) / n
        self.B_rem -= costs.sum()
        self.t += 1


class HedgeAgent(Agent):
    """Hedge agent for online learning"""

    def __init__(self, K: int, learning_rate: float) -> None:
        self.K: int = K
        self.learning_rate: float = learning_rate
        self.weights: np.ndarray = np.ones(K)
        self.x_t: np.ndarray = np.ones(K) / K
        self.a_t: Optional[int] = None
        self.rng = np.random.default_rng()
        self.t: int = 0

    def pull_arm(self) -> int:
        self.x_t = self.weights / np.sum(self.weights)
        self.a_t = int(self.rng.choice(np.arange(self.K), p=self.x_t))
        return int(self.a_t)

    def update(self, l_t: np.ndarray) -> None:
        self.weights *= np.exp(-self.learning_rate * l_t)
        self.t += 1


class FFPrimalDualPricingAgent(Agent):
    """Primal-Dual agent with Full-Feedback for non-stationary pricing"""

    def __init__(self, prices: np.ndarray, T: int, B: float, eta: float) -> None:
        self.prices: np.ndarray = np.array(prices)
        self.K: int = len(prices)
        self.T: int = T
        self.inventory: float = B
        self.eta: float = eta
        self.rng = np.random.default_rng()
        self.hedge: HedgeAgent = HedgeAgent(
            self.K, np.sqrt(np.log(self.K) / T))
        self.rho: float = B / T
        self.lmbd: float = 1.0
        self.t: int = 0
        self.pull_counts: np.ndarray = np.zeros(self.K, int)
        self.last_arm: Optional[int] = None

    def pull_arm(self) -> Optional[int]:
        if self.inventory < 1:
            self.last_arm = None
            return self.last_arm
        self.last_arm = self.hedge.pull_arm()
        return self.last_arm

    def update(self, v_t: float) -> Tuple[float, float]:
        sale_mask: np.ndarray = (self.prices <= v_t).astype(float)
        f_full: np.ndarray = self.prices * sale_mask
        c_full: np.ndarray = sale_mask

        L: np.ndarray = f_full - self.lmbd * (c_full - self.rho)
        f_max: float = self.prices.max()
        L_up: float = f_max - self.lmbd * (0 - self.rho)
        L_low: float = 0.0 - self.lmbd * (1 - self.rho)

        rescaled: np.ndarray = (L - L_low) / (L_up - L_low + 1e-12)
        losses: np.ndarray = 1.0 - rescaled
        self.hedge.update(losses)

        if self.last_arm is not None:
            c_t: int = 1 if self.prices[self.last_arm] <= v_t else 0
            f_t: float = self.prices[self.last_arm] * c_t
            self.inventory -= c_t
            self.pull_counts[self.last_arm] += 1
        else:
            c_t = 0
            f_t = 0.0

        self.lmbd = np.clip(self.lmbd - self.eta * (self.rho - c_t),
                            a_min=0.0, a_max=1.0 / self.rho)
        self.t += 1
        return f_t, float(c_t)


class MultiProductFFPrimalDualPricingAgent(Agent):
    """Primal-Dual agent with Full-Feedback for multi-product"""

    def __init__(self, prices: List[np.ndarray], T: int, B: float, n_products: int, eta: float) -> None:
        # Assuming all products have the same price grid
        self.prices: np.ndarray = prices[0]
        self.K: int = len(prices[0])
        self.T: int = T
        self.n_products: int = n_products
        self.B: float = B
        self.rho: float = B / (n_products * T)
        self.eta: float = eta
        self.rng = np.random.default_rng()
        self.hedges: List[HedgeAgent] = [HedgeAgent(self.K, np.sqrt(np.log(self.K) / T))
                                         for _ in range(n_products)]
        self.lmbd: float = 1.0

    def pull_arm(self) -> List[Optional[int]]:
        if self.B < 1:
            return [None] * self.n_products
        arms: List[Optional[int]] = [hedge.pull_arm() for hedge in self.hedges]
        return arms

    def update(self, v_t: np.ndarray) -> Tuple[float, int]:
        arms: List[Optional[int]] = self.pull_arm()
        total_revenue: float = 0.0
        total_units_sold: int = 0
        losses: List[np.ndarray] = []
        p_max: float = self.prices.max()
        L_up: float = p_max - self.lmbd * (0 - self.rho)
        L_low: float = 0.0 - self.lmbd * (1 - self.rho)
        norm_factor: float = L_up - L_low + 1e-12

        for j in range(self.n_products):
            arm: Optional[int] = arms[j]
            if arm is None:
                losses.append(np.zeros(self.K))
                continue

            p_chosen: float = self.prices[arm]
            # v_t[j] is assumed to be a numpy array; take the first element after flattening
            val_j: float = float(v_t[j])

            sale: int = 1 if p_chosen <= val_j else 0
            f_val: float = p_chosen * sale
            total_revenue += f_val
            total_units_sold += sale

            would_sell: np.ndarray = (self.prices <= val_j).astype(float)
            f_vec: np.ndarray = self.prices * would_sell
            L_vec: np.ndarray = f_vec - self.lmbd * (would_sell - self.rho)
            loss_vec: np.ndarray = 1.0 - (L_vec - L_low) / norm_factor
            loss_vec[-1] = 1.0
            losses.append(loss_vec)

        for j in range(self.n_products):
            self.hedges[j].update(losses[j])

        self.B -= total_units_sold
        self.lmbd = np.clip(self.lmbd - self.eta * (self.rho * self.n_products - total_units_sold),
                            a_min=0.0, a_max=1 / self.rho)
        return total_revenue, total_units_sold


class SlidingWindowConstrainedCombinatorialUCBAgent(ConstrainedCombinatorialUCBAgent):
    """Constrained Combinatorial UCB agent with Sliding Window for non-stationarity"""

    def __init__(self, price_grid: List[np.ndarray], B: float, T: int, window_size: int, alpha: float = 2) -> None:
        super().__init__(price_grid, B, T, alpha)
        self.window_size: int = window_size
        self.samples: List[List[deque]] = [[deque() for _ in range(K)]
                                           for K in self.Ks]

    def pull_arm(self) -> Optional[Tuple[int, ...]]:
        if self.B_rem < 1 or self.t >= self.T:
            return None

        f_ucb: List[np.ndarray] = []
        c_lcb: List[np.ndarray] = []
        for j in range(self.N):
            K: int = self.Ks[j]
            f_j_vals: List[float] = []
            c_j_vals: List[float] = []
            n_j_vals: List[int] = []
            for k in range(K):
                while self.samples[j][k] and self.samples[j][k][0][0] < self.t - self.window_size:
                    self.samples[j][k].popleft()
                samples: deque = self.samples[j][k]
                n: int = len(samples)
                avg_reward: float = np.mean(
                    [item[1] for item in samples]) if n > 0 else 0.0
                avg_cost: float = np.mean(
                    [item[2] for item in samples]) if n > 0 else 0.0
                n_j_vals.append(n)
                f_j_vals.append(avg_reward)
                c_j_vals.append(avg_cost)
            n_j: np.ndarray = np.array(n_j_vals)
            f_j: np.ndarray = np.array(f_j_vals)
            c_j: np.ndarray = np.array(c_j_vals)
            bonus: np.ndarray = np.sqrt(
                self.alpha * np.log(max(self.t, 1)) / np.maximum(1, n_j))
            bonus[n_j == 0] = self.T
            bonus[-1] = 0.0
            f_ucb.append(f_j + bonus)
            c_lcb.append(np.clip(c_j - bonus, 0.0, 1.0))
        marginals: List[np.ndarray] = self._solve_marginal_lp(
            f_ucb, c_lcb, self.rho)
        choice: Tuple[int, ...] = tuple(
            int(self.rng.choice(self.Ks[j], p=marginals[j])) for j in range(self.N))
        self.current_choice = choice
        return choice

    def update(self, rewards: np.ndarray, costs: np.ndarray) -> None:
        for j, idx in enumerate(self.current_choice):
            self.samples[j][idx].append((self.t, rewards[j], costs[j]))
        self.B_rem -= costs.sum()
        self.t += 1

#########


class Exp3PAgent:
    """EXP3.P sub-agent for single product"""

    def __init__(self, K: int, T: int, delta: float = 0.1):
        self.K = K
        self.gamma = min(1.0, math.sqrt(
            (K * math.log(K / delta)) / ((math.e - 1) * T)))
        self.alpha = (math.log(K / delta) / K) * \
            math.sqrt(1 / ((math.e - 1) * T))
        self.weights = np.ones(K)
        self.probs = np.ones(K) / K
        self.t = 0

    def get_probs(self) -> np.ndarray:
        W = np.sum(self.weights)
        base = (1 - self.gamma) * (self.weights / W)
        return base + self.gamma / self.K

    def pull_arm(self) -> int:
        self.probs = self.get_probs()
        choice = int(np.random.choice(self.K, p=self.probs))
        self.t += 1
        return choice

    def update(self, chosen: int, reward: float) -> None:
        # reward in [0,1]
        p = self.probs[chosen]
        x_hat = reward / p
        # correction term for variance
        correction = self.alpha / (p * math.sqrt(self.K * self.t))
        self.weights[chosen] *= math.exp((self.gamma *
                                         x_hat + correction) / self.K)


class MultiProductPDExp3PricingAgent:
    """Primal-Dual agent using EXP3.P bandit feedback for multi-product pricing"""

    def __init__(
        self,
        prices: List[np.ndarray],
        T: int,
        B: float,
        n_products: int,
        delta: float = 0.1,
        eta: Optional[float] = None
    ) -> None:
        self.prices = prices[0]  # assume same grid
        self.K = len(self.prices)
        self.T = T
        self.n_products = n_products
        self.B = B
        self.rho = B / (n_products * T)
        # EXP3.P parameters
        self.delta = delta
        self.sub_agents = [Exp3PAgent(self.K, T, delta)
                           for _ in range(n_products)]
        # Dual step size
        self.eta = eta if eta is not None else 1 / math.sqrt(n_products * T)
        self.lmbd = 1.0
        self.t = 0

    def pull_arm(self) -> List[Optional[int]]:
        if self.B < 1:
            return [None] * self.n_products
        choices = [agent.pull_arm() for agent in self.sub_agents]
        return choices

    def update(self, values: np.ndarray) -> Tuple[float, int]:
        # Bandit feedback: only see chosen arms' sales
        choices = self.pull_arm()
        total_revenue = 0.0
        total_sales = 0

        p_max = self.prices.max()
        for j, agent in enumerate(self.sub_agents):
            choice = choices[j]
            if choice is None:
                continue
            price = self.prices[choice]
            val = float(values[j])
            sale = 1.0 if price <= val else 0.0

            # primal reward: price * sale
            reward = price * sale
            total_revenue += reward
            total_sales += int(sale)

            # dual penalty term: (sale - rho)
            penalty = self.lmbd * (sale - self.rho)
            # net reward normalized to [0,1]
            net = reward - penalty
            net_norm = (net + p_max) / (2 * p_max)

            # update EXP3.P sub-agent
            agent.probs = agent.get_probs()
            agent.update(choice, net_norm)

        # update dual variable lambda
        self.lmbd = np.clip(
            self.lmbd - self.eta * (self.rho * self.n_products - total_sales),
            a_min=0.0, a_max=1/self.rho
        )

        self.B -= total_sales
        self.t += 1

        return total_revenue, total_sales
