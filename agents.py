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


class UCB1(Agent):
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


class UCBBudget(Agent):
    """Constrained UCB agent for pricing with budget"""

    def __init__(self, K: int, B: float, T: int, alpha: float = 2, adaptive_rho: bool = False) -> None:
        self.K: int = K
        self.T: int = T
        self.alpha: float = alpha
        self.a_t: Optional[int] = None
        self.avg_f: np.ndarray = np.zeros(K)
        self.avg_c: np.ndarray = np.zeros(K)
        self.N_pulls: np.ndarray = np.zeros(K)
        self.rng = np.random.default_rng()
        self.rem_budget: float = B
        self.B: float = B
        self.rho: float = B / T
        self.t: int = 0
        self.adaptive_rho: bool = adaptive_rho

    def pull_arm(self) -> Optional[int]:
        if self.rem_budget <= 0:
            self.a_t = None
            return None

        if self.t < self.K:
            self.a_t = self.t
            return self.a_t

        f_ucbs = self.avg_f + \
            np.sqrt(self.alpha * np.log(self.t) / self.N_pulls)
        c_lcbs = self.avg_c - \
            np.sqrt(self.alpha * np.log(self.t) / self.N_pulls)
        # TODO negative cost doesnt make sense, note on google doc
        c_lcbs = np.clip(c_lcbs, 0.0, 1.0)
        f_ucbs[-1] = 0.0
        c_lcbs[-1] = 0.0

        gamma_t = self.compute_opt(f_ucbs, c_lcbs)
        self.a_t = int(self.rng.choice(self.K, p=gamma_t))
        return self.a_t

    def compute_opt(self, f_ucbs: np.ndarray, c_lcbs: np.ndarray) -> np.ndarray:
        c = -f_ucbs
        A_ub = [c_lcbs]
        if not self.adaptive_rho:
            b_ub = [self.rho]
        else:
            b_ub = [self.rem_budget / (self.T - self.t + 1)]
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

    def __init__(self, price_grid: List[np.ndarray], B: float, T: int, alpha: float = 2, adaptive_rho: bool = False) -> None:
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
        self.current_choice: List[int] = []
        self.adaptive_rho: bool = adaptive_rho

    def _solve_marginal_lp(self, f_ucb: List[np.ndarray], c_lcb: List[np.ndarray]) -> List[np.ndarray]:
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
        if not self.adaptive_rho:
            b_ub = np.array([self.rho])
        else:
            b_ub = np.array([self.B_rem / (self.T - self.t + 1)])
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
    '''

    def _solve_marginal_lp(self, f_ucb: List[np.ndarray], c_lcb: List[np.ndarray], rho: float) -> np.ndarray:
        """
        Solves an LP over the full set of superarms.
        Each superarm is a tuple (k1, k2, ..., kN) with one index for each product.

        The LP is:
        maximize   sum_i f_super[i] * y_i    (or minimize - sum_i f_super[i] * y_i)
        s.t.       sum_i c_super[i] * y_i <= rho,
                    sum_i y_i = 1,
                    y_i >= 0   for all i.

        Returns:
            y: 1D numpy array giving the optimal probability over superarms.
        """
        N = len(f_ucb)
        # Build the list of all possible combinations: one index per product
        superarm_indices = list(itertools.product(
            *[range(len(f_ucb[j])) for j in range(N)]))
        num_superarms = len(superarm_indices)

        # Compute total revenue and cost for each superarm
        f_super = np.empty(num_superarms)
        c_super = np.empty(num_superarms)
        for i, indices in enumerate(superarm_indices):
            # revenue is sum of f_ucb values and cost is sum of c_lcb values for that combination
            f_super[i] = sum(f_ucb[j][indices[j]] for j in range(N))
            c_super[i] = sum(c_lcb[j][indices[j]] for j in range(N))

        # Define the LP:
        # Objective: minimize -f_super^T y
        c_obj = -f_super
        # Equality constraint: sum_i y_i = 1
        A_eq = np.ones((1, num_superarms))
        b_eq = np.array([1])
        # Inequality constraint: sum_i c_super[i] * y_i <= rho
        A_ub = c_super.reshape(1, -1)
        b_ub = np.array([rho])
        # y_i >= 0. The upper bound can be left at 1.
        bounds = [(0, 1)] * num_superarms

        res = linprog(c=c_obj, A_ub=A_ub, b_ub=b_ub,
                      A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                      method='highs-ds', options={"tol": 1e-6})
        if not res.success and not (res.status == 15 and res.primal_status == 0):
            raise RuntimeError(f"LP failed: {res.message}")

        # res.x is the probability distribution over superarms.
        return res.x
    '''

    def pull_arm(self) -> Optional[List[int]]:
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
            f_ucb, c_lcb)
        choice: List[int] = [
            int(self.rng.choice(self.Ks[j], p=marginals[j])) for j in range(self.N)]
        self.current_choice = choice
        return choice
    '''
    def pull_arm(self) -> Optional[List[int]]:
        if self.B_rem < 1 or self.t >= self.T:
            return None

        f_ucb: List[np.ndarray] = []
        c_lcb: List[np.ndarray] = []
        for j in range(self.N):
            n_j = self.N_pulls[j]
            f_j = self.avg_f[j]
            c_j = self.avg_c[j]
            # Compute bonus; when not pulled, set bonus high so that option is forced to be selected
            bonus = np.sqrt(
                self.alpha * np.log(max(self.t, 1)) / np.maximum(1, n_j))
            bonus[n_j == 0] = self.T
            # Last price option (often a dummy) has no bonus
            bonus[-1] = 0.0
            f_ucb.append(f_j + bonus)
            c_lcb.append(np.clip(c_j - bonus, 0.0, 1.0))

        # Use the superarm LP to get a joint distribution over all products.
        y = self._solve_marginal_lp(f_ucb, c_lcb, self.rho)

        # Reconstruct the list of all possible superarms.
        superarm_indices = list(
            itertools.product(*[range(k) for k in self.Ks]))
        # Sample one superarm according to distribution y.
        chosen_idx = self.rng.choice(len(y), p=y)
        choice = list(superarm_indices[chosen_idx])
        self.current_choice = choice
        return choice
    '''

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
        self.B: float = B
        self.eta: float = eta
        self.rng = np.random.default_rng()
        self.hedge: HedgeAgent = HedgeAgent(
            self.K, np.sqrt(np.log(self.K) / T))
        self.rho: float = B / T
        self.lmbd: float = 1.0
        self.t: int = 0
        self.pull_counts: np.ndarray = np.zeros(self.K, int)
        self.last_arm: Optional[int] = None
        # New histories for plotting
        self.lmbd_history: List[float] = []
        self.hedge_weight_history: List[np.ndarray] = []

    def pull_arm(self) -> Optional[int]:
        if self.B < 1:
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
            self.B -= c_t
            self.pull_counts[self.last_arm] += 1
        else:
            c_t = 0
            f_t = 0.0

        # Update lambda and record it
        self.lmbd = np.clip(self.lmbd - self.eta * (self.rho - c_t),
                            a_min=0.0, a_max=1.0 / self.rho)
        self.lmbd_history.append(self.lmbd)
        # Record a copy of the current hedge weights
        self.hedge_weight_history.append(self.hedge.weights.copy())
        self.t += 1
        return f_t, float(c_t)


class BanditFeedbackPrimalDual(Agent):
    """Primal-Dual agent with Bandit Feedback for non-stationary pricing using EXP3.P."""

    def __init__(self, prices: np.ndarray, T: int, B: float) -> None:
        self.prices: np.ndarray = np.array(prices)
        self.K: int = len(prices)
        self.T: int = T
        self.B: float = B
        self.eta: float = 1 / np.sqrt(T)
        self.rng = np.random.default_rng()
        # Use EXP3.P as the primal (hedge) agent with a given delta
        self.exp3p: Exp3PAgent = Exp3PAgent(K=self.K, T=self.T, delta=0.05)
        self.rho: float = B / T
        self.lmbd: float = 1.0
        self.t: int = 0
        self.pull_counts: np.ndarray = np.zeros(self.K, int)
        self.last_arm: Optional[int] = None
        # Histories for plotting
        self.lmbd_history: List[float] = []
        self.exp3p_weight_history: List[np.ndarray] = []

    def pull_arm(self) -> Optional[int]:
        if self.B < 1:
            self.last_arm = None
            return self.last_arm
        self.last_arm = self.exp3p.pull_arm()
        return self.last_arm

    def update(self, v_t: float) -> Tuple[float, float]:
        if self.last_arm is None:
            return 0.0, 0.0

        price_chosen = self.prices[self.last_arm]
        c_t = 1 if price_chosen <= v_t else 0

        f_t = price_chosen * c_t
        self.B -= c_t
        self.pull_counts[self.last_arm] += 1

        net = f_t - self.lmbd * (c_t - self.rho)

        # Normalize
        p_max = self.prices.max()
        L_up = p_max - self.lmbd * (0 - self.rho)
        L_low = 0.0 - self.lmbd * (1 - self.rho)
        norm_factor = L_up - L_low + 1e-12
        net_norm = (net - L_low) / norm_factor

        # Update the EXP3.P sub-agent using the bandit reward feedback for the chosen arm
        self.exp3p.probs = self.exp3p._compute_probs()
        self.exp3p.update(self.last_arm, net_norm)

        # Update the dual variable lambda
        self.lmbd = np.clip(self.lmbd - self.eta *
                            (self.rho - c_t), a_min=0.0, a_max=1.0 / self.rho)
        self.lmbd_history.append(self.lmbd)

        self.t += 1
        return f_t, float(c_t)


class MultiProductFFPrimalDualPricingAgent(Agent):
    """Primal-Dual agent with Full-Feedback for multi-product pricing"""

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
        self.hedges: List[HedgeAgent] = [
            HedgeAgent(self.K, np.sqrt(np.log(self.K) / T))
            for _ in range(n_products)
        ]
        self.lmbd: float = 1.0
        self.lmbd_history: List[float] = []
        self.hedge_prob_history: List[List[np.ndarray]] = [
            [] for _ in range(n_products)]

    def pull_arm(self) -> Optional[List[int]]:
        if self.B < 1:
            return None
        arms: List[int] = [hedge.pull_arm() for hedge in self.hedges]
        return arms

    def update(self, v_t: np.ndarray) -> Tuple[float, int]:
        arms: List[int] | None = self.pull_arm()
        total_revenue: float = 0.0
        total_units_sold: int = 0
        losses: List[np.ndarray] = []
        p_max: float = self.prices.max()
        L_up: float = p_max - self.lmbd * (0 - self.rho)
        L_low: float = 0.0 - self.lmbd * (1 - self.rho)
        norm_factor: float = L_up - L_low + 1e-12

        for j in range(self.n_products):
            if arms is None:
                losses.append(np.zeros(self.K))
                continue
            arm: int = arms[j]

            p_chosen: float = self.prices[arm]
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
            prob_j = self.hedges[j].weights / np.sum(self.hedges[j].weights)
            self.hedge_prob_history[j].append(prob_j.copy())

        self.B -= total_units_sold
        self.lmbd = np.clip(self.lmbd - self.eta * (self.rho * self.n_products - total_units_sold),
                            a_min=0.0, a_max=1 / self.rho)
        self.lmbd_history.append(self.lmbd)
        return total_revenue, total_units_sold


class SlidingWindowConstrainedCombinatorialUCBAgent(ConstrainedCombinatorialUCBAgent):
    """Constrained Combinatorial UCB agent with Sliding Window for non-stationarity"""

    def __init__(self, price_grid: List[np.ndarray], B: float, T: int, window_size: int, alpha: float = 2) -> None:
        super().__init__(price_grid, B, T, alpha)
        self.window_size: int = window_size
        self.samples: List[List[deque]] = [[deque() for _ in range(K)]
                                           for K in self.Ks]

    def pull_arm(self) -> Optional[List[int]]:
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
                avg_reward: float = float(np.mean(
                    [item[1] for item in samples])) if n > 0 else 0.0
                avg_cost: float = float(np.mean(
                    [item[2] for item in samples])) if n > 0 else 0.0
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
            f_ucb, c_lcb)
        choice: List[int] = [
            int(self.rng.choice(self.Ks[j], p=marginals[j])) for j in range(self.N)]
        self.current_choice = choice
        return choice

    def update(self, rewards: np.ndarray, costs: np.ndarray) -> None:
        for j, idx in enumerate(self.current_choice):
            self.samples[j][idx].append((self.t, rewards[j], costs[j]))
        self.B_rem -= costs.sum()
        self.t += 1

#########


class Exp3PAgent(Agent):
    """
    EXP3.P agent for adversarial bandits with high-probability regret bounds.

    Applies the Exp3.P algorithm with the following hyperparameter settings:
      - η = log(K/δ) / (T * K)
      - γ = 0.95 * log(K) / (T * K)
      - β = K * log(K) / T

    Theorem:
      The Exp3.P algorithm applied to an adversarial MAB problem with K arms and the above
      parameters suffers a regret of:
          R_T ≤ 5.15 * T * K * log(K/δ)
      with probability at least 1 − δ.

    On each round t:
      1. Compute sampling distribution p_t:
           p_{i,t} = (1-γ)*exp(η * G_i) / Σ_j exp(η * G_j) + γ/K
      2. Draw I_t ∼ p_t and observe reward g ∈ [0,1]
      3. Form importance-weighted reward plus bias:
           x̂ = (g + β) / p_{I_t,t}
           Define B as the vector with B_i = β / p_{i,t} for all arms.
      4. Update cumulative pseudo-rewards:
           G_{I_t} ← G_{I_t} + x̂
           For each i ≠ I_t, update G_i ← G_i + B_i

    References:
      - https://trovo.faculty.polimi.it/02source/olam_2022/2022_05_11_Lez_4_MAB.pdf
    """

    def __init__(self, K: int, T: int, delta: float = 0.1):
        self.K = K
        self.T = T
        self.delta = delta
        self.eta = math.log(K / delta) / (T * K)
        self.gamma = 0.95 * math.log(K) / (T * K)
        self.beta = K * math.log(K) / T
        self.G = np.zeros(K)
        self.probs = np.ones(K) / K

    def _compute_probs(self) -> np.ndarray:
        expG = np.exp(self.eta * self.G)
        base = (1 - self.gamma) * (expG / expG.sum())
        probs = base + self.gamma / self.K
        return probs / probs.sum()

    def pull_arm(self) -> int:
        self.probs = self._compute_probs()
        choice = int(np.random.choice(self.K, p=self.probs))
        return choice

    def update(self, chosen: int, reward: float) -> None:
        for i in range(self.K):
            if i != chosen:
                self.G[i] += self.beta / self.probs[i]
            else:
                self.G[i] += (reward + self.beta) / self.probs[i]


class MultiProductPDExp3PricingAgent(Agent):
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
        self.last_choices: Optional[List[int]] = None
        # EXP3.P parameters
        self.delta = delta
        self.exp3p_agents = [Exp3PAgent(self.K, T, delta)
                             for _ in range(n_products)]
        # Dual step size
        self.eta = eta if eta is not None else 1 / math.sqrt(n_products * T)
        self.lmbd = 1.0
        self.lmbd_history: List[float] = []

    def pull_arm(self) -> Optional[List[int]]:
        if self.B < 1:
            return None
        self.last_choices = [agent.pull_arm() for agent in self.exp3p_agents]
        return self.last_choices

    def update(self, values: np.ndarray) -> Tuple[float, int]:
        total_revenue = 0.0
        total_sales = 0
        p_max = self.prices.max()
        for j, agent in enumerate(self.exp3p_agents):
            if self.last_choices is None:
                continue
            choice = self.last_choices[j]
            price = self.prices[choice]
            val = float(values[j])
            cost = 1.0 if price <= val else 0.0

            reward = price * cost
            total_revenue += reward
            total_sales += cost

            net = reward - self.lmbd * (cost - self.rho)

            L_up = p_max - self.lmbd * (0 - self.rho)
            L_low = 0.0 - self.lmbd * (1 - self.rho)
            norm_factor = L_up - L_low + 1e-12
            net_norm = (net - L_low) / norm_factor

            # update EXP3.P sub-agent based on the stored choice
            agent.probs = agent._compute_probs()
            agent.update(choice, net_norm)

        self.lmbd = np.clip(
            self.lmbd - self.eta * (self.rho * self.n_products - total_sales),
            a_min=0.0, a_max=1/self.rho
        )
        self.B -= total_sales
        self.lmbd_history.append(self.lmbd)
        return total_revenue, total_sales
