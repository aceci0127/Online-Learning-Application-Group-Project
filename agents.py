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
    def pull_arm(self) -> Union[int, List[Optional[int]], Tuple[int, ...], None]:
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
        self.rng = np.random.default_rng()

        # Standardized variables
        self.current_action: Optional[int] = None
        self.average_rewards: np.ndarray = np.zeros(K)
        self.pull_counts: np.ndarray = np.zeros(K)
        self.t: int = 0

        # History tracking
        self.action_history: List[Optional[int]] = []
        self.reward_history: List[float] = []

    def pull_arm(self) -> int:
        if self.t < self.K:
            self.current_action = self.t
        else:
            ucbs = self.average_rewards + self.range * \
                np.sqrt(2 * np.log(self.t) / self.pull_counts)
            self.current_action = int(np.argmax(ucbs))

        self.action_history.append(self.current_action)
        return int(self.current_action)

    def update(self, reward: float) -> None:
        if self.current_action is None:
            return

        self.pull_counts[self.current_action] += 1
        self.average_rewards[self.current_action] += (
            reward - self.average_rewards[self.current_action]) / self.pull_counts[self.current_action]

        self.reward_history.append(reward)
        self.t += 1


class UCBBudget(Agent):
    """Constrained UCB agent for pricing with budget"""

    def __init__(self, K: int, B: float, T: int, alpha: float = 2, adaptive_rho: bool = False) -> None:
        self.K: int = K
        self.T: int = T
        self.alpha: float = alpha
        self.rng = np.random.default_rng()

        # Standardized variables
        self.current_action: Optional[int] = None
        self.average_rewards: np.ndarray = np.zeros(K)
        self.average_costs: np.ndarray = np.zeros(K)
        self.pull_counts: np.ndarray = np.zeros(K)
        self.remaining_budget: float = B
        self.initial_budget: float = B
        self.rho: float = B / T
        self.t: int = 0
        self.adaptive_rho: bool = adaptive_rho

        # History tracking
        self.action_history: List[Optional[int]] = []
        self.reward_history: List[float] = []
        self.cost_history: List[float] = []

    def pull_arm(self) -> Optional[int]:
        if self.remaining_budget <= 0:
            self.current_action = None
            self.action_history.append(self.current_action)
            return None

        if self.t < self.K:
            self.current_action = self.t
        else:
            f_ucbs = self.average_rewards + \
                np.sqrt(self.alpha * np.log(self.t) / self.pull_counts)
            c_lcbs = self.average_costs - \
                np.sqrt(self.alpha * np.log(self.t) / self.pull_counts)
            c_lcbs = np.clip(c_lcbs, 0.0, 1.0)
            f_ucbs[-1] = 0.0
            c_lcbs[-1] = 0.0

            gamma_t = self.compute_opt(f_ucbs, c_lcbs)
            self.current_action = int(self.rng.choice(self.K, p=gamma_t))

        self.action_history.append(self.current_action)
        return self.current_action

    def compute_opt(self, f_ucbs: np.ndarray, c_lcbs: np.ndarray) -> np.ndarray:
        c = -f_ucbs
        A_ub = [c_lcbs]
        if not self.adaptive_rho:
            b_ub = [self.rho]
        else:
            b_ub = [self.remaining_budget / (self.T - self.t + 1)]
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

    def update(self, reward: float, cost: float) -> None:
        if self.current_action is None:
            return

        self.pull_counts[self.current_action] += 1
        self.average_rewards[self.current_action] += (reward - self.average_rewards[self.current_action]
                                                      ) / self.pull_counts[self.current_action]
        self.average_costs[self.current_action] += (cost - self.average_costs[self.current_action]
                                                    ) / self.pull_counts[self.current_action]

        self.remaining_budget -= cost
        self.reward_history.append(reward)
        self.cost_history.append(cost)
        self.t += 1


class ConstrainedCombinatorialUCBAgent(Agent):
    """Constrained Combinatorial UCB agent for multi-product"""

    def __init__(self, price_grid: List[np.ndarray], B: float, T: int, alpha: float = 2, adaptive_rho: bool = False) -> None:
        self.price_grid: List[np.ndarray] = price_grid
        self.N: int = len(price_grid)
        self.K_list: List[int] = [len(pg) for pg in price_grid]
        self.T: int = T
        self.alpha: float = alpha
        self.rng = np.random.default_rng()
        self.remaining_budget: float = B
        self.initial_budget: float = B
        self.rho: float = B / T
        self.t: int = 0
        self.adaptive_rho: bool = adaptive_rho

        # Pull counts and averages
        self.pull_counts: List[np.ndarray] = [np.zeros(K) for K in self.K_list]
        self.average_rewards: List[np.ndarray] = [
            np.zeros(K) for K in self.K_list]
        self.average_costs: List[np.ndarray] = [
            np.zeros(K) for K in self.K_list]
        self.current_action: List[int] = []

        # History tracking
        self.action_history: List[List[int]] = []
        self.reward_history: List[np.ndarray] = []
        self.cost_history: List[np.ndarray] = []

    def _solve_marginal_lp(self, f_ucb: List[np.ndarray], c_lcb: List[np.ndarray]) -> List[np.ndarray]:
        f_flat: np.ndarray = np.concatenate(f_ucb)
        c_flat: np.ndarray = np.concatenate(c_lcb)
        num_vars: int = len(f_flat)
        c_obj: np.ndarray = -f_flat

        A_eq = np.zeros((self.N, num_vars))
        b_eq = np.ones(self.N)
        offset = 0
        for j, K in enumerate(self.K_list):
            A_eq[j, offset:offset + K] = 1
            offset += K

        A_ub = c_flat.reshape(1, -1)
        if not self.adaptive_rho:
            b_ub = np.array([self.rho])
        else:
            b_ub = np.array([self.remaining_budget / (self.T - self.t + 1)])
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
        for K in self.K_list:
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
        if self.remaining_budget < 1 or self.t >= self.T:
            self.current_action = []
            self.action_history.append(self.current_action.copy())
            return None

        f_ucb: List[np.ndarray] = []
        c_lcb: List[np.ndarray] = []
        for j in range(self.N):
            n_j: np.ndarray = self.pull_counts[j]
            f_j: np.ndarray = self.average_rewards[j]
            c_j: np.ndarray = self.average_costs[j]

            bonus: np.ndarray = np.sqrt(
                self.alpha * np.log(max(self.t, 1)) / np.maximum(1, n_j))
            bonus[n_j == 0] = self.T
            bonus[-1] = 0.0

            f_ucb.append(f_j + bonus)
            c_lcb.append(np.clip(c_j - bonus, 0.0, 1.0))

        marginals: List[np.ndarray] = self._solve_marginal_lp(f_ucb, c_lcb)
        choice: List[int] = [
            int(self.rng.choice(self.K_list[j], p=marginals[j])) for j in range(self.N)]
        self.current_action = choice
        self.action_history.append(self.current_action.copy())
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
        
        # Update action tracking
        self.current_choice = choice
        self.action_history.append(choice.copy())
        
        return choice
    '''

    def update(self, rewards: np.ndarray, costs: np.ndarray) -> None:
        for j, idx in enumerate(self.current_action):
            self.pull_counts[j][idx] += 1
            n: float = self.pull_counts[j][idx]

            self.average_rewards[j][idx] += (rewards[j] -
                                             self.average_rewards[j][idx]) / n
            self.average_costs[j][idx] += (costs[j] -
                                           self.average_costs[j][idx]) / n

        self.remaining_budget -= costs.sum()

        self.reward_history.append(rewards.copy())
        self.cost_history.append(costs.copy())

        self.t += 1


class HedgeAgent(Agent):
    """Hedge agent for online learning"""

    def __init__(self, K: int, learning_rate: float) -> None:
        self.K: int = K
        self.learning_rate: float = learning_rate
        self.rng = np.random.default_rng()

        # Standardized variables
        self.weights: np.ndarray = np.ones(K)
        self.probabilities: np.ndarray = np.ones(K) / K
        self.current_action: Optional[int] = None
        self.t: int = 0

        # History tracking
        self.action_history: List[Optional[int]] = []
        self.loss_history: List[np.ndarray] = []
        self.weight_history: List[np.ndarray] = []

    def pull_arm(self) -> int:
        self.probabilities = self.weights / np.sum(self.weights)
        self.current_action = int(self.rng.choice(
            np.arange(self.K), p=self.probabilities))
        self.action_history.append(self.current_action)
        return int(self.current_action)

    def update(self, losses: np.ndarray) -> None:
        self.weights *= np.exp(-self.learning_rate * losses)
        self.loss_history.append(losses.copy())
        self.weight_history.append(self.weights.copy())
        self.t += 1


class FFPrimalDualPricingAgent(Agent):
    """Primal-Dual agent with Full-Feedback for non-stationary pricing"""

    def __init__(self, prices: np.ndarray, T: int, B: float, eta: float) -> None:
        self.prices: np.ndarray = np.array(prices)
        self.K: int = len(prices)
        self.T: int = T
        self.eta: float = eta
        self.rng = np.random.default_rng()

        # Standardized variables
        self.remaining_budget: float = B
        self.initial_budget: float = B
        self.rho: float = B / T
        self.lmbd: float = 1.0
        self.t: int = 0

        self.hedge: HedgeAgent = HedgeAgent(
            self.K, np.sqrt(np.log(self.K) / T))
        self.pull_counts: np.ndarray = np.zeros(self.K, int)
        self.current_action: Optional[int] = None

        # History tracking
        self.action_history: List[Optional[int]] = []
        self.lmbd_history: List[float] = []
        self.hedge_weight_history: List[np.ndarray] = []

    def pull_arm(self) -> Optional[int]:
        if self.remaining_budget < 1:
            self.current_action = None
        else:
            self.current_action = self.hedge.pull_arm()

        self.action_history.append(self.current_action)
        return self.current_action

    def update(self, valuation: float) -> None:
        sale_mask: np.ndarray = (self.prices <= valuation).astype(float)
        f_full: np.ndarray = self.prices * sale_mask
        c_full: np.ndarray = sale_mask

        L: np.ndarray = f_full - self.lmbd * (c_full - self.rho)
        f_max: float = self.prices.max()
        L_up: float = f_max - self.lmbd * (0 - self.rho)
        L_low: float = 0.0 - self.lmbd * (1 - self.rho)

        rescaled: np.ndarray = (L - L_low) / (L_up - L_low + 1e-12)
        losses: np.ndarray = 1.0 - rescaled
        self.hedge.update(losses)

        if self.current_action is not None:
            cost: int = 1 if self.prices[self.current_action] <= valuation else 0
            self.remaining_budget -= cost
            self.pull_counts[self.current_action] += 1
        else:
            cost = 0

        # Update lambda and record it
        self.lmbd = np.clip(self.lmbd - self.eta * (self.rho - cost),
                            a_min=0.0, a_max=1.0 / self.rho)
        self.lmbd_history.append(self.lmbd)
        self.hedge_weight_history.append(self.hedge.weights.copy())
        self.t += 1


class BanditFeedbackPrimalDual(Agent):
    """Primal-Dual agent with Bandit Feedback for non-stationary pricing using EXP3.P."""

    def __init__(self, prices: np.ndarray, T: int, B: float) -> None:
        self.prices: np.ndarray = np.array(prices)
        self.K: int = len(prices)
        self.T: int = T
        self.eta: float = 1 / np.sqrt(T)
        self.rng = np.random.default_rng()

        # Standardized variables
        self.remaining_budget: float = B
        self.initial_budget: float = B
        self.rho: float = B / T
        self.lmbd: float = 1.0
        self.t: int = 0

        self.exp3p: Exp3PAgent = Exp3PAgent(K=self.K, T=self.T, delta=0.05)
        self.pull_counts: np.ndarray = np.zeros(self.K, int)
        self.current_action: Optional[int] = None

        # History tracking
        self.action_history: List[Optional[int]] = []
        self.lmbd_history: List[float] = []
        self.exp3p_weight_history: List[np.ndarray] = []

    def pull_arm(self) -> Optional[int]:
        if self.remaining_budget < 1:
            self.current_action = None
        else:
            self.current_action = self.exp3p.pull_arm()

        self.action_history.append(self.current_action)
        return self.current_action

    def update(self, valuation: float) -> None:
        if self.current_action is None:
            return

        price_chosen = self.prices[self.current_action]
        cost = 1 if price_chosen <= valuation else 0
        reward = price_chosen * cost

        self.remaining_budget -= cost
        self.pull_counts[self.current_action] += 1

        net = reward - self.lmbd * (cost - self.rho)

        # Normalize
        p_max = self.prices.max()
        L_up = p_max - self.lmbd * (0 - self.rho)
        L_low = 0.0 - self.lmbd * (1 - self.rho)
        norm_factor = L_up - L_low + 1e-12
        net_norm = (net - L_low) / norm_factor

        # Update the EXP3.P sub-agent
        self.exp3p.probabilities = self.exp3p._compute_probs()
        self.exp3p.update(self.current_action, net_norm)

        # Update the dual variable lambda
        self.lmbd = np.clip(self.lmbd - self.eta * (self.rho - cost),
                            a_min=0.0, a_max=1.0 / self.rho)
        self.lmbd_history.append(self.lmbd)
        self.t += 1


class MultiProductFFPrimalDualPricingAgent(Agent):
    """Primal-Dual agent with Full-Feedback for multi-product pricing"""

    def __init__(self, prices: List[np.ndarray], T: int, B: float, n_products: int, eta: float) -> None:
        # Assuming all products have the same price grid
        self.prices: np.ndarray = prices[0]
        self.K: int = len(prices[0])
        self.T: int = T
        self.n_products: int = n_products
        self.eta: float = eta
        self.rng = np.random.default_rng()

        # Standardized variables
        self.remaining_budget: float = B
        self.initial_budget: float = B
        self.rho: float = B / (n_products * T)
        self.lmbd: float = 1.0
        self.t: int = 0

        self.hedges: List[HedgeAgent] = [
            HedgeAgent(self.K, np.sqrt(np.log(self.K) / T))
            for _ in range(n_products)
        ]
        self.current_action: Optional[List[int]] = None

        # History tracking
        self.action_history: List[Optional[List[int]]] = []
        self.lmbd_history: List[float] = []
        self.hedge_prob_history: List[List[np.ndarray]] = [
            [] for _ in range(n_products)]

    def pull_arm(self) -> Optional[List[int]]:
        if self.remaining_budget < 1:
            self.current_action = None
        else:
            self.current_action = [hedge.pull_arm() for hedge in self.hedges]

        self.action_history.append(
            self.current_action.copy() if self.current_action else None)
        return self.current_action

    def update(self, valuations: np.ndarray) -> None:
        if self.current_action is None:
            return

        total_revenue: float = 0.0
        total_units_sold: int = 0
        losses: List[np.ndarray] = []
        p_max: float = self.prices.max()
        L_up: float = p_max - self.lmbd * (0 - self.rho)
        L_low: float = 0.0 - self.lmbd * (1 - self.rho)
        norm_factor: float = L_up - L_low + 1e-12

        for j in range(self.n_products):
            arm: int = self.current_action[j]
            p_chosen: float = self.prices[arm]
            val_j: float = float(valuations[j])
            sale: int = 1 if p_chosen <= val_j else 0
            revenue: float = p_chosen * sale
            total_revenue += revenue
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

        self.remaining_budget -= total_units_sold
        self.lmbd = np.clip(self.lmbd - self.eta * (self.rho * self.n_products - total_units_sold),
                            a_min=0.0, a_max=1 / self.rho)
        self.lmbd_history.append(self.lmbd)
        self.t += 1


class SlidingWindowConstrainedCombinatorialUCBAgent(ConstrainedCombinatorialUCBAgent):
    """Constrained Combinatorial UCB agent with Sliding Window for non-stationarity"""

    def __init__(self, price_grid: List[np.ndarray], B: float, T: int, window_size: int, alpha: float = 2) -> None:
        super().__init__(price_grid, B, T, alpha)
        self.window_size: int = window_size
        self.samples: List[List[deque]] = [[deque() for _ in range(K)]
                                           for K in self.K_list]

    def pull_arm(self) -> Optional[List[int]]:
        if self.remaining_budget < 1 or self.t >= self.T:
            self.current_action = []
            self.action_history.append(self.current_action.copy())
            return None

        f_ucb: List[np.ndarray] = []
        c_lcb: List[np.ndarray] = []
        for j in range(self.N):
            K: int = self.K_list[j]
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

        marginals: List[np.ndarray] = self._solve_marginal_lp(f_ucb, c_lcb)
        choice: List[int] = [
            int(self.rng.choice(self.K_list[j], p=marginals[j])) for j in range(self.N)]
        self.current_action = choice
        self.action_history.append(self.current_action.copy())
        return choice

    def update(self, rewards: np.ndarray, costs: np.ndarray) -> None:
        if not self.current_action:
            return

        for j, idx in enumerate(self.current_action):
            self.samples[j][idx].append((self.t, rewards[j], costs[j]))

        self.remaining_budget -= costs.sum()
        self.reward_history.append(rewards.copy())
        self.cost_history.append(costs.copy())
        self.t += 1


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
        self.K: int = K
        self.T: int = T
        self.delta: float = delta
        self.rng = np.random.default_rng()

        # Standardized variables
        self.eta: float = math.log(K / delta) / (T * K)
        self.gamma: float = 0.95 * math.log(K) / (T * K)
        self.beta: float = K * math.log(K) / T
        self.G: np.ndarray = np.zeros(K)
        self.probabilities: np.ndarray = np.ones(K) / K
        self.current_action: Optional[int] = None
        self.t: int = 0

        # History tracking
        self.action_history: List[Optional[int]] = []
        self.reward_history: List[float] = []
        self.weight_history: List[np.ndarray] = []

    def _compute_probs(self) -> np.ndarray:
        expG = np.exp(self.eta * self.G)
        base = (1 - self.gamma) * (expG / expG.sum())
        probs = base + self.gamma / self.K
        return probs / probs.sum()

    def pull_arm(self) -> int:
        self.probabilities = self._compute_probs()
        self.current_action = int(
            self.rng.choice(self.K, p=self.probabilities))
        self.action_history.append(self.current_action)
        return self.current_action

    def update(self, chosen_arm: int, reward: float) -> None:
        for i in range(self.K):
            if i != chosen_arm:
                self.G[i] += self.beta / self.probabilities[i]
            else:
                self.G[i] += (reward + self.beta) / self.probabilities[i]

        self.reward_history.append(reward)
        self.weight_history.append(self.G.copy())
        self.t += 1


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
        self.prices: np.ndarray = prices[0]  # assume same grid
        self.K: int = len(self.prices)
        self.T: int = T
        self.n_products: int = n_products
        self.delta: float = delta
        self.rng = np.random.default_rng()

        # Standardized variables
        self.remaining_budget: float = B
        self.initial_budget: float = B
        self.rho: float = B / (n_products * T)
        self.eta: float = eta if eta is not None else 1 / \
            math.sqrt(n_products * T)
        self.lmbd: float = 1.0
        self.t: int = 0

        self.exp3p_agents: List[Exp3PAgent] = [
            Exp3PAgent(self.K, T, delta) for _ in range(n_products)]
        self.current_action: Optional[List[int]] = None

        # History tracking
        self.action_history: List[Optional[List[int]]] = []
        self.lmbd_history: List[float] = []

    def pull_arm(self) -> Optional[List[int]]:
        if self.remaining_budget < 1:
            self.current_action = None
        else:
            self.current_action = [agent.pull_arm()
                                   for agent in self.exp3p_agents]

        self.action_history.append(
            self.current_action.copy() if self.current_action else None)
        return self.current_action

    def update(self, valuations: np.ndarray) -> None:
        if self.current_action is None:
            return

        total_revenue: float = 0.0
        total_sales: int = 0
        p_max: float = self.prices.max()

        for j, agent in enumerate(self.exp3p_agents):
            choice: int = self.current_action[j]
            price: float = self.prices[choice]
            val: float = float(valuations[j])
            cost: float = 1.0 if price <= val else 0.0

            reward: float = price * cost
            total_revenue += reward
            total_sales += int(cost)

            net = reward - self.lmbd * (cost - self.rho)

            L_up = p_max - self.lmbd * (0 - self.rho)
            L_low = 0.0 - self.lmbd * (1 - self.rho)
            norm_factor = L_up - L_low + 1e-12
            net_norm = (net - L_low) / norm_factor

            # update EXP3.P sub-agent based on the stored choice
            agent.probabilities = agent._compute_probs()
            agent.update(choice, net_norm)

        self.lmbd = np.clip(
            self.lmbd - self.eta * (self.rho * self.n_products - total_sales),
            a_min=0.0, a_max=1/self.rho
        )
        self.remaining_budget -= total_sales
        self.lmbd_history.append(self.lmbd)
        self.t += 1
