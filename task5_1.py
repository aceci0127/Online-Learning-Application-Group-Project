import numpy as np
import matplotlib.pyplot as plt

def generate_flattened_valuation_data(T=1000, K=5, M=2, transition_frac=0.1, concentration=50, rng = None):
    
    
    # Generate random target means per interval and product
    target_means = rng.uniform(0.2, 0.8, size=(K, M))
    
    # Precompute interval length and transition window size
    L = T // K
    delta = int(transition_frac * L)
    
    expected_means = np.zeros((T, M))
    valuations = np.zeros((T, M))
    
    for t in range(T):
        k = min(t // L, K - 1)
        start = k * L
        end = start + L
        
        # Compute current mean by piecewise constant + ramp
        if t < start + delta and k > 0:
            frac = (t - (start - delta)) / (2 * delta)
            mean = (1 - frac) * target_means[k - 1] + frac * target_means[k]
        elif t >= end - delta and k < K - 1:
            frac = (t - (end - delta)) / (2 * delta)
            mean = (1 - frac) * target_means[k] + frac * target_means[k + 1]
        else:
            mean = target_means[k]
        
        # Beta parameters with controlled concentration
        a = mean * concentration
        b = (1 - mean) * concentration
        
        expected_means[t] = mean
        valuations[t] = rng.beta(a, b)
    
    return expected_means, valuations


class MultiProductPricingEnvironment:
    def __init__(self, price_grid, T,num_windows, rng=None):
        self.price_grid = price_grid
        self.N = len(price_grid)
        self.T = T
        self.t = 0
        self.num_windows = num_windows
        self.rng = rng if rng is not None else np.random.default_rng()
        self.expectations, self.vals = generate_flattened_valuation_data(T, self.num_windows, self.N, transition_frac=0.1, concentration=50, rng=self.rng)

    def round(self, price_indices):
        vs = self.vals[self.t]
        rewards = np.zeros(self.N)
        costs = np.zeros(self.N)
        for j, idx in enumerate(price_indices):
            p = self.price_grid[j][idx]
            if vs[j] >= p:
                rewards[j] = p
                costs[j] = 1.0
        self.t += 1
        return rewards, costs
from collections import deque

class ConstrainedCombinatorialUCBAgent:
    def __init__(self, price_grid, B, T, alpha=1, window_size=1000):
        ...  # ...existing initialization code...
        self.price_grid = price_grid
        self.N = len(price_grid)
        self.Ks = [len(pg) for pg in price_grid]
        self.B_rem = B
        self.B = B
        self.T = T
        self.t = 0
        self.alpha = alpha
        self.rng = np.random.default_rng()

        self.window_size = window_size
        # For each (product j, price k), we store a deque of tuples: (time, reward, cost)
        self.samples = [
            [deque() for _ in range(K)]
            for K in self.Ks
        ]
        self.current_choice = None

    def pull_arm(self):
        if self.B_rem < 1 or self.t >= self.T:
            return None

        rho = self.B_rem / (self.T - self.t)
        f_ucb = []
        c_lcb = []
        # For each product
        for j in range(self.N):
            K = self.Ks[j]
            f_j_vals = []
            c_j_vals = []
            n_j_vals = []
            for k in range(K):
                # Remove samples older than time threshold self.t - window_size
                while self.samples[j][k] and self.samples[j][k][0][0] < self.t - self.window_size:
                    self.samples[j][k].popleft()
                samples = self.samples[j][k]
                n = len(samples)
                avg_reward = np.mean([item[1] for item in samples]) if n > 0 else 0.0
                avg_cost   = np.mean([item[2] for item in samples]) if n > 0 else 0.0
                n_j_vals.append(n)
                f_j_vals.append(avg_reward)
                c_j_vals.append(avg_cost)
            n_j = np.array(n_j_vals)
            f_j = np.array(f_j_vals)
            c_j = np.array(c_j_vals)

            bonus = np.sqrt(self.alpha * np.log(max(self.t, 1)) / np.maximum(1, n_j))
            bonus[n_j == 0] = self.T   # force exploration for unseen arms
            bonus[-1] = 0.0           # last dummy price has no bonus

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
            # Append current round with reward and cost; no need for explicit pop as pull_arm cleans outdated samples.
            self.samples[j][idx].append((self.t, rewards[j], costs[j]))
        self.B_rem -= costs.sum()
        self.t += 1



