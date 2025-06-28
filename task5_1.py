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



