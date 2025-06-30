import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import beta
from data_generators import (
    generate_beta_valuations,
    generate_valuations,
    generate_piecewise_beta_valuations,
    generate_simple_tv_mv_gauss,
    generate_piecewise_tv_mv_gauss,
    generate_flattened_valuation_data
)


class Environment(ABC):
    """Abstract base class for all environments"""

    @abstractmethod
    def round(self, action):
        """Execute a round of the environment given an action"""
        pass


class PricingEnvironment(Environment):
    """Environment for simple pricing with uniform valuations"""

    def __init__(self, prices, T, rng=None):
        self.prices = np.array(prices)
        self.m = len(prices)
        self.T = T
        self.t = 0

        if rng is None:
            rng = np.random.default_rng()
        self._rng = rng
        self.valuations = self._rng.uniform(0, 1, size=T)

    def round(self, arm_index):
        chosen_price = self.prices[arm_index]
        v = self.valuations[self.t]
        revenue = chosen_price if v >= chosen_price else 0.0
        self.t += 1
        return revenue


class BudgetedPricingEnvironment(Environment):
    """Environment for pricing with limited budget"""

    def __init__(self, prices, T, distribution='uniform', rng=None):
        self.prices = np.array(prices)
        self.T = T
        self.t = 0

        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

        if distribution == 'beta':
            self.alpha = 60
            self.beta = 40
            self.vals = beta(self.alpha, self.beta).rvs(
                size=T, random_state=rng)
        elif distribution == 'uniform':
            self.vals = rng.uniform(0, 1, size=T)

    def round(self, price_index):
        p = self.prices[price_index]
        sale = self.vals[self.t] >= p
        reward = p if sale else 0.0
        cost = 1.0 if sale else 0.0
        self.t += 1
        return reward, cost


class NonStationaryBudgetedPricingEnvironment(Environment):
    """Environment for non-stationary pricing with full feedback"""

    def __init__(self, prices, T, shock_prob, freq=None, num_regimes= 10000,
                 valuation_type='piecewise_beta', rng=None):
        self.prices = np.array(prices)
        self.T = T
        self.t = 0
        self.shock_prob = shock_prob
        self.freq = freq

        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

        # Generate valuations based on the specified type
        if valuation_type == 'beta':
            self.valuations = generate_beta_valuations(
                T, shock_prob, freq, rng=rng)
        elif valuation_type == 'sinusoidal':
            self.valuations = generate_valuations(T, shock_prob, freq, rng=rng)
        elif valuation_type == 'piecewise_beta':
            self.valuations = generate_piecewise_beta_valuations(
                T, shock_prob, num_regimes, rng=rng)
        else:
            raise ValueError(
                f"Tipo di valutazione non supportato: {valuation_type}")
    
    def bandit_round(self, price_index):
        """Round con feedback bandit (solo reward della scelta)"""
        p = self.prices[price_index]
        sale = self.valuations[self.t] >= p
        reward = p if sale else 0.0
        cost = 1.0 if sale else 0.0
        self.t += 1
        return reward, cost

    def round(self):
        """Round con feedback completo (ritorna la valutazione)"""
        valuation = self.valuations[self.t]
        self.t += 1
        return valuation

    def compute_sell_probabilities(self):
        """Calcola le probabilità di vendita per ogni prezzo"""
        sell_probabilities = np.array([
            sum(p <= self.valuations) / self.T for p in self.prices
        ])
        return sell_probabilities


class MultiProductPricingEnvironment(Environment):
    """Environment for multi-product pricing"""

    def __init__(self, price_grid, T, rng=None):
        self.price_grid = price_grid
        self.N = len(price_grid)
        self.T = T
        self.t = 0
        self.rng = rng if rng is not None else np.random.default_rng()
        self.vals = self.rng.uniform(0, 1, size=(T, self.N))

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


class MultiProductBudgetedPricingEnvironment(Environment):
    """Environment for multi-product pricing with budget and full feedback"""

    def __init__(self, prices, T, m, valuation_params, valuation_type='simple_tv', rng=None):
        self.prices = np.array(prices)
        self.T = T
        self.m = m
        self.t = 0
        self.rng = rng or np.random.default_rng(0)

        # Genera le valutazioni in base al tipo
        if valuation_type == 'simple_tv':
            mu0, A, f, phi = valuation_params['mu0'], valuation_params[
                'A'], valuation_params['f'], valuation_params['phi']
            sigma0, A_sigma, phi_sigma, rho0 = valuation_params['sigma0'], valuation_params[
                'A_sigma'], valuation_params['phi_sigma'], valuation_params['rho0']
            self.V, _ = generate_simple_tv_mv_gauss(
                T, m, mu0, A, f, phi, sigma0, A_sigma, phi_sigma, rho0, rng=rng)
        elif valuation_type == 'piecewise_tv':
            num_regimes = valuation_params.get('num_regimes', 10000)
            self.V, _ = generate_piecewise_tv_mv_gauss(
                T, m, num_regimes, rng=rng)
        else:
            raise ValueError(
                f"Tipo di valutazione non supportato: {valuation_type}")

    def round(self):
        if self.t >= self.T:
            raise RuntimeError("Orizzonte temporale superato!")
        v_t = self.V[self.t]
        self.t += 1
        return v_t


class SlidingWindowMultiProductEnvironment(Environment):
    """Multi-product environment with sliding window for non-stationarity"""

    def __init__(self, price_grid, T, num_windows, rng=None):
        self.price_grid = price_grid
        self.N = len(price_grid)
        self.T = T
        self.t = 0
        self.num_windows = num_windows
        self.rng = rng if rng is not None else np.random.default_rng()

        self.expectations, self.vals = generate_flattened_valuation_data(
            T, self.num_windows, self.N, transition_frac=0.1,
            concentration=50, rng=self.rng
        )

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
