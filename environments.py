import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import beta
from typing import List, Optional, Tuple, Union
from data_generators import (
    generate_beta_valuations,
    generate_sinusoidal_valuations,
    generate_piecewise_beta_valuations,
    generate_simple_tv_mv_gauss,
    generate_piecewise_tv_mv_gauss,
    generate_smooth_valuation_data
)
from runner import Distribution


class Environment(ABC):
    """Abstract base class for all environments"""

    @abstractmethod
    def round(self, action: object) -> object:
        """Execute a round of the environment given an action"""
        pass


class PricingEnvironment(Environment):
    """Environment for simple pricing with uniform valuations"""

    def __init__(self, prices: Union[List[float], np.ndarray], T: int, rng: Optional[np.random.Generator] = None, distribution: Distribution = Distribution.UNIFORM) -> None:
        self.prices: np.ndarray = np.array(prices)
        self.m: int = len(prices)
        self.T: int = T
        self.t: int = 0
        if rng is None:
            rng = np.random.default_rng()
        self._rng: np.random.Generator = rng
        if distribution == Distribution.BETA_SINUSOIDAL:
            self.valuations: np.ndarray = self._rng.beta(0.5, 2, size=T)
        elif distribution == Distribution.UNIFORM:
            self.valuations = self._rng.uniform(0, 1, size=T)
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

    def round(self, arm_index: int) -> float:
        chosen_price: float = self.prices[arm_index]
        v: float = float(self.valuations[self.t])
        revenue: float = chosen_price if v >= chosen_price else 0.0
        self.t += 1
        return revenue


class BudgetedPricingEnvironment(Environment):
    """Environment for pricing with limited budget"""

    def __init__(self, prices: Union[List[float], np.ndarray], T: int, distribution: Distribution = Distribution.UNIFORM,
                 rng: Optional[np.random.Generator] = None) -> None:
        self.prices: np.ndarray = np.array(prices)
        self.T: int = T
        self.t: int = 0
        if rng is None:
            rng = np.random.default_rng()
        self._rng: np.random.Generator = rng
        if distribution == Distribution.BETA_SINUSOIDAL:
            self.vals: np.ndarray = self._rng.beta(0.5, 2, size=T)
        elif distribution == Distribution.UNIFORM:
            self.vals = self._rng.uniform(0, 1, size=T)
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

    def round(self, price_index: int) -> Tuple[float, float]:
        p: float = self.prices[price_index]
        sale: bool = self.vals[self.t] >= p
        reward: float = p if sale else 0.0
        cost: float = 1.0 if sale else 0.0
        self.t += 1
        return reward, cost


class NonStationaryBudgetedPricingEnvironment(Environment):
    """Environment for non-stationary pricing with full feedback"""

    def __init__(self, prices: Union[List[float], np.ndarray], T: int, shock_prob: float,
                 freq: int, num_regimes: int = 10000,
                 distribution: Distribution = Distribution.BETA_SINUSOIDAL, rng: Optional[np.random.Generator] = None) -> None:
        self.prices: np.ndarray = np.array(prices)
        self.T: int = T
        self.t: int = 0
        self.freq: int = freq
        self.rng: np.random.Generator = rng or np.random.default_rng()
        if distribution == Distribution.BETA_SINUSOIDAL:
            self.valuations: np.ndarray = generate_beta_valuations(
                T, freq, rng=self.rng)
        elif distribution == Distribution.UNIFORM_SINUSOIDAL:
            self.valuations = generate_sinusoidal_valuations(
                T, freq, rng=self.rng, shock_prob=shock_prob)
        elif distribution == Distribution.PIECEWISE_BETA:
            self.valuations = generate_piecewise_beta_valuations(
                T, shock_prob, num_regimes, rng=self.rng)
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

    def bandit_round(self, price_index: int) -> Tuple[float, float]:
        """Round with bandit feedback (only reward for the chosen arm)"""
        p: float = self.prices[price_index]
        sale: bool = self.valuations[self.t] >= p
        reward: float = p if sale else 0.0
        cost: float = 1.0 if sale else 0.0
        self.t += 1
        return reward, cost

    def round(self, round: int) -> float:
        """Round with full feedback (returns the valuation)"""
        valuation: float = float(self.valuations[self.t])
        self.t += 1
        return valuation

    def compute_sell_probabilities(self) -> np.ndarray:
        """Compute sell probabilities for each price"""
        sell_probabilities: np.ndarray = np.array([
            float(np.sum(p <= self.valuations)) / self.T for p in self.prices
        ])
        return sell_probabilities


class MultiProductPricingEnvironment(Environment):
    """Environment for multi-product pricing"""

    def __init__(self, price_grid: List[np.ndarray], T: int,
                 rng: Optional[np.random.Generator] = None, distribution: Distribution = Distribution.UNIFORM) -> None:
        self.price_grid: List[np.ndarray] = price_grid
        self.N: int = len(price_grid)
        self.T: int = T
        self.t: int = 0
        self.rng: np.random.Generator = rng if rng is not None else np.random.default_rng()

        if distribution == Distribution.UNIFORM:
            self.vals = self.rng.uniform(0, 1, size=(T, self.N))
        elif distribution == Distribution.BETA_SINUSOIDAL:
            self.vals = self.rng.beta(0.5, 2, size=(T, self.N))
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

    def round(self, price_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        vs: np.ndarray = self.vals[self.t]
        rewards: np.ndarray = np.zeros(self.N)
        costs: np.ndarray = np.zeros(self.N)
        for j, idx in enumerate(price_indices):
            p: float = self.price_grid[j][idx]
            if vs[j] >= p:
                rewards[j] = p
                costs[j] = 1.0
        self.t += 1
        return rewards, costs


class MultiProductBudgetedPricingEnvironment(Environment):
    """Environment for multi-product pricing with budget and full feedback"""

    def __init__(self, T: int, n_products: int, valuation_params: dict,
                 rng: np.random.Generator, distribution: Distribution = Distribution.SIMPLE_TV) -> None:
        self.T: int = T
        self.n_products: int = n_products
        self.t: int = 0
        self.rng: np.random.Generator = rng
        if distribution == Distribution.SIMPLE_TV:
            mu0: float = valuation_params['mu0']
            A: float = valuation_params['A']
            f: float = valuation_params['f']
            phi: float = valuation_params['phi']
            sigma0: float = valuation_params['sigma0']
            A_sigma: float = valuation_params['A_sigma']
            phi_sigma: float = valuation_params['phi_sigma']
            rho0: float = valuation_params['rho0']
            self.valuations, _ = generate_simple_tv_mv_gauss(
                T,
                n_products,
                mu0=mu0,
                A=A,
                f=f,
                phi=phi,
                sigma0=sigma0,
                A_sigma=A_sigma,
                phi_sigma=phi_sigma,
                rho0=rho0,
                rng=self.rng
            )
        elif distribution == Distribution.PIECEWISE_TV:
            num_regimes: int = valuation_params.get('num_regimes', 10000)
            self.valuations, _ = generate_piecewise_tv_mv_gauss(
                T, n_products, num_regimes, rng=self.rng)
        else:
            raise ValueError(
                f"Tipo di valutazione non supportato: {distribution}")

    def round(self, round: int) -> np.ndarray:
        if self.t >= self.T:
            raise RuntimeError("Orizzonte temporale superato!")
        v_t: np.ndarray = self.valuations[self.t]
        self.t += 1
        return v_t


class SmoothMultiProductPricingEnvironment:
    def __init__(self, price_grid: List[np.ndarray], T: int, n_products: int, num_windows: int,
                 rng: np.random.Generator, distribution: Distribution = Distribution.SMOOTH) -> None:
        self.price_grid: List[np.ndarray] = price_grid
        self.N: int = len(price_grid)
        self.T: int = T
        self.t: int = 0
        self.rng: np.random.Generator = rng
        self.n_products: int = n_products
        self.num_windows: int = num_windows
        self.distribution: Distribution = distribution

        if distribution == Distribution.SMOOTH:
            self.expected_means, self.valuations = generate_smooth_valuation_data(
                T, K=self.num_windows, M=self.n_products,
                concentration=10, rng=self.rng
            )
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

    def round(self, price_indices: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        vs: np.ndarray = self.valuations[self.t]
        rewards: np.ndarray = np.zeros(self.N)
        costs: np.ndarray = np.zeros(self.N)
        for j, idx in enumerate(price_indices):
            p: float = self.price_grid[j][idx]
            if vs[j] >= p:
                rewards[j] = p
                costs[j] = 1.0
        self.t += 1
        return rewards, costs, vs
