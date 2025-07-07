import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import beta
from typing import List, Optional, Tuple, Union
from data_generators import (
    generate_beta_valuations,
    generate_valuations,
    generate_piecewise_beta_valuations,
    generate_simple_tv_mv_gauss,
    generate_piecewise_tv_mv_gauss,
    generate_flattened_valuation_data
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
        if distribution == Distribution.BETA:
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
        if distribution == Distribution.BETA:
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
                 freq: Optional[int] = None, num_regimes: int = 10000,
                 valuation_type: str = 'piecewise_beta',
                 rng: Optional[np.random.Generator] = None) -> None:
        self.prices: np.ndarray = np.array(prices)
        self.T: int = T
        self.t: int = 0
        self.shock_prob: float = shock_prob
        self.freq: Optional[int] = freq
        if rng is None:
            rng = np.random.default_rng()
        self.rng: np.random.Generator = rng
        if valuation_type == 'beta':
            self.valuations: np.ndarray = generate_beta_valuations(
                T, shock_prob, freq or 1.0, rng=rng)
        elif valuation_type == 'sinusoidal':
            self.valuations = generate_valuations(
                T, shock_prob, freq or 1.0, rng=rng)
        elif valuation_type == 'piecewise_beta':
            self.valuations = generate_piecewise_beta_valuations(
                T, shock_prob, num_regimes, rng=rng)
        else:
            raise ValueError(
                f"Tipo di valutazione non supportato: {valuation_type}")

    def bandit_round(self, price_index: int) -> Tuple[float, float]:
        """Round with bandit feedback (only reward for the chosen arm)"""
        p: float = self.prices[price_index]
        sale: bool = self.valuations[self.t] >= p
        reward: float = p if sale else 0.0
        cost: float = 1.0 if sale else 0.0
        self.t += 1
        return reward, cost

    def round(self) -> float:
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
            self.vals: np.ndarray = self.rng.uniform(0, 1, size=(T, self.N))
        elif distribution == Distribution.BETA:
            self.vals: np.ndarray = self.rng.beta(0.5, 2, size=(T, self.N))
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

    def __init__(self, prices: Union[List[float], np.ndarray], T: int, m: int, valuation_params: dict,
                 valuation_type: str = 'simple_tv',
                 rng: Optional[np.random.Generator] = None) -> None:
        self.prices: np.ndarray = np.array(prices)
        self.T: int = T
        self.m: int = m
        self.t: int = 0
        self.rng: np.random.Generator = rng or np.random.default_rng(0)
        if valuation_type == 'simple_tv':
            mu0: float = valuation_params['mu0']
            A: float = valuation_params['A']
            f: float = valuation_params['f']
            phi: float = valuation_params['phi']
            sigma0: float = valuation_params['sigma0']
            A_sigma: float = valuation_params['A_sigma']
            phi_sigma: float = valuation_params['phi_sigma']
            rho0: float = valuation_params['rho0']
            self.V, _ = generate_simple_tv_mv_gauss(
                T, m, mu0, A, f, phi, sigma0, A_sigma, phi_sigma, rho0, rng=rng)
        elif valuation_type == 'piecewise_tv':
            num_regimes: int = valuation_params.get('num_regimes', 10000)
            self.V, _ = generate_piecewise_tv_mv_gauss(
                T, m, num_regimes, rng=rng)
        else:
            raise ValueError(
                f"Tipo di valutazione non supportato: {valuation_type}")

    def round(self) -> np.ndarray:
        if self.t >= self.T:
            raise RuntimeError("Orizzonte temporale superato!")
        v_t: np.ndarray = self.V[self.t]
        self.t += 1
        return v_t


class SmoothMultiProductPricingEnvironment:
    def __init__(self, price_grid: List[np.ndarray], T: int, valuations: np.ndarray,
                 rng: np.random.Generator) -> None:
        self.price_grid: List[np.ndarray] = price_grid
        self.N: int = len(price_grid)
        self.T: int = T
        self.t: int = 0
        self.rng: np.random.Generator = rng
        self.vals: np.ndarray = valuations

    def round(self, price_indices: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        vs: np.ndarray = self.vals[self.t]
        rewards: np.ndarray = np.zeros(self.N)
        costs: np.ndarray = np.zeros(self.N)
        for j, idx in enumerate(price_indices):
            p: float = self.price_grid[j][idx]
            if vs[j] >= p:
                rewards[j] = p
                costs[j] = 1.0
        self.t += 1
        return rewards, costs, vs
