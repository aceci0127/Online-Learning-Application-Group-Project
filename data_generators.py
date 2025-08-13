from scipy.stats import truncnorm
import numpy as np
from scipy.stats import beta, truncnorm
from collections import deque
from typing import List, Optional, Tuple, Any


def pattern(t: int, T: int, freq: float) -> float:
    """Sinusoidal time-varying pattern: 1 - |sin(freq * t / T)|"""
    return 1 - abs(np.sin(freq * t / T))


def generate_beta_valuations(T: int, freq: int, rng: np.random.Generator) -> np.ndarray:
    """Generate Beta valuations with oscillating parameters"""
    valuations: np.ndarray = np.empty(T)
    for t in range(T):
        # Oscillating Alpha and Beta parameters
        alpha_t = 1 + 4 * (0.5 + 0.5 * np.sin(freq * np.pi * t / T))
        beta_t = 1 + 4 * (0.5 + 0.5 * np.cos(freq * np.pi * t / T))
        valuations[t] = rng.beta(alpha_t, beta_t)
    return valuations


def generate_sinusoidal_valuations(T: int, freq: int, rng: np.random.Generator, shock_prob: float) -> np.ndarray:
    """Generate sequence of valuations with occasional adversarial shocks"""
    valuations: np.ndarray = np.empty(T)
    for t in range(T):
        a_t = pattern(t, T, freq)
        if rng.random() < shock_prob:
            valuations[t] = rng.choice([0.0, a_t])
        else:
            valuations[t] = rng.uniform(0, a_t)
    return valuations


def generate_piecewise_beta_valuations(T: int, shock_prob: float, num_regimes: int, rng) -> np.ndarray:
    """Generate piecewise-stationary Beta distribution valuations"""
    valuations: np.ndarray = np.empty(T)
    regime_length: int = T // num_regimes
    for regime in range(num_regimes):
        start: int = regime * regime_length
        end: int = T if regime == num_regimes - \
            1 else (regime + 1) * regime_length
        # Sample new parameters for the current regime
        alpha = rng.uniform(1, 100)
        beta_param = rng.uniform(1, 100)
        for t in range(start, end):
            if rng.random() < shock_prob:
                valuations[t] = rng.choice([0.0, 1.0])
            else:
                valuations[t] = rng.beta(alpha, beta_param)
    return valuations


def generate_stationary_correlated_gauss(T: int, m: int, mu: float, sigma: float, rho: float, rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate stationary correlated Gaussian valuations"""
    rng = rng or np.random.default_rng(0)
    # Build the constant correlation matrix
    R: np.ndarray = np.eye(m) + (1 - np.eye(m)) * rho
    # Calculate the covariance matrix using sigma
    Sigma: np.ndarray = np.diag([sigma] * m) @ R @ np.diag([sigma] * m)
    # Generate T samples from the multivariate normal distribution
    V: np.ndarray = rng.multivariate_normal(mean=[mu] * m, cov=Sigma, size=T)
    V = np.clip(V, 0, 1)  # Ensure valuations are in [0, 1]
    R_ts: np.ndarray = np.repeat(R[np.newaxis, :, :], T, axis=0)
    return V, R_ts


def generate_simple_tv_mv_gauss(T: int, m: int, mu0: float, A: float, f: float, phi: float, sigma0: float, A_sigma: float, phi_sigma: float, rho0: float, rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate time-varying multivariate Gaussian valuations with sinusoidal modulation"""
    rng = rng or np.random.default_rng(0)
    V: np.ndarray = np.empty((T, m))
    R: np.ndarray = np.eye(m) + (1 - np.eye(m)) * rho0  # constant correlation
    for t in range(T):
        mu_t = mu0 + A * np.sin(2 * np.pi * f * t / T + phi)
        sigma_t = sigma0 + A_sigma * np.sin(2 * np.pi * f * t / T + phi_sigma)
        Sigma: np.ndarray = np.diag([sigma_t] * m) @ R @ np.diag([sigma_t] * m)
        sample: np.ndarray = rng.multivariate_normal([mu_t] * m, Sigma)
        V[t] = np.clip(sample, 0, 1)
    R_ts: np.ndarray = np.repeat(R[np.newaxis, :, :], T, axis=0)
    return V, R_ts


def generate_piecewise_tv_mv_gauss(T: int, m: int, num_regimes: int,
                                   mu_low: float = 0.3, mu_high: float = 0.7,
                                   sigma_low: float = 0.05, sigma_high: float = 0.15,
                                   rho_low: float = -0.3, rho_high: float = 0.3,
                                   rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate piecewise-stationary multivariate Gaussian valuations"""
    rng = rng or np.random.default_rng(0)
    V: np.ndarray = np.empty((T, m))
    R_ts: np.ndarray = np.empty((T, m, m))
    regime_length: int = T // num_regimes
    for regime in range(num_regimes):
        start: int = regime * regime_length
        end: int = T if regime == num_regimes - \
            1 else (regime + 1) * regime_length
        # Sample specific parameters for the current regime
        mu_reg: np.ndarray = rng.uniform(mu_low, mu_high, size=m)
        sigma_reg: np.ndarray = rng.uniform(sigma_low, sigma_high, size=m)
        rho: float = rng.uniform(rho_low, rho_high)
        # Build correlation and covariance matrices
        R: np.ndarray = np.eye(m) + (np.ones((m, m)) - np.eye(m)) * rho
        Sigma: np.ndarray = np.diag(sigma_reg) @ R @ np.diag(sigma_reg)
        for t in range(start, end):
            sample: np.ndarray = rng.multivariate_normal(mu_reg, Sigma)
            V[t] = np.clip(sample, 0, 1)
            R_ts[t] = R
    return V, R_ts


def generate_smooth_valuation_data(T: int, K: int, M: int = 1, concentration: float = 50,
                                   rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate smooth valuation data where the distribution in each window is constant,
    but windows are connected by a random walk to produce slowly changing distributions.

    T: total number of time steps.
    K: number of windows.
    M: number of products.
    concentration: concentration parameter for the Beta distribution.

    Each window uses its target mean, which is created as a slight step from the previous one.
    A higher concentration leads to a distribution that is more peaked around the mean (lower variance), 
    while a lower concentration produces a flatter distribution (higher variance).
    """
    rng = rng if rng is not None else np.random.default_rng()
    target_means: np.ndarray = np.empty((K, M))
    target_means[0] = rng.uniform(0.2, 0.8, size=M)
    for k in range(1, K):
        step: np.ndarray = rng.normal(
            0, 0.02, size=M)  # slight change per window
        target_means[k] = np.clip(target_means[k-1] + step, 0.0, 1.0)
    L: int = T // K
    expected_means: np.ndarray = np.zeros((T, M))
    valuations: np.ndarray = np.zeros((T, M))
    for k in range(K):
        start: int = k * L
        end: int = T if k == K - 1 else (k + 1) * L
        mean = target_means[k]
        expected_means[start:end] = mean
        a = mean * concentration
        b = (1 - mean) * concentration
        valuations[start:end] = rng.beta(a, b, size=(end - start, M))
    return expected_means, valuations


def generate_independent_valuation_data(T: int, M: int = 1, concentration: float = 50,
                                        rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate independent valuation data for each time step.

    T: total number of time steps.
    M: number of products.
    concentration: concentration parameter for the Beta distribution.

    Each time step draws its target mean independently from a uniform distribution.
    """
    rng = rng if rng is not None else np.random.default_rng()
    # Independently sample target means for each time step
    expected_means = rng.uniform(0.2, 0.8, size=(T, M))
    valuations = np.empty((T, M))
    for t in range(T):
        mean = expected_means[t]
        a = mean * concentration
        b = (1 - mean) * concentration
        valuations[t] = rng.beta(a, b, size=M)
    return expected_means, valuations
