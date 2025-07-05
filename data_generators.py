import numpy as np
from scipy.stats import beta, truncnorm
from collections import deque
from typing import List, Optional, Tuple, Any


def pattern(t: int, T: int, freq: float) -> float:
    """Sinusoidal time-varying pattern: 1 - |sin(freq * t / T)|"""
    return 1 - abs(np.sin(freq * t / T))


def generate_beta_valuations(T: int, shock_prob: float, freq: float, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Generate Beta valuations with oscillating parameters"""
    if rng is None:
        rng = np.random.default_rng()
    valuations: np.ndarray = np.empty(T)
    for t in range(T):
        # Oscillating Alpha and Beta parameters
        alpha_t = 1 + 4 * (0.5 + 0.5 * np.sin(freq * np.pi * t / T))
        beta_t = 1 + 4 * (0.5 + 0.5 * np.cos(freq * np.pi * t / T))
        valuations[t] = rng.beta(alpha_t, beta_t)
    return valuations


def generate_valuations(T: int, shock_prob: float, freq: float, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Generate sequence of valuations with occasional adversarial shocks"""
    if rng is None:
        rng = np.random.default_rng()
    valuations: np.ndarray = np.empty(T)
    for t in range(T):
        a_t = pattern(t, T, freq)
        if rng.random() < shock_prob:
            valuations[t] = rng.choice([0.0, a_t])
        else:
            valuations[t] = rng.uniform(0, a_t)
    return valuations


def generate_piecewise_beta_valuations(T: int, shock_prob: float, num_regimes: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Generate piecewise-stationary Beta distribution valuations"""
    if rng is None:
        rng = np.random.default_rng()
    valuations: np.ndarray = np.empty(T)
    regime_length: int = T // num_regimes
    for regime in range(num_regimes):
        start: int = regime * regime_length
        end: int = T if regime == num_regimes - \
            1 else (regime + 1) * regime_length
        # Sample new parameters for the current regime
        alpha = rng.uniform(1, 50)
        beta_param = rng.uniform(1, 50)
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


def generate_flattened_valuation_data(T: int = 1000, K: int = 5, M: int = 2, transition_frac: float = 0.01,
                                      concentration: float = 50, rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate flattened valuation data for sliding window"""
    if rng is None:
        rng = np.random.default_rng()
    target_means: np.ndarray = rng.uniform(0.2, 0.8, size=(K, M))
    L: int = T // K
    delta: int = int(transition_frac * L)
    expected_means: np.ndarray = np.zeros((T, M))
    valuations: np.ndarray = np.zeros((T, M))
    for t in range(T):
        k: int = min(t // L, K - 1)
        start: int = k * L
        end: int = start + L
        if t < start + delta and k > 0:
            frac: float = (t - (start - delta)) / (2 * delta)
            mean: np.ndarray = (1 - frac) * \
                target_means[k - 1] + frac * target_means[k]
        elif t >= end - delta and k < K - 1:
            frac: float = (t - (end - delta)) / (2 * delta)
            mean: np.ndarray = (1 - frac) * \
                target_means[k] + frac * target_means[k + 1]
        else:
            mean = target_means[k]
        a = mean * concentration
        b = (1 - mean) * concentration
        expected_means[t] = mean
        valuations[t] = rng.beta(a, b)
    return expected_means, valuations


def generate_one_sided_valuation_data(T: int, segments: int, products: int, max_jump: float = 0.05, transition_frac: float = 0.2, concentration: float = 50, rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    rng = rng or np.random.default_rng()
    L: int = T // segments
    delta: int = int(transition_frac * L)
    seg_means: np.ndarray = np.empty((segments, products))
    seg_means[0] = rng.uniform(0.2, 0.8, size=products)
    for s in range(1, segments):
        step: np.ndarray = rng.normal(0, max_jump, size=products)
        seg_means[s] = np.clip(seg_means[s-1] + step, 0.0, 1.0)
    expected_means: np.ndarray = np.zeros((T, products))
    for t in range(T):
        s: int = min(t // L, segments - 1)
        start: int = s * L
        end: int = start + L
        if s < segments - 1 and t >= end - delta:
            frac: float = (t - (end - delta)) / delta
            expected_means[t] = (1 - frac) * seg_means[s] + \
                frac * seg_means[s+1]
        else:
            expected_means[t] = seg_means[s]
    a = expected_means * concentration
    b = (1 - expected_means) * concentration
    a = np.clip(a, 1e-6, None)
    b = np.clip(b, 1e-6, None)
    valuations: np.ndarray = rng.beta(a, b)
    return expected_means, valuations


def generate_smooth_valuation_data(T: int, K: int, M: int = 1, max_jump: float = 0.02, transition_frac: float = 0.0, concentration: float = 50, rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    rng = rng if rng is not None else np.random.default_rng()
    target_means: np.ndarray = np.empty((K, M))
    target_means[0] = rng.uniform(0.2, 0.8, size=M)
    for k in range(1, K):
        step: np.ndarray = rng.normal(0, max_jump, size=M)
        target_means[k] = np.clip(target_means[k-1] + step, 0.0, 1.0)
    L: int = T // K
    delta: int = int(transition_frac * L)
    expected_means: np.ndarray = np.zeros((T, M))
    valuations: np.ndarray = np.zeros((T, M))
    for t in range(T):
        k: int = min(t // L, K - 1)
        start: int = k * L
        end: int = start + L
        if delta > 0 and k > 0 and start <= t < start + delta:
            frac: float = (t - start) / delta
            mean: np.ndarray = (1 - frac) * \
                target_means[k-1] + frac * target_means[k]
        elif delta > 0 and k < K - 1 and end - delta <= t < end:
            frac: float = (t - (end - delta)) / delta
            mean: np.ndarray = (1 - frac) * \
                target_means[k] + frac * target_means[k+1]
        else:
            mean = target_means[k]
        a = mean * concentration
        b = (1 - mean) * concentration
        expected_means[t] = mean
        valuations[t] = rng.beta(a, b)
    return expected_means, valuations
