import numpy as np
from scipy.stats import beta, truncnorm
from collections import deque


def pattern(t, T, freq):
    """Sinusoidal time-varying pattern: 1 - |sin(freq * t / T)|"""
    return 1 - abs(np.sin(freq * t / T))


def generate_beta_valuations(T, shock_prob, freq, rng=None):
    """Generate Beta valuations with oscillating parameters"""
    if rng is None:
        rng = np.random.default_rng()

    valuations = np.empty(T)
    for t in range(T):
        # Oscillating Alpha and Beta parameters
        alpha_t = 1 + 4 * (0.5 + 0.5 * np.sin(freq*np.pi * t / T))
        beta_t = 1 + 4 * (0.5 + 0.5 * np.cos(freq*np.pi * t / T))
        valuations[t] = rng.beta(alpha_t, beta_t)
    return valuations


def generate_valuations(T, shock_prob, freq, rng=None):
    """Generate sequence of valuations with occasional adversarial shocks"""
    if rng is None:
        rng = np.random.default_rng()

    valuations = np.empty(T)
    for t in range(T):
        a_t = pattern(t, T, freq)
        if rng.random() < shock_prob:
            valuations[t] = rng.choice([0.0, a_t])
        else:
            valuations[t] = rng.uniform(0, a_t)
    return valuations


def generate_piecewise_beta_valuations(T, shock_prob, num_regimes, rng=None):
    """Generate piecewise-stationary Beta distribution valuations"""
    if rng is None:
        rng = np.random.default_rng()

    valuations = np.empty(T)
    regime_length = T // num_regimes

    for regime in range(num_regimes):
        start = regime * regime_length
        end = T if regime == num_regimes - 1 else (regime + 1) * regime_length

        # Sample new parameters for the current regime
        alpha = rng.uniform(1, 50)
        beta_param = rng.uniform(1, 50)

        for t in range(start, end):
            if rng.random() < shock_prob:
                valuations[t] = rng.choice([0.0, 1.0])
            else:
                valuations[t] = rng.beta(alpha, beta_param)

    return valuations


def generate_stationary_correlated_gauss(T, m, mu, sigma, rho, rng=None):
    """Generate stationary correlated Gaussian valuations"""
    rng = rng or np.random.default_rng(0)

    # Build the constant correlation matrix
    R = np.eye(m) + (1 - np.eye(m)) * rho
    # Calculate the covariance matrix using sigma
    Sigma = np.diag([sigma]*m) @ R @ np.diag([sigma]*m)
    # Generate T samples from the multivariate normal distribution
    V = rng.multivariate_normal(mean=[mu]*m, cov=Sigma, size=T)
    V = np.clip(V, 0, 1)  # Assicura che le valutazioni siano in [0, 1]
    R_ts = np.repeat(R[np.newaxis, :, :], T, axis=0)
    return V, R_ts


def generate_simple_tv_mv_gauss(T, m, mu0, A, f, phi, sigma0, A_sigma, phi_sigma, rho0, rng=None):
    """Generate time-varying multivariate Gaussian valuations with sinusoidal modulation"""
    rng = rng or np.random.default_rng(0)
    V = np.empty((T, m))
    R = np.eye(m) + (1 - np.eye(m)) * rho0  # constant correlation

    for t in range(T):
        mu_t = mu0 + A * np.sin(2 * np.pi * f * t / T + phi)
        sigma_t = sigma0 + A_sigma * np.sin(2 * np.pi * f * t / T + phi_sigma)
        Sigma = np.diag([sigma_t]*m) @ R @ np.diag([sigma_t]*m)
        sample = rng.multivariate_normal(mu_t, Sigma)
        V[t] = np.clip(sample, 0, 1)

    R_ts = np.repeat(R[np.newaxis, :, :], T, axis=0)
    return V, R_ts


def generate_piecewise_tv_mv_gauss(T, m, num_regimes,
                                   mu_low=0.3, mu_high=0.7,
                                   sigma_low=0.05, sigma_high=0.15,
                                   rho_low=-0.3, rho_high=0.3,
                                   rng=None):
    """Generate piecewise-stationary multivariate Gaussian valuations"""
    rng = rng or np.random.default_rng(0)
    V = np.empty((T, m))
    R_ts = np.empty((T, m, m))
    regime_length = T // num_regimes

    for regime in range(num_regimes):
        start = regime * regime_length
        end = T if regime == num_regimes - 1 else (regime + 1) * regime_length

        # Sample specific parameters for the current regime
        mu_reg = rng.uniform(mu_low, mu_high, size=m)
        sigma_reg = rng.uniform(sigma_low, sigma_high, size=m)
        rho = rng.uniform(rho_low, rho_high)

        # Build correlation and covariance matrices
        R = np.eye(m) + (np.ones((m, m)) - np.eye(m)) * rho
        Sigma = np.diag(sigma_reg) @ R @ np.diag(sigma_reg)

        # Generate samples for this regime
        for t in range(start, end):
            sample = rng.multivariate_normal(mu_reg, Sigma)
            V[t] = np.clip(sample, 0, 1)
            R_ts[t] = R

    return V, R_ts


def generate_flattened_valuation_data(T=1000, K=5, M=2, transition_frac=0.01,
                                      concentration=50, rng=None):
    """Generate flattened valuation data for sliding window"""
    if rng is None:
        rng = np.random.default_rng()

    # Generate random target means for interval and product
    target_means = rng.uniform(0.2, 0.8, size=(K, M))

    # Precalculate interval length and transition window size
    L = T // K
    delta = int(transition_frac * L)

    expected_means = np.zeros((T, M))
    valuations = np.zeros((T, M))

    for t in range(T):
        k = min(t // L, K - 1)
        start = k * L
        end = start + L

        # Calculate current mean through piecewise constant + ramp
        if t < start + delta and k > 0:
            frac = (t - (start - delta)) / (2 * delta)
            mean = (1 - frac) * target_means[k - 1] + frac * target_means[k]
        elif t >= end - delta and k < K - 1:
            frac = (t - (end - delta)) / (2 * delta)
            mean = (1 - frac) * target_means[k] + frac * target_means[k + 1]
        else:
            mean = target_means[k]

        a = mean * concentration
        b = (1 - mean) * concentration

        expected_means[t] = mean
        valuations[t] = rng.beta(a, b)

    return expected_means, valuations


def generate_one_sided_valuation_data(T, segments, products, max_jump=0.05, transition_frac=0.2, concentration=50, rng=None):
    rng = rng or np.random.default_rng()
    L = T // segments
    delta = int(transition_frac * L)

    # 1) Sample segment means
    seg_means = np.empty((segments, products))
    seg_means[0] = rng.uniform(0.2, 0.8, size=products)
    for s in range(1, segments):
        step = rng.normal(0, max_jump, size=products)
        seg_means[s] = np.clip(seg_means[s-1] + step, 0.0, 1.0)

    # 2) Build full expected_means with only "ramp out" at end of each segment
    expected_means = np.zeros((T, products))
    for t in range(T):
        s = min(t // L, segments - 1)
        start = s * L
        end = start + L
        if s < segments - 1 and t >= end - delta:
            # ramp from seg_means[s] to seg_means[s+1]
            frac = (t - (end - delta)) / delta
            expected_means[t] = (1 - frac) * seg_means[s] + \
                frac * seg_means[s+1]
        else:
            expected_means[t] = seg_means[s]

    # 3) Sample valuations via Beta(conc·mean, conc·(1-mean))
    a = expected_means * concentration
    b = (1 - expected_means) * concentration
    a = np.clip(a, 1e-6, None)
    b = np.clip(b, 1e-6, None)
    valuations = rng.beta(a, b)

    return expected_means, valuations


def generate_smooth_valuation_data(T, K, M=1, max_jump=0.02, transition_frac=0.0, concentration=50, rng=None):
    rng = rng if rng is not None else np.random.default_rng()

    # 1) correlated target means by random-walk
    target_means = np.empty((K, M))
    target_means[0] = rng.uniform(0.2, 0.8, size=M)
    for k in range(1, K):
        step = rng.normal(0, max_jump, size=M)
        target_means[k] = np.clip(target_means[k-1] + step, 0.0, 1.0)

    # 2) build the piecewise + ramp
    L = T // K
    delta = int(transition_frac * L)
    expected_means = np.zeros((T, M))
    valuations = np.zeros((T, M))
    for t in range(T):
        k = min(t // L, K-1)
        start = k * L
        end = start + L
        if delta > 0 and k > 0 and start <= t < start + delta:
            frac = (t - start) / delta
            mean = (1 - frac) * target_means[k-1] + frac * target_means[k]
        elif delta > 0 and k < K-1 and end - delta <= t < end:
            frac = (t - (end - delta)) / delta
            mean = (1 - frac) * target_means[k] + frac * target_means[k+1]
        else:
            mean = target_means[k]

        a = mean * concentration
        b = (1 - mean) * concentration
        expected_means[t] = mean
        valuations[t] = rng.beta(a, b)

    return expected_means, valuations
