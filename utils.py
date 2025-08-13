import numpy as np
from scipy.optimize import linprog
from numpy.typing import NDArray


def compute_clairvoyant_single_product(prices, sell_probabilities, budget, horizon):
    """
    Calcola la soluzione chiarveggente per singolo prodotto usando LP.

    Args:
        prices: array dei prezzi disponibili
        sell_probabilities: probabilità di vendita per ogni prezzo
        budget: budget totale
        horizon: orizzonte temporale

    Returns:
        expected_utility: utilità attesa per round
        gamma: distribuzione ottimale sui prezzi
        expected_cost: costo atteso per round
    """
    rho = budget / horizon

    c = -(prices * sell_probabilities)
    A_ub = [sell_probabilities]
    b_ub = [rho]
    A_eq = [np.ones(len(prices))]
    b_eq = [1]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1))

    if not res.success:
        raise ValueError("LP fallito: " + res.message)

    gamma = res.x
    expected_utility = -res.fun
    expected_cost = np.sum(sell_probabilities * gamma)
    print(f"Expected utility: {expected_utility:.4f}")
    print(f"Optimal distribution (gamma): {gamma}")
    print(f"Expected cost: {expected_cost:.4f}")

    return expected_utility, gamma, expected_cost


def compute_sell_probabilities_multi(V, prices):
    """
    Calcola le probabilità di vendita per più prodotti.

    Args:
        V: array di valutazioni (T, m)
        prices: array dei prezzi

    Returns:
        s: matrice delle probabilità di vendita (m, K)
    """
    T_env, m_env = V.shape
    K = len(prices)
    s = np.zeros((m_env, K))

    for j in range(m_env):
        for i, p in enumerate(prices):
            s[j, i] = np.sum(V[:, j] >= p) / T_env

    return s


def solve_clairvoyant_lp(price_grid, B, T, f_true, c_true):
    """
    Solves the clairvoyant per-round LP for given price grids, budget, and horizon.

    Args:
        price_grid: list of 1D numpy arrays, each array is the prices for one product.
        B: total inventory budget.
        T: total number of rounds.

    Returns:
        optimal_per_round: the optimal expected revenue per round.
    """
    # pacing constraint
    rho = B / T

    f_flat = np.concatenate(f_true)
    c_flat = np.concatenate(c_true)
    num_vars = len(f_flat)

    # Objective: maximize sum(f_flat * x) -> minimize -f_flat @ x
    c_obj = -f_flat

    # Equality constraints: for each product j, its marginal sums to 1
    N = len(price_grid)
    A_eq = np.zeros((N, num_vars))
    b_eq = np.ones(N)
    offset = 0
    for j in range(N):
        K = len(price_grid[j])
        A_eq[j, offset:offset+K] = 1
        offset += K

    # Inequality: total expected consumption <= rho
    A_ub = c_flat.reshape(1, -1)
    b_ub = np.array([rho])

    # Variable bounds: x >= 0
    bounds = [(0, None)] * num_vars

    # Solve using SciPy's linprog
    res = linprog(c=c_obj, A_ub=A_ub, b_ub=b_ub,
                  A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                  method='highs')
    expected_cost = np.sum(c_flat * res.x)
    print(f"Expected cost: {expected_cost:.4f}")
    print(f"Optimal expected revenue per round: {-res.fun:.4f}")
    print(f"Optimal distribution (simplex): {res.x}")
    if res.success:
        optimal_per_round = -res.fun
        simplex = res.x
        return optimal_per_round, simplex
    else:
        raise ValueError("LP did not solve successfully: " + res.message)


def compute_extended_clairvoyant(V, prices, total_inventory):
    """
    Extended clairvoyant solution for multi-product pricing.

    Args:
        V: valuations (T, m)
        prices: array of prices
        total_inventory: total inventory

    Returns:
        expected_utility: expected utility per round
        gamma: optimal distribution (m, K)
        expected_cost: expected cost per round
    """
    T_env, m_env = V.shape
    K = len(prices)
    pacing_rate = total_inventory / T_env

    s = compute_sell_probabilities_multi(V, prices)

    # Appiattisce variabili nella formulazione LP
    c = -(np.tile(prices, m_env) * s.flatten())
    A_ub = np.array([s.flatten()])
    b_ub = np.array([pacing_rate])

    A_eq = []
    b_eq = []
    for j in range(m_env):
        eq = np.zeros(m_env * K)
        eq[j*K:(j+1)*K] = 1
        A_eq.append(eq)
        b_eq.append(1)

    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)
    bounds = [(0, 1)] * (m_env * K)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                  A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")

    if not res.success:
        raise ValueError("LP fallito: " + res.message)

    gamma = res.x.reshape((m_env, K))
    expected_utility = -res.fun
    expected_cost = np.dot(s.flatten(), res.x)
    print(f"Expected utility: {expected_utility:.4f}")
    print(f"Optimal distribution (gamma): {gamma}")
    print(f"Expected cost: {expected_cost:.4f}")

    return expected_utility, gamma, expected_cost


def create_default_prices() -> NDArray[np.float64]:
    """Crea array di prezzi di default per gli esperimenti"""
    return np.array([0.2, 0.256, 0.311, 0.367, 0.422, 0.478,
                     0.553, 0.589, 0.644, 0.7, 0.756, 0.811,
                     0.867, 0.922, 0.98, 1.001])


def create_simple_prices() -> NDArray[np.float64]:
    """Crea array di prezzi semplice per esperimenti base"""
    return np.array([0.1, 0.2, 0.3, 0.5, 0.7, 0.8])
