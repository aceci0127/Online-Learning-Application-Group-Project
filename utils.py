import itertools
from typing import Tuple, List
import numpy as np
from scipy.optimize import linprog
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
from typing import List
from numpy.typing import NDArray


def compute_expected_revenues(prices, mu=0.8, sigma=0.2, lower=0., upper=1.):
    """
    Calcola E[p * 1{V>=p}] per V ~ TruncNorm(mu, sigma^2) su [lower, upper].
    """
    a, b = (lower - mu) / sigma, (upper - mu) / sigma
    dist = truncnorm(a, b, loc=mu, scale=sigma)
    revs = []
    for p in prices:
        prob_accept = 1.0 - dist.cdf(p)
        revs.append(p * prob_accept)
    return np.array(revs)


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


def solve_clairvoyant_lp_superarm(price_grid: List[np.ndarray], B: float, T: int, f_true: List[np.ndarray], c_true: List[np.ndarray]) -> Tuple[float, np.ndarray]:
    """
    Solves the clairvoyant per-round LP using the superarm formulation.

    Args:
        price_grid: list of 1D numpy arrays, one for each product, containing the available prices.
        B: total inventory budget.
        T: total number of rounds.
        f_true: list of 1D numpy arrays; each f_true[j] gives the true expected revenue (e.g. price*sell probability) for product j.
        c_true: list of 1D numpy arrays; each c_true[j] gives the true expected consumption (e.g. sell probability) for product j.

    Returns:
        optimal_per_round: the optimal expected revenue per round.
        simplex: 1D numpy array with the optimal probability distribution over superarms.
    """
    # Compute per-round budget (pacing constraint)
    rho = B / T

    N = len(price_grid)  # number of products
    # Create list of all possible superarm indices
    superarm_indices = list(itertools.product(
        *[range(len(price_grid[j])) for j in range(N)]))
    num_superarms = len(superarm_indices)

    # Compute total expected revenue and total cost for each superarm
    f_super = np.empty(num_superarms)
    c_super = np.empty(num_superarms)
    for i, indices in enumerate(superarm_indices):
        # f_super[i] is the sum over products of f_true[j][index_j]
        f_super[i] = sum(f_true[j][indices[j]] for j in range(N))
        # c_super[i] is the sum over products of c_true[j][indices[j]]
        c_super[i] = sum(c_true[j][indices[j]] for j in range(N))

    # Define LP: maximize sum(f_super * y) subject to sum(y) = 1 and sum(c_super * y) <= rho
    # We express this as: minimize c_obj = -f_super^T y
    c_obj = -f_super
    # Equality constraint: sum_i y_i = 1
    A_eq = np.ones((1, num_superarms))
    b_eq = np.array([1])
    # Inequality constraint: sum_i c_super[i] * y[i] <= rho
    A_ub = c_super.reshape(1, -1)
    b_ub = np.array([rho])
    # Bounds: y[i] >= 0 (upper bound can be left at 1)
    bounds = [(0, 1)] * num_superarms

    res = linprog(c=c_obj, A_ub=A_ub, b_ub=b_ub,
                  A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                  method='highs')
    if not res.success:
        raise ValueError(
            "Superarm LP did not solve successfully: " + res.message)

    # The optimal expected revenue per round is -res.fun
    optimal_per_round = -res.fun
    simplex = res.x
    print(f"Superarm LP Expected cost: {np.sum(c_super * res.x):.4f}")
    print(
        f"Superarm LP Optimal expected revenue per round: {optimal_per_round:.4f}")
    print(f"Superarm LP Optimal distribution (simplex): {simplex}")
    return optimal_per_round, simplex


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


def compute_clairvoyant_sequence(valuations, prices, budget):
    """
    Calcola la sequenza di prezzi chiarveggente dato conoscenza completa delle valutazioni.

    Args:
        valuations: array delle valutazioni
        prices: prezzi disponibili
        budget: budget massimo (unità massime vendibili)

    Returns:
        price_sequence: sequenza di prezzi da postare
        total_revenue: ricavo totale
    """
    prices = np.sort(prices)
    T = len(valuations)
    candidate = np.empty(T)

    # Per ogni round, calcola il miglior prezzo che non supera la valutazione
    for t, v in enumerate(valuations):
        allowed = prices[prices <= v]
        if allowed.size > 0:
            candidate[t] = allowed[-1]
        else:
            candidate[t] = -np.inf

    # Seleziona B round con ricavo candidato più alto
    sorted_indices = np.argsort(candidate)[::-1]
    selected = np.zeros(T, dtype=bool)
    count = 0

    for i in sorted_indices:
        if candidate[i] > -np.inf and count < budget:
            selected[i] = True
            count += 1
        else:
            selected[i] = False

    # Per round selezionati, carica il prezzo candidato; altrimenti, imposta prezzo dummy
    dummy_price = prices[-1] + 1
    price_sequence = [candidate[t] if selected[t]
                      else dummy_price for t in range(T)]
    total_revenue = sum(candidate[t] for t in range(T) if selected[t])

    return price_sequence, total_revenue


def plot_results(regrets_data, units_data, n_trials, title_prefix=""):
    """
    Crea grafici per i risultati degli esperimenti.

    Args:
        regrets_data: dati del regret cumulativo
        units_data: dati delle unità vendute cumulative
        n_trials: numero di trial
        title_prefix: prefisso per i titoli
    """
    min_rounds = min(len(reg) for reg in regrets_data)
    regrets_arr = np.array([reg[:min_rounds] for reg in regrets_data])
    units_arr = np.array([us[:min_rounds] for us in units_data])

    avg_regret = regrets_arr.mean(axis=0)
    se_regret = regrets_arr.std(axis=0) / np.sqrt(n_trials)

    avg_units = units_arr.mean(axis=0)
    se_units = units_arr.std(axis=0) / np.sqrt(n_trials)

    plt.figure(figsize=(12, 4))

    # Plot regret cumulativo
    plt.subplot(1, 2, 1)
    plt.plot(avg_regret, label="Regret Cumulativo Medio")
    plt.fill_between(np.arange(min_rounds),
                     avg_regret - se_regret,
                     avg_regret + se_regret,
                     alpha=0.3, label="±1 SE")
    plt.xlabel("Round")
    plt.ylabel("Regret Cumulativo")
    plt.title(f"{title_prefix}Regret Cumulativo")
    plt.legend()

    # Plot unità vendute cumulative
    plt.subplot(1, 2, 2)
    plt.plot(avg_units, label="Unità Vendute Cumulative Medie")
    plt.fill_between(np.arange(min_rounds),
                     avg_units - se_units,
                     avg_units + se_units,
                     alpha=0.3, label="±1 SE")
    plt.xlabel("Round")
    plt.ylabel("Unità Vendute Cumulative")
    plt.title(f"{title_prefix}Unità Vendute Cumulative")
    plt.legend()

    plt.tight_layout()
    plt.show()

    return avg_regret, avg_units, min_rounds


def print_final_results(avg_regret, avg_units, min_rounds, final_rewards, agent=None):
    """
    Stampa i risultati finali dell'esperimento.

    Args:
        avg_regret: regret medio
        avg_units: unità medie vendute
        min_rounds: numero minimo di round
        final_rewards: ricavi finali
        agent: agente (opzionale, per statistiche aggiuntive)
    """
    print(f"\nRisultati Finali:")
    print(f"Regret medio per round: {avg_regret[-1]/min_rounds:.4f}")
    print(f"Unità vendute cumulative medie: {avg_units[-1]:.2f}")

    if final_rewards:
        final_rewards = np.array(final_rewards)
        print(f"Ricavo cumulativo medio: {np.mean(final_rewards):.2f}")

    if agent and hasattr(agent, 'N_pulls'):
        print(f"Conteggi pull: {agent.N_pulls}")
    elif agent and hasattr(agent, 'pull_counts'):
        print(f"Conteggi pull: {agent.pull_counts}")


def create_default_prices() -> NDArray[np.float64]:
    """Crea array di prezzi di default per gli esperimenti"""
    return np.array([0.2, 0.256, 0.311, 0.367, 0.422, 0.478,
                     0.553, 0.589, 0.644, 0.7, 0.756, 0.811,
                     0.867, 0.922, 0.98, 1.001])


def create_simple_prices() -> NDArray[np.float64]:
    """Crea array di prezzi semplice per esperimenti base"""
    return np.array([0.1, 0.2, 0.3, 0.5, 0.7, 0.8])


def create_multiproduct_price_grid(base_prices: NDArray[np.float64], num_products: int) -> NDArray[np.float64]:
    """
    Create price grid for multi-product.

    Args:
        base_prices: base prices
        num_products: number of products

    Returns:
        price_grid: list of arrays of prices for each product
    """
    return base_prices
