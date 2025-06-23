import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# ----------------------------
# Environment Class (unchanged)
# ----------------------------
class MultiProductPricingEnvironment:
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





# ----------------------------
# UCB Agent (with LP inside)
# ----------------------------
class ConstrainedCombinatorialUCBAgent:
    def __init__(self, price_grid, B, T, alpha=1):
        self.price_grid = price_grid
        self.N = len(price_grid)
        self.Ks = [len(pg) for pg in price_grid]
        self.B_rem = B
        self.T = T
        self.t = 0
        self.alpha = alpha
        self.rng = np.random.default_rng()

        # stats per (product j, price k)
        self.N_pulls = [np.zeros(K) for K in self.Ks]
        self.avg_f    = [np.zeros(K) for K in self.Ks]
        self.avg_c    = [np.zeros(K) for K in self.Ks]
        self.current_choice = None

    def _solve_marginal_lp(self, f_ucb, c_lcb, rho):
        """
        Solve the per-round LP:
          max_x sum_j f_ucb[j]·x_j
          s.t. ∑_k x_{j,k} = 1  ∀j,
               ∑_{j,k} c_lcb[j][k]·x_{j,k} ≤ rho,
               x ≥ 0.
        Returns a list of marginals [x_0, x_1, …, x_{N-1}].
        """
        # flatten objective and constraints
        f_flat = np.concatenate(f_ucb)
        c_flat = np.concatenate(c_lcb)
        num_vars = len(f_flat)

        # objective for linprog: minimize -f_flat @ x
        c_obj = -f_flat

        # equality: per-product sums = 1
        A_eq = np.zeros((self.N, num_vars))
        b_eq = np.ones(self.N)
        offset = 0
        for j, K in enumerate(self.Ks):
            A_eq[j, offset:offset+K] = 1
            offset += K

        # inequality: total consumption ≤ rho
        A_ub = c_flat.reshape(1, -1)
        b_ub = np.array([rho])

        bounds = [(0, None)] * num_vars

        res = linprog(c=c_obj,
                      A_ub=A_ub, b_ub=b_ub,
                      A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds,
                      method='highs')

        if not res.success:
            raise RuntimeError("LP failed: " + res.message)

        x_flat = res.x
        # un-flatten to list of arrays
        marginals = []
        offset = 0
        for K in self.Ks:
            marginals.append(x_flat[offset:offset+K])
            offset += K

        return marginals

    def pull_arm(self):
        if self.B_rem < 1 or self.t >= self.T:
            return None

        rho = self.B_rem / (self.T - self.t)

        # build UCB/LCB arrays
        f_ucb = []
        c_lcb = []
        for j in range(self.N):
            n_j = self.N_pulls[j]
            f_j = self.avg_f[j]
            c_j = self.avg_c[j]

            bonus = np.sqrt(self.alpha * np.log(self.T) / np.maximum(1, n_j))
            bonus[n_j == 0] = self.T        # force exploring unseen arms

            f_ucb.append(f_j + bonus)
            c_lcb.append(np.clip(c_j - bonus, 0.0, 1.0))

        # solve LP to get marginals
        marginals = self._solve_marginal_lp(f_ucb, c_lcb, self.B_rem / (self.T - self.t + 1) )
        #print(f"Round {self.t}: marginals = {marginals}, B_rem = {self.B_rem:.2f}")

        # sample a joint price vector
        choice = tuple(
            self.rng.choice(self.Ks[j], p=marginals[j])
            for j in range(self.N)
        )

        self.current_choice = choice
        return choice

    def update(self, rewards, costs):
        for j, idx in enumerate(self.current_choice):
            self.N_pulls[j][idx] += 1
            n = self.N_pulls[j][idx]
            # incremental average:
            self.avg_f[j][idx] += (rewards[j] - self.avg_f[j][idx]) / n
            self.avg_c[j][idx] += (costs[j] - self.avg_c[j][idx]) / n
        self.B_rem -= costs.sum()
        self.t += 1

def solve_clairvoyant_lp(price_grid, B, T):
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
    
    # Compute true expected reward and consumption for uniform[0,1] valuations
    f_true = [p * np.maximum(0, (1 - p)) for p in price_grid]  # expected revenue
    c_true = [np.maximum(0,1 - p) for p in price_grid]        # expected consumption/probability
    print(f_true, c_true)
    # Flatten variables for LP
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
    
    if res.success:
        optimal_per_round = -res.fun
        simplex = res.x
        return optimal_per_round , simplex
    else:
        raise ValueError("LP did not solve successfully: " + res.message)


    
# ----------------------------
# Example of running
# ----------------------------
if __name__ == "__main__":
    N_products = 3
    base_prices = np.array([0.2, 0.5, 0.9])
    dummy_price = 1.001
    price_grid = [
    np.concatenate([base_prices, [dummy_price]])
    for _ in range(N_products)
    ]

    T = 10000
    B = 15000
    seed = 18

    clair_reward ,simplex = solve_clairvoyant_lp(price_grid, B, T)
    print(f"Clairvoyant expected reward per round: {clair_reward:.4f} and simplex: {simplex}")
    n_trials = 10
    all_regrets, all_units = [], []

    for trial in range(n_trials):
        print(f"Trial {trial+1}/{n_trials}...")
        rng = np.random.default_rng(seed + trial)
        env = MultiProductPricingEnvironment(price_grid, T, rng=rng)
        agent = ConstrainedCombinatorialUCBAgent(price_grid, B, T)

        cum_regret, cum_units = 0.0, 0
        regrets, units = [], []

        for t in range(T):
            choice = agent.pull_arm()
            if choice is None:
                print(f"Trial {trial+1}: Ran out of budget or time at round {t}.")
                break
            rewards, costs = env.round(choice)
            agent.update(rewards, costs)

            actual_rew = rewards.sum()
            actual_units = costs.sum()
            cum_regret += (clair_reward - actual_rew)
            cum_units += actual_units
            regrets.append(cum_regret)
            units.append(cum_units)

        all_regrets.append(regrets)
        all_units.append(units)

    # Aggregate and plot
    min_len = min(len(r) for r in all_regrets)
    avg_regret = np.mean([r[:min_len] for r in all_regrets], axis=0)
    sd_regret  = np.std([r[:min_len] for r in all_regrets], axis=0)
    avg_units  = np.mean([u[:min_len] for u in all_units], axis=0)
    sd_units   = np.std([u[:min_len] for u in all_units], axis=0)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(avg_regret, label="Avg Cumul. Regret")
    plt.fill_between(np.arange(min_len),
                     avg_regret - sd_regret/np.sqrt(n_trials),
                     avg_regret + sd_regret/np.sqrt(n_trials), alpha=0.3)
    plt.xlabel("Round"); plt.ylabel("Cumul. Regret")
    plt.title("Cumulative Regret")
    plt.legend()

    plt.subplot(1, 2, 2)
    import matplotlib.ticker as ticker
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    plt.plot(avg_units, label="Avg Cumul. Units Sold")
    plt.fill_between(np.arange(min_len),
                     avg_units - sd_units/np.sqrt(n_trials),
                     avg_units + sd_units/np.sqrt(n_trials), alpha=0.3)
    plt.xlabel("Round"); plt.ylabel("Cumul. Units")
    plt.title("Cumulative Units Sold")
    plt.legend()

    plt.tight_layout()
    plt.show()

    print(f"Final avg units sold: {avg_units[-1]:.2f}")
    print(f"Final avg regret per round: {avg_regret[-1]/min_len:.4f}")

