import numpy as np
from scipy.stats import truncnorm
from scipy.optimize import linprog
import matplotlib.pyplot as plt


class BudgetedPricingEnvironment:
    def __init__(self, prices, T, budget, mu=0.8, sigma=0.2, rng=None):
        self.prices = np.array(prices)
        self.T = T
        self.B = budget
        self.remaining_budget = budget
        self.t = 0
        
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng
        
        # truncated normal valuations
        a, b = (0 - mu) / sigma, (1 - mu) / sigma
        self.vals = truncnorm(a, b, loc=mu, scale=sigma).rvs(size=T, random_state=rng)
    
    def round(self, price_index):
        p = self.prices[price_index]
        sale = self.vals[self.t] >= p
        reward = p if sale else 0.0
        cost = 1.0 if sale else 0.0
        if sale:
            self.remaining_budget -= 1
        self.t += 1
        return reward, cost

class ConstrainedUCBPricingAgent:
    def __init__(self, prices, T, B, alpha=1.0, rng=None):
        self.prices = np.array(prices)
        self.m = len(prices)
        self.T = T
        self.B = B
        
        # statistics
        self.counts = np.zeros(self.m, dtype=int)
        self.sum_reward = np.zeros(self.m, dtype=float)
        self.sum_cost = np.zeros(self.m, dtype=float)
        self.t = 0
        
        self.alpha = alpha
        
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng
    
    def estimate_bounds(self):
        f_ucb = np.full(self.m, np.inf)
        c_lcb = np.full(self.m, -np.inf)
        for i in range(self.m):
            n = self.counts[i]
            if n > 0:
                mu_r = self.sum_reward[i] / n
                mu_c = self.sum_cost[i] / n
                bonus = self.alpha * np.log(self.t +1) / n
                f_ucb[i] = mu_r + bonus
                c_lcb[i] = mu_c - bonus
        return f_ucb, c_lcb
    
    def select_distribution(self, f_ucb, c_lcb, remaining_budget):
        ####THIS iS WRONG, FIX IT
        rhs = remaining_budget / (self.T - self.t) if self.T > self.t else 0.0
        c = -f_ucb  # linprog minimizes
        A = np.vstack([c_lcb, np.ones(self.m)])
        b = np.array([rhs, 1.0])
        bounds = [(0, 1) for _ in range(self.m)]
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        
        
        if res.success:
            return res.x
        else:
            return np.ones(self.m) / self.m
    
    def pull_arm(self, env):
        if self.t < self.m:
            return self.t  # initialize
        f_ucb, c_lcb = self.estimate_bounds()
        gamma = self.select_distribution(f_ucb, c_lcb, env.remaining_budget)
        return self.rng.choice(self.m, p=gamma)
    
    def update(self, arm, reward, cost):
        self.counts[arm] += 1
        self.sum_reward[arm] += reward
        self.sum_cost[arm] += cost
        self.t += 1

import numpy as np
from scipy.stats import truncnorm
from scipy.optimize import linprog

def compute_expected_revenues_truncnorm(prices, mu=0.8, sigma=0.2, lower=0., upper=1.):
    """
    Compute:
      f_true[p] = E[p * 1{V>=p}]
      c_true[p] = E[1{V>=p}]
    for V ~ TruncNorm(mu, sigma^2) on [lower, upper].
    """
    a, b = (lower - mu) / sigma, (upper - mu) / sigma
    dist = truncnorm(a, b, loc=mu, scale=sigma)

    f_true = np.array([p * (1 - dist.cdf(p)) for p in prices])
    c_true = np.array([(1 - dist.cdf(p))       for p in prices])
    return f_true, c_true

def compute_true_clairvoyant(T, B, prices, mu=0.8, sigma=0.2):
    """
    Solves the LP:
      max   f_true^T γ
      s.t.  c_true^T γ <= B / T
            sum(γ) = 1, γ >= 0

    Returns:
      γ_opt          -- optimal mixing over prices
      opt_per_round  -- expected revenue per round = f_true^T γ_opt
      clairvoyant_total -- T * opt_per_round
    """
    ### THIS IS WRONG, FIX IT
    f_true, c_true = compute_expected_revenues_truncnorm(prices, mu, sigma)
    m = len(prices)

    # We minimize -f_true^T γ
    c_lp = -f_true
    # Constraints: c_true^T γ <= B/T  and  sum(γ)=1
    A_ub = np.vstack([c_true])
    b_ub = np.array([B/T])

    A_eq = np.ones((1, m))
    b_eq = np.array([1.0])

    res = linprog(
    c_lp,
    A_ub=A_ub, b_ub=b_ub,
    A_eq=A_eq, b_eq=b_eq,
    bounds=[(0,1)]*m,
    method='highs'
    )
    if not res.success:
        raise RuntimeError("LP solver failed to find clairvoyant distribution.")

    gamma_opt = res.x
    opt_per_round = f_true.dot(gamma_opt)
    clairvoyant_total = T * opt_per_round

    return gamma_opt, opt_per_round, clairvoyant_total

if __name__ == "__main__":
    # Example parameters
    prices = np.array([0.2, 0.3, 0.4, 0.6, 0.7, 0.8])
    T = 100_000
    B = 30_000
    mu, sigma = 0.8, 0.2
    seed = 18

    gamma_opt, opt_per_round, clairvoyant_total = compute_true_clairvoyant(
        T, B, prices, mu, sigma
    )

    print("Prices:", prices)
    print(f"Budget B = {B}, Horizon T = {T}\n")
    print("Clairvoyant optimal distribution γ:")
    for p, weight in zip(prices, gamma_opt):
        print(f"  Price {p:.1f}: γ = {weight:.4f}")
    print(f"\nOPT per round (expected revenue): {opt_per_round:.6f}")
    print(f"Total clairvoyant reward (T * OPT): {clairvoyant_total:.2f}")
    n_trials = 10
    # simulate
    np.random.seed(seed)
    all_regrets = []
    for trial in range(n_trials):
        # new RNG per trial for both env and reproducibility
        rng = np.random.RandomState(seed + trial)
        env = PricingEnvironment(prices, T, rng=rng)
        agent = UCB1PricingAgent(prices, T)

        regrets = []
        cum_regret = 0.0
        for t in range(T):
            arm = agent.pull_arm()
            r = env.round(arm)
            agent.update(r)

            # instantaneous regret = clairvoyant reward − actual
            instant_regret = expected_revenues[best_idx] - r
            cum_regret += instant_regret
            regrets.append(cum_regret)

        all_regrets.append(regrets)

    all_regrets = np.array(all_regrets)
    avg_regret = all_regrets.mean(axis=0)
    sd_regret = all_regrets.std(axis=0)

    # plot
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(avg_regret, label="Average Cumulative Regret")
    plt.fill_between(
        np.arange(T),
        avg_regret - sd_regret / np.sqrt(n_trials),
        avg_regret + sd_regret / np.sqrt(n_trials),
        alpha=0.3,
        label="±1 SE"
    )
    plt.xlabel("t")
    plt.ylabel("Cumulative Regret")
    plt.title("UCB1 Pricing: Cumulative Regret")
    plt.legend()

    plt.subplot(1, 2, 2)
    labels = [f"{p:.1f}" for p in prices]
    plt.barh(labels, agent.N_pulls)
    plt.xlabel("Number of Pulls")
    plt.ylabel("Price")
    plt.title("Final Allocation of Pulls")

    plt.tight_layout()
    plt.show()

    print("\nFinal Results:")
    print(f"Average regret per round: {avg_regret[-1]/T:.4f}")
    print("Empirical average revenues:", np.round(agent.average_rewards, 4))
    print("Pull counts:", agent.N_pulls)
    print("True expected revenues:", np.round(expected_revenues, 4))

