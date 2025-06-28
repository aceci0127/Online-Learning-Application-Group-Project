import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from collections import Counter

# --- The multi-product environment and agent definitions (excerpts) ---
def generate_stationary_correlated_gauss(T, m, mu, sigma, rho, rng=None):
    """
    Generate stationary correlated Gaussian valuations.
    
    Parameters:
        T (int): Number of rounds.
        m (int): Number of products.
        mu (float): Mean valuation for each product.
        sigma (float): Standard deviation for each product.
        rho (float): Constant correlation coefficient among products.
        rng (np.random.Generator, optional): Random number generator.
        
    Returns:
        V (np.ndarray): Array of shape (T, m) with generated valuations clipped to [0, 1].
        R_ts (np.ndarray): Repeated correlation matrix of shape (T, m, m).
    """
    rng = rng or np.random.default_rng(0)
    # Build the constant correlation matrix.
    R = np.eye(m) + (1 - np.eye(m)) * rho
    # Compute the covariance matrix using sigma.
    Sigma = np.diag([sigma]*m) @ R @ np.diag([sigma]*m)
    # Generate T samples from the multivariate normal distribution.
    V = rng.multivariate_normal(mean=[mu]*m, cov=Sigma, size=T)
    V = np.clip(V, 0, 1)  # Ensure valuations are within [0, 1].
    R_ts = np.repeat(R[np.newaxis, :, :], T, axis=0)
    return V, R_ts

# Generate time-varying multivariate Gaussian valuations with sinusoidal modulation.
def generate_simple_tv_mv_gauss(T, m,
                                mu0, A, f, phi,
                                sigma0, A_sigma, phi_sigma, rho0,
                                rng=None):
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
## PIECEWISE STATIONARY MULTIVARIATE GAUSSIAN VALUATIONS ##
def generate_piecewise_tv_mv_gauss(T, m, num_regimes,
                                     mu_low=0.3, mu_high=0.7,
                                     sigma_low=0.05, sigma_high=0.15,
                                     rho_low=-0.3, rho_high=0.3,
                                     rng=None):
    """
    Generate T rounds of m-dimensional correlated valuations from a piecewise-stationary multivariate Gaussian.
    
    The time horizon is split into num_regimes epochs. In each regime:
      - A mean vector for the m products is sampled uniformly from [mu_low, mu_high].
      - A standard deviation vector is sampled uniformly from [sigma_low, sigma_high].
      - A constant off-diagonal correlation is sampled uniformly from [rho_low, rho_high],
        which is used to form a correlation matrix.
      - Valuations are drawn from the multivariate Gaussian with these parameters and then clipped to [0,1].
      
    Parameters:
        T           : int
                      Total number of rounds.
        m           : int
                      Number of products (dimensions).
        num_regimes : int
                      Number of piecewise regimes.
        mu_low,
        mu_high    : float
                      Lower and upper bound for each product's mean.
        sigma_low,
        sigma_high : float
                      Lower and upper bound for each product's standard deviation.
        rho_low,
        rho_high   : float
                      Lower and upper bound for the common off-diagonal correlations.
        rng         : Generator, optional
                      NumPy random generator.
    
    Returns:
        V   : ndarray, shape (T, m)
              Generated valuations (clipped to [0,1]).
        R_ts: ndarray, shape (T, m, m)
              Correlation matrix used for each round.
    """
    rng = rng or np.random.default_rng(0)
    V = np.empty((T, m))
    R_ts = np.empty((T, m, m))
    regime_length = T // num_regimes
    
    for regime in range(num_regimes):
        start = regime * regime_length
        end = T if regime == num_regimes - 1 else (regime + 1) * regime_length
        
        # Sample regime-specific parameters.
        mu_reg = rng.uniform(mu_low, mu_high, size=m)
        sigma_reg = rng.uniform(sigma_low, sigma_high, size=m)
        rho = rng.uniform(rho_low, rho_high)
        
        # Construct correlation and covariance matrices.
        R = np.eye(m) + (np.ones((m, m)) - np.eye(m)) * rho
        Sigma = np.diag(sigma_reg) @ R @ np.diag(sigma_reg)
        
        # Generate samples for this regime.
        for t in range(start, end):
            sample = rng.multivariate_normal(mu_reg, Sigma)
            V[t] = np.clip(sample, 0, 1)
            R_ts[t] = R
            
    return V, R_ts






class MultiProductBudgetedPricingEnvironment:
    def __init__(self, prices, T, m, V , mu0, A, f, phi, sigma0, A_sigma, phi_sigma, rho0, rng=None):
        self.prices = np.array(prices)
        self.T = T
        self.m = m
        self.t = 0
        self.rng = rng or np.random.default_rng(0)
        self.V = V
        
    def full_feedback_round(self):
        if self.t >= self.T:
            raise RuntimeError("Time horizon exceeded!")
        v_t = self.V[self.t]
        self.t += 1
        return v_t

class HedgeAgent:
    def __init__(self, K, learning_rate, rng):
        self.K = K
        self.learning_rate = learning_rate
        self.weights = np.ones(K)
        self.x_t = np.ones(K) / K
        self.rng = rng
    def pull_arm(self):
        self.x_t = self.weights / np.sum(self.weights)
        arm = self.rng.choice(np.arange(self.K), p=self.x_t)
        return arm
    def update(self, loss):
        self.weights *= np.exp(-self.learning_rate * loss)

class MultiProductFFPrimalDualPricingAgent:
    def __init__(self, prices, T, B, m, rng, eta):
        """
        prices: array of allowed prices (arms), including a dummy arm >1
        T: time horizon
        B: total inventory
        m: number of products
        eta: dual step size
        """
        self.prices = np.array(prices)
        self.K = len(prices)
        self.T = T
        self.m = m
        self.B = B
        # global pacing rate per product
        self.rho = self.rho = B / (m * T)
        self.eta = eta

        self.rng = rng
        self.hedges = [HedgeAgent(self.K, np.sqrt(np.log(self.K)/T), self.rng)
                       for _ in range(m)]
        
        self.lmbd = 1.0
        self.inventory = B

        # -- Debug: record chosen and sold prices per product --
        self.debug_chosen_prices = [[] for _ in range(m)]
        self.debug_sold_prices = [[] for _ in range(m)]

    def pull_arms(self):
        if self.inventory < 1:
            return [None] * self.m
        arms = [hedge.pull_arm() for hedge in self.hedges]
        return arms

    def update(self, v_t):
        arms = self.pull_arms()
        total_revenue = 0.0
        total_units_sold = 0
        losses = []
        # For normalization: assume prices in [0,1] for genuine arms, dummy >1.
        p_max = self.prices.max()
        L_up = p_max - self.lmbd * (0 - self.rho)
        L_low = 0.0 - self.lmbd * (1 - self.rho)
        norm_factor = L_up - L_low + 1e-12
        
        for j in range(self.m):
            arm = arms[j]
            # Record debug info even if no arm is pulled.
            if arm is None:
                self.debug_chosen_prices[j].append(None)
                self.debug_sold_prices[j].append(None)
                losses.append(np.zeros(self.K))
                continue
            p_chosen = self.prices[arm]
            # Record chosen price
            self.debug_chosen_prices[j].append(p_chosen)
            sale = 1 if p_chosen <= v_t[j] else 0
            # Record sold price if sale occurs; otherwise record None.
            self.debug_sold_prices[j].append(p_chosen if sale == 1 else None)
            
            f = p_chosen * sale
            total_revenue += f
            total_units_sold += sale
            
            # 1. which arms would sell?
            would_sell = (self.prices <= v_t[j]).astype(float)      # shape (K,)
            # 2. revenue vector f_i = p_i * sale_i
            f_vec     = self.prices * would_sell                   # shape (K,)
            # 3. the per-arm primal utility L_i = f_i - λ*(sale_i - ρ)
            L_vec     = f_vec - self.lmbd * (would_sell - self.rho) # shape (K,)
            # 4. normalize into [0,1] losses
            loss_vec  = 1.0 - (L_vec - L_low) / norm_factor          # shape (K,)
            # 5. push into your list
            
            loss_vec[-1] = 1.0 #trying to avoid dummy arm
            losses.append(loss_vec)
            
        for j in range(self.m):
            self.hedges[j].update(losses[j])
            
        self.inventory -= total_units_sold
        self.lmbd = np.clip(self.lmbd - self.eta * (self.rho * self.m - total_units_sold),
                              a_min=0.0 , a_max= 1 / self.rho)
        return total_revenue, total_units_sold

# --- Simulation of cumulative regret, similar to task3_1.py ---

if __name__ == "__main__":
    # Set simulation parameters
    T = 200_000      # number of rounds
    m = 4      # number of products
    B = 80_000       # total inventory
    seed = 42
    n_trials = 2
    rng_global = np.random.default_rng(seed)
    
    # Prices: dummy arm > 1 (represents 'do not sell')
    prices = np.array([0.2, 0.256, 0.3, 0.4, 0.5, 0.6, 0.7 , 0.8 ,0.85, 0.9 ,0.95, 1.1])
    
    # Environment parameters for generating multivariate valuations
    mu0, A, f = 0.5, 0.1, 100
    phi = np.zeros(m)
    sigma0, A_sigma, phi_sigma, rho0 = 0.1, 0.1, 0, 0.6
    '''
    # First, compute the clairvoyant benchmark per round by computing sell probabilities.
    # We generate valuations once for the clairvoyant.
    V, _ = generate_simple_tv_mv_gauss(T, m, mu0, A, f, phi, sigma0, A_sigma, phi_sigma, rho0, rng=rng_global)
    V, _ = generate_piecewise_tv_mv_gauss(T, m, num_regimes=10000,rng = rng_global)
    
    #V ,_ = generate_stationary_correlated_gauss(T, m, mu0, sigma0, rho0, rng=rng_global) #testing purposes
    '''
    # Compute sell probabilities per product and price.
    def compute_sell_probabilities_multi(V, prices):
        T_env, m_env = V.shape
        K = len(prices)
        s = np.zeros((m_env, K))
        for j in range(m_env):
            for i, p in enumerate(prices):
                s[j, i] = np.sum(V[:, j] >= p) / T_env
        return s

    def compute_extended_clairvoyant(V, prices, total_inventory):
        T_env, m_env = V.shape
        K = len(prices)
        pacing_rate = total_inventory / T_env  # budget per round (global)
        s = compute_sell_probabilities_multi(V, prices)
        print(f"Sell probabilities (s): {s}")
        # Flatten variables in LP formulation.
        c = - (np.tile(prices, m_env) * s.flatten())
        A_ub = np.array([s.flatten()])
        b_ub = np.array([pacing_rate])
        A_eq = []
        b_eq = []
        for j in range(m_env):
            eq = np.zeros(m_env*K)
            eq[j*K:(j+1)*K] = 1
            A_eq.append(eq)
            b_eq.append(1)
        A_eq = np.array(A_eq)
        b_eq = np.array(b_eq)
        bounds = [(0, 1)] * (m_env*K)
        res = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub,
                               A_eq=A_eq, b_eq=b_eq,
                               bounds=bounds, method="highs")
        if res.success:
            gamma = res.x.reshape((m_env,K))
            expected_clairvoyant_utility = -res.fun
            expected_cost = np.dot(s.flatten(), res.x)
        else:
            raise ValueError("LP failed to solve.")
        return expected_clairvoyant_utility, gamma, expected_cost
    '''
    exp_util, gamma, exp_cost = compute_extended_clairvoyant(V, prices, B)
    print(f"Gamma : {gamma:}")
    print(f"Expected clairvoyant revenue per round: {exp_util:.4f}")
    print(f"Expected cost (inv. consumption per round): {exp_cost:.4f}")
    '''
    # --- Simulation across multiple trials ---
    all_regrets = []
    all_units_sold = []
    final_rewards = []
    
    for trial in range(n_trials):
        trial_rng = np.random.default_rng(seed + trial)
        # Create environment and agent for this trial.
        V, _ = generate_simple_tv_mv_gauss(T, m, mu0, A, f, phi, sigma0, A_sigma, phi_sigma, rho0, rng=trial_rng)
        V, _ = generate_piecewise_tv_mv_gauss(T, m, num_regimes=10000,rng = trial_rng)
        
        exp_util, gamma, exp_cost = compute_extended_clairvoyant(V, prices, B)
        print(f"Gamma : {gamma:}")
        print(f"Expected clairvoyant revenue per round: {exp_util:.4f}")
        print(f"Expected cost (inv. consumption per round): {exp_cost:.4f}")
        
        env = MultiProductBudgetedPricingEnvironment(prices, T, m, V ,mu0, A, f, phi,
                                                     sigma0, A_sigma, phi_sigma, rho0,
                                                     rng=trial_rng)
        eta = 1 / np.sqrt(T)
        agent = MultiProductFFPrimalDualPricingAgent(prices, T, B, m, trial_rng, eta)
        
        regrets = []
        units_sold = []
        cum_reward = 0.0
        cum_regret = 0.0
        cum_units = 0
        
        for t in range(T):
            if agent.inventory < 1:
                # Budget exhausted.
                print(f"Trial {trial+1}: Budget exhausted at round {t}.")
                break
            v_t = env.full_feedback_round()    # vector of valuations for m products
            reward, sold = agent.update(v_t)
            cum_reward += reward
            # Instantaneous regret is the gap between clairvoyant per-round revenue and obtained revenue.
            instant_regret = exp_util - reward
            cum_regret += instant_regret
            cum_units += sold
            regrets.append(cum_regret)
            units_sold.append(cum_units)
        
        all_regrets.append(regrets)
        all_units_sold.append(units_sold)
        final_rewards.append(cum_reward)
        print(f"Trial {trial+1}: Final cumulative reward = {cum_reward:.2f}")
        print(f"Trial {trial+1}: Final remaining budget = {agent.inventory}")
        
        # --- Debug Report per Product for this trial ---
        print("\nDebug Report for Trial", trial+1)
        for j in range(m):
            chosen_counter = Counter(x for x in agent.debug_chosen_prices[j] if x is not None)
            sold_counter = Counter(x for x in agent.debug_sold_prices[j] if x is not None)
            print(f"Product {j+1}:")
            print(f"   Chosen prices frequency: {dict(chosen_counter)}")
            print(f"   Sold prices frequency:   {dict(sold_counter)}")
    
    # Truncate arrays to the minimum rounds reached among trials.
    min_rounds = min(len(reg) for reg in all_regrets)
    regrets_arr = np.array([reg[:min_rounds] for reg in all_regrets])
    units_arr = np.array([us[:min_rounds] for us in all_units_sold])
    
    avg_regret = regrets_arr.mean(axis=0)
    se_regret = regrets_arr.std(axis=0) / np.sqrt(n_trials)
    
    avg_units = units_arr.mean(axis=0)
    se_units = units_arr.std(axis=0) / np.sqrt(n_trials)
    
    # Plot cumulative regret.
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(avg_regret, label="Average Cumulative Regret")
    plt.fill_between(np.arange(min_rounds), avg_regret - se_regret, avg_regret + se_regret, alpha=0.3, label="± 1 SE")
    plt.xlabel("Rounds")
    plt.ylabel("Cumulative Regret")
    plt.title("Cumulative Regret (Multi-Product)")
    plt.legend()
    
    # Plot cumulative units sold.
    plt.subplot(1, 2, 2)
    plt.plot(avg_units, label="Average Cumulative Units Sold")
    plt.fill_between(np.arange(min_rounds), avg_units - se_units, avg_units + se_units, alpha=0.3, label="± 1 SE")
    plt.xlabel("Rounds")
    plt.ylabel("Cumulative Units Sold")
    plt.title("Cumulative Units Sold (Multi-Product)")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print("\nFinal Results:")
    print(f"Average final cumulative reward: {np.mean(final_rewards):.2f}")
    print(f"Average regret per round: {avg_regret[-1]/min_rounds:.4f}")