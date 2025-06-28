import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
### SINUSOIDAL PATTERN ###

def pattern(t, T, freq):
    """Time-varying upper bound: 1 - |sin(freq * t / T)|."""
    return 1 - abs(np.sin(freq * t / T))
def generate_beta_valuations(T, shock_prob, freq, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    valuations = np.empty(T)
    for t in range(T):
        a_t = pattern(t, T, freq)
        # let alpha, beta oscillate or trend
        alpha_t = 1 + 4 * (0.5 + 0.5 * np.sin(10*np.pi * t / T))
        beta_t  = 1 + 4 * (0.5 + 0.5 * np.cos(10*np.pi * t / T))
        '''
        if rng.random() < shock_prob:
            valuations[t] = rng.choice([0.0, a_t])
        else:
            # sample Beta then scale with a_t
            valuations[t] = a_t * rng.beta(alpha_t, beta_t)
        '''
        valuations[t] = rng.beta(alpha_t, beta_t)
    return valuations

def generate_valuations(T, shock_prob, freq, rng=None):
    """
    Generate a sequence of T valuations with occasional adversarial shocks.
    
    - a(t) = pattern(t, T, freq)
    - With probability shock_prob, v_t is either 0 or a(t) (extreme).
    - Otherwise v_t ~ Uniform(0, a(t)).
    """
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

### PIECEWISE STATIONARY BETA DISTRIBUTION ###

def generate_piecewise_beta_valuations(T, shock_prob, num_regimes, rng=None):
    """
    Generate T valuations from a piecewise-stationary Beta distribution.
    
    The time horizon is divided into num_regimes epochs. In each epoch a new pair 
    of Beta parameters (alpha, beta) is sampled uniformly from [1, 20] to induce large jumps.
    
    With probability shock_prob, the valuation is set to an extreme (0 or 1) in a given round.
    Otherwise, the valuation is drawn from Beta(alpha, beta).
    
    This creates a highly non-stationary environment where the best price can jump 
    drastically between regimes.
    """
    if rng is None:
        rng = np.random.default_rng()

    valuations = np.empty(T)
    regime_length = T // num_regimes
    for regime in range(num_regimes):
        start = regime * regime_length
        end = T if regime == num_regimes - 1 else (regime + 1) * regime_length
        # Sample new parameters for the current regime
        alpha = rng.uniform(1, 50)
        beta  = rng.uniform(1, 50)
        for t in range(start, end):
            if rng.random() < shock_prob:
                valuations[t] = rng.choice([0.0, 1.0])
            else:
                valuations[t] = rng.beta(alpha, beta)
    return valuations
### CLAIRVOYANT PRICING SEQUENCE ###
def compute_clairvoyant_sequence(valuations, prices, B):
    """
    Given full knowledge of the valuations (clairvoyant) and allowed prices,
    compute the best sequence of prices under the budget constraint B (max units sold).
    
    For each round t, define the candidate revenue as the best allowed price (largest p <= v_t).
    Then, select the B rounds with the highest candidate revenues (ignoring rounds where no sale is possible)
    and set the price to the candidate value. For rounds not selected, set a dummy high price (e.g. max(prices)+ε)
    to enforce no sale.
    
    Returns:
        price_sequence: list of prices to be posted at each round t
        total_revenue: sum of revenues in rounds where a sale is made.
    """
    prices = np.sort(prices)  # ensure prices are in ascending order
    T = len(valuations)
    candidate = np.empty(T)
    
    # For each round, compute best price that is not above the valuation.
    # If no allowed price can be set (i.e. valuation < smallest allowed), mark candidate as -inf.
    for t, v in enumerate(valuations):
        allowed = prices[prices <= v]
        if allowed.size > 0:
            candidate[t] = allowed[-1]
        else:
            candidate[t] = -np.inf  # no sale is possible

    # Select B rounds with highest candidate revenue.
    # (Rounds with candidate == -inf are naturally not selected.)
    sorted_indices = np.argsort(candidate)[::-1]  # sort rounds descending by candidate revenue
    selected = np.zeros(T, dtype=bool)
    count = 0
    for i in sorted_indices:
        if candidate[i] > -np.inf and count < B:
            selected[i] = True
            count += 1
        else:
            selected[i] = False
            
    # For rounds selected, charge the candidate price; otherwise, set a dummy price above any possible valuation.
    dummy_price = prices[-1] + 1  # assuming valuations are in [0, 1], this will prevent a sale.
    price_sequence = [candidate[t] if selected[t] else dummy_price for t in range(T)]
    total_revenue = sum(candidate[t] for t in range(T) if selected[t])
    
    return price_sequence, total_revenue
class BudgetedPricingEnvironment:
    def __init__(self, prices, T, shock_prob , freq, rng=None):
        self.prices = np.array(prices)
        self.T = T
        self.t = 0
        self.shock_prob = shock_prob
        self.freq = freq
        
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng
        '''
        # truncated normal valuations
        a, b = (0 - mu) / sigma, (1 - mu) / sigma
        self.vals = truncnorm(a, b, loc=mu, scale=sigma).rvs(size=T, random_state=rng)
        '''
        #self.valuations = generate_beta_valuations(T, shock_prob, freq, rng=rng)
        #self.valuations = np.clip(self.valuations, 0, 1)  # ensure valuations are in [0, 1]
        #self.valuations = generate_valuations(T, shock_prob, freq, rng=rng)
        self.valuations = generate_piecewise_beta_valuations(T, shock_prob, 10000 , rng=rng)
    
    def bandit_round(self, price_index):
        p = self.prices[price_index]
        sale = self.valuations[self.t] >= p
        reward = p if sale else 0.0
        cost = 1.0 if sale else 0.0
        self.t += 1
        return reward, cost
    def full_feedback_round(self):
        """
        Full feedback round: returns the valuation.
        """
        valuation = self.valuations[self.t]
        self.t += 1
        return valuation
    def compute_sell_probabilities(self):
        """
        Compute the probability of selling at each price point.
        This is based on the current valuations and the prices.
        """
        sell_probabilities = np.array([sum(p <= self.valuations)/self.T for p in self.prices])
        return sell_probabilities

def compute_clairvoyant(sell_probabilities, prices, rho):
    """
    Compute the clairvoyant solution given the sell probabilities and prices.
    This is a greedy approach that selects the best price points based on expected revenue.
    Returns:
        expected_clairvoyant_utility : expected revenue per round under the clairvoyant policy,
        gamma                        : the probability distribution over prices,
        expected_cost                : expected units sold per round.
    """
    c = -(prices)*sell_probabilities
    A_ub = [sell_probabilities]
    b_ub = [rho]
    A_eq = [np.ones(len(prices))]
    b_eq = [1]
    
    res = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0,1))
    gamma = res.x
    expected_clairvoyant_utility = -res.fun
    expected_cost = np.sum(sell_probabilities * gamma)
    
    return expected_clairvoyant_utility, gamma, expected_cost
    

class HedgeAgent:
    def __init__(self, K, learning_rate, rng):
        self.K = K
        self.learning_rate = learning_rate
        self.weights = np.ones(K)
        self.x_t = np.ones(K)/K
        self.a_t = None
        self.rng = rng if rng is not None else np.random
        self.t = 0

    def pull_arm(self):
        self.x_t = self.weights/sum(self.weights)
        self.a_t = self.rng.choice(np.arange(self.K), p=self.x_t)
        return self.a_t
    
    def update(self, l_t):
        self.weights *= np.exp(-self.learning_rate*l_t)
        self.t += 1

import numpy as np

class FFPrimalDualPricingAgent:
    def __init__(self, prices, T, B, rng  ,eta):
        """
        prices: array of allowed prices (arms)
        T: time horizon
        inventory: total units you can sell
        eta: dual‐step size
        """
        self.prices = np.array(prices)
        self.K = len(prices)
        self.T = T
        self.inventory = B
        self.eta = eta
        # Use provided rng or default to np.random
        self.rng = rng if rng is not None else np.random

        # Hedge over K arms, η = sqrt(log K / T)
        self.hedge = HedgeAgent(self.K, np.sqrt(np.log(self.K) / T),rng=self.rng)

        # pacing rate ρ = inventory / T
        self.rho = B / T

        # Lagrange multiplier
        self.lmbd = 1.0

        # book‐keeping
        self.t = 0
        self.pull_counts = np.zeros(self.K, int)
        self.last_arm = None

    def pull_arm(self):
        if self.inventory < 1:
            self.last_arm = None
            return self.last_arm
        self.last_arm = self.hedge.pull_arm()
        return self.last_arm

    def update(self, v_t):
        """
        v_t: the buyer’s valuation this round (full feedback)
        """
        # --- 1) full‐feedback sales mask & revenues for all arms
        sale_mask = (self.prices <= v_t).astype(float)
        f_full = self.prices * sale_mask       # revenue if that price were chosen
        c_full = sale_mask                     # units sold (0/1)

        # --- 2) Lagrangian payoff L_i = f_i − λ (c_i − ρ)
        L = f_full - self.lmbd * (c_full - self.rho)

        # --- 3) find extreme possible L values for normalization
        f_max = self.prices.max()
        # worst case consumption shift is −ρ (when c_i=0) → L_up
        L_up  = f_max   - self.lmbd * (0    - self.rho)
        # best case consumption shift is +(1−ρ) (when c_i=1) → L_low
        L_low = 0.0     - self.lmbd * (1.0  - self.rho)

        # --- 4) rescale L into [0,1] and convert to losses
        rescaled = (L - L_low) / (L_up - L_low + 1e-12)
        losses   = 1.0 - rescaled

        # --- 5) Hedge update (minimize these losses)
        self.hedge.update(losses)

        # --- 6) actual consumption & inventory update
        if self.last_arm is not None:
            c_t = 1 if self.prices[self.last_arm] <= v_t else 0
            f_t = self.prices[self.last_arm] * c_t
            
            self.inventory -= c_t
            self.pull_counts[self.last_arm] += 1
        else:
            c_t = 0

        # --- 7) dual‐step on λ:  λ ← [λ − η(ρ − c_t)]₊
        self.lmbd = np.clip(self.lmbd - self.eta * (self.rho - c_t),
                            a_min=0.0,
                            a_max=1.0/self.rho)

        self.t += 1
        return f_t, c_t  # return revenue and units sold (0/1)



if __name__ == "__main__":
    # Parameters
    

    

    
    
    # Example parameters
    prices = np.array([0.2, 0.256, 0.311, 0.367, 0.422, 0.478, 0.533, 0.589, 0.644, 0.7, 0.756, 0.811, 0.867, 0.922, 0.98, 1.001]) #1 is dummy
    
    T = 80_000
    B = 20_000
    seed = 17
    shock_prob = 0.50
    freq = 100
    
    
    
    '''
    plt.figure()
    plt.plot(range(T), env.vals)
    plt.xlabel('Time step t')
    plt.ylabel('Valuation v_t')
    plt.title('Time-varying valuations with occasional adversarial shocks')
    plt.show()
    '''
    
    

    
    
    n_trials = 5
    # simulate
    
    
    all_regrets = []
    all_units_sold = []
    
    final_reward = []
    for trial in range(n_trials):
        rng = np.random.RandomState(seed + trial)
        env = BudgetedPricingEnvironment(prices, T, shock_prob, freq,
                                    rng=rng)
        sell_probabilities = env.compute_sell_probabilities()
        expected_clairvoyant_utility , gamma , expected_cost_round = compute_clairvoyant(sell_probabilities,prices,B/T)
        print(f"Expected clairvoyant utility: {expected_clairvoyant_utility:.4f}, gamma: {gamma} , expected cost per round: {expected_cost_round:.4f}")
        
        best_prices_sequence , _ = compute_clairvoyant_sequence(env.valuations, prices, B)
        agent = FFPrimalDualPricingAgent(prices, T, B, rng = rng , eta= 1/np.sqrt(T))

        regrets = []
        units_sold = []
        cum_reward = 0.0
        cum_regret = 0.0
        cum_unit_sold = 0
        for t in range(T):
            arm = agent.pull_arm()
            if arm is None:
                print(f"Trial {trial+1}: Budget exhausted at round {t}.")
                break
                
            valuation = env.full_feedback_round()
            reward , sold = agent.update(valuation)

            cum_reward += reward
            # instantaneous regret = clairvoyant reward − actual
            instant_regret = expected_clairvoyant_utility - reward
            
            #instant_regret = best_prices_sequence[t] - reward
           
            
            cum_regret += instant_regret
            cum_unit_sold += sold
            regrets.append(cum_regret)
            units_sold.append(cum_unit_sold)
            

        all_regrets.append(regrets)
        all_units_sold.append(units_sold)
        final_reward.append(cum_reward)

    
    min_rounds = min(len(reg) for reg in all_regrets)
    all_regrets_fixed = np.array([reg[:min_rounds] for reg in all_regrets])
    all_units_sold_fixed = np.array([us[:min_rounds] for us in all_units_sold])

    avg_regret = all_regrets_fixed.mean(axis=0)
    sd_regret = all_regrets_fixed.std(axis=0)

    avg_units_sold = all_units_sold_fixed.mean(axis=0)
    sd_units_sold = all_units_sold_fixed.std(axis=0)

    # plot cumulative regret using min_rounds along x-axis
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(avg_regret, label="Average Cumulative Regret")
    plt.fill_between(
        np.arange(min_rounds),
        avg_regret - sd_regret / np.sqrt(n_trials),
        avg_regret + sd_regret / np.sqrt(n_trials),
        alpha=0.3,
        label="±1 SE"
    )
    plt.xlabel("t")
    plt.ylabel("Cumulative Regret")
    plt.title("UCB1 Pricing: Cumulative Regret")
    plt.legend()

    # plot cumulative units sold using min_rounds
    plt.subplot(1, 2, 2)
    import matplotlib.ticker as ticker
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    plt.plot(avg_units_sold, label="Average Cumulative Units Sold")
    plt.fill_between(
        np.arange(min_rounds),
        avg_units_sold - sd_units_sold / np.sqrt(n_trials),
        avg_units_sold + sd_units_sold / np.sqrt(n_trials),
        alpha=0.3,
        label="±1 SE"
    )
    plt.xlabel("t")
    plt.ylabel("Cumulative Units Sold")
    plt.title("Cumulative Units Sold Over Time")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Print average cumulative units sold at the final round
    print(f"Average cumulative units sold: {avg_units_sold[-1]:.2f}")

    print("\nFinal Results:")
    print(f"Average regret per round: {avg_regret[-1]/T:.4f}")
    print("Pull counts:", agent.pull_counts)
    final_reward = np.array(final_reward)
    print("Average cum reward:", np.mean(final_reward))
    basline_reward = expected_clairvoyant_utility * T
    print(f"Baseline reward: {basline_reward:.2f}")

