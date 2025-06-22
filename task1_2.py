import numpy as np
from scipy.stats import truncnorm
from scipy import optimize
import matplotlib.pyplot as plt


class BudgetedPricingEnvironment:
    def __init__(self, prices, T, rng=None):
        self.prices = np.array(prices)
        self.T = T
        self.t = 0
        
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng
        '''
        # truncated normal valuations
        a, b = (0 - mu) / sigma, (1 - mu) / sigma
        self.vals = truncnorm(a, b, loc=mu, scale=sigma).rvs(size=T, random_state=rng)
        '''
        self.vals = rng.uniform(0, 1, size=T)  # uniform valuations for simplicity
    
    def round(self, price_index):
        p = self.prices[price_index]
        sale = self.vals[self.t] >= p
        reward = p if sale else 0.0
        cost = 1.0 if sale else 0.0
        self.t += 1
        return reward, cost

class ConstrainedUCBPricingAgent:
    def __init__(self, K, B, T, range=1):
        self.K = K
        self.T = T
        self.range = range
        self.a_t = None # it's an index, not the actual bid
        self.avg_f = np.zeros(K)
        self.avg_c = np.zeros(K)
        self.N_pulls = np.zeros(K)
        self.budget = B
        self.rho = B/T
        self.t = 0
    
    def pull_arm(self):
        if self.budget < 1:
            print("Budget exhausted, cannot pull any more arms.")
            self.a_t = None
            return None
        if self.t < self.K:
            self.a_t = self.t 
        else:
            f_ucbs = self.avg_f + self.range*np.sqrt(2*np.log(self.T)/self.N_pulls)
            c_lcbs = self.avg_c - self.range*np.sqrt(2*np.log(self.T)/self.N_pulls)
            gamma_t = self.compute_opt(f_ucbs, c_lcbs)
            self.a_t = np.random.choice(self.K, p=gamma_t)
        return self.a_t

    def compute_opt(self, f_ucbs, c_lcbs):
        if np.sum(c_lcbs <= np.zeros(len(c_lcbs))):
            gamma = np.zeros(len(f_ucbs))
            gamma[np.argmax(f_ucbs)] = 1
            return gamma
        c = -f_ucbs
        A_ub = [c_lcbs]
        b_ub = [self.rho]
        A_eq = [np.ones(self.K)]
        b_eq = [1]
        res = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0,1))
        gamma = res.x
        return gamma
    
    def update(self, f_t, c_t):
        self.N_pulls[self.a_t] += 1
        self.avg_f[self.a_t] += (f_t - self.avg_f[self.a_t])/self.N_pulls[self.a_t]
        self.avg_c[self.a_t] += (c_t - self.avg_c[self.a_t])/self.N_pulls[self.a_t]
        self.budget -= c_t
        self.t += 1

import numpy as np
from scipy.stats import truncnorm


## Linear Program
def compute_clairvoyant(available_prices, rho, sell_probabilities):
    c = -(available_prices)*sell_probabilities
    A_ub = [available_prices*sell_probabilities]
    b_ub = [rho]
    A_eq = [np.ones(len(available_prices))]
    b_eq = [1]
    res = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0,1))
    gamma = res.x
    
    return gamma, -res.fun



    

if __name__ == "__main__":
    # Example parameters
    prices = np.array([0.2, 0.5,0.6, 0.8])
    T = 10_000
    B = 5_000
    seed = 18

    sell_probabilities = 1-prices
    print("sell_probabilities:", sell_probabilities)
    expected_reward = prices * sell_probabilities
    print("expected_reward:", expected_reward)

    gamma, expected_clairvoyant_utility = compute_clairvoyant(prices,B/T,sell_probabilities)
    print(gamma)
    n_trials = 10
    # simulate
    np.random.seed(seed)
    all_regrets = []
    all_units_sold = []
    for trial in range(n_trials):
        # new RNG per trial for both env and reproducibility
        rng = np.random.RandomState(seed + trial)
        env = BudgetedPricingEnvironment(prices, T, rng=rng)
        agent = ConstrainedUCBPricingAgent(len(prices), B, T, range=1)

        regrets = []
        units_sold = []
        cum_regret = 0.0
        cum_unit_sold = 0
        for t in range(T):
            arm = agent.pull_arm()
            if arm is None:
                break
            reward,sold = env.round(arm)
            agent.update(reward,sold)

            # instantaneous regret = clairvoyant reward − actual
            instant_regret = expected_clairvoyant_utility - reward
            cum_regret += instant_regret
            cum_unit_sold += sold
            regrets.append(cum_regret)
            units_sold.append(cum_unit_sold)
            

        all_regrets.append(regrets)
        all_units_sold.append(units_sold)

    '''
    min_rounds = min(len(reg) for reg in all_regrets)
    all_regrets_fixed = np.array([reg[:min_rounds] for reg in all_regrets])
    all_units_sold_fixed = np.array([us[:min_rounds] for us in all_units_sold])

    avg_regret = all_regrets_fixed.mean(axis=0)
    sd_regret = all_regrets_fixed.std(axis=0)

    avg_units_sold = all_units_sold_fixed.mean(axis=0)
    sd_units_sold = all_units_sold_fixed.std(axis=0)
    '''
    all_regrets = np.array(all_regrets)
    all_units_sold = np.array(all_units_sold)
    
    avg_regret = all_regrets.mean(axis=0)
    sd_regret = all_regrets.std(axis=0)
    
    avg_units_sold = all_units_sold.mean(axis=0)
    sd_units_sold = all_units_sold.std(axis=0)

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

    # plot cumulative units sold
    import matplotlib.ticker as ticker

    plt.figure(figsize=(12, 4))
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    plt.plot(avg_units_sold, label="Average Cumulative Units Sold")
    plt.fill_between(
        np.arange(len(avg_units_sold)),
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
    print("Pull counts:", agent.N_pulls)


