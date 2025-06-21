import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class Environment:
    def __init__(self):
        pass

    def round(self, a_t):
        pass

class PricingEnvironment(Environment):
    def __init__(self, prices, T):
        self.prices = prices  # Array of allowed prices P = {p1, p2, ..., pm}
        self.m = len(prices)  # Number of price arms
        self.valuations = np.random.normal(0.8, 0.2, size=T)
        self.valuations = np.clip(self.valuations, 0, 1)  # Ensure valuations are in [0, 1]
        self.t = 0

    def round(self, arm_index):
        # arm_index is the index of the chosen price in self.prices
        chosen_price = self.prices[arm_index]
        # Revenue is chosen_price if buyer accepts (v_t >= price), 0 otherwise
        revenue = chosen_price if self.valuations[self.t] >= chosen_price else 0
        self.t += 1
        return revenue

class Agent:
    def __init__(self):
        pass
    def pull_arm(self):
        pass
    def update(self, r_t):
        pass

class UCB1PricingAgent(Agent):
    def __init__(self, prices, T):
        self.prices = prices  # Array of allowed prices
        self.m = len(prices)  # Number of price arms
        self.T = T
        self.t = 0  # Current round (0-indexed)
        
        # UCB1 state variables
        self.N_pulls = np.zeros(self.m)  # n_i(t): number of times each price tried
        self.cumulative_revenue = np.zeros(self.m)  # S_i(t): cumulative revenue per price
        self.average_rewards = np.zeros(self.m)  # mu_hat_i(t): empirical average reward
        
        self.chosen_arm = None
    
    def pull_arm(self):
        # Initialization phase: try each price once
        if self.t < self.m:
            self.chosen_arm = self.t
        else:
            # Compute UCB values for each arm
            ucb_values = np.zeros(self.m)
            for i in range(self.m):
                if self.N_pulls[i] > 0:
                    confidence_width = np.sqrt(2 * np.log(self.T) / self.N_pulls[i])
                    ucb_values[i] = self.average_rewards[i] + confidence_width
                else:
                    ucb_values[i] = float('inf')  # Unvisited arms get highest priority
            
            # Choose arm with highest UCB value
            self.chosen_arm = np.argmax(ucb_values)
        
        return self.chosen_arm
    
    def update(self, revenue):
        # Update statistics for the chosen arm
        arm = self.chosen_arm
        self.N_pulls[arm] += 1
        self.cumulative_revenue[arm] += revenue
        self.average_rewards[arm] = self.cumulative_revenue[arm] / self.N_pulls[arm]
        self.t += 1

# Compute expected revenues for each price
def compute_expected_revenues(prices, mu=0.8, sigma=0.2, lower=0, upper=1):
    """
    Compute expected revenue for each price given truncated normal valuations.
    
    Args:
        prices: array of candidate prices
        mu, sigma: parameters of the underlying normal distribution
        lower, upper: truncation bounds
    """
    expected_revenues = []
    
    for p in prices:
        # For truncated normal, P(V >= p) = (1 - CDF_truncated(p))
        # Use scipy.stats.truncnorm for truncated normal distribution
        
        # Standardize the bounds for truncnorm
        a = (lower - mu) / sigma  # standardized lower bound
        b = (upper - mu) / sigma  # standardized upper bound
        
        # Create truncated normal distribution
        truncated_norm = stats.truncnorm(a, b, loc=mu, scale=sigma)
        
        # P(V >= p) = 1 - CDF(p)
        prob_accept = 1 - truncated_norm.cdf(p)
        
        # Expected revenue = price × probability of acceptance
        expected_revenue = p * prob_accept
        expected_revenues.append(expected_revenue)
    
    return np.array(expected_revenues)

prices = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
m = len(prices)
T = 10000
seed = 18

expected_revenues = compute_expected_revenues(prices)
best_price_index = np.argmax(expected_revenues)
best_price = prices[best_price_index]

print(f'Prices: {prices}')
print(f'Expected revenues: {expected_revenues}')
print(f'Best price: {best_price} (index {best_price_index})')

expected_clairvoyant_rewards = np.repeat(expected_revenues[best_price_index], T)

n_trials = 10
regret_per_trial = []

np.random.seed(seed)
for trial in range(n_trials):
    env = PricingEnvironment(prices, T)
    ucb_agent = UCB1PricingAgent(prices, T)

    agent_rewards = np.array([])

    for t in range(T):
        a_t = ucb_agent.pull_arm()
        r_t = env.round(a_t)
        ucb_agent.update(r_t)

        agent_rewards = np.append(agent_rewards, r_t)

    cumulative_regret = np.cumsum(expected_clairvoyant_rewards - agent_rewards)
    regret_per_trial.append(cumulative_regret)

regret_per_trial = np.array(regret_per_trial)

average_regret = regret_per_trial.mean(axis=0)
regret_sd = regret_per_trial.std(axis=0)

plt.figure(figsize=(12, 4))

# Plot cumulative regret
plt.subplot(1, 2, 1)
plt.plot(np.arange(T), average_regret, label='Average Regret')
plt.title('Cumulative Regret of UCB1 Pricing')
plt.fill_between(np.arange(T),
                average_regret - regret_sd / np.sqrt(n_trials),
                average_regret + regret_sd / np.sqrt(n_trials),
                alpha=0.3,
                label='Uncertainty')
plt.xlabel('$t$')
plt.ylabel('Cumulative Regret')
plt.legend()

# Plot number of pulls per price
plt.subplot(1, 2, 2)
price_labels = [f'{p:.1f}' for p in prices]
plt.barh(y=price_labels, width=ucb_agent.N_pulls)
plt.title('Number of Pulls per Price (UCB1 Pricing)')
plt.xlabel('Number of Pulls')
plt.ylabel('Price')

plt.tight_layout()
plt.show()

print(f"\nFinal Results:")
print(f"Average regret per round: {average_regret[-1]/T:.4f}")
print(f"Final empirical average revenues: {ucb_agent.average_rewards}")
print(f"Number of times each price was chosen: {ucb_agent.N_pulls}")
print(f"True expected revenues: {expected_revenues}")