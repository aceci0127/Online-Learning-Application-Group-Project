import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize

def pattern(t, T, freq):
    """Time-varying upper bound: 1 - |sin(freq * t / T)|."""
    return 1 - abs(np.sin(freq * t / T))

def generate_valuations(T, shock_prob, freq):
    """
    Generate a sequence of T valuations with occasional adversarial shocks.
    
    - a(t) = pattern(t, T, freq)
    - With probability shock_prob, v_t is either 0 or a(t) (extreme).
    - Otherwise v_t ~ Uniform(0, a(t)).
    """
    valuations = np.empty(T)
    for t in range(T):
        a_t = pattern(t, T, freq)
        if np.random.rand() < shock_prob:
            valuations[t] = np.random.choice([0.0, a_t])
        else:
            valuations[t] = np.random.uniform(0, a_t)
    return valuations

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
        self.vals = generate_valuations(T, self.shock_prob, self.freq)
    
    def round(self, price_index):
        p = self.prices[price_index]
        sale = self.vals[self.t] >= p
        reward = p if sale else 0.0
        cost = 1.0 if sale else 0.0
        self.t += 1
        return reward, cost
    def compute_sell_probabilities(self):
        """
        Compute the probability of selling at each price point.
        This is based on the current valuations and the prices.
        """
        sell_probabilities = np.array([sum(p <= self.vals)/self.T for p in self.prices])
        return sell_probabilities

def compute_clairvoyant(sell_probabilities, prices, rho):
    """
    Compute the clairvoyant solution given the sell probabilities and prices.
    This is a greedy approach that selects the best price points based on expected revenue.
    """
    
    ## Linear Program
    c = -(prices)*sell_probabilities
    A_ub = [sell_probabilities]
    b_ub = [rho]
    A_eq = [np.ones(len(prices))]
    b_eq = [1]
    res = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0,1))
    gamma = res.x
    expected_clairvoyant_utility = -res.fun
    return expected_clairvoyant_utility, gamma
    





# Parameters
T = 1000
shock_prob = 0.05
freq = 8

# Generate and plot
values = generate_valuations(T, shock_prob, freq)

plt.figure()
plt.plot(range(T), values)
plt.xlabel('Time step t')
plt.ylabel('Valuation v_t')
plt.title('Time-varying valuations with occasional adversarial shocks')
plt.show()


