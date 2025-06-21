import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

class Environment:
    def round(self, a_t):
        raise NotImplementedError

class PricingEnvironment(Environment):
    def __init__(self, prices, T, rng=None, mu=0.8, sigma=0.2):
        """
        prices: array of allowed prices
        T: horizon
        rng: numpy RandomState or Generator for reproducibility
        mu, sigma: parameters of the underlying normal before truncation
        """
        self.prices = np.array(prices)
        self.m = len(prices)
        self.T = T
        self.t = 0

        # set up RNG
        if rng is None:
            rng = np.random.default_rng()
        self._rng = rng

        # compute truncation bounds for scipy’s truncnorm
        a, b = (0 - mu) / sigma, (1 - mu) / sigma
        # draw truly truncated Normal(mu, sigma^2) on [0,1]
        self.valuations = truncnorm(a, b, loc=mu, scale=sigma).rvs(
            size=T, random_state=rng
        )

    def round(self, arm_index):
        chosen_price = self.prices[arm_index]
        v = self.valuations[self.t]
        revenue = chosen_price if v >= chosen_price else 0.0
        self.t += 1
        return revenue

class Agent:
    def pull_arm(self):
        raise NotImplementedError
    def update(self, r_t):
        raise NotImplementedError

class UCB1PricingAgent(Agent):
    def __init__(self, prices, T):
        self.prices = np.array(prices)
        self.m = len(prices)
        self.T = T
        self.t = 0  # rounds completed

        # statistics
        self.N_pulls = np.zeros(self.m, dtype=int)
        self.cumulative_revenue = np.zeros(self.m, dtype=float)
        self.average_rewards = np.zeros(self.m, dtype=float)
        self.chosen_arm = None

    def pull_arm(self):
        # explore each arm once
        if self.t < self.m:
            self.chosen_arm = self.t
        else:
            ucb_vals = np.zeros(self.m)
            for i in range(self.m):
                if self.N_pulls[i] > 0:
                    # use log(self.t) so bonus shrinks over time
                    bonus = np.sqrt(2 * np.log(self.t) / self.N_pulls[i])
                    ucb_vals[i] = self.average_rewards[i] + bonus
                else:
                    ucb_vals[i] = np.inf
            self.chosen_arm = np.argmax(ucb_vals)
        return self.chosen_arm

    def update(self, revenue):
        arm = self.chosen_arm
        self.N_pulls[arm] += 1
        self.cumulative_revenue[arm] += revenue
        self.average_rewards[arm] = self.cumulative_revenue[arm] / self.N_pulls[arm]
        self.t += 1

def compute_expected_revenues(prices, mu=0.8, sigma=0.2, lower=0., upper=1.):
    """
    Compute E[p * 1{V>=p}] for V ~ TruncNorm(mu, sigma^2) on [lower, upper].
    """
    a, b = (lower - mu) / sigma, (upper - mu) / sigma
    dist = truncnorm(a, b, loc=mu, scale=sigma)
    revs = []
    for p in prices:
        prob_accept = 1.0 - dist.cdf(p)
        revs.append(p * prob_accept)
    return np.array(revs)


if __name__ == "__main__":
    # parameters
    prices = [0.2 , 0.3 , 0.4 , 0.6, 0.7, 0.8]
    T = 100_000
    seed = 18
    n_trials = 10

    # analytic benchmark (clairvoyant)
    expected_revenues = compute_expected_revenues(prices)
    best_idx = np.argmax(expected_revenues)
    best_price = prices[best_idx]
    print("Prices:", prices)
    print("Expected revenues:", np.round(expected_revenues, 6))
    print(f"Clairvoyant best price: {best_price} (idx {best_idx})\n")

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
