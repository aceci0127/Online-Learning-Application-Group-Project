import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any
import time
from enum import Enum

class Distribution(Enum):
    NORMAL = "normal"
    UNIFORM = "uniform"
    EXPONENTIAL = "exponential"
    BETA = "beta"

@dataclass
class ExperimentConfig:
    """Standard configuration for all experiments"""
    task_name: str
    horizon: int
    n_trials: int
    seed: int
    prices: np.ndarray
    n_products: int = 1
    n_windows: int = 1
    distribution: Distribution = Distribution.UNIFORM
    budget: Optional[float] = None

    def __post_init__(self):
        pass


@dataclass
class ExperimentResult:

    """Standard result container for all experiments"""
    config: ExperimentConfig
    regrets: List[List[float]]
    units_sold: List[List[int]]
    final_rewards: List[float]
    execution_time: float
    clairvoyant_reward: Optional[float] = None
    final_agents: List = field(default_factory=list)

    def __post_init__(self):
        pass


class StandardExperimentRunner:
    """Standard experiment runner that works with any environment/agent combination"""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def create_environment(self, trial_seed: int):
        """Factory method for creating environments - override for specific tasks"""
        raise NotImplementedError(
            "Subclasses must implement create_environment")

    def create_agent(self):
        """Factory method for creating agents - override for specific tasks"""
        raise NotImplementedError("Subclasses must implement create_agent")

    def compute_clairvoyant_reward(self) -> float:
        """Compute clairvoyant reward - override for specific tasks"""
        raise NotImplementedError(
            "Subclasses must implement compute_clairvoyant_reward")

    def extract_metrics(self, result) -> Tuple[float, float]:
        """Extract reward and cost from environment result - override for specific tasks"""
        if isinstance(result, tuple) and len(result) >= 2:
            reward_raw = result[0]
            cost_raw = result[1]
        elif isinstance(result, tuple) and len(result) == 1:
            reward_raw = result[0]
            cost_raw = 1.0
        else:
            reward_raw = result
            cost_raw = 1.0

        reward = float(np.sum(reward_raw))
        cost = float(np.sum(cost_raw))

        return reward, cost

    def run_single_trial(self, trial: int) -> Tuple[List[float], List[int], float, Any]:
        """Run a single trial and return regrets, units sold, final reward, and final agent"""
        trial_seed = self.config.seed + trial
        np.random.seed(trial_seed)

        env = self.create_environment(trial_seed)
        agent = self.create_agent()

        regrets = []
        units_sold = []
        cum_reward = 0.0
        cum_regret = 0.0
        cum_units = 0.0

        clairvoyant_reward = self.compute_clairvoyant_reward()

        for t in range(self.config.horizon):
            action = agent.pull_arm()

            if action is None:
                print(f"Trial {trial+1}: Agent stopped at round {t}.", end=" ")
                break

            result = env.round(action)

            if isinstance(result, tuple):
                agent.update(*result)
            else:
                agent.update(result)

            reward, cost = self.extract_metrics(result)

            cum_reward += reward
            cum_units += cost

            instant_regret = clairvoyant_reward - reward
            cum_regret += instant_regret

            regrets.append(cum_regret)
            units_sold.append(int(cum_units))

        return regrets, units_sold, cum_reward, agent

    def run_experiment(self) -> ExperimentResult:
        """Run the complete experiment"""
        print(f"Running {self.config.task_name}")
        print(
            f"Horizon: {self.config.horizon}, Trials: {self.config.n_trials}")

        start_time = time.time()

        all_regrets = []
        all_units_sold = []
        final_rewards = []
        final_agents = []

        for trial in range(self.config.n_trials):
            print(f"Trial {trial+1}/{self.config.n_trials}...", end=" ")
            regrets, units_sold, final_reward, agent = self.run_single_trial(
                trial)
            all_regrets.append(regrets)
            all_units_sold.append(units_sold)
            final_rewards.append(final_reward)
            final_agents.append(agent)
            print("✓")

        execution_time = time.time() - start_time

        result = ExperimentResult(
            config=self.config,
            regrets=all_regrets,
            units_sold=all_units_sold,
            final_rewards=final_rewards,
            execution_time=execution_time,
            clairvoyant_reward=self.compute_clairvoyant_reward(),
            final_agents=final_agents
        )

        return result


print("Experiment framework created successfully!")
