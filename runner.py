import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any, Union
import time
from enum import Enum


class Distribution(Enum):
    NORMAL = "normal"
    UNIFORM = "uniform"
    EXPONENTIAL = "exponential"
    BETA_SINUSOIDAL = "beta"
    UNIFORM_SINUSOIDAL = "sinusoidal"
    PIECEWISE_BETA = "piecewise_beta"
    SIMPLE_TV = "simple_tv"
    PIECEWISE_TV = "piecewise_tv"
    PIECEWISE_SINUSOIDAL = "piecewise_sinusoidal"
    SMOOTH = "smooth"


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
    distribution: Union[Distribution,
                        List[Distribution]] = Distribution.UNIFORM
    budget: float = 1000.0
    adaptive_rho: bool = False

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

        self.env = self.create_environment(trial_seed)
        self.agent = self.create_agent()

        regrets = []
        units_sold = []
        cum_reward = 0.0
        cum_regret = 0.0
        cum_units = 0.0

        clairvoyant_reward = self.compute_clairvoyant_reward()

        for t in range(self.config.horizon):
            action = self.agent.pull_arm()

            if action is None:
                print(f"Trial {trial+1}: Agent stopped at round {t}.", end=" ")
                break

            env_result = self.env.round(action)

            if isinstance(env_result, tuple):
                agent_result = self.agent.update(*env_result)
            else:
                agent_result = self.agent.update(env_result)

            if agent_result is not None:
                reward, cost = self.extract_metrics(agent_result)
            else:
                reward, cost = self.extract_metrics(env_result)

            cum_reward += reward
            cum_units += cost

            instant_regret = clairvoyant_reward - reward
            cum_regret += instant_regret

            regrets.append(cum_regret)
            units_sold.append(int(cum_units))

        return regrets, units_sold, cum_reward, self.agent

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


class MultiDistributionRunner:
    """Wrapper for running experiments with multiple distributions"""

    def __init__(self, runner_class, config: ExperimentConfig):
        self.runner_class = runner_class
        self.config = config
        self.results = {}
        self.distribution_names = []

    def run_experiment(self):
        """Run experiments for all distributions"""
        # Check if distribution is a list
        if isinstance(self.config.distribution, list):
            distributions = self.config.distribution
        else:
            distributions = [self.config.distribution]

        # Create names for distributions
        distribution_names = []
        for dist in distributions:
            if dist == Distribution.UNIFORM:
                distribution_names.append("Uniform")
            elif dist == Distribution.BETA_SINUSOIDAL:
                distribution_names.append("Beta")
            elif dist == Distribution.NORMAL:
                distribution_names.append("Normal")
            elif dist == Distribution.EXPONENTIAL:
                distribution_names.append("Exponential")
            elif dist == Distribution.UNIFORM_SINUSOIDAL:
                distribution_names.append("Sinusoidal")
            elif dist == Distribution.PIECEWISE_BETA:
                distribution_names.append("Piecewise Beta")
            elif dist == Distribution.SIMPLE_TV:
                distribution_names.append("Simple TV")
            elif dist == Distribution.PIECEWISE_TV:
                distribution_names.append("Piecewise TV")
            elif dist == Distribution.PIECEWISE_SINUSOIDAL:
                distribution_names.append("Piecewise Sinusoidal")
            elif dist == Distribution.SMOOTH:
                distribution_names.append("Smooth")
            else:
                distribution_names.append(str(dist))

        self.distribution_names = distribution_names

        # Run experiment for each distribution
        for dist, name in zip(distributions, distribution_names):
            print(f"\n=== Eseguendo esperimento con distribuzione {name} ===")

            single_config = ExperimentConfig(
                task_name=f"{self.config.task_name} ({name})",
                horizon=self.config.horizon,
                n_trials=self.config.n_trials,
                seed=self.config.seed,
                prices=self.config.prices,
                n_products=self.config.n_products,
                n_windows=self.config.n_windows,
                distribution=dist,
                budget=self.config.budget,
                adaptive_rho=self.config.adaptive_rho
            )

            runner = self.runner_class(single_config)
            result = runner.run_experiment()
            self.results[name] = result

            print(f"Esperimento {name} completato")

        return self.results

    def plot_comparison(self, show_units: bool = False, show_budget: bool = True):
        """Plot comparison of all distributions using the same style as StandardPlotter"""
        import matplotlib.pyplot as plt

        if not self.results:
            print("Nessun risultato da plottare. Esegui prima run_experiment().")
            return

        # Find minimum length across all trials and distributions
        min_rounds = 0
        for result in self.results.values():
            trial_min = min(len(regrets) for regrets in result.regrets)
            if min_rounds == 0:
                min_rounds = trial_min
            else:
                min_rounds = min(min_rounds, trial_min)

        # Create plots with same style as StandardPlotter
        fig_width = 16 if show_units else 12
        fig, axes = plt.subplots(
            1, 2 if show_units else 1, figsize=(fig_width, 6))

        if not show_units:
            axes = [axes]

        # Colors for different distributions
        colors = ['blue', 'red', 'green', 'orange',
                  'purple', 'brown', 'pink', 'gray']

        # Plot regret for all distributions
        for i, (name, result) in enumerate(self.results.items()):
            color = colors[i % len(colors)]

            # Truncate to minimum length and convert to arrays (same as StandardPlotter)
            truncated_regrets = np.array(
                [regrets[:min_rounds] for regrets in result.regrets])

            # Compute statistics (same as StandardPlotter)
            avg_regret = truncated_regrets.mean(axis=0)
            std_regret = truncated_regrets.std(axis=0)

            # Plot with same style as StandardPlotter
            axes[0].plot(avg_regret, color=color, linewidth=2,
                         label=f"{name} Distribution")

            # Add error bands if multiple trials (same as StandardPlotter)
            if result.config.n_trials > 1:
                se_regret = std_regret / np.sqrt(result.config.n_trials)
                axes[0].fill_between(
                    range(min_rounds),
                    avg_regret - se_regret,
                    avg_regret + se_regret,
                    alpha=0.3,
                    color=color
                )

        # Format regret plot exactly like StandardPlotter
        axes[0].set_xlabel("Round")
        axes[0].set_ylabel("Cumulative Regret")
        axes[0].set_title(f"{self.config.task_name}: Cumulative Regret")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Units sold plot (if requested)
        if show_units:
            for i, (name, result) in enumerate(self.results.items()):
                color = colors[i % len(colors)]

                # Truncate units to minimum length
                truncated_units = np.array(
                    [units[:min_rounds] for units in result.units_sold])
                avg_units = truncated_units.mean(axis=0)

                axes[1].plot(avg_units, color=color, linewidth=2,
                             label=f"{name} Distribution")

            # Format units plot exactly like StandardPlotter
            axes[1].set_xlabel("Round")
            axes[1].set_ylabel("Cumulative Units Sold")
            axes[1].set_title(f"{self.config.task_name}: Units Sold")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            # Add budget line if available (same as StandardPlotter)
            if self.config.budget is not None and show_budget:
                axes[1].axhline(y=self.config.budget, color='r', linestyle='--',
                                label=f"Budget: {self.config.budget}")
                axes[1].legend()

        plt.tight_layout()
        plt.show()

        return min_rounds

    def print_analysis(self):
        """Print detailed analysis for all distributions"""
        print(f"\n=== ANALISI DETTAGLIATA ===")
        for name, result in self.results.items():
            print(f"\n--- Distribuzione {name} ---")
            from plotter import StandardAnalyzer
            analysis = StandardAnalyzer.analyze_results(result)
            StandardAnalyzer.print_analysis(result, analysis)
            if hasattr(result.final_agents[0], 'average_rewards'):
                print(
                    f"Empirical average rewards: {np.round(result.final_agents[0].average_rewards, 4)}")

        # Confronto finale
        print(f"\n=== CONFRONTO FINALE ===")
        for name, result in self.results.items():
            final_regret = np.mean([regrets[-1]
                                   for regrets in result.regrets if regrets])
            final_reward = np.mean(result.final_rewards)
            print(
                f"{name}: Final Average Regret = {final_regret:.4f}, Final Average Reward = {final_reward:.4f}")

    def plot_arm_distributions(self):
        """Plot arm distribution for each distribution"""
        from plotter import StandardPlotter

        for name, result in self.results.items():
            try:
                if hasattr(result.final_agents[0], 'N_pulls') and isinstance(result.final_agents[0].N_pulls, list):
                    # Multi-product
                    price_grid = [np.concatenate(
                        [self.config.prices, [1.001]]) for _ in range(self.config.n_products)]
                    StandardPlotter.plot_arm_distribution(
                        result.final_agents[0],
                        price_grid,
                        f"Arm Distribution - {name}"
                    )
                else:
                    # Singolo prodotto
                    StandardPlotter.plot_arm_distribution(
                        result.final_agents[0],
                        self.config.prices,
                        f"Arm Distribution - {name}"
                    )
            except Exception as e:
                print(f"Errore nel plottare arm distribution per {name}: {e}")


print("Experiment framework created successfully!")
