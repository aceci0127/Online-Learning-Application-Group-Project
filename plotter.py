import numpy as np
import matplotlib.pyplot as plt
from runner import ExperimentResult
from numpy.typing import NDArray
from typing import List, Dict, Any


class StandardPlotter:
    """Standard plotting utilities for all experiments"""

    @staticmethod
    def plot_arm_distribution(agent, prices: NDArray[np.float64] | List[NDArray[np.float64]], title: str = "Arm Distribution"):
        """Plot the distribution of arm pulls"""
        if hasattr(agent, 'N_pulls'):
            plt.figure(figsize=(10, 6))
            if isinstance(agent.N_pulls, np.ndarray) and agent.N_pulls.ndim == 1:
                # Single product
                labels = [f"{p:.3f}" for p in prices]
                plt.bar(labels, agent.N_pulls)
                plt.xlabel("Price")
                plt.ylabel("Number of Pulls")
                plt.title(title)
                plt.xticks(rotation=45)
            else:
                # Multi-product
                n_products = len(agent.N_pulls)
                fig, axes = plt.subplots(
                    1, n_products, figsize=(4*n_products, 4))
                if n_products == 1:
                    axes = [axes]

                for j in range(n_products):
                    labels = [f"{p:.3f}" for p in prices[j]]
                    axes[j].bar(labels, agent.N_pulls[j])
                    axes[j].set_xlabel("Price")
                    axes[j].set_ylabel("Number of Pulls")
                    axes[j].set_title(f"Product {j+1}")
                    axes[j].tick_params(axis='x', rotation=45)

                plt.suptitle(title)
            plt.tight_layout()
            plt.show()

    @staticmethod
    def plot_lambda(agent, title: str = "Lambda (λ) Over Time"):
        """
        Plot the evolution of lambda (λ) recorded in the agent.
        Works for agents that record 'lmbd_history'.
        """
        if not hasattr(agent, "lmbd_history") or not agent.lmbd_history:
            print("No lambda history recorded in the agent.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(agent.lmbd_history, 'm-', linewidth=2, label="λ")
        plt.xlabel("Update Number")
        plt.ylabel("λ")
        plt.title(title)
        plt.legend()
        plt.grid(alpha=0.4)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_hedge_probabilities(agent, title: str = "Hedge Probabilities Over Time"):
        """
        For agents that record hedge weights (e.g. FFPrimalDualPricingAgent),
        plot the evolution of the normalized probabilities over time.
        """
        if not hasattr(agent, "hedge_weight_history") or not agent.hedge_weight_history:
            print("No hedge history recorded in the agent.")
            return

        # Compute probabilities by normalizing the recorded weights at each update.
        prob_history = np.array([w / np.sum(w)
                                for w in agent.hedge_weight_history])
        num_updates, K = prob_history.shape

        plt.figure(figsize=(10, 6))
        for k in range(K):
            plt.plot(prob_history[:, k], label=f"Prob[{k}]")
        plt.xlabel("Update Number")
        plt.ylabel("Probability")
        plt.title(title)
        plt.legend()
        plt.grid(alpha=0.4)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_multi_hedge_probabilities(agent, title: str = "Multi-Product Hedge Probabilities Over Time"):
        """
        Plot the evolution of normalized hedge probabilities over time for each product.
        Assumes agent.hedge_prob_history is a list of lists, one per product.
        """
        if not hasattr(agent, "hedge_prob_history") or not any(agent.hedge_prob_history):
            print("No hedge probability history recorded in the agent.")
            return

        n_products = len(agent.hedge_prob_history)
        fig, axes = plt.subplots(n_products, 1, figsize=(
            10, 4 * n_products), sharex=True)
        if n_products == 1:
            axes = [axes]

        for j in range(n_products):
            # Convert list of arrays for product j into a 2D array: (updates, K)
            prob_history = np.array(agent.hedge_prob_history[j])
            num_updates, K = prob_history.shape
            for k in range(K):
                axes[j].plot(prob_history[:, k],
                             label=f"Product {j+1} | Arm {k}")
            axes[j].set_xlabel("Update Number")
            axes[j].set_ylabel("Probability")
            axes[j].set_title(title)
            axes[j].grid(alpha=0.4)
        plt.tight_layout()
        plt.show()


class StandardAnalyzer:
    """Standard analysis utilities for all experiments"""

    @staticmethod
    def analyze_results(result: ExperimentResult) -> Dict[str, Any]:
        """Standard analysis of experiment results"""
        min_rounds = min(len(regrets) for regrets in result.regrets)

        # Basic statistics
        final_regrets = [regrets[-1]
                         if regrets else 0 for regrets in result.regrets]
        avg_regret_per_round = np.mean(
            final_regrets) / min_rounds if min_rounds > 0 else 0
        avg_final_reward = np.mean(result.final_rewards)
        std_final_reward = np.std(result.final_rewards)

        # Efficiency metrics
        efficiency = None
        if result.clairvoyant_reward is not None and result.clairvoyant_reward > 0:
            baseline_total = result.clairvoyant_reward * min_rounds
            efficiency = 100 * avg_final_reward / baseline_total

        analysis = {
            'min_rounds': min_rounds,
            'avg_regret_per_round': avg_regret_per_round,
            'avg_final_reward': avg_final_reward,
            'std_final_reward': std_final_reward,
            'efficiency_percent': efficiency,
            'execution_time': result.execution_time
        }

        return analysis

    @staticmethod
    def print_analysis(result: ExperimentResult, analysis: Dict[str, Any]):
        """Print standard analysis results"""
        print(f"\n{'='*60}")
        print(f"RESULTS FOR {result.config.task_name}")
        print(f"{'='*60}")
        print(f"Execution time: {analysis['execution_time']:.2f} seconds")
        print(f"Completed rounds: {analysis['min_rounds']}")
        print(
            f"Average regret per round: {analysis['avg_regret_per_round']:.6f}")
        print(
            f"Average final reward: {analysis['avg_final_reward']:.2f} ± {analysis['std_final_reward']:.2f}")

        if analysis['efficiency_percent'] is not None:
            print(
                f"Efficiency vs clairvoyant: {analysis['efficiency_percent']:.1f}%")

        if result.config.budget is not None:
            final_units = [
                units[-1] if units else 0 for units in result.units_sold]
            avg_units = np.mean(final_units)
            print(
                f"Average units sold: {avg_units:.2f}/{result.config.budget} ({100*avg_units/result.config.budget:.1f}%)")

        print(f"{'='*60}")
