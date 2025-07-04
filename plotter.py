import numpy as np
import matplotlib.pyplot as plt
from runner import ExperimentResult
from numpy.typing import NDArray
from numpy.typing import NDArray
from typing import List, Dict, Any

class StandardPlotter:
    """Standard plotting utilities for all experiments"""
    
    @staticmethod
    def plot_experiment_results(result: ExperimentResult, show_units: bool = True):
        """Standard plotting for experiment results"""
        all_regrets = result.regrets
        all_units = result.units_sold
        
        # Find minimum length across trials
        min_rounds = min(len(regrets) for regrets in result.regrets)
        
        # Truncate to minimum length and convert to arrays
        truncated_regrets = np.array([regrets[:min_rounds] for regrets in all_regrets])
        truncated_units = np.array([units[:min_rounds] for units in all_units])
        
        # Compute statistics
        avg_regret = truncated_regrets.mean(axis=0)
        std_regret = truncated_regrets.std(axis=0)
        avg_units = truncated_units.mean(axis=0)
        
        # Create plots
        fig_width = 16 if show_units else 12
        fig, axes = plt.subplots(1, 2 if show_units else 1, figsize=(fig_width, 6))
        
        if not show_units:
            axes = [axes]
            
        # Regret plot
        axes[0].plot(avg_regret, 'b-', linewidth=2, label="Average Cumulative Regret")
        if result.config.n_trials > 1:
            se_regret = std_regret / np.sqrt(result.config.n_trials)
            axes[0].fill_between(
                range(min_rounds),
                avg_regret - se_regret,
                avg_regret + se_regret,
                alpha=0.3,
                color='blue',
                label="±1 SE"
            )
        axes[0].set_xlabel("Round")
        axes[0].set_ylabel("Cumulative Regret")
        axes[0].set_title(f"{result.config.task_name}: Cumulative Regret")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Units sold plot
        if show_units:
            axes[1].plot(avg_units, 'g-', linewidth=2, label="Average Units Sold")
            axes[1].set_xlabel("Round")
            axes[1].set_ylabel("Cumulative Units Sold")
            axes[1].set_title(f"{result.config.task_name}: Units Sold")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # Add budget line if available
            if result.config.budget is not None:
                axes[1].axhline(y=result.config.budget, color='r', linestyle='--', 
                              label=f"Budget: {result.config.budget}")
                axes[1].legend()
        
        plt.tight_layout()
        plt.show()
        
        return avg_regret, avg_units, min_rounds
    
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
                fig, axes = plt.subplots(1, n_products, figsize=(4*n_products, 4))
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

class StandardAnalyzer:
    """Standard analysis utilities for all experiments"""
    
    @staticmethod
    def analyze_results(result: ExperimentResult) -> Dict[str, Any]:
        """Standard analysis of experiment results"""
        min_rounds = min(len(regrets) for regrets in result.regrets)
        
        # Basic statistics
        final_regrets = [regrets[-1] if regrets else 0 for regrets in result.regrets]
        avg_regret_per_round = np.mean(final_regrets) / min_rounds if min_rounds > 0 else 0
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
        print(f"Average regret per round: {analysis['avg_regret_per_round']:.6f}")
        print(f"Average final reward: {analysis['avg_final_reward']:.2f} ± {analysis['std_final_reward']:.2f}")
        
        if analysis['efficiency_percent'] is not None:
            print(f"Efficiency vs clairvoyant: {analysis['efficiency_percent']:.1f}%")
        
        if result.config.budget is not None:
            final_units = [units[-1] if units else 0 for units in result.units_sold]
            avg_units = np.mean(final_units)
            print(f"Average units sold: {avg_units:.2f}/{result.config.budget} ({100*avg_units/result.config.budget:.1f}%)")
        
        print(f"{'='*60}")

print("Plotting and analysis utilities created successfully!")

