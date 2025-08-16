import numpy as np
import matplotlib.pyplot as plt
from data_generators import generate_smooth_valuation_data, generate_independent_valuation_data


def main():
    T = 20000           # total number of time steps
    K = 5              # number of windows
    M = 1              # number of products; adjust if needed
    concentration = 50
    trial_seed = 18
    rng = np.random.default_rng(trial_seed)

    # --- Plot Smooth Valuation Data ---
    expected_means, valuations = generate_smooth_valuation_data(
        T, K, M, concentration, rng
    )
    L = T // K   # window length

    fig, axes = plt.subplots(K, 1, figsize=(8, 2 * K), sharex=True)
    if K == 1:
        axes = [axes]
    for k in range(K):
        start = k * L
        end = T if k == K - 1 else (k + 1) * L
        window_vals = valuations[start:end, 0]  # using product 0
        ax = axes[k]
        ax.hist(window_vals, bins=30, alpha=0.7,
                density=True, label=f"Smooth Window {k + 1}")
        ax.set_title(
            f"Smooth Data - Window {k + 1}: Time steps {start} to {end}")
        ax.legend()
    plt.xlabel("Valuation")
    plt.tight_layout()
    plt.show()

    # --- Plot Independent Valuation Data ---
    # For consistency, we partition T into K windows also here.
    expected_means_ind, valuations_ind = generate_independent_valuation_data(
        T=T, K=K, M=M, concentration=concentration, rng=rng
    )
    fig2, axes2 = plt.subplots(K, 1, figsize=(8, 2 * K), sharex=True)
    if K == 1:
        axes2 = [axes2]
    for k in range(K):
        start = k * L
        end = T if k == K - 1 else (k + 1) * L
        window_vals = valuations_ind[start:end, 0]
        ax2 = axes2[k]
        ax2.hist(window_vals, bins=30, alpha=0.7, density=True,
                 label=f"Independent Window {k + 1}")
        ax2.set_title(
            f"Independent Data - Window {k + 1}: Time steps {start} to {end}")
        ax2.legend()
    plt.xlabel("Valuation")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
