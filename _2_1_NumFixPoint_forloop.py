"""
_2_1_NumFixPoint_forloop.py
---------------------------
Sweep over a range of k values at a fixed theta, count the number of distinct
fixed-point attractors for each (k, theta) pair, and save the results to a
compressed NumPy archive (.npz).

Output filename convention:
    results_theta<theta>_S<N_species>_IT<iterations>.npz
"""

import numpy as np
from tqdm import tqdm
from _1_NumFixPoint_fun import simulate_trajectories, count_distinct_attractors

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
N_species  = 50          # total number of species  (S)
n_sims     = 10_000      # number of independent initial conditions per k value
iterations = 100_000     # total number of Euler integration steps (IT)
theta      = 0.16        # fixed scale parameter of the Gamma distribution

k_values = np.linspace(0.1, 6.0, 20)   # k values to sweep

# ---------------------------------------------------------------------------
# Run simulations
# ---------------------------------------------------------------------------
results = []

print(
    f"Sweeping {len(k_values)} k values | "
    f"theta={theta} | S={N_species} | IT={iterations} | n_sims={n_sims}"
)

for k in tqdm(k_values):
    # Simulate all trajectories for this (k, theta) pair
    final_N = simulate_trajectories(k, theta, N_species, n_sims, iterations)

    # Count distinct attractors using the d2 criterion
    num_fp = count_distinct_attractors(final_N)
    results.append(num_fp)

results = np.array(results)

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
output_filename = (
    f"results_theta{theta}_S{N_species}_IT{iterations}.npz"
)

np.savez(
    output_filename,
    k_values=k_values,
    num_fixed_points=results,
    theta=np.array([theta]),
    N_species=np.array([N_species]),
    iterations=np.array([iterations]),
    n_sims=np.array([n_sims]),
)

print(f"Results saved to: {output_filename}")