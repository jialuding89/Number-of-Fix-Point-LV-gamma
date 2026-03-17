"""
_2_2_NumFixPoint_fx_k_theta.py
------------------------------
Run a single simulation for one specific (k, theta) pair.
Designed to be submitted as a Slurm array job where k and theta are passed
as command-line arguments from the .sh submission script.

Usage (standalone):
    python _2_2_NumFixPoint_fx_k_theta.py <k> <theta> [N_species] [iterations]

Output filename convention:
    result_k<k>_theta<theta>_S<N_species>_IT<iterations>.npz

Each output directory/file therefore encodes all four key parameters and
contains exactly one value: the number of distinct fixed points.
"""

import sys
import numpy as np
from _1_NumFixPoint_fun import simulate_trajectories, count_distinct_attractors

# ---------------------------------------------------------------------------
# Parse command-line arguments
# ---------------------------------------------------------------------------
if len(sys.argv) < 3:
    print("Usage: python _2_2_NumFixPoint_fx_k_theta.py <k> <theta> [S] [IT]")
    sys.exit(1)

k_val      = float(sys.argv[1])
theta      = float(sys.argv[2])
N_species  = int(sys.argv[3])   if len(sys.argv) > 3 else 50
iterations = int(sys.argv[4])   if len(sys.argv) > 4 else 100_000
n_sims     = int(sys.argv[5])   if len(sys.argv) > 5 else 10_000

print(
    f"Running: k={k_val}, theta={theta}, S={N_species}, "
    f"IT={iterations}, n_sims={n_sims}"
)

# ---------------------------------------------------------------------------
# Simulate & count
# ---------------------------------------------------------------------------
final_N = simulate_trajectories(k_val, theta, N_species, n_sims, iterations)
num_fp  = count_distinct_attractors(final_N)

print(f"Number of distinct fixed points: {num_fp}")

# ---------------------------------------------------------------------------
# Save result
# ---------------------------------------------------------------------------
# Format floats to avoid dots in filenames (e.g. 0.16 → 0p16)
def fmt(x):
    return f"{x:.4g}".replace(".", "p")

output_filename = (
    f"result_k{fmt(k_val)}_theta{fmt(theta)}_S{N_species}_IT{iterations}.npz"
)

np.savez(
    output_filename,
    k=np.array([k_val]),
    theta=np.array([theta]),
    N_species=np.array([N_species]),
    iterations=np.array([iterations]),
    n_sims=np.array([n_sims]),
    num_fixed_points=np.array([num_fp]),
)

print(f"Result saved to: {output_filename}")