"""
_3_2_NumFixPoint_plot.py
------------------------
Collect all individual result files produced by _2_2_NumFixPoint_fx_k_theta.py
(one file per (k, theta) pair), aggregate the data, and plot the number of
distinct fixed points as a function of k — one curve per theta value.

Expected filename pattern (produced by _2_2_NumFixPoint_fx_k_theta.py):
    result_k<k>_theta<theta>_S<S>_IT<IT>.npz

Place all downloaded .npz files in the same directory as this script, or set
RESULTS_DIR below.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_DIR = "."          # directory containing the downloaded .npz files
GLOB_PATTERN = os.path.join(RESULTS_DIR, "result_k*_theta*_S*_IT*.npz")

# ---------------------------------------------------------------------------
# Load all result files
# ---------------------------------------------------------------------------
files = sorted(glob.glob(GLOB_PATTERN))

if not files:
    raise FileNotFoundError(
        f"No result files found matching:\n  {GLOB_PATTERN}\n"
        "Download the .npz files from the cluster first."
    )

print(f"Found {len(files)} result file(s).")

# Group by (theta, S, IT) so we can draw one curve per theta value
# data_by_theta[theta] = list of (k, num_fp) tuples
data_by_theta = defaultdict(list)

for fpath in files:
    data      = np.load(fpath)
    k         = float(data["k"][0])
    theta     = float(data["theta"][0])
    num_fp    = int(data["num_fixed_points"][0])
    N_species = int(data["N_species"][0])
    iters     = int(data["iterations"][0])
    data_by_theta[(theta, N_species, iters)].append((k, num_fp))

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
colors = plt.cm.tab10.colors
fig, ax = plt.subplots(figsize=(10, 6))

for idx, ((theta, S, IT), pairs) in enumerate(sorted(data_by_theta.items())):
    pairs_sorted = sorted(pairs, key=lambda x: x[0])   # sort by k
    k_arr  = np.array([p[0] for p in pairs_sorted])
    fp_arr = np.array([p[1] for p in pairs_sorted])

    color = colors[idx % len(colors)]
    ax.plot(
        k_arr, fp_arr,
        "o-",
        color=color,
        markersize=6,
        linewidth=1.5,
        label=f"$\\theta$={theta},  S={S},  IT={IT:,}",
    )

ax.set_xlabel("Parameter $k$", fontsize=13)
ax.set_ylabel("Number of Distinct Fixed Points (based on $d_2$)", fontsize=13)
ax.set_title("Ecological Multi-stability Analysis", fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()

output_fig = os.path.join(RESULTS_DIR, "NumFixPoint_combined.png")
fig.savefig(output_fig, dpi=150)
print(f"Figure saved to: {output_fig}")

plt.show()
