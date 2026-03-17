"""
_3_2_NumFixPoint_plot.py
------------------------
Collect all individual result files produced by _2_2_NumFixPoint_fx_k_theta.py
(one file per (k, theta) pair), aggregate the data, and plot the normalised
number of distinct fixed points (num_fp / S) as a function of k —
one curve per (theta, S) combination.

Expected filename pattern (produced by _2_2_NumFixPoint_fx_k_theta.py):
    result_k<k>_theta<theta>_S<S>_IT<IT>.npz

Place all downloaded .npz files in the same directory as this script, or set
RESULTS_DIR below.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import defaultdict

# ---------------------------------------------------------------------------
# Publication-quality matplotlib settings
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family":        "serif",
    "font.size":          16,
    "axes.titlesize":     20,
    "axes.labelsize":     18,
    "xtick.labelsize":    15,
    "ytick.labelsize":    15,
    "legend.fontsize":    14,
    "lines.linewidth":    2.5,
    "lines.markersize":   8,
    "axes.linewidth":     1.5,
    "xtick.major.width":  1.5,
    "ytick.major.width":  1.5,
    "xtick.major.size":   6,
    "ytick.major.size":   6,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
})

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_DIR  = "."
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

data_by_group = defaultdict(list)   # key: (theta, S, IT)

for fpath in files:
    data      = np.load(fpath)
    k         = float(data["k"][0])
    theta     = float(data["theta"][0])
    num_fp    = int(data["num_fixed_points"][0])
    N_species = int(data["N_species"][0])
    iters     = int(data["iterations"][0])
    n_sims    = int(data["n_sims"][0]) if "n_sims" in data else 10_000
    data_by_group[(theta, N_species, iters, n_sims)].append((k, num_fp))

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
colors  = plt.cm.tab10.colors
markers = ["o", "s", "^", "D", "v", "P", "*"]

fig, ax = plt.subplots(figsize=(10, 7))

for idx, ((theta, S, IT, n_sims), pairs) in enumerate(sorted(data_by_group.items())):
    pairs_sorted = sorted(pairs, key=lambda x: x[0])
    k_arr  = np.array([p[0] for p in pairs_sorted])
    fp_arr = np.array([p[1] for p in pairs_sorted])

    # Normalise: proportion of initial conditions that lead to distinct outcomes
    fp_norm = fp_arr / n_sims

    color  = colors[idx % len(colors)]
    marker = markers[idx % len(markers)]

    ax.plot(
        k_arr, fp_norm,
        marker + "-",
        color=color,
        label=f"$\\theta={theta}$,  $S={S}$",
    )

ax.set_xlabel("Interaction strength parameter $k$")
ax.set_ylabel("Proportion of distinct outcomes\n(Number of FP / $n_{sims}$)")
ax.set_title("Ecological Multi-stability in Lotka–Volterra Model")

ax.set_ylim(bottom=0)
ax.legend(framealpha=0.9, edgecolor="grey")
ax.grid(True, alpha=0.3, linestyle="--")

for spine in ax.spines.values():
    spine.set_linewidth(1.5)

output_fig = os.path.join(RESULTS_DIR, "NumFixPoint_combined.png")
fig.savefig(output_fig)
print(f"Figure saved to: {output_fig}")

plt.show()
