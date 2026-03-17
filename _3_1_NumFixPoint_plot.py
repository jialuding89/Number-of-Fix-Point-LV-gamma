"""
_3_1_NumFixPoint_plot.py
------------------------
Load the .npz result file produced by _2_1_NumFixPoint_forloop.py and plot
the normalised number of distinct fixed-point attractors (num_fp / n_sims)
as a function of k.
"""

import os
import glob
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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
# Locate result file
# ---------------------------------------------------------------------------
if len(sys.argv) > 1:
    npz_path = sys.argv[1]
else:
    candidates = glob.glob("results_theta*_S*_IT*.npz")
    if not candidates:
        raise FileNotFoundError(
            "No result file found. Run _2_1_NumFixPoint_forloop.py first, "
            "or pass the file path as an argument."
        )
    npz_path = max(candidates, key=lambda p: os.path.getmtime(p))

print(f"Loading: {npz_path}")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
data       = np.load(npz_path)
k_values   = data["k_values"]
results    = data["num_fixed_points"]
theta      = float(data["theta"][0])
N_species  = int(data["N_species"][0])
iterations = int(data["iterations"][0])

# Normalise: proportion of initial conditions converging to distinct outcomes
n_sims  = 10_000
fp_norm = results / n_sims

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 7))

ax.plot(k_values, fp_norm, "o-", color="navy")

ax.set_xlabel("Interaction strength parameter $k$")
ax.set_ylabel("Proportion of distinct outcomes\n(Number of FP / $n_{sims}$)")
ax.set_title(
    f"Ecological Multi-stability in Lotka–Volterra Model\n"
    f"$\\theta={theta}$,  $S={N_species}$,  IT={iterations:,}"
)

ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3, linestyle="--")

for spine in ax.spines.values():
    spine.set_linewidth(1.5)

output_fig = npz_path.replace(".npz", ".png")
fig.savefig(output_fig)
print(f"Figure saved to: {output_fig}")

plt.show()