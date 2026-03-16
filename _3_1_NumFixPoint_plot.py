"""
_3_1_NumFixPoint_plot.py
------------------------
Load the .npz result file produced by _2_1_NumFixPoint_forloop.py and plot
the number of distinct fixed-point attractors as a function of k.
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import sys

# ---------------------------------------------------------------------------
# Locate result file
# ---------------------------------------------------------------------------
# Accept an explicit path as a command-line argument, otherwise auto-detect
if len(sys.argv) > 1:
    npz_path = sys.argv[1]
else:
    candidates = glob.glob("results_theta*_S*_IT*.npz")
    if not candidates:
        raise FileNotFoundError(
            "No result file found. Run _2_1_NumFixPoint_forloop.py first, "
            "or pass the file path as an argument."
        )
    # If multiple files exist, use the most recently modified one
    npz_path = max(candidates, key=lambda p: __import__("os").path.getmtime(p))

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

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(k_values, results, "o-", color="navy", markersize=6, linewidth=1.5)

ax.set_xlabel("Parameter $k$", fontsize=13)
ax.set_ylabel("Number of Distinct Fixed Points (based on $d_2$)", fontsize=13)
ax.set_title(
    f"Ecological Multi-stability Analysis\n"
    f"$\\theta$={theta},  $S$={N_species},  IT={iterations:,}",
    fontsize=13,
)
ax.grid(True, alpha=0.3)
fig.tight_layout()

output_fig = npz_path.replace(".npz", ".png")
fig.savefig(output_fig, dpi=150)
print(f"Figure saved to: {output_fig}")

plt.show()
