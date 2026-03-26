"""
_3_1_NumFixPoint_plot.py
------------------------
Load one or more .npz result files produced by _2_1_NumFixPoint_forloop.py
and plot the normalised number of distinct fixed-point attractors (num_fp /
n_sims) as a function of k — one curve per file.

Interactively lists all available result files and asks which ones to plot.
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
# Find all available result files
# ---------------------------------------------------------------------------
RESULTS_DIR  = "."
candidates   = sorted(glob.glob(os.path.join(RESULTS_DIR, "results_theta*_S*_IT*.npz")))

if not candidates:
    raise FileNotFoundError(
        "No result files found matching: results_theta*_S*_IT*.npz\n"
        "Run _2_1_NumFixPoint_forloop.py first."
    )

# ---------------------------------------------------------------------------
# Interactive file selection
# ---------------------------------------------------------------------------
print("=" * 60)
print(f"Found {len(candidates)} result file(s):")
print("=" * 60)
for i, f in enumerate(candidates):
    d = np.load(f)
    theta  = float(d["theta"][0])
    S      = int(d["N_species"][0])
    IT     = int(d["iterations"][0])
    n_sims = int(d["n_sims"][0]) if "n_sims" in d else 10_000
    print(f"  [{i}] {os.path.basename(f)}   "
          f"theta={theta}, S={S}, IT={IT:,}, n_sims={n_sims:,}")

print()
print("Enter the indices of files to plot, separated by commas.")
print("Press Enter to plot ALL files.")
raw = input("Select files: ").strip()

if raw == "":
    selected_files = candidates
else:
    indices = [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]
    selected_files = [candidates[i] for i in indices if 0 <= i < len(candidates)]

if not selected_files:
    raise ValueError("No valid files selected.")

print(f"\nPlotting {len(selected_files)} curve(s)...\n")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
colors  = plt.cm.tab10.colors
markers = ["o", "s", "^", "D", "v", "P", "*"]

fig, ax = plt.subplots(figsize=(10, 7))

for idx, fpath in enumerate(selected_files):
    data       = np.load(fpath)
    k_values   = data["k_values"]
    results    = data["num_fixed_points"]
    theta      = float(data["theta"][0])
    N_species  = int(data["N_species"][0])
    iterations = int(data["iterations"][0])
    n_sims     = int(data["n_sims"][0]) if "n_sims" in data else 10_000

    fp_norm = results / n_sims

    color  = colors[idx % len(colors)]
    marker = markers[idx % len(markers)]

    ax.plot(
        k_values, fp_norm,
        marker + "-",
        color=color,
        label=f"$\\theta={theta}$,  $S={N_species}$,  IT={iterations:,},  $n_{{sims}}={n_sims:,}$",
    )

ax.set_xlabel("Interaction strength parameter $k$")
ax.set_ylabel("Proportion of distinct outcomes\n(Number of FP / $n_{sims}$)")
ax.set_title("Ecological Multi-stability in Lotka–Volterra Model")
ax.set_ylim(bottom=0)
ax.legend(framealpha=0.6, edgecolor="grey", loc="upper left")
ax.grid(True, alpha=0.3, linestyle="--")

for spine in ax.spines.values():
    spine.set_linewidth(1.5)

output_fig = os.path.join(RESULTS_DIR, "NumFixPoint_combined_local.png")
fig.savefig(output_fig)
print(f"Figure saved to: {output_fig}")

plt.show()