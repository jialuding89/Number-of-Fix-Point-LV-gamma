"""
_3_2_NumFixPoint_plot.py
------------------------
Collect all individual result files produced by _2_2_NumFixPoint_fx_k_theta.py
(one file per (k, theta) pair), aggregate the data, and plot the normalised
number of distinct fixed points (num_fp / n_sims) as a function of k.

Interactively asks which parameter values to include before plotting.
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
GLOB_PATTERN = os.path.join(RESULTS_DIR, "result_CL_k*_theta*_S*_IT*.npz")

# ---------------------------------------------------------------------------
# Load ALL result files first
# ---------------------------------------------------------------------------
files = sorted(glob.glob(GLOB_PATTERN))
if not files:
    raise FileNotFoundError(
        f"No result files found matching:\n  {GLOB_PATTERN}\n"
        "Download the .npz files from the cluster first."
    )

print(f"Found {len(files)} result file(s) in total.\n")

# Read all files and collect available parameter values
all_data     = []
all_thetas   = set()
all_S        = set()
all_IT       = set()
all_n_sims   = set()

for fpath in files:
    data      = np.load(fpath)
    k         = float(data["k"][0])
    theta     = float(data["theta"][0])
    num_fp    = int(data["num_fixed_points"][0])
    N_species = int(data["N_species"][0])
    iters     = int(data["iterations"][0])
    n_sims    = int(data["n_sims"][0]) if "n_sims" in data else 10_000
    all_data.append((k, theta, num_fp, N_species, iters, n_sims))
    all_thetas.add(theta)
    all_S.add(N_species)
    all_IT.add(iters)
    all_n_sims.add(n_sims)

# ---------------------------------------------------------------------------
# Interactive parameter selection
# ---------------------------------------------------------------------------
def ask_subset(param_name, available):
    """Print available values and ask user which ones to include."""
    sorted_vals = sorted(available)
    print(f"Available {param_name} values: {sorted_vals}")
    print(f"  Enter values separated by commas, or press Enter to select ALL")
    raw = input(f"  Select {param_name}: ").strip()
    if raw == "":
        return set(sorted_vals)
    selected = set()
    for v in raw.split(","):
        v = v.strip()
        # Match by converting to same type
        for sv in sorted_vals:
            if str(sv) == v or f"{sv:g}" == v or str(int(sv)) == v:
                selected.add(sv)
    if not selected:
        print(f"  No valid values selected, using ALL.")
        return set(sorted_vals)
    print(f"  Selected: {sorted(selected)}\n")
    return selected

print("=" * 60)
print("Select which parameter values to include in the plot:")
print("=" * 60)

selected_thetas = ask_subset("theta",  all_thetas)
selected_S      = ask_subset("S",      all_S)
selected_IT     = ask_subset("IT",     all_IT)
selected_nsims  = ask_subset("n_sims", all_n_sims)

# ---------------------------------------------------------------------------
# Filter data by selected parameters
# ---------------------------------------------------------------------------
data_by_group = defaultdict(list)

for (k, theta, num_fp, N_species, iters, n_sims) in all_data:
    if (theta    in selected_thetas and
        N_species in selected_S      and
        iters     in selected_IT     and
        n_sims    in selected_nsims):
        data_by_group[(theta, N_species, iters, n_sims)].append((k, num_fp))

if not data_by_group:
    raise ValueError("No data matched the selected parameters. Please re-run and try different values.")

print(f"\nPlotting {len(data_by_group)} curve(s)...\n")

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

    fp_norm = fp_arr / n_sims

    color  = colors[idx % len(colors)]
    marker = markers[idx % len(markers)]

    ax.plot(
        k_arr, fp_norm,
        marker + "-",
        color=color,
        label=f"$\\theta={theta}$,  $S={S}$,  IT={IT:,},  $n_{{sims}}={n_sims:,}$",
    )

ax.set_xlabel("Interaction strength parameter $k$")
ax.set_ylabel("Proportion of distinct outcomes\n(Number of FP / $n_{sims}$)")
ax.set_title("Ecological Multi-stability in Lotka–Volterra Model")
ax.set_ylim(bottom=0)
ax.legend(framealpha=0.9, edgecolor="grey", loc="upper right")
ax.grid(True, alpha=0.3, linestyle="--")

for spine in ax.spines.values():
    spine.set_linewidth(1.5)

output_fig = os.path.join(RESULTS_DIR, "NumFixPoint_combined.png")
fig.savefig(output_fig)
print(f"Figure saved to: {output_fig}")

plt.show()