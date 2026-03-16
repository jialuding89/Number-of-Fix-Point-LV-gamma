# Number-of-Fix-Point-LV-gamma
# Number of Fixed Points in Lotka-Volterra System with Gamma-distributed Interactions

This repository investigates **ecological multi-stability** in a generalised Lotka-Volterra (LV) competition model where inter-species interaction strengths are drawn from a **Gamma distribution** with shape parameter *k* and scale parameter *θ*.

The central question is: **how does the number of distinct fixed-point attractors change as a function of *k* and *θ*?**

---

## Model

Each species *i* follows logistic growth with pairwise competitive interactions:

$$\frac{dN_i}{dt} = r_i N_i (1 - N_i) - N_i \sum_{j \neq i} \alpha_{ij} N_j$$

where:
- $r_i = 1$ for all species
- $\alpha_{ij} \sim \text{Gamma}(k, \theta)$ for $i \neq j$,  $\alpha_{ii} = 0$
- Integration uses the Euler method with step $dt = 0.01$

Fixed points are detected by running many independent trajectories from random initial conditions and counting distinct final states using a distance threshold $d_2$.

---

## File Structure

```
.
├── _1_NumFixPoint_fun.py            # Core functions: simulation + attractor counting
├── _2_1_NumFixPoint_forloop.py      # Local run: sweep k values at fixed θ → one .npz
├── _2_2_NumFixPoint_fx_k_theta.py   # Slurm run: single (k, θ) pair → one .npz
├── _3_1_NumFixPoint_plot.py         # Plot results from _2_1 output
├── _3_2_NumFixPoint_plot.py         # Plot results aggregated from multiple _2_2 outputs
├── submit_NumFixPoint.sh            # Slurm array job submission script
└── README.md
```

---

## Workflow

### Option A — Local (single θ, sweep over k)

```bash
# 1. Edit parameters in _2_1_NumFixPoint_forloop.py (theta, N_species, iterations, k_values)
python _2_1_NumFixPoint_forloop.py

# 2. Plot
python _3_1_NumFixPoint_plot.py
```

Output: `results_theta<θ>_S<S>_IT<IT>.npz` + `.png`

---

### Option B — HPC / Slurm (one job per (k, θ) pair, fully parallelised)

```bash
# 1. Upload files to cluster
scp _1_NumFixPoint_fun.py _2_2_NumFixPoint_fx_k_theta.py submit_NumFixPoint.sh \
    user@cluster:/scratch/users/<username>/files/

# 2. Edit K_VALUES, THETA_VALUES, S, IT in submit_NumFixPoint.sh

# 3. Submit array job (one task per k value)
sbatch submit_NumFixPoint.sh

# 4. Download all results locally
scp user@cluster:/scratch/users/<username>/files/result_*.npz ./results/

# 5. Plot all results (one curve per θ)
python _3_2_NumFixPoint_plot.py
```

Each Slurm task produces one file:
`result_k<k>_theta<θ>_S<S>_IT<IT>.npz`

---

## Parameters

| Symbol | Variable | Description | Default |
|--------|----------|-------------|---------|
| *k* | `k_val` | Shape parameter of Gamma distribution | swept 0.1 → 6.0 |
| *θ* | `theta` | Scale parameter of Gamma distribution | 0.16 |
| *S* | `N_species` | Total number of species | 50 |
| IT | `iterations` | Number of Euler integration steps | 100,000 |
| — | `n_sims` | Number of independent initial conditions | 10,000 |
| — | `dt` | Integration time step | 0.01 |

---

## Output File Naming Convention

| Script | Output filename |
|--------|----------------|
| `_2_1_` | `results_theta<θ>_S<S>_IT<IT>.npz` |
| `_2_2_` | `result_k<k>_theta<θ>_S<S>_IT<IT>.npz` |

---

## Dependencies

```bash
pip install numpy numba tqdm matplotlib
```

| Package | Purpose |
|---------|---------|
| `numpy` | Array operations and file I/O |
| `numba` | JIT compilation for fast parallel simulation |
| `tqdm` | Progress bar (local runs) |
| `matplotlib` | Plotting |

Python ≥ 3.11 recommended.

---

## Cluster Configuration (KCL ERC HPC)

- **Partition:** `cpu`
- **Time limit:** 2 days
- **CPUs per task:** 4 (used by Numba parallel threads)
- **Memory:** 8 GB per task
- Email notifications on job `END` and `FAIL`

---

## Author

Jialu Ding — King's College London