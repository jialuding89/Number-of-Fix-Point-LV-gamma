# _1_NumFixPoint_fun_CompleteLinkage.py
import numpy as np
from numba import njit
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


@njit(parallel=True)
def simulate_trajectories(k_val, theta, N_species, n_sims, iterations, dt=0.01):
    """
    Simulate ecological dynamics for a community of N_species species.

    Parameters
    ----------
    k_val      : float  - shape parameter of the Gamma distribution for interaction strengths
    theta      : float  - scale parameter of the Gamma distribution
    N_species  : int    - number of species (S)
    n_sims     : int    - number of independent initial conditions (trajectories)
    iterations : int    - number of Euler integration steps (IT)
    dt         : float  - integration time step

    Returns
    -------
    N : ndarray, shape (n_sims, N_species)
        Final population state of every trajectory.
    """
    # Growth rates (all set to 1)
    r = np.ones(N_species, dtype=np.float64)

    # Interaction matrix drawn from Gamma(k, theta); self-interaction set to zero
    alpha = np.random.gamma(k_val, theta, size=(N_species, N_species))
    for i in range(N_species):
        alpha[i, i] = 0.0

    # Random initial conditions uniformly distributed in (0.01, 1.0)
    N = np.random.uniform(0.01, 1.0, size=(n_sims, N_species))

    # Euler integration
    for _ in range(iterations):
        interact = N @ alpha.T                        # pairwise competitive pressure
        dNdt = r * N * (1.0 - N) - N * interact      # logistic growth + competition
        N += dNdt * dt
        N = np.maximum(N, 0.0)                        # enforce non-negativity

    return N


def count_distinct_attractors(final_states, N_species, d2_threshold=1e-2):
    """
    Count the number of distinct fixed-point attractors using complete-linkage
    hierarchical clustering.

    Algorithm:
    - Compute ALL pairwise Euclidean distances between final states
    - Build a dendrogram using complete linkage:
      two clusters merge only if ALL point pairs between them are < d2_threshold
    - Cut the dendrogram at d2_threshold to get cluster labels
    - Count the number of distinct clusters

    Advantage over greedy: result is order-independent and strictly respects
    the d2 threshold for every pair of points within a cluster.

    Limitation: O(n^2) memory and time — slow for large n_sims (e.g. 10,000).

    Parameters
    ----------
    final_states   : ndarray, shape (n_sims, N_species)
    d2_threshold   : float  - Euclidean distance threshold (default 1e-3)

    Returns
    -------
    int : number of distinct attractors detected
    """
    if len(final_states) == 0:
        return 0

    if len(final_states) == 1:
        return 1

    # 直接用传入的 N_species 缩放
    dist_condensed = pdist(final_states, metric="euclidean") / np.sqrt(N_species)

    Z = linkage(dist_condensed, method="complete")
    labels = fcluster(Z, t=d2_threshold, criterion="distance")
    return int(np.max(labels))
