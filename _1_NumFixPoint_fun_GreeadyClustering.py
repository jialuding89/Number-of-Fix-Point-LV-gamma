#_1_NumFixPoint_fun_GreeadyClustering.py
import numpy as np
from numba import njit


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


def count_distinct_attractors(final_states, d2_threshold=1e-2):
    """
    Count the number of distinct fixed-point attractors using greedy clustering.

    Algorithm:
    - Go through each final state one by one
    - Compare its d2 distance to the centroid of every existing cluster
    - If distance < d2_threshold: assign to the nearest cluster and update
      its centroid as the running average
    - Otherwise: open a new cluster

    Limitation: centroid drifts as new points are merged in, so the result
    may depend on the order of trajectories.

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

    centroids = [final_states[0]]
    counts    = [1]

    for state in final_states[1:]:
        dists = np.array([np.linalg.norm(state - c) for c in centroids])
        nearest = int(np.argmin(dists))
        if dists[nearest] < d2_threshold:
            # Update centroid as running average
            counts[nearest] += 1
            centroids[nearest] = (
                centroids[nearest] + (state - centroids[nearest]) / counts[nearest]
            )
        else:
            centroids.append(state.copy())
            counts.append(1)

    return len(centroids)