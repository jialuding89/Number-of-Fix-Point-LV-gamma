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
    # ← 每次调用这个函数都会生成一个新的 alpha 矩阵
    alpha = np.random.gamma(k_val, theta, size=(N_species, N_species))
    for i in range(N_species):
        alpha[i, i] = 0.0

    # Random initial conditions uniformly distributed in (0.01, 1.0)
    # ← 10,000 条轨迹都跑在同一个 alpha 上
    N = np.random.uniform(0.01, 1.0, size=(n_sims, N_species))

    # Euler integration
    for _ in range(iterations):
        interact = N @ alpha.T                        # pairwise competitive pressure
        dNdt = r * N * (1.0 - N) - N * interact      # logistic growth + competition
        N += dNdt * dt
        N = np.maximum(N, 0.0)                        # enforce non-negativity

    return N


def count_distinct_attractors(final_states, d2_threshold=1e-3):
    """
    Count the number of distinct fixed-point attractors in the final states.

    Two states are considered identical if their Euclidean distance is below
    d2_threshold.  This is approximated by rounding to 3 decimal places and
    using np.unique.

    Parameters
    ----------
    final_states   : ndarray, shape (n_sims, N_species)
    d2_threshold   : float  - distance threshold (default 1e-3)

    Returns
    -------
    int : number of distinct attractors detected
    """
    if len(final_states) == 0:
        return 0

    # Rounding maps nearby states to the same grid point → fast proxy for d2
    unique_points = np.unique(np.round(final_states, 3), axis=0)
    return len(unique_points)
