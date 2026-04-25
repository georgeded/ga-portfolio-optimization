"""
Genetic Algorithm for cardinality-constrained portfolio optimization.

Chromosome representation:
    A real-valued weight vector of length N (universe size).
    Exactly K entries are non-zero (the selected stocks).
    Non-zero weights satisfy: w_i ∈ [W_MIN, W_MAX] and sum to 1.
"""

import numpy as np

# ── Fixed Design Decisions (not tuned by Optuna) ─────────────────────────────
K_MIN      = 10      # minimum number of stocks to hold (Chang et al. 2000)
K_MAX      = 30      # maximum number of stocks to hold (Chang et al. 2000)
W_MIN      = 0.02    # minimum weight per selected stock (Step 7)
W_MAX      = 0.15    # maximum weight per selected stock (Jagannathan & Ma 2003)
POP_SIZE   = 100     # population size P (Eiben & Smith 2015)
N_GENS     = 200     # maximum generations (computational budget)
ELITE_FRAC = 0.05    # elitism fraction (Eiben & Smith 2015)
LOCAL_ITER = 5       # local refinement iterations (computational budget)
LOCAL_STEP = 0.01    # local refinement step size δ
EARLY_STOP = 20      # stagnation patience in generations
TOURNAMENT = 3       # tournament selection size k (Eiben & Smith 2015)

# ── Hyperparameters (tuned by Optuna, defaults from literature) ───────────────
PC         = 0.8     # crossover probability — Optuna searches [0.6, 0.95]
PM         = 0.1     # mutation probability  — Optuna searches [0.01, 0.3]
SIGMA_M    = 0.05    # Gaussian mutation std — Optuna searches [0.01, 0.15]
LAMBDA     = 0.0     # turnover penalty λ    — Optuna searches [0.0, 2.0]

DEBUG = False

def project_bounded_simplex(v, lower, upper, tol=1e-12):
    """
    Exact projection onto the bounded simplex

    sum(w)=1
    lower <= w_i <= upper

    Implemented via bisection on the Lagrange multiplier,
    following bounded-simplex projection methods
    (e.g. Duchi et al. 2008; Wang & Carreira-Perpinán 2013).

    This replaces the iterative clip-normalize repair
    described in the methodology, but enforces the same
    constraints exactly and guarantees feasibility.
    """

    v = np.asarray(v, dtype=float)

    n = len(v)

    if not (n*lower <= 1 <= n*upper):
        raise ValueError("Infeasible bounds.")

    def projected_sum(lam):
        return np.clip(v - lam, lower, upper).sum()

    # bracket lambda
    lo = np.min(v) - upper
    hi = np.max(v) - lower

    while hi-lo > tol:
        mid = 0.5*(lo+hi)

        if projected_sum(mid) > 1:
            lo = mid
        else:
            hi = mid

    lam = 0.5*(lo+hi)

    w = np.clip(v - lam, lower, upper)

    # tiny cleanup
    w /= w.sum()

    return w

# ── Repair Operator ───────────────────────────────────────────────────────────

def repair(weights: np.ndarray,
           rng:     np.random.Generator,
           depth: int = 0) -> np.ndarray:
    """
    Repair a weight vector to satisfy all constraints.

    Constraints enforced in order:
    1. Cardinality: K_MIN <= number of non-zero weights <= K_MAX
    2. Weight clipping: W_MIN <= w_i <= W_MAX for selected stocks
    3. Normalization: weights sum to 1.0
    4. Re-check bounds after normalization (iterative until convergence)

    The order matters — normalization after clipping may re-violate bounds,
    so steps 2-3 repeat until stable (max 10 iterations).

    Args:
        weights: raw weight vector of length N (may violate constraints)
        rng:     numpy Generator for random choices

    Returns:
        Feasible weight vector satisfying all constraints
    """
    if depth > 1:
        raise ValueError("Repair recursion exceeded depth.")
    
    N = len(weights)
    w = weights.copy()

    # ── Step 1: Cardinality repair ────────────────────────────────────────────
    selected   = np.nonzero(w > 0)[0]
    n_selected = len(selected)

    if n_selected > K_MAX:
        # Keep only the K_MAX largest weights, zero out the rest
        keep = np.argsort(w)[::-1][:K_MAX]
        mask = np.zeros(N, dtype=bool)
        mask[keep] = True
        w[~mask] = 0.0

    elif n_selected < K_MIN:
        # Activate random zero-weight stocks until we reach K_MIN
        zeros      = np.nonzero(w == 0)[0]
        needed = K_MIN - n_selected
        if len(zeros) >= needed:
            activate    = rng.choice(zeros, size=needed, replace=False)
            w[activate] = W_MIN
        else:
            # Edge case: not enough stocks — activate all available zeros
            w[zeros] = W_MIN

   # ── Steps 2–3: Bounded-simplex projection
    # Replaces iterative clip → normalize → repeat repair
    selected   = np.nonzero(w > 0)[0]

    w[selected] = project_bounded_simplex(w[selected], W_MIN, W_MAX)

    # ── Step 4: Final cardinality check ──────────────────────────────────────
    selected   = np.nonzero(w > 0)[0]
    n_selected = len(selected)

    if n_selected < K_MIN:
        zeros      = np.nonzero(w == 0)[0]
        needed = K_MIN - n_selected
        if len(zeros) >= needed:
            activate    = rng.choice(zeros, size=needed, replace=False)
            w[activate] = W_MIN
        return repair(w, rng, depth+1)  # one recursive call to fix

    if n_selected > K_MAX:
        keep = np.argsort(w)[::-1][:K_MAX]
        mask = np.zeros(N, dtype=bool)
        mask[keep] = True
        w[~mask] = 0.0
        return repair(w, rng, depth+1)  # one recursive call to fix

    # Zero out tiny numerical noise
    w[w <= 1e-10] = 0.0
    selected = w > 1e-10

    if DEBUG:
        assert abs(w.sum()-1) < 1e-8
        assert K_MIN <= selected.sum() <= K_MAX
        assert np.all(w[selected] >= W_MIN-1e-8)
        assert np.all(w[selected] <= W_MAX+1e-8)

    w[w <= 1e-10] = 0.0

    return w


# ── Population Initialization ─────────────────────────────────────────────────

def initialize_population(n_assets:   int,
                           rng: np.random.Generator) -> np.ndarray:
    """
    Initialize a population of POP_SIZE feasible portfolios.

    Each portfolio:
    1. Randomly selects K stocks (K drawn uniformly from [K_MIN, K_MAX])
    2. Assigns random weights from Dirichlet distribution
    3. Repaired to satisfy all constraints

    Args:
        n_assets:   number of stocks in the universe at this rebalancing date
        rng: numpy Generator for reproducibility
    Note:
    Parallel GA runs should use independent seeds
    (e.g. base_seed + run_id) in runner.py.
    
    Returns:
        population array of shape (POP_SIZE, n_assets)
    """
    population = np.zeros((POP_SIZE, n_assets))

    for i in range(POP_SIZE):
        # Random cardinality K ∈ [K_MIN, K_MAX]
        K = rng.integers(K_MIN, K_MAX + 1)

        # Random stock selection
        selected = rng.choice(n_assets, size=K, replace=False)

        # Random weights from Dirichlet (uniform over simplex)
        raw_weights = rng.dirichlet(np.ones(K))

        # Place into full vector
        w          = np.zeros(n_assets)
        w[selected] = raw_weights

        # Repair to enforce all constraints
        population[i] = repair(w, rng)

    return population