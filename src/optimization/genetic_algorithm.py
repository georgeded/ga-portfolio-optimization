"""
Genetic Algorithm for cardinality-constrained portfolio optimization.

Chromosome representation:
    A real-valued weight vector of length N (universe size).
    Exactly K entries are non-zero (the selected stocks).
    Non-zero weights satisfy: w_i ∈ [W_MIN, W_MAX] and sum to 1.
"""

import numpy as np
from src.evaluation.metrics import portfolio_turnover

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

PC      = 0.6054
PM      = 0.1370
SIGMA_M = 0.1469
LAMBDA  = 1.8437

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
        raise ValueError("Repair recursion exceeded 1 call.")
    
    N = len(weights)
    w = weights.copy()

    # ── Step 1: Cardinality repair ────────────────────────────────────────────
    selected   = np.nonzero(w > 0)[0]
    n_selected = len(selected)

    if n_selected > K_MAX:
        # Keep only the K_MAX largest weights, zero out the rest
        keep = np.argpartition(w, -K_MAX)[-K_MAX:]
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
        keep = np.argpartition(w, -K_MAX)[-K_MAX:]
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

# ── Fitness Function ──────────────────────────────────────────────────────────

def fitness(weights:      np.ndarray,
            mu:           np.ndarray,
            sigma:        np.ndarray,
            prev_weights: np.ndarray | None = None,
            lambda_:      float = LAMBDA) -> float:
    """
    GA fitness function.

    F(w) = Sharpe(w, mu, sigma) − λ × Turnover(w, prev_w)

    Sharpe is computed from in-sample estimates (60-month rolling window),
    not from realised returns. Monthly (non-annualised) Sharpe is used:
    annualisation by √12 is a constant scaling that cancels in the ratio
    and leaves the optimisation landscape unchanged.

    Turnover penalty is 0.0 for the first rebalancing period (prev_weights
    is None) — there is no prior portfolio to trade from. This is distinct
    from the evaluation convention (turnover = 1.0 at t=0), which applies
    only when computing reported transaction costs, not during optimisation.

    Args:
        weights:      feasible weight vector of length N (K non-zero entries)
        mu:           in-sample mean excess return vector, length N
        sigma:        in-sample covariance matrix, shape (N, N)
        prev_weights: pre-rebalance weight vector aligned to current universe
                      (after drift, length N); None for the first period
        lambda_:      turnover penalty coefficient (Optuna-tuned, default 0.0)

    Returns:
        Scalar fitness value (higher is better)
    """
    port_return = float(weights @ mu)
    port_var    = float(weights @ sigma @ weights)

    if port_var <= 0.0:
        sharpe = 0.0
    else:
        sharpe = port_return / np.sqrt(port_var)

    if prev_weights is None or lambda_ < 1e-12:
        penalty = 0.0
    else:
        penalty = lambda_ * portfolio_turnover(weights, prev_weights)

    return sharpe - penalty


# ── Tournament Selection ──────────────────────────────────────────────────────

def tournament_select(population: np.ndarray,
                      fitnesses:  np.ndarray,
                      rng:        np.random.Generator,
                      k:          int = TOURNAMENT) -> np.ndarray:
    """
    Tournament selection: sample k individuals without replacement,
    return a copy of the one with the highest fitness.

    Called twice per crossover event to produce two parents. Using
    replace=False prevents the degenerate case where the same individual
    fills all tournament slots, which would make selection pressure
    dependent on population size in a non-meaningful way.

    Args:
        population: weight matrix of shape (POP_SIZE, N)
        fitnesses:  pre-computed fitness array of shape (POP_SIZE,)
        rng:        numpy Generator for reproducibility
        k:          tournament size (default TOURNAMENT = 3)

    Returns:
        Copy of the winner's weight vector, shape (N,)
    """
    k = min(k, len(population))
    indices = rng.choice(len(population), size=k, replace=False)
    winner  = indices[np.argmax(fitnesses[indices])]
    return population[winner].copy()

# ── Crossover Operator ────────────────────────────────────────────────────────

def _make_child(w1: np.ndarray,
                w2: np.ndarray,
                rng: np.random.Generator) -> np.ndarray:
    """
    Produce one offspring from two parents.

    Steps:
    1. Compute union of active stocks from both parents.
    2. Draw child cardinality K_child uniformly from [K_MIN, K_MAX].
    3. Sample K_child stocks from union without replacement, weighted by
       average parental weight (intersection stocks get 2× the sampling
       probability of exclusive stocks, reflecting genetic consensus).
    4. Blend weights using a single shared α ~ Uniform(0, 1).
       For exclusive stocks, the absent parent contributes weight 0.
    5. Repair to enforce all constraints.

    Args:
        w1: feasible weight vector for parent 1, length N
        w2: feasible weight vector for parent 2, length N
        rng: numpy Generator

    Returns:
        Feasible child weight vector, length N
    """
    N     = len(w1)
    union = np.nonzero((w1 > 0) | (w2 > 0))[0]
    if len(union) == 0:
        return repair(np.ones(len(w1))/len(w1), rng)

    # Draw child cardinality independently — crossover is primary source
    # of cardinality exploration across [K_MIN, K_MAX]
    k_child = rng.integers(K_MIN, K_MAX + 1)
    k_child = min(k_child, len(union))  # cap at union size (never binding
                                         # with K<=30 and N~867, but required
                                         # for correctness)

    # Weight-biased sampling: intersection stocks have 2× the probability
    # of stocks held by only one parent
    probs = (w1[union] + w2[union]) / 2.0
    s = probs.sum()
    if s <= 1e-12:
        probs = np.ones(len(union)) / len(union)
    else:
        probs /= s

    selected = rng.choice(union, size=k_child, replace=False, p=probs)

    # Single shared α: coherent directional blend toward one parent
    alpha   = rng.uniform(0.0, 1.0)
    w_child = np.zeros(N)
    w_child[selected] = alpha * w1[selected] + (1.0 - alpha) * w2[selected]

    # Near-zero cleanup before repair: exclusive stocks scaled by small alpha
    # can produce phantom weights that confuse cardinality counting in repair
    w_child[w_child < 1e-10] = 0.0

    return repair(w_child, rng)


def crossover(p1: np.ndarray,
              p2: np.ndarray,
              rng: np.random.Generator,
              pc: float = PC) -> tuple[np.ndarray, np.ndarray]:
    """
    Two-child crossover with probability pc.

    If crossover does not fire, copies of both parents are returned.
    Two children are produced per crossover event by swapping the parent
    roles, doubling information extracted from each tournament pair.

    Args:
        p1:  feasible weight vector for parent 1, length N
        p2:  feasible weight vector for parent 2, length N
        rng: numpy Generator
        pc:  crossover probability (default PC = 0.8)

    Returns:
        Tuple of two feasible child weight vectors, each length N
    """
    if rng.random() > pc:
        return p1.copy(), p2.copy()

    child1 = _make_child(p1, p2, rng)
    child2 = _make_child(p2, p1, rng)
    return child1, child2


# ── Mutation Operator ─────────────────────────────────────────────────────────

def mutate(w: np.ndarray,
           rng: np.random.Generator,
           pm: float = PM) -> np.ndarray:
    """
    Apply Gaussian weight mutation and/or asset-swap mutation.

    Two independent Bernoulli draws, each with probability pm:

    (a) Gaussian weight mutation: add N(0, SIGMA_M) noise to all currently
        selected (non-zero) weights. Weights pushed negative are zeroed.
        SIGMA_M = 0.02 ≈ 1/6 of the weight range [W_MIN, W_MAX], consistent
        with standard bounded real-valued mutation calibration (Eiben & Smith
        2015). Optuna searches [0.01, 0.15].

    (b) Asset-swap mutation: remove one randomly chosen held stock (set to 0),
        add one randomly chosen unheld stock at W_MIN. Preserves cardinality
        exactly before repair. Skipped if no unheld stocks exist.

    A single repair() is applied at the end if any operator fired.
    Applying repair after each operator independently would dampen both
    perturbations and produce less varied offspring.

    pm is interpreted as per-chromosome probability, not per-gene:
        P(no mutation)      = (1 - pm)^2
        P(Gaussian only)    = pm(1 - pm)
        P(swap only)        = pm(1 - pm)
        P(both)             = pm^2

    Args:
        w:   feasible weight vector, length N
        rng: numpy Generator
        pm:  per-operator mutation probability (default PM = 0.1)

    Returns:
        Feasible weight vector (mutated or unchanged copy)
    """
    w = w.copy()
    gaussian_fired = rng.random() < pm
    swap_fired     = rng.random() < pm

    if gaussian_fired:
        selected = w > 0
        w[selected] += rng.normal(0.0, SIGMA_M, size=selected.sum())
        # Zero out negatives: these are stocks pushed below 0 by noise.
        # Treated as dropped holdings; repair re-checks cardinality.
        w[w < 0.0] = 0.0
        w[w > 1.0] = 1.0

    if swap_fired:
        # Operate on current w (may be dirty from Gaussian step).
        # w > 0 and w == 0 are meaningful even on non-normalised vectors:
        # Gaussian adds to existing non-zero weights, never activates true zeros.
        non_zero = np.nonzero(w > 0)[0]
        zero = np.nonzero(w == 0)[0]
        if len(non_zero) > 0 and len(zero) > 0:
            w[rng.choice(non_zero)] = 0.0
            w[rng.choice(zero)]     = W_MIN

    if gaussian_fired or swap_fired:
        # Near-zero cleanup: Gaussian noise can leave phantom tiny positives
        # that are not true holdings but confuse cardinality counting in repair.
        w[w < 1e-10] = 0.0
        try:
            w = repair(w, rng)
        except ValueError:
            w = repair(np.ones_like(w) / len(w), rng)

    return w

# ── Local Refinement ──────────────────────────────────────────────────────────
 
def local_refine(w:            np.ndarray,
                 mu:           np.ndarray,
                 sigma:        np.ndarray,
                 prev_weights: np.ndarray | None,
                 rng:          np.random.Generator) -> tuple[np.ndarray, float]:
    """
    Pairwise weight-shift hill-climber applied to a single feasible chromosome.
 
    At each iteration, randomly selects two held stocks (i, j) and shifts
    LOCAL_STEP weight from i to j, subject to W_MIN/W_MAX bounds. The shift
    is accepted only if fitness improves (greedy acceptance). Sum and
    cardinality are preserved by construction — no repair is needed.
 
    Args:
        w:            feasible weight vector, length N
        mu:           in-sample mean excess return vector, length N
        sigma:        in-sample covariance matrix, shape (N, N)
        prev_weights: pre-rebalance weights aligned to current universe,
                      or None for the first period
        rng:          numpy Generator
 
    Returns:
        Tuple of (refined feasible weight vector of length N, its fitness value).
        Fitness is returned to avoid recomputation in the caller.
    """
    w_best = w.copy()
    f_best = fitness(w_best, mu, sigma, prev_weights, LAMBDA)
 
    for _ in range(LOCAL_ITER):
        held = np.nonzero(w_best > 1e-10)[0]
        if len(held) < 2:
            break
 
        i, j = rng.choice(held, size=2, replace=False)
 
        # Maximum transferable amount that keeps both weights in [W_MIN, W_MAX]
        delta = min(LOCAL_STEP,
                    w_best[i] - W_MIN,
                    W_MAX     - w_best[j])
        if delta <= 0:
            continue
 
        w_cand     = w_best.copy()
        w_cand[i] -= delta
        w_cand[j] += delta
 
        f_cand = fitness(w_cand, mu, sigma, prev_weights, LAMBDA)
        if f_cand > f_best:
            w_best = w_cand
            f_best = f_cand
 
    return w_best, f_best
 
 
# ── Main GA Loop ──────────────────────────────────────────────────────────────
 
def run_ga(n_assets:     int,
           mu:           np.ndarray,
           sigma:        np.ndarray,
           prev_weights: np.ndarray | None,
           rng:          np.random.Generator,
           return_history: bool = False) -> np.ndarray:
    """
    Full generational GA with elitism, local refinement, and early stopping.
 
    At each generation:
      1. Elite chromosomes are preserved unchanged (except the single best,
         which receives local refinement).
      2. Offspring fill the remaining slots via tournament selection,
         crossover, and mutation.
      3. Fitness is carried over for elites and recomputed for offspring only.
      4. Early stopping triggers after EARLY_STOP consecutive generations
         with improvement < 1e-6.
 
    Args:
        n_assets:     number of stocks in the current universe
        mu:           in-sample mean excess return vector, length n_assets
        sigma:        in-sample covariance matrix, shape (n_assets, n_assets)
        prev_weights: pre-rebalance weights aligned to current universe,
                      or None for the first rebalancing period
        rng:          numpy Generator (seed set by runner per run)
 
    Returns:
        Best feasible weight vector found across all generations, length n_assets
    """
    # ── Initialization ────────────────────────────────────────────────────────
    population = initialize_population(n_assets, rng)
    fitnesses  = np.array([
        fitness(population[i], mu, sigma, prev_weights, LAMBDA)
        for i in range(POP_SIZE)
    ])
 
    n_elite     = max(1, int(POP_SIZE * ELITE_FRAC))
    best_idx    = int(np.argmax(fitnesses))
    best_ever_w = population[best_idx].copy()
    best_ever_f = fitnesses[best_idx]
    stagnation  = 0
    fitness_history = []
 
    # ── Generation loop ───────────────────────────────────────────────────────
    for _ in range(N_GENS):
 
        # Step 1: Extract elites (sorted ascending; last entry is best)
        elite_idx = np.argsort(fitnesses)[-n_elite:]
        elites    = population[elite_idx].copy()
        elite_fit = fitnesses[elite_idx].copy()
 
        # Step 2: Local refinement on the single best individual.
        # local_refine returns (w, f) — fitness is reused directly,
        # avoiding one redundant evaluation.
        elites[-1], elite_fit[-1] = local_refine(
            elites[-1], mu, sigma, prev_weights, rng
        )
 
        # Step 3: Build next generation — elites occupy slots 0..n_elite-1
        new_pop = np.empty_like(population)
        new_pop[:n_elite] = elites
 
        # Step 4: Fill remaining slots with offspring
        i = n_elite
        while i < POP_SIZE:
            p1     = tournament_select(population, fitnesses, rng)
            p2     = tournament_select(population, fitnesses, rng)
            c1, c2 = crossover(p1, p2, rng, pc=PC)
            c1     = mutate(c1, rng, pm=PM)
            c2     = mutate(c2, rng, pm=PM)
            new_pop[i] = c1
            i += 1
            if i < POP_SIZE:
                new_pop[i] = c2
                i += 1
 
        population = new_pop
 
        # Step 5: Update fitness — carry over elite, recompute offspring only
        new_fit = np.empty(POP_SIZE)
        new_fit[:n_elite] = elite_fit
        for i in range(n_elite, POP_SIZE):
            new_fit[i] = fitness(
                population[i], mu, sigma, prev_weights, LAMBDA
            )
        fitnesses = new_fit
 
        # Step 6: Track best-ever and early stopping
        gen_best_idx = int(np.argmax(fitnesses))
        gen_best_f   = fitnesses[gen_best_idx]
 
        if gen_best_f - best_ever_f > 1e-6:
            best_ever_f = gen_best_f
            best_ever_w = population[gen_best_idx].copy()
            stagnation  = 0
        else:
            stagnation += 1

        fitness_history.append(best_ever_f)
        if stagnation >= EARLY_STOP:
            break
 
    if return_history:
        return best_ever_w, fitness_history
    return best_ever_w