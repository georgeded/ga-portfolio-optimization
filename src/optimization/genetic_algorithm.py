"""
Genetic Algorithm for cardinality-constrained portfolio optimization.
Chromosome: real-valued weight vector of length N (universe size).
Exactly K entries are non-zero, with w_i ∈ [W_MIN, W_MAX] summing to 1.
"""

import numpy as np
from src.evaluation.metrics import portfolio_turnover

# not tuned by Optuna
K_MIN      = 10
K_MAX      = 30
W_MIN      = 0.02
W_MAX      = 0.15
POP_SIZE   = 100
N_GENS     = 200
ELITE_FRAC = 0.05    # top fraction preserved unchanged each generation
LOCAL_ITER = 5       # hill-climber steps per elite refinement
LOCAL_STEP = 0.01    # weight-shift step size δ
EARLY_STOP = 20      # stagnation patience (generations)
TOURNAMENT = 3       # tournament size

PC      = 0.6054
PM      = 0.1370
SIGMA_M = 0.1469
LAMBDA  = 1.8437

DEBUG = False

def project_bounded_simplex(v, lower, upper, tol=1e-12):
    """
    Exact projection onto {Σw=1, lower ≤ w_i ≤ upper} via bisection on the Lagrange multiplier.
    Replaces iterative clip-normalize; enforces constraints in a single pass.
    """
    v = np.asarray(v, dtype=float)
    n = len(v)

    if not (n*lower <= 1 <= n*upper):
        raise ValueError("Infeasible bounds.")

    def projected_sum(lam):
        return np.clip(v - lam, lower, upper).sum()

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
    w /= w.sum()

    return w


def repair(weights: np.ndarray, rng: np.random.Generator, depth: int = 0) -> np.ndarray:
    """
    Cardinality first (K_MIN ≤ K ≤ K_MAX), then bounded-simplex projection.
    At most one recursive call for the rare case where projection drops a weight below W_MIN.
    """
    if depth > 1:
        raise ValueError("Repair recursion exceeded 1 call.")

    N = len(weights)
    w = weights.copy()

    selected = np.nonzero(w > 0)[0]
    n_selected = len(selected)

    if n_selected > K_MAX:
        keep = np.argpartition(w, -K_MAX)[-K_MAX:]
        mask = np.zeros(N, dtype=bool)
        mask[keep] = True
        w[~mask] = 0.0

    elif n_selected < K_MIN:
        zeros = np.nonzero(w == 0)[0]
        needed = K_MIN - n_selected
        if len(zeros) >= needed:
            activate = rng.choice(zeros, size=needed, replace=False)
            w[activate] = W_MIN
        else:
            # edge case: not enough stocks — activate all available zeros
            w[zeros] = W_MIN

    # bounded-simplex projection — replaces iterative clip → normalize → repeat
    selected = np.nonzero(w > 0)[0]
    w[selected] = project_bounded_simplex(w[selected], W_MIN, W_MAX)

    selected = np.nonzero(w > 0)[0]
    n_selected = len(selected)

    if n_selected < K_MIN:
        zeros = np.nonzero(w == 0)[0]
        needed = K_MIN - n_selected
        if len(zeros) >= needed:
            activate = rng.choice(zeros, size=needed, replace=False)
            w[activate] = W_MIN
        return repair(w, rng, depth+1)

    if n_selected > K_MAX:
        keep = np.argpartition(w, -K_MAX)[-K_MAX:]
        mask = np.zeros(N, dtype=bool)
        mask[keep] = True
        w[~mask] = 0.0
        return repair(w, rng, depth+1)

    # zero out tiny numerical noise
    w[w <= 1e-10] = 0.0
    selected = w > 1e-10

    if DEBUG:
        assert abs(w.sum()-1) < 1e-8
        assert K_MIN <= selected.sum() <= K_MAX
        assert np.all(w[selected] >= W_MIN-1e-8)
        assert np.all(w[selected] <= W_MAX+1e-8)

    w[w <= 1e-10] = 0.0

    return w


def initialize_population(n_assets: int, rng: np.random.Generator) -> np.ndarray:
    """
    K ~ Uniform[K_MIN, K_MAX] stocks per individual; Dirichlet weights; repair applied.
    Seeds must be independent across parallel runs (runner.py uses BASE_SEED + i).
    """
    population = np.zeros((POP_SIZE, n_assets))

    for i in range(POP_SIZE):
        K = rng.integers(K_MIN, K_MAX + 1)
        selected = rng.choice(n_assets, size=K, replace=False)

        # Dirichlet gives uniform distribution over the weight simplex
        raw_weights = rng.dirichlet(np.ones(K))

        w = np.zeros(n_assets)
        w[selected] = raw_weights
        population[i] = repair(w, rng)

    return population

def fitness(weights: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
            prev_weights: np.ndarray | None = None, lambda_: float = LAMBDA) -> float:
    """
    F(w) = Sharpe(w) − λ·Turnover(w, prev_w), using monthly (non-annualised) Sharpe.
    √12 cancels in the ratio so annualisation doesn't change the optimisation landscape.
    At t=0 (prev_weights=None), penalty=0. Distinct from evaluation convention (turnover=1.0 at t=0).
    """
    port_return = float(weights @ mu)
    port_var = float(weights @ sigma @ weights)

    if port_var <= 0.0:
        sharpe = 0.0
    else:
        sharpe = port_return / np.sqrt(port_var)

    if prev_weights is None or lambda_ < 1e-12:
        penalty = 0.0
    else:
        penalty = lambda_ * portfolio_turnover(weights, prev_weights)

    return sharpe - penalty


def tournament_select(population: np.ndarray, fitnesses: np.ndarray,
                      rng: np.random.Generator, k: int = TOURNAMENT) -> np.ndarray:
    """
    Sample k individuals (replace=False), return the best.
    replace=False avoids the degenerate case where all slots are filled by the same individual.
    """
    k = min(k, len(population))
    indices = rng.choice(len(population), size=k, replace=False)
    winner = indices[np.argmax(fitnesses[indices])]
    return population[winner].copy()

def _make_child(w1: np.ndarray, w2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Union of active stocks from both parents; child cardinality drawn independently.
    Stocks sampled proportional to average parental weight (intersection gets 2× probability).
    Weights blended with shared α ~ Uniform(0,1); repair enforces all constraints.
    """
    N = len(w1)
    union = np.nonzero((w1 > 0) | (w2 > 0))[0]
    if len(union) == 0:
        return repair(np.ones(len(w1))/len(w1), rng)

    # cardinality varies by crossover — primary source of K exploration across [K_MIN, K_MAX]
    k_child = rng.integers(K_MIN, K_MAX + 1)
    k_child = min(k_child, len(union))  # never binding with K≤30, N~867, but required

    probs = (w1[union] + w2[union]) / 2.0
    s = probs.sum()
    if s <= 1e-12:
        probs = np.ones(len(union)) / len(union)
    else:
        probs /= s

    selected = rng.choice(union, size=k_child, replace=False, p=probs)

    # single shared α: coherent directional blend toward one parent
    alpha = rng.uniform(0.0, 1.0)
    w_child = np.zeros(N)
    w_child[selected] = alpha * w1[selected] + (1.0 - alpha) * w2[selected]

    # phantom tiny weights from small-α exclusive stocks confuse repair's cardinality count
    w_child[w_child < 1e-10] = 0.0

    return repair(w_child, rng)


def crossover(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator,
              pc: float = PC) -> tuple[np.ndarray, np.ndarray]:
    """
    Fires with probability pc; returns parent copies otherwise.
    Two children per event (parent roles swapped) to extract more information per tournament pair.
    """
    if rng.random() > pc:
        return p1.copy(), p2.copy()

    child1 = _make_child(p1, p2, rng)
    child2 = _make_child(p2, p1, rng)
    return child1, child2


def mutate(w: np.ndarray, rng: np.random.Generator, pm: float = PM) -> np.ndarray:
    """
    Two independent Bernoulli draws (each with probability pm):
    (a) Gaussian: add N(0, SIGMA_M) noise to selected weights; negatives zeroed (treated as dropped).
    (b) Swap: remove one held stock (→0), add one unheld stock at W_MIN; preserves cardinality pre-repair.
    Single repair at end — not after each operator — to avoid dampening both perturbations.
    """
    w = w.copy()
    gaussian_fired = rng.random() < pm
    swap_fired = rng.random() < pm

    if gaussian_fired:
        selected = w > 0
        w[selected] += rng.normal(0.0, SIGMA_M, size=selected.sum())
        w[w < 0.0] = 0.0
        w[w > 1.0] = 1.0

    if swap_fired:
        non_zero = np.nonzero(w > 0)[0]
        zero = np.nonzero(w == 0)[0]
        if len(non_zero) > 0 and len(zero) > 0:
            w[rng.choice(non_zero)] = 0.0
            w[rng.choice(zero)] = W_MIN

    if gaussian_fired or swap_fired:
        # phantom tiny positives from Gaussian noise confuse repair
        w[w < 1e-10] = 0.0
        try:
            w = repair(w, rng)
        except ValueError:
            w = repair(np.ones_like(w) / len(w), rng)

    return w


def local_refine(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                 prev_weights: np.ndarray | None, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """
    Greedy pairwise weight-shift (LOCAL_STEP from i → j) for LOCAL_ITER steps.
    Accepted only on fitness improvement. Sum and cardinality preserved — no repair needed.
    Returns (refined_w, fitness) to avoid recomputation in caller.
    """
    w_best = w.copy()
    f_best = fitness(w_best, mu, sigma, prev_weights, LAMBDA)

    for _ in range(LOCAL_ITER):
        held = np.nonzero(w_best > 1e-10)[0]
        if len(held) < 2:
            break

        i, j = rng.choice(held, size=2, replace=False)

        # maximum transferable amount that keeps both weights in [W_MIN, W_MAX]
        delta = min(LOCAL_STEP,
                    w_best[i] - W_MIN,
                    W_MAX - w_best[j])
        if delta <= 0:
            continue

        w_cand = w_best.copy()
        w_cand[i] -= delta
        w_cand[j] += delta

        f_cand = fitness(w_cand, mu, sigma, prev_weights, LAMBDA)
        if f_cand > f_best:
            w_best = w_cand
            f_best = f_cand

    return w_best, f_best


def run_ga(n_assets: int, mu: np.ndarray, sigma: np.ndarray,
           prev_weights: np.ndarray | None, rng: np.random.Generator,
           return_history: bool = False) -> np.ndarray:
    """
    Generational GA: elites preserved (best one gets local refinement), rest via selection/crossover/mutation.
    Fitness carried over for elites; recomputed for offspring only.
    Early stop after EARLY_STOP consecutive generations with improvement < 1e-6.
    """
    population = initialize_population(n_assets, rng)
    fitnesses = np.array([
        fitness(population[i], mu, sigma, prev_weights, LAMBDA)
        for i in range(POP_SIZE)
    ])

    n_elite = max(1, int(POP_SIZE * ELITE_FRAC))
    best_idx = int(np.argmax(fitnesses))
    best_ever_w = population[best_idx].copy()
    best_ever_f = fitnesses[best_idx]
    stagnation = 0
    fitness_history = []

    for _ in range(N_GENS):

        # sorted ascending — last entry is the best
        elite_idx = np.argsort(fitnesses)[-n_elite:]
        elites = population[elite_idx].copy()
        elite_fit = fitnesses[elite_idx].copy()

        # local_refine returns (w, f) — fitness reused, avoiding one redundant evaluation
        elites[-1], elite_fit[-1] = local_refine(
            elites[-1], mu, sigma, prev_weights, rng
        )

        # elites occupy slots 0..n_elite-1
        new_pop = np.empty_like(population)
        new_pop[:n_elite] = elites

        i = n_elite
        while i < POP_SIZE:
            p1 = tournament_select(population, fitnesses, rng)
            p2 = tournament_select(population, fitnesses, rng)
            c1, c2 = crossover(p1, p2, rng, pc=PC)
            c1 = mutate(c1, rng, pm=PM)
            c2 = mutate(c2, rng, pm=PM)
            new_pop[i] = c1
            i += 1
            if i < POP_SIZE:
                new_pop[i] = c2
                i += 1

        population = new_pop

        # carry over elite fitness; recompute offspring only
        new_fit = np.empty(POP_SIZE)
        new_fit[:n_elite] = elite_fit
        for i in range(n_elite, POP_SIZE):
            new_fit[i] = fitness(
                population[i], mu, sigma, prev_weights, LAMBDA
            )
        fitnesses = new_fit

        gen_best_idx = int(np.argmax(fitnesses))
        gen_best_f = fitnesses[gen_best_idx]

        if gen_best_f - best_ever_f > 1e-6:
            best_ever_f = gen_best_f
            best_ever_w = population[gen_best_idx].copy()
            stagnation = 0
        else:
            stagnation += 1

        fitness_history.append(best_ever_f)
        if stagnation >= EARLY_STOP:
            break

    if return_history:
        return best_ever_w, fitness_history
    return best_ever_w
