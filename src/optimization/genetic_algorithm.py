"""
Genetic Algorithm for cardinality-constrained portfolio optimization.
Chromosome: real-valued weight vector of length N (universe size).
Exactly K entries are non-zero, with w_i in [W_MIN, W_MAX] summing to 1.
"""

import numpy as np
from src.evaluation.metrics import portfolio_turnover

# not tuned by Optuna
K_MIN = 10
K_MAX = 30
W_MIN = 0.02
W_MAX = 0.15
POP_SIZE = 100
N_GENS = 200
ELITE_FRAC = 0.05  # top fraction kept each generation
LOCAL_ITER = 5  # hill-climber steps per elite refinement
LOCAL_STEP = 0.01  # weight-shift step size delta
EARLY_STOP = 20  # stagnation patience in generations
TOURNAMENT = 3  # tournament size

PC = 0.6054
PM = 0.1370
SIGMA_M = 0.1469
LAMBDA = 1.8437

DEBUG = False

def project_bounded_simplex(v, lower, upper, tol=1e-12):
    """Project weights onto the feasible simplex."""
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
    """Fix a chromosome so it satisfies all constraints."""
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
            # Small universe fallback.
            w[zeros] = W_MIN

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
    """Random feasible starting population."""
    population = np.zeros((POP_SIZE, n_assets))

    for i in range(POP_SIZE):
        K = rng.integers(K_MIN, K_MAX + 1)
        selected = rng.choice(n_assets, size=K, replace=False)

        raw_weights = rng.dirichlet(np.ones(K))

        w = np.zeros(n_assets)
        w[selected] = raw_weights
        population[i] = repair(w, rng)

    return population

def fitness(weights: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
            prev_weights: np.ndarray | None = None, lambda_: float = LAMBDA) -> float:
    """Sharpe minus turnover penalty."""
    # using only the held stocks avoids polluting the Sharpe with the
    # full-universe covariance, zeros in weights contribute nothing but noise
    held = weights > 0
    if not held.any():
        return 0.0
    w_held = weights[held]
    mu_held = mu[held]
    sigma_held = sigma[np.ix_(held, held)]

    port_return = float(w_held @ mu_held)
    port_var = float(w_held @ sigma_held @ w_held)

    if port_var <= 0.0:
        sharpe = 0.0
    else:
        sharpe = port_return / np.sqrt(port_var)

    if prev_weights is None or lambda_ < 1e-12:
        penalty = 0.0
    else:
        # turnover on full vectors so stocks exiting the portfolio count
        penalty = lambda_ * portfolio_turnover(weights, prev_weights)

    return sharpe - penalty


def tournament_select(population: np.ndarray, fitnesses: np.ndarray,
                      rng: np.random.Generator, k: int = TOURNAMENT) -> np.ndarray:
    """Pick the best member of a random tournament."""
    k = min(k, len(population))
    indices = rng.choice(len(population), size=k, replace=False)
    winner = indices[np.argmax(fitnesses[indices])]
    return population[winner].copy()

def _make_child(w1: np.ndarray, w2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Build one child from the parents' active holdings."""
    N = len(w1)
    union = np.nonzero((w1 > 0) | (w2 > 0))[0]
    if len(union) == 0:
        return repair(np.ones(len(w1))/len(w1), rng)

    k_child = rng.integers(K_MIN, K_MAX + 1)
    k_child = min(k_child, len(union))

    probs = (w1[union] + w2[union]) / 2.0
    s = probs.sum()
    if s <= 1e-12:
        probs = np.ones(len(union)) / len(union)
    else:
        probs /= s

    selected = rng.choice(union, size=k_child, replace=False, p=probs)

    alpha = rng.uniform(0.0, 1.0)
    w_child = np.zeros(N)
    w_child[selected] = alpha * w1[selected] + (1.0 - alpha) * w2[selected]

    w_child[w_child < 1e-10] = 0.0

    return repair(w_child, rng)


def crossover(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator,
              pc: float = PC) -> tuple[np.ndarray, np.ndarray]:
    """Crossover operator."""
    if rng.random() > pc:
        return p1.copy(), p2.copy()

    child1 = _make_child(p1, p2, rng)
    child2 = _make_child(p2, p1, rng)
    return child1, child2


def mutate(w: np.ndarray, rng: np.random.Generator, pm: float = PM) -> np.ndarray:
    """Weight noise plus occasional asset swap."""
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
        w[w < 1e-10] = 0.0
        try:
            w = repair(w, rng)
        except ValueError:
            w = repair(np.ones_like(w) / len(w), rng)

    return w


def local_refine(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                 prev_weights: np.ndarray | None, rng: np.random.Generator,
                 lambda_: float = LAMBDA) -> tuple[np.ndarray, float]:
    """Small greedy weight shifts on the current holdings."""
    w_best = w.copy()
    f_best = fitness(w_best, mu, sigma, prev_weights, lambda_)

    for _ in range(LOCAL_ITER):
        held = np.nonzero(w_best > 1e-10)[0]
        if len(held) < 2:
            break

        i, j = rng.choice(held, size=2, replace=False)

        delta = min(LOCAL_STEP,
                    w_best[i] - W_MIN,
                    W_MAX - w_best[j])
        if delta <= 0:
            continue

        w_cand = w_best.copy()
        w_cand[i] -= delta
        w_cand[j] += delta

        f_cand = fitness(w_cand, mu, sigma, prev_weights, lambda_)
        if f_cand > f_best:
            w_best = w_cand
            f_best = f_cand

    return w_best, f_best


def run_ga(n_assets: int, mu: np.ndarray, sigma: np.ndarray,
           prev_weights: np.ndarray | None, rng: np.random.Generator,
           return_history: bool = False) -> np.ndarray:
    """Run the genetic algorithm for one rebalance date."""
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

        elite_idx = np.argsort(fitnesses)[-n_elite:]
        elites = population[elite_idx].copy()
        elite_fit = fitnesses[elite_idx].copy()

        elites[-1], elite_fit[-1] = local_refine(
            elites[-1], mu, sigma, prev_weights, rng, lambda_=LAMBDA
        )

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
