"""
tests/test_genetic_algorithm.py
Research-grade validation for src/optimization/genetic_algorithm.py

Components tested:
    project_bounded_simplex  — bounded simplex projection (mathematical core)
    repair                   — constraint enforcement operator
    initialize_population    — feasible population generation
    fitness                  — objective function with turnover penalty
    tournament_select        — parent selection operator
    crossover                — two-child crossover operator
    mutate                   — Gaussian + asset-swap mutation operator

Run with:
    python3 -m pytest tests/test_genetic_algorithm.py -v
"""

import numpy as np
import pytest

from src.optimization.genetic_algorithm import (
    project_bounded_simplex,
    repair,
    initialize_population,
    fitness,
    tournament_select,
    crossover,
    mutate,
    K_MIN, K_MAX, W_MIN, W_MAX, POP_SIZE, TOURNAMENT, LAMBDA,
)
from src.evaluation.metrics import portfolio_turnover


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_feasible_portfolio(rng, n=50):
    """Return one verified-feasible portfolio of length n."""
    w = np.zeros(n)
    selected = rng.choice(n, size=K_MIN, replace=False)
    w[selected] = 1.0 / K_MIN
    return repair(w, rng)


def two_feasible(rng, n=100):
    """Return two independent feasible portfolios of length n."""
    pop = initialize_population(n, rng)
    return pop[0], pop[1]


def make_mu_sigma(n=5, monthly_ret=0.01, monthly_std=0.04):
    """Simple diagonal mu/sigma for fitness tests."""
    mu    = np.full(n, monthly_ret)
    sigma = np.diag(np.full(n, monthly_std ** 2))
    return mu, sigma


# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    return np.random.default_rng(42)


# project_bounded_simplex
# ─────────────────────────────────────────────────────────────────────────────

class TestProjectBoundedSimplex:

    def test_output_sums_to_one(self):
        """Projected vector must sum to exactly 1."""
        rng = np.random.default_rng(0)
        for _ in range(100):
            v = rng.uniform(0, 1, 20)
            w = project_bounded_simplex(v, W_MIN, W_MAX)
            assert abs(w.sum() - 1.0) < 1e-10, f"sum={w.sum()}"

    def test_bounds_respected(self):
        """All entries must satisfy lower <= w_i <= upper."""
        rng = np.random.default_rng(1)
        for _ in range(100):
            n = rng.integers(5, 30)
            v = rng.uniform(0, 1, n)
            lower = 1.0 / (2 * n)
            upper = min(0.9, 1.0 / max(1, n // 3))
            if n * lower > 1 or n * upper < 1:
                continue
            w = project_bounded_simplex(v, lower, upper)
            assert np.all(w >= lower - 1e-10)
            assert np.all(w <= upper + 1e-10)

    def test_already_feasible_vector_unchanged(self):
        """A vector already on the bounded simplex must be returned unchanged."""
        K = 15
        v = np.full(K, 1.0 / K)
        w = project_bounded_simplex(v, W_MIN, W_MAX)
        np.testing.assert_allclose(w, v, atol=1e-10)

    def test_infeasible_bounds_raises(self):
        """Both infeasible cases must raise ValueError:
        (a) n * lower > 1  — lower bound too large to satisfy sum=1
        (b) n * upper < 1  — upper bound too small to satisfy sum=1
        """
        v = np.full(5, 0.2)

        # Case (a): n * lower > 1  (5 * 0.3 = 1.5 > 1)
        with pytest.raises(ValueError):
            project_bounded_simplex(v, lower=0.3, upper=0.5)

        # Case (b): n * upper < 1  (5 * 0.1 = 0.5 < 1)
        with pytest.raises(ValueError):
            project_bounded_simplex(v, lower=0.01, upper=0.1)

    def test_extreme_concentration_resolved(self):
        """A vector with all weight on one entry must be spread to satisfy lower."""
        v = np.zeros(10)
        v[0] = 1.0
        w = project_bounded_simplex(v, W_MIN, W_MAX)
        assert abs(w.sum() - 1.0) < 1e-10
        assert np.all(w >= W_MIN - 1e-10)
        assert np.all(w <= W_MAX + 1e-10)


# repair
# ─────────────────────────────────────────────────────────────────────────────

class TestRepair:

    def _assert_feasible(self, w):
        selected = w > 0
        assert K_MIN <= selected.sum() <= K_MAX, \
            f"Cardinality {selected.sum()} outside [{K_MIN}, {K_MAX}]"
        assert abs(w.sum() - 1.0) < 1e-8, f"Weights sum to {w.sum()}"
        assert np.all(w[selected] >= W_MIN - 1e-8), \
            f"Min weight {w[selected].min()} < W_MIN={W_MIN}"
        assert np.all(w[selected] <= W_MAX + 1e-8), \
            f"Max weight {w[selected].max()} > W_MAX={W_MAX}"

    def test_too_many_stocks_reduced_to_k_max(self, rng):
        """K_MAX + 5 non-zero entries must be reduced to at most K_MAX."""
        N = 100
        w = np.zeros(N)
        idx = rng.choice(N, size=K_MAX + 5, replace=False)
        w[idx] = 1.0 / (K_MAX + 5)
        result = repair(w, rng)
        self._assert_feasible(result)
        assert (result > 0).sum() <= K_MAX

    def test_too_few_stocks_raised_to_k_min(self, rng):
        """K_MIN - 1 non-zero entries must be raised to at least K_MIN."""
        N = 100
        w = np.zeros(N)
        idx = rng.choice(N, size=K_MIN - 1, replace=False)
        w[idx] = 1.0 / (K_MIN - 1)
        result = repair(w, rng)
        self._assert_feasible(result)
        assert (result > 0).sum() >= K_MIN

    def test_all_constraints_simultaneously_satisfied(self, rng):
        """Output must satisfy cardinality, bounds, and sum=1 simultaneously."""
        N = 200
        w = np.zeros(N)
        w[:50] = 0.001   # too many stocks, weights below W_MIN
        w[0]   = 0.9     # weight above W_MAX
        result = repair(w, rng)
        self._assert_feasible(result)

    def test_idempotence(self, rng):
        """repair(repair(w)) must equal repair(w) independent of RNG state."""
        N    = 200
        rng2 = np.random.default_rng(99)
        for _ in range(200):
            w = rng2.dirichlet(np.ones(N))
            zero_mask = rng2.random(N) < 0.85
            w[zero_mask] = 0.0
            if w.sum() > 0:
                w /= w.sum()

            # Use different seeds for first and second repair call to expose
            # any stochastic inconsistency — true idempotence must hold
            # regardless of which random choices are made during repair.
            r1 = repair(w,         np.random.default_rng(0))
            r2 = repair(r1.copy(), np.random.default_rng(1))
            np.testing.assert_allclose(r1, r2, atol=1e-10,
                                       err_msg="repair is not idempotent")

    def test_fuzz_random_inputs(self):
        """repair must produce feasible output for 10,000 random inputs."""
        N        = 867
        rng_fuzz = np.random.default_rng(123)
        violations = 0
        for _ in range(10_000):
            w = rng_fuzz.dirichlet(np.ones(N))
            zero_mask = rng_fuzz.random(N) < 0.97
            w[zero_mask] = 0.0
            total = w.sum()
            if total > 0:
                w /= total
            else:
                w[rng_fuzz.integers(N)] = 1.0

            result   = repair(w, rng_fuzz)
            selected = result > 0
            if not (K_MIN <= selected.sum() <= K_MAX
                    and abs(result.sum() - 1.0) < 1e-8
                    and (result[selected] >= W_MIN - 1e-8).all()
                    and (result[selected] <= W_MAX + 1e-8).all()):
                violations += 1

        assert violations == 0, \
            f"{violations}/10,000 repair outputs violated constraints"

    def test_valid_input_passes_through_unchanged(self, rng):
        """A vector already satisfying all constraints must be returned unchanged."""
        N      = 50
        w      = make_feasible_portfolio(rng, N)
        result = repair(w.copy(), rng)
        np.testing.assert_allclose(result, w, atol=1e-10)

    def test_no_nan_in_output(self):
        """repair must never produce NaN values for any input."""
        N        = 867
        rng_fuzz = np.random.default_rng(777)
        for _ in range(1_000):
            w = rng_fuzz.dirichlet(np.ones(N))
            zero_mask = rng_fuzz.random(N) < 0.97
            w[zero_mask] = 0.0
            total = w.sum()
            if total > 0:
                w /= total
            else:
                w[rng_fuzz.integers(N)] = 1.0

            result = repair(w, rng_fuzz)
            assert not np.any(np.isnan(result)), \
                "repair produced NaN output"


# initialize_population
# ─────────────────────────────────────────────────────────────────────────────

class TestInitializePopulation:

    def _all_feasible(self, pop):
        for i in range(len(pop)):
            w        = pop[i]
            selected = w > 0
            assert K_MIN <= selected.sum() <= K_MAX, \
                f"Row {i}: cardinality {selected.sum()}"
            assert abs(w.sum() - 1.0) < 1e-8, \
                f"Row {i}: sum={w.sum()}"
            assert np.all(w[selected] >= W_MIN - 1e-8), \
                f"Row {i}: min weight {w[selected].min()}"
            assert np.all(w[selected] <= W_MAX + 1e-8), \
                f"Row {i}: max weight {w[selected].max()}"

    def test_all_rows_feasible(self, rng):
        """All POP_SIZE rows must satisfy every constraint."""
        pop = initialize_population(867, rng)
        assert pop.shape == (POP_SIZE, 867)
        self._all_feasible(pop)

    def test_output_shape(self, rng):
        """Shape must be (POP_SIZE, N) exactly."""
        for N in [50, 200, 867]:
            pop = initialize_population(N, rng)
            assert pop.shape == (POP_SIZE, N), \
                f"Expected ({POP_SIZE}, {N}), got {pop.shape}"

    def test_no_duplicate_rows(self, rng):
        """Initial population must have very few duplicate chromosomes (< 5 allowed)."""
        pop        = initialize_population(200, rng)
        seen       = set()
        duplicates = 0
        for row in pop:
            h = row.tobytes()
            if h in seen:
                duplicates += 1
            seen.add(h)
        assert duplicates < 5, \
            f"Too many duplicate chromosomes in initial population: {duplicates}"

    def test_cardinality_spread_across_range(self, rng):
        """Cardinalities should span multiple values across [K_MIN, K_MAX]."""
        pop           = initialize_population(200, rng)
        cardinalities = np.array([(row > 0).sum() for row in pop])
        assert cardinalities.min() >= K_MIN
        assert cardinalities.max() <= K_MAX
        assert len(np.unique(cardinalities)) >= 3, \
            f"Only {len(np.unique(cardinalities))} distinct cardinalities"


# fitness
# ─────────────────────────────────────────────────────────────────────────────

class TestFitness:

    def test_lambda_zero_equals_sharpe(self):
        """With λ=0, fitness must equal in-sample (non-annualised) Sharpe."""
        mu, sigma = make_mu_sigma()
        w         = np.full(5, 0.2)
        expected  = float((w @ mu) / np.sqrt(w @ sigma @ w))
        result    = fitness(w, mu, sigma, prev_weights=None, lambda_=0.0)
        assert abs(result - expected) < 1e-10

    def test_first_period_no_penalty(self):
        """prev_weights=None must produce zero penalty regardless of λ."""
        mu, sigma = make_mu_sigma()
        w         = np.full(5, 0.2)
        f_none    = fitness(w, mu, sigma, prev_weights=None, lambda_=2.0)
        f_zero    = fitness(w, mu, sigma, prev_weights=None, lambda_=0.0)
        assert abs(f_none - f_zero) < 1e-10

    def test_penalty_proportional_to_lambda(self):
        """F(λ=0) − F(λ=1) must equal exactly turnover(w, prev)."""
        mu, sigma = make_mu_sigma()
        w         = np.full(5, 0.2)
        prev      = np.array([0.3, 0.2, 0.1, 0.2, 0.2])
        to        = portfolio_turnover(w, prev)
        f0        = fitness(w, mu, sigma, prev_weights=prev, lambda_=0.0)
        f1        = fitness(w, mu, sigma, prev_weights=prev, lambda_=1.0)
        assert abs((f0 - f1) - to) < 1e-10

    def test_identical_prev_weights_no_penalty(self):
        """Zero turnover must not change fitness even at high λ."""
        mu, sigma     = make_mu_sigma()
        w             = np.full(5, 0.2)
        f_high_lambda = fitness(w, mu, sigma, prev_weights=w.copy(), lambda_=5.0)
        f_no_prev     = fitness(w, mu, sigma, prev_weights=None,      lambda_=0.0)
        assert abs(f_high_lambda - f_no_prev) < 1e-10

    def test_zero_variance_returns_zero(self):
        """Portfolio variance = 0 must return 0.0, not inf or NaN."""
        mu     = np.full(5, 0.01)
        sigma  = np.zeros((5, 5))
        w      = np.full(5, 0.2)
        result = fitness(w, mu, sigma)
        assert abs(result - 0.0) < 1e-10
        assert not np.isnan(result)

    def test_penalty_strictly_reduces_fitness(self):
        """Non-zero λ with non-zero turnover must strictly reduce fitness."""
        mu, sigma   = make_mu_sigma()
        w           = np.full(5, 0.2)
        prev        = np.array([0.1, 0.3, 0.1, 0.3, 0.2])
        assert portfolio_turnover(w, prev) > 0
        f_penalised   = fitness(w, mu, sigma, prev_weights=prev, lambda_=1.0)
        f_unpenalised = fitness(w, mu, sigma, prev_weights=prev, lambda_=0.0)
        assert f_penalised < f_unpenalised

    def test_higher_return_dominates_same_risk(self):
        """Doubling mu with same sigma must strictly increase fitness."""
        mu, sigma = make_mu_sigma()
        w         = np.full(5, 0.2)
        assert fitness(w, mu * 2, sigma, lambda_=0.0) > \
               fitness(w, mu,     sigma, lambda_=0.0)

    def test_higher_risk_reduces_fitness_same_return(self):
        """Quadrupling sigma (doubling vol) with same mu must strictly decrease fitness."""
        mu, sigma = make_mu_sigma()
        w         = np.full(5, 0.2)
        assert fitness(w, mu, sigma * 4, lambda_=0.0) < \
               fitness(w, mu, sigma,     lambda_=0.0)


# tournament_select
# ─────────────────────────────────────────────────────────────────────────────

class TestTournamentSelect:

    def _make_pop_fitnesses(self, n=20, d=10, seed=0):
        rng       = np.random.default_rng(seed)
        pop       = rng.dirichlet(np.ones(d), size=n)
        fitnesses = rng.random(n)
        return pop, fitnesses

    def test_winner_is_in_population(self):
        """Winner must be identical to exactly one population row."""
        rng            = np.random.default_rng(5)
        pop, fitnesses = self._make_pop_fitnesses()
        for _ in range(200):
            winner  = tournament_select(pop, fitnesses, rng)
            matches = np.all(np.isclose(pop, winner), axis=1)
            assert matches.sum() == 1, "Winner not found in population"

    def test_returns_copy_not_view(self):
        """Mutating the returned winner must not corrupt the population."""
        rng            = np.random.default_rng(6)
        pop, fitnesses = self._make_pop_fitnesses()
        original       = pop.copy()
        winner         = tournament_select(pop, fitnesses, rng)
        winner[:]      = 999.0
        np.testing.assert_array_equal(pop, original)

    def test_selection_pressure_empirical(self):
        """
        Best individual must be selected significantly more often than chance.
        Threshold is set relative to uniform baseline (1/n) to remain valid
        if TOURNAMENT changes.
        """
        n         = 20
        rng       = np.random.default_rng(7)
        pop       = np.eye(n)
        fitnesses = np.arange(n, dtype=float)
        wins      = sum(
            np.array_equal(
                tournament_select(pop, fitnesses, rng, k=TOURNAMENT),
                pop[-1]
            )
            for _ in range(5_000)
        )
        win_rate = wins / 5_000
        assert win_rate > (1 / n) * 2, \
            f"Selection pressure too low: best chosen {win_rate:.2%} of time " \
            f"(threshold: {(1/n)*2:.2%})"

    def test_k1_tournament_is_uniform_sampling(self):
        """k=1 reduces to uniform random selection — all individuals must appear."""
        rng       = np.random.default_rng(8)
        n         = 10
        pop       = np.eye(n)
        fitnesses = np.ones(n)
        seen      = set()
        for _ in range(500):
            winner = tournament_select(pop, fitnesses, rng, k=1)
            seen.add(tuple(winner))
        assert len(seen) == n, \
            f"k=1 tournament did not explore full population: saw {len(seen)}/{n}"

    def test_output_shape_matches_chromosome(self):
        """Output shape must match one chromosome row."""
        rng = np.random.default_rng(9)
        for N in [20, 200, 867]:
            pop       = rng.dirichlet(np.ones(N), size=POP_SIZE)
            fitnesses = rng.random(POP_SIZE)
            winner    = tournament_select(pop, fitnesses, rng)
            assert winner.shape == (N,)


# crossover
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossover:

    def _assert_feasible(self, w, label=""):
        sel = w > 0
        assert K_MIN <= sel.sum() <= K_MAX,     f"{label} cardinality {sel.sum()}"
        assert abs(w.sum() - 1.0) < 1e-8,      f"{label} sum={w.sum()}"
        assert np.all(w[sel] >= W_MIN - 1e-8),  f"{label} min {w[sel].min()}"
        assert np.all(w[sel] <= W_MAX + 1e-8),  f"{label} max {w[sel].max()}"

    def test_children_feasible(self, rng):
        """Both children must satisfy all constraints."""
        p1, p2 = two_feasible(rng)
        c1, c2 = crossover(p1, p2, rng, pc=1.0)
        self._assert_feasible(c1, "c1")
        self._assert_feasible(c2, "c2")

    def test_pc_zero_returns_parent_copies(self, rng):
        """pc=0 must return exact copies of both parents unchanged."""
        p1, p2 = two_feasible(rng)
        c1, c2 = crossover(p1, p2, rng, pc=0.0)
        np.testing.assert_array_equal(c1, p1)
        np.testing.assert_array_equal(c2, p2)

    def test_child_stocks_subset_of_union(self, rng):
        """Child holdings must come exclusively from the union of parental holdings."""
        p1, p2 = two_feasible(rng, n=200)
        union  = (p1 > 0) | (p2 > 0)
        for _ in range(50):
            c1, c2 = crossover(p1, p2, rng, pc=1.0)
            assert np.all((c1 > 0) <= union), "c1 holds stock outside parental union"
            assert np.all((c2 > 0) <= union), "c2 holds stock outside parental union"

    def test_returns_copies_not_views(self, rng):
        """Mutating a child must not corrupt the parent arrays."""
        p1, p2  = two_feasible(rng)
        p1_snap = p1.copy()
        c1, _   = crossover(p1, p2, rng, pc=1.0)
        c1[:]   = 999.0
        np.testing.assert_array_equal(p1, p1_snap)

    def test_identical_parents_still_feasible(self, rng):
        """Crossover of a portfolio with itself must produce a feasible child."""
        p1, _  = two_feasible(rng)
        c1, c2 = crossover(p1, p1, rng, pc=1.0)
        self._assert_feasible(c1, "identical-parent c1")
        self._assert_feasible(c2, "identical-parent c2")

    def test_cardinality_varies_across_children(self, rng):
        """
        Random K_child draw should produce multiple distinct cardinalities —
        crossover must be the source of cardinality exploration.
        """
        p1, p2 = two_feasible(rng, n=200)
        cards  = set()
        for _ in range(200):
            c1, c2 = crossover(p1, p2, rng, pc=1.0)
            cards.add(int((c1 > 0).sum()))
            cards.add(int((c2 > 0).sum()))
        assert len(cards) >= 5, \
            f"Only {len(cards)} distinct cardinalities — K_child draw may be broken"

    def test_stress_feasibility(self):
        """5,000 crossover calls (10,000 children) at N=867 must all be feasible."""
        rng        = np.random.default_rng(0)
        pop        = initialize_population(867, rng)
        violations = 0
        for i in range(5_000):
            p1, p2 = pop[i % POP_SIZE], pop[(i + 1) % POP_SIZE]
            c1, c2 = crossover(p1, p2, rng, pc=1.0)
            for c in (c1, c2):
                sel = c > 0
                if not (K_MIN <= sel.sum() <= K_MAX
                        and abs(c.sum() - 1.0) < 1e-8
                        and (c[sel] >= W_MIN - 1e-8).all()
                        and (c[sel] <= W_MAX + 1e-8).all()):
                    violations += 1
        assert violations == 0, \
            f"{violations}/10,000 crossover outputs infeasible"

    def test_symmetry(self, rng):
        """crossover(p1, p2) and crossover(p2, p1) must both produce feasible outputs."""
        p1, p2   = two_feasible(rng, n=200)
        c1a, c2a = crossover(p1, p2, rng, pc=1.0)
        c1b, c2b = crossover(p2, p1, rng, pc=1.0)
        for label, c in [("p1p2-c1", c1a), ("p1p2-c2", c2a),
                          ("p2p1-c1", c1b), ("p2p1-c2", c2b)]:
            self._assert_feasible(c, label)


# mutate
# ─────────────────────────────────────────────────────────────────────────────

class TestMutate:

    def _assert_feasible(self, w, label=""):
        sel = w > 0
        assert K_MIN <= sel.sum() <= K_MAX,     f"{label} cardinality {sel.sum()}"
        assert abs(w.sum() - 1.0) < 1e-8,      f"{label} sum={w.sum()}"
        assert np.all(w[sel] >= W_MIN - 1e-8),  f"{label} min {w[sel].min()}"
        assert np.all(w[sel] <= W_MAX + 1e-8),  f"{label} max {w[sel].max()}"

    def test_output_feasible(self, rng):
        """Output must satisfy all constraints at pm=1.0."""
        p1, _ = two_feasible(rng)
        self._assert_feasible(mutate(p1, rng, pm=1.0), "pm=1")

    def test_pm_zero_returns_unchanged_copy(self, rng):
        """pm=0 must return an exact copy with no modification."""
        p1, _ = two_feasible(rng)
        result = mutate(p1, rng, pm=0.0)
        np.testing.assert_array_equal(result, p1)

    def test_returns_copy_not_view(self, rng):
        """Mutating the output must not corrupt the input."""
        p1, _     = two_feasible(rng)
        original  = p1.copy()
        result    = mutate(p1, rng, pm=0.0)
        result[:] = 999.0
        np.testing.assert_array_equal(p1, original)

    def test_pm_one_usually_changes_output(self):
        """pm=1.0 must produce output that differs from input in >= 90% of calls."""
        rng   = np.random.default_rng(5)
        p1, _ = two_feasible(rng, n=200)
        unchanged = sum(
            np.array_equal(mutate(p1, np.random.default_rng(i), pm=1.0), p1)
            for i in range(100)
        )
        assert unchanged / 100 < 0.1, \
            f"pm=1.0 left output unchanged in {unchanged}/100 calls"

    def test_cardinality_bounds_respected(self, rng):
        """Mutation output must always respect K_MIN <= nonzero weights <= K_MAX."""
        pop        = initialize_population(200, rng)
        violations = 0
        for i in range(500):
            result = mutate(pop[i % POP_SIZE], rng, pm=1.0)
            k      = (result > 0).sum()
            if not (K_MIN <= k <= K_MAX):
                violations += 1
        assert violations == 0, \
            f"{violations}/500 mutate outputs violated cardinality bounds"

    def test_stress_feasibility(self):
        """10,000 mutation calls at N=867 must all produce feasible outputs."""
        rng        = np.random.default_rng(7)
        pop        = initialize_population(867, rng)
        violations = 0
        for i in range(10_000):
            result = mutate(pop[i % POP_SIZE], rng, pm=1.0)
            sel    = result > 0
            if not (K_MIN <= sel.sum() <= K_MAX
                    and abs(result.sum() - 1.0) < 1e-8
                    and (result[sel] >= W_MIN - 1e-8).all()
                    and (result[sel] <= W_MAX + 1e-8).all()):
                violations += 1
        assert violations == 0, \
            f"{violations}/10,000 mutate outputs infeasible"

    def test_sigma_m_upper_bound_still_feasible(self):
        """
        At SIGMA_M=0.15 (Optuna upper bound), repair must handle aggressive
        perturbations — feasibility must hold even at maximum noise level.
        """
        import src.optimization.genetic_algorithm as ga
        original   = ga.SIGMA_M
        ga.SIGMA_M = 0.15
        try:
            rng        = np.random.default_rng(8)
            pop        = initialize_population(867, rng)
            violations = 0
            for i in range(2_000):
                result = mutate(pop[i % POP_SIZE], rng, pm=1.0)
                sel    = result > 0
                if not (K_MIN <= sel.sum() <= K_MAX
                        and abs(result.sum() - 1.0) < 1e-8
                        and (result[sel] >= W_MIN - 1e-8).all()
                        and (result[sel] <= W_MAX + 1e-8).all()):
                    violations += 1
            assert violations == 0, \
                f"{violations}/2,000 infeasible at SIGMA_M=0.15"
        finally:
            ga.SIGMA_M = original


# local_refine
# ─────────────────────────────────────────────────────────────────────────────

from src.optimization.genetic_algorithm import local_refine, run_ga


def make_estimation_data(n=50, t=60, seed=0):
    """Synthetic mu and sigma for local_refine and run_ga tests."""
    rng        = np.random.default_rng(seed)
    ret_matrix = rng.normal(0.008, 0.05, size=(t, n))
    mu         = ret_matrix.mean(axis=0)
    sigma      = np.cov(ret_matrix, rowvar=False)
    return mu, sigma


class TestLocalRefine:

    N = 100  # universe size — must match two_feasible default

    def _assert_feasible(self, w, label=""):
        sel = w > 0
        assert K_MIN <= sel.sum() <= K_MAX,     f"{label} cardinality {sel.sum()}"
        assert abs(w.sum() - 1.0) < 1e-8,      f"{label} sum={w.sum()}"
        assert np.all(w[sel] >= W_MIN - 1e-8),  f"{label} min {w[sel].min()}"
        assert np.all(w[sel] <= W_MAX + 1e-8),  f"{label} max {w[sel].max()}"

    def test_output_feasible(self, rng):
        """Refined chromosome must satisfy all constraints."""
        mu, sigma  = make_estimation_data(n=self.N)
        p1, _      = two_feasible(rng, n=self.N)
        w_out, _   = local_refine(p1, mu, sigma, None, rng)
        self._assert_feasible(w_out, "local_refine")

    def test_returns_tuple(self, rng):
        """local_refine must return (w, f) — caller unpacks both."""
        mu, sigma = make_estimation_data(n=self.N)
        p1, _     = two_feasible(rng, n=self.N)
        result    = local_refine(p1, mu, sigma, None, rng)
        assert isinstance(result, tuple) and len(result) == 2, \
            "local_refine must return a (w, f) tuple"
        w_out, f_out = result
        assert isinstance(f_out, float), "fitness return must be a float"
        assert w_out.shape == p1.shape

    def test_fitness_non_decreasing(self, rng):
        """Returned fitness must be >= input fitness (greedy acceptance)."""
        mu, sigma = make_estimation_data(n=self.N)
        p1, _     = two_feasible(rng, n=self.N)
        f_in      = fitness(p1, mu, sigma, None, LAMBDA)
        _, f_out  = local_refine(p1, mu, sigma, None, rng)
        assert f_out >= f_in - 1e-12, \
            f"local_refine decreased fitness: {f_in:.6f} -> {f_out:.6f}"

    def test_returned_fitness_matches_weight(self, rng):
        """The returned f must equal fitness(w_out) — not a stale value."""
        mu, sigma    = make_estimation_data(n=self.N)
        p1, _        = two_feasible(rng, n=self.N)
        w_out, f_out = local_refine(p1, mu, sigma, None, rng)
        f_recomputed = fitness(w_out, mu, sigma, None, LAMBDA)
        assert abs(f_out - f_recomputed) < 1e-10, \
            "Returned fitness does not match fitness(w_out)"

    def test_cardinality_preserved(self, rng):
        """Pairwise weight shift must not change the number of held stocks."""
        mu, sigma = make_estimation_data(n=self.N)
        p1, _     = two_feasible(rng, n=self.N)
        k_in      = (p1 > 0).sum()
        w_out, _  = local_refine(p1, mu, sigma, None, rng)
        k_out     = (w_out > 0).sum()
        assert k_in == k_out, \
            f"local_refine changed cardinality: {k_in} -> {k_out}"

    def test_unchanged_input(self, rng):
        """local_refine must not mutate the input chromosome."""
        mu, sigma = make_estimation_data(n=self.N)
        p1, _     = two_feasible(rng, n=self.N)
        p1_snap   = p1.copy()
        local_refine(p1, mu, sigma, None, rng)
        np.testing.assert_array_equal(p1, p1_snap)

    def test_prev_weights_path(self, rng):
        """local_refine must accept prev_weights without error."""
        mu, sigma    = make_estimation_data(n=self.N)
        p1, p2       = two_feasible(rng, n=self.N)
        w_out, f_out = local_refine(p1, mu, sigma, p2, rng)
        self._assert_feasible(w_out, "with prev_weights")
        assert isinstance(f_out, float)


# run_ga
# ─────────────────────────────────────────────────────────────────────────────

class TestRunGA:

    def _assert_feasible(self, w, label=""):
        sel = w > 0
        assert K_MIN <= sel.sum() <= K_MAX,     f"{label} cardinality {sel.sum()}"
        assert abs(w.sum() - 1.0) < 1e-8,      f"{label} sum={w.sum()}"
        assert np.all(w[sel] >= W_MIN - 1e-8),  f"{label} min {w[sel].min()}"
        assert np.all(w[sel] <= W_MAX + 1e-8),  f"{label} max {w[sel].max()}"

    def test_output_feasible(self):
        """run_ga must return a feasible chromosome."""
        mu, sigma = make_estimation_data(n=50)
        w = run_ga(50, mu, sigma, None, np.random.default_rng(0))
        self._assert_feasible(w, "run_ga")

    def test_output_shape(self):
        """Output shape must match n_assets."""
        for n in [50, 100]:
            mu, sigma = make_estimation_data(n=n)
            w = run_ga(n, mu, sigma, None, np.random.default_rng(1))
            assert w.shape == (n,), f"Expected ({n},), got {w.shape}"

    def test_fitness_positive_for_positive_mu(self):
        """With positive expected returns, GA should find a portfolio with positive fitness."""
        mu, sigma = make_estimation_data(n=50, seed=1)
        # Force clearly positive mu
        mu = np.abs(mu) + 0.005
        w  = run_ga(50, mu, sigma, None, np.random.default_rng(2))
        f  = fitness(w, mu, sigma, None, LAMBDA)
        assert f > 0, f"Expected positive fitness, got {f:.6f}"

    def test_deterministic_with_same_seed(self):
        """Same seed must produce identical output."""
        mu, sigma = make_estimation_data(n=50)
        w1 = run_ga(50, mu, sigma, None, np.random.default_rng(99))
        w2 = run_ga(50, mu, sigma, None, np.random.default_rng(99))
        np.testing.assert_array_equal(w1, w2)

    def test_different_seeds_differ(self):
        """Different seeds should produce different portfolios."""
        mu, sigma = make_estimation_data(n=50)
        w1 = run_ga(50, mu, sigma, None, np.random.default_rng(10))
        w2 = run_ga(50, mu, sigma, None, np.random.default_rng(11))
        assert not np.array_equal(w1, w2), \
            "Different seeds produced identical output — RNG not propagated"

    def test_prev_weights_path(self):
        """run_ga must complete without error when prev_weights is provided."""
        mu, sigma = make_estimation_data(n=50)
        w1        = run_ga(50, mu, sigma, None, np.random.default_rng(3))
        w2        = run_ga(50, mu, sigma, w1,   np.random.default_rng(4))
        self._assert_feasible(w2, "run_ga with prev_weights")

    def test_no_nan_in_output(self):
        """run_ga must never return NaN values."""
        mu, sigma = make_estimation_data(n=50)
        w = run_ga(50, mu, sigma, None, np.random.default_rng(5))
        assert not np.any(np.isnan(w)), "run_ga returned NaN weights"