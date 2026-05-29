import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

import src.optimization.genetic_algorithm as ga_module
from src.optimization.genetic_algorithm import run_ga, fitness as ga_fitness
from src.optimization.runner import N_RUNNERS, BASE_SEED
from src.evaluation.metrics import (
    TRANSACTION_COST,
    herfindahl_index,
    portfolio_turnover,
    transaction_cost,
)
from src.utils.data import load_data
from src.utils.portfolio import (
    compute_drift_weights,
    get_estimation_window,
    get_monthly_returns,
    get_rf_for_month,
    print_results,
)

K_VALUES = [10, 15, 20, 25, 30]
OUTPUT_DIR = "results/k_sensitivity"


def _set_affinity(core_id: int) -> None:
    try:
        allowed = sorted(os.sched_getaffinity(0))
        cpu = allowed[core_id % len(allowed)]
        os.sched_setaffinity(0, {cpu})
    except (AttributeError, OSError):
        pass


def _run_single_fixed_k(args: tuple) -> np.ndarray:
    (n_assets, mu, sigma, prev_weights, seed, n_gens,
     pc, pm, sigma_m, lambda_, k_min, k_max) = args

    ga_module.K_MIN = k_min
    ga_module.K_MAX = k_max
    ga_module.N_GENS = n_gens
    ga_module.PC = pc
    ga_module.PM = pm
    ga_module.SIGMA_M = sigma_m
    ga_module.LAMBDA = lambda_

    rng = np.random.default_rng(seed)
    return run_ga(n_assets, mu, sigma, prev_weights, rng)


def _run_fixed_k_affinity(args: tuple) -> np.ndarray:
    core_id, actual_args = args
    _set_affinity(core_id)
    return _run_single_fixed_k(actual_args)


def _process_period_k(
    t: pd.Timestamp,
    apply_date: pd.Timestamp,
    universe: pd.DataFrame,
    returns: pd.DataFrame,
    prev_weights: np.ndarray | None,
    prev_permnos: list | None,
    n_runs: int,
    n_gens: int,
    pool: mp.pool.Pool,
    gamma: float,
    params: dict,
    k: int,
) -> tuple[dict | None, np.ndarray | None, list | None]:
    eligible = universe[universe["date"] == t]["permno"].tolist()
    if not eligible:
        return None, prev_weights, prev_permnos

    mu, sigma, valid_permnos = get_estimation_window(returns, eligible, t)
    if mu is None or len(valid_permnos) < 2:
        return None, prev_weights, prev_permnos

    n_assets = len(valid_permnos)

    if prev_weights is not None and prev_permnos is not None:
        pw_raw = (pd.Series(prev_weights, index=prev_permnos)
                  .reindex(valid_permnos, fill_value=0.0)
                  .values)
        raw_total = pw_raw.sum()
        if raw_total > 0:
            pw_aligned = pw_raw / raw_total
        else:
            pw_aligned = np.ones(n_assets) / n_assets
    else:
        pw_raw = None
        pw_aligned = None

    ga_args_list = [
        (n_assets, mu, sigma, pw_aligned,
         BASE_SEED + i, n_gens,
         params["pc"], params["pm"], params["sigma_m"], params["lambda_"],
         k, k)
        for i in range(n_runs)
    ]
    indexed_args = [(i % N_RUNNERS, arg) for i, arg in enumerate(ga_args_list)]
    all_weights: list[np.ndarray] = pool.map(_run_fixed_k_affinity, indexed_args)

    month_ret = get_monthly_returns(returns, valid_permnos, apply_date)
    stock_rets = month_ret.values
    rf = get_rf_for_month(returns, apply_date)

    gross_returns = np.array([float(w @ stock_rets) for w in all_weights])

    in_sample_fitnesses = np.array([
        ga_fitness(w, mu, sigma, pw_aligned, params["lambda_"])
        for w in all_weights
    ])
    median_fitness = float(np.median(in_sample_fitnesses))
    canonical_i = int(np.argmin(np.abs(in_sample_fitnesses - median_fitness)))
    canon_w = all_weights[canonical_i]

    canon_gross = float(canon_w @ stock_rets)
    portfolio_gross = canon_gross
    portfolio_excess = portfolio_gross - rf

    if pw_raw is not None:
        turnover = portfolio_turnover(canon_w, pw_raw)
    else:
        turnover = 1.0

    cost = transaction_cost(turnover, gamma)
    net_excess = portfolio_excess - cost

    result = {
        "date": apply_date,
        "n_stocks": int((canon_w > 0).sum()),
        "portfolio_ret": portfolio_gross,
        "excess_ret": portfolio_excess,
        "rf": rf,
        "net_excess_ret": net_excess,
        "turnover": turnover,
        "cost": cost,
        "hhi": herfindahl_index(canon_w),
        "gross_ret_std": float(np.std(gross_returns)),
        "gross_ret_iqr": float(np.percentile(gross_returns, 75) -
                               np.percentile(gross_returns, 25)),
        "gross_ret_min": float(np.min(gross_returns)),
        "gross_ret_max": float(np.max(gross_returns)),
    }

    new_weights = compute_drift_weights(canon_w, stock_rets)
    return result, new_weights, valid_permnos


def run_fixed_k(
    k: int,
    n_runs: int = 8,
    n_gens: int = 200,
    gamma: float = TRANSACTION_COST,
    seed: int = 1000,
    universe: pd.DataFrame | None = None,
    returns: pd.DataFrame | None = None,
) -> pd.DataFrame:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    orig_k_min = ga_module.K_MIN
    orig_k_max = ga_module.K_MAX

    try:
        ga_module.K_MIN = k
        ga_module.K_MAX = k

        params = {
            "pc": ga_module.PC,
            "pm": ga_module.PM,
            "sigma_m": ga_module.SIGMA_M,
            "lambda_": ga_module.LAMBDA,
        }

        if universe is None or returns is None:
            universe, returns = load_data()

        rebalance_dates = sorted(universe["date"].unique())
        all_return_dates = sorted(returns["date"].unique())

        results: list[dict] = []
        prev_weights: np.ndarray | None = None
        prev_permnos: list | None = None

        t0_total = time.time()

        with mp.Pool(processes=N_RUNNERS) as pool:
            for t in tqdm(rebalance_dates, desc=f"K={k:02d}", position=0, leave=True):
                apply_dates = [
                    d for d in all_return_dates
                    if d.year == t.year and d.month == t.month
                ]
                if not apply_dates:
                    continue
                apply_date = apply_dates[0]

                t0 = time.time()
                result, prev_weights, prev_permnos = _process_period_k(
                    t, apply_date, universe, returns,
                    prev_weights, prev_permnos,
                    n_runs, n_gens, pool, gamma, params, k,
                )
                elapsed = time.time() - t0

                if result is None:
                    continue

                results.append(result)
                tqdm.write(
                    f"{apply_date.date()} | "
                    f"ret={result['portfolio_ret']*100:+.2f}% "
                    f"excess={result['excess_ret']*100:+.2f}% "
                    f"turnover={result['turnover']*100:.1f}% "
                    f"K_actual={result['n_stocks']} "
                    f"({elapsed:.1f}s)",
                    file=sys.stderr,
                )

        total_elapsed = time.time() - t0_total
        print(f"\nK={k}: completed {len(results)} periods in "
              f"{total_elapsed/60:.1f} min")

        df = pd.DataFrame(results)
        out_path = os.path.join(OUTPUT_DIR, f"k{k:02d}_results.parquet")
        df.to_parquet(out_path, index=False)
        print(f"Saved: {out_path}")

        print_results(df, f"GA (K={k}, FIXED)")

        return df

    finally:
        ga_module.K_MIN = orig_k_min
        ga_module.K_MAX = orig_k_max


def run_all_k(
    n_runs: int = 8,
    n_gens: int = 200,
    gamma: float = TRANSACTION_COST,
) -> dict:
    results = {}
    t0_total = time.time()

    print("Loading data (once for all K values)...")
    shared_universe, shared_returns = load_data()

    for k in K_VALUES:
        print(f"\nRunning K={k} ({K_VALUES.index(k)+1}/{len(K_VALUES)})")
        t0 = time.time()
        results[k] = run_fixed_k(
            k,
            n_runs=n_runs,
            n_gens=n_gens,
            gamma=gamma,
            universe=shared_universe,
            returns=shared_returns,
        )
        elapsed = time.time() - t0
        print(f"K={k} finished in {elapsed/60:.1f} min")

    total = time.time() - t0_total
    print(f"\nAll K values complete. Total runtime: {total/60:.1f} min")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GA K-sensitivity analysis")
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Run a single K value. If omitted, runs all K values.",
    )
    parser.add_argument("--n-runs", type=int, default=8)
    parser.add_argument("--n-gens", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=TRANSACTION_COST)
    args = parser.parse_args()

    if args.k is not None:
        if args.k not in K_VALUES:
            print(f"Warning: K={args.k} is not in K_VALUES={K_VALUES}. Running anyway.")
        run_fixed_k(args.k, n_runs=args.n_runs, n_gens=args.n_gens, gamma=args.gamma)
    else:
        run_all_k(n_runs=args.n_runs, n_gens=args.n_gens, gamma=args.gamma)
