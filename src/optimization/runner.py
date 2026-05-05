"""
Out-of-sample GA experiment runner (252 monthly periods, Jan 2005 - Dec 2025).

Each period: build mu/sigma from 60-month window -> run N_RUNS parallel GA instances
-> report median realised gross return -> use weights of the median-fitness run as the
canonical portfolio for drift/turnover/prev_weights in the next period.
Checkpoints after every period so cloud jobs can resume safely.

python3 -m src.optimization.runner (full run)
python3 -m src.optimization.runner --debug (3 runs, 50 gens, 10 periods)
python3 -m src.optimization.runner --clear-checkpoint (restart from scratch)
python3 -m src.optimization.runner --n-runs 5 --n-gens 100 --max-periods 20
"""

import argparse
import multiprocessing as mp
from multiprocessing.pool import Pool
import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

import src.optimization.genetic_algorithm as ga_module
from src.optimization.genetic_algorithm import run_ga, fitness as ga_fitness
from src.evaluation.metrics import (
    portfolio_turnover,
    herfindahl_index,
    transaction_cost,
    TRANSACTION_COST,
)
from src.utils.data import load_data
from src.utils.portfolio import (
    get_estimation_window,
    get_monthly_returns,
    get_rf_for_month,
    compute_drift_weights,
    align_drifted_weights,
    print_results,
)

N_RUNS     = 30
BASE_SEED  = 1000        # run i uses seed BASE_SEED + i
OUTPUT_DIR = "results/ga"
CHECKPOINT         = os.path.join(OUTPUT_DIR, "checkpoint.parquet")
FINAL_OUT          = os.path.join(OUTPUT_DIR, "ga_results.parquet")
WEIGHTS_CHECKPOINT = os.path.join(OUTPUT_DIR, "last_weights.npy")
PERMNOS_CHECKPOINT = os.path.join(OUTPUT_DIR, "last_permnos.npy")


# module-level for multiprocessing pickling
def _run_single(args: tuple) -> np.ndarray:
    """args: (n_assets, mu, sigma, prev_weights, seed, n_gens)"""
    n_assets, mu, sigma, prev_weights, seed, n_gens = args

    original_n_gens = ga_module.N_GENS
    ga_module.N_GENS = n_gens
    try:
        rng = np.random.default_rng(seed)
        w = run_ga(n_assets, mu, sigma, prev_weights, rng)
    finally:
        ga_module.N_GENS = original_n_gens

    return w


def _process_period(
    t: pd.Timestamp,
    apply_date: pd.Timestamp,
    universe: pd.DataFrame,
    returns: pd.DataFrame,
    prev_weights: np.ndarray | None,
    prev_permnos: list | None,
    n_runs: int,
    n_gens: int,
    pool: Pool,
    gamma: float,
) -> tuple[dict, np.ndarray, list]:
    """Returns (result_row, new_drifted_weights, new_permnos); drifted weights aligned to valid_permnos."""
    eligible = universe[universe["date"] == t]["permno"].tolist()
    if len(eligible) == 0:
        return None, prev_weights, prev_permnos

    mu, sigma, valid_permnos = get_estimation_window(returns, eligible, t)
    if mu is None or len(valid_permnos) < 2:
        return None, prev_weights, prev_permnos

    n_assets = len(valid_permnos)

    if prev_weights is not None and prev_permnos is not None:
        prev_ret_series = get_monthly_returns(returns, prev_permnos, apply_date)
        drifted = compute_drift_weights(prev_weights, prev_ret_series.values)
        pw_aligned = align_drifted_weights(drifted, prev_permnos, valid_permnos)
    else:
        pw_aligned = None

    worker_args = [
        (n_assets, mu, sigma, pw_aligned, BASE_SEED + i, n_gens)
        for i in range(n_runs)
    ]
    all_weights: list[np.ndarray] = pool.map(_run_single, worker_args)

    month_ret = get_monthly_returns(returns, valid_permnos, apply_date)
    stock_rets = month_ret.values
    rf = get_rf_for_month(returns, apply_date)

    gross_returns = np.array([float(w @ stock_rets) for w in all_weights])

    median_ret = float(np.median(gross_returns))
    portfolio_gross = median_ret
    portfolio_excess = portfolio_gross - rf

    # canonical = run at median IN-SAMPLE fitness (not realised return) — avoids ex-post
    # return dependency for prev_weights. pw_aligned ensures penalty is consistent
    # with what run_ga optimised against.
    in_sample_fitnesses = np.array([
        ga_fitness(w, mu, sigma, pw_aligned, ga_module.LAMBDA)
        for w in all_weights
    ])
    median_fitness = float(np.median(in_sample_fitnesses))
    canonical_i = int(np.argmin(np.abs(in_sample_fitnesses - median_fitness)))
    canon_w = all_weights[canonical_i]

    if pw_aligned is not None:
        turnover = portfolio_turnover(canon_w, pw_aligned)
    else:
        turnover = 1.0   # first-period convention — consistent with benchmarks

    cost = transaction_cost(turnover, gamma)
    net_excess = portfolio_excess - cost

    result = {
        "date":           apply_date,
        "n_stocks":       int((canon_w > 0).sum()),
        "portfolio_ret":  portfolio_gross,
        "excess_ret":     portfolio_excess,
        "rf":             rf,
        "net_excess_ret": net_excess,
        "turnover":       turnover,
        "cost":           cost,
        "hhi":            herfindahl_index(canon_w),
        # cross-run dispersion — addresses stochasticity threat to validity
        "gross_ret_std":  float(np.std(gross_returns)),
        "gross_ret_iqr":  float(np.percentile(gross_returns, 75) -
                               np.percentile(gross_returns, 25)),
        "gross_ret_min":  float(np.min(gross_returns)),
        "gross_ret_max":  float(np.max(gross_returns)),
    }

    new_weights = compute_drift_weights(canon_w, stock_rets)
    new_permnos = valid_permnos

    return result, new_weights, new_permnos


def _save_checkpoint(completed_results: list[dict], prev_weights: np.ndarray,
                     prev_permnos: list) -> None:
    """Save results parquet + canonical weights state after every period."""
    pd.DataFrame(completed_results).to_parquet(CHECKPOINT, index=False)
    np.save(WEIGHTS_CHECKPOINT, prev_weights)
    np.save(PERMNOS_CHECKPOINT, np.array(prev_permnos, dtype=object))


def _load_checkpoint() -> tuple[list[dict], set, np.ndarray | None, list | None]:
    """Load checkpoint state. prev_weights/prev_permnos are None if weight file is missing."""
    ckpt = pd.read_parquet(CHECKPOINT)
    completed_results = ckpt.to_dict("records")
    completed_dates = set(ckpt["date"].tolist())

    prev_weights = None
    prev_permnos = None
    if os.path.exists(WEIGHTS_CHECKPOINT) and os.path.exists(PERMNOS_CHECKPOINT):
        prev_weights = np.load(WEIGHTS_CHECKPOINT)
        prev_permnos = np.load(PERMNOS_CHECKPOINT, allow_pickle=True).tolist()

    return completed_results, completed_dates, prev_weights, prev_permnos


def _clear_checkpoint() -> None:
    """Delete all checkpoint files."""
    for path in [CHECKPOINT, FINAL_OUT, WEIGHTS_CHECKPOINT, PERMNOS_CHECKPOINT]:
        if os.path.exists(path):
            os.remove(path)
    print("Checkpoint cleared.")


def run(n_runs: int = N_RUNS, n_gens: int = ga_module.N_GENS,
        max_periods: int | None = None, gamma: float = TRANSACTION_COST,
        clear_checkpoint: bool = False) -> pd.DataFrame:
    """Run the full OOS GA experiment with checkpointing. Returns monthly results DataFrame."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if clear_checkpoint:
        _clear_checkpoint()

    universe, returns = load_data()

    rebalance_dates = sorted(universe["date"].unique())
    all_return_dates = sorted(returns["date"].unique())

    if max_periods is not None:
        rebalance_dates = rebalance_dates[:max_periods]

    completed_results: list[dict] = []
    completed_dates: set = set()
    prev_weights: np.ndarray | None = None
    prev_permnos: list | None = None

    if os.path.exists(CHECKPOINT):
        completed_results, completed_dates, prev_weights, prev_permnos = \
            _load_checkpoint()
        print(f"Resuming from checkpoint: {len(completed_results)} periods done.")
        if prev_weights is not None:
            print("  prev_weights state recovered — turnover will be accurate.")
        else:
            print("  prev_weights not found — first resumed period uses turnover=1.0.")

    remaining = [t for t in rebalance_dates if t not in completed_dates]
    print(f"Periods to process : {len(remaining)}")
    print(f"Parallel runs      : {n_runs}  |  Max generations: {n_gens}")

    n_workers = min(n_runs, mp.cpu_count())
    t0_total = time.time()

    with mp.Pool(processes=n_workers) as pool:
        for t in tqdm(remaining, desc="GA out-of-sample"):

            apply_dates = [
                d for d in all_return_dates
                if d.year == t.year and d.month == t.month
            ]
            if not apply_dates:
                continue
            apply_date = apply_dates[0]

            t0 = time.time()
            result, prev_weights, prev_permnos = _process_period(
                t, apply_date, universe, returns,
                prev_weights, prev_permnos,
                n_runs, n_gens, pool, gamma,
            )
            elapsed = time.time() - t0

            if result is None:
                continue

            completed_results.append(result)

            _save_checkpoint(completed_results, prev_weights, prev_permnos)

            tqdm.write(
                f"  {apply_date.date()} | "
                f"ret={result['portfolio_ret']*100:+.2f}%  "
                f"excess={result['excess_ret']*100:+.2f}%  "
                f"turnover={result['turnover']*100:.1f}%  "
                f"K={result['n_stocks']}  "
                f"({elapsed:.1f}s)"
            )

    total_elapsed = time.time() - t0_total
    print(f"\nTotal runtime: {total_elapsed/60:.1f} min")

    results_df = pd.DataFrame(completed_results)
    results_df.to_parquet(FINAL_OUT, index=False)
    print(f"Saved to {FINAL_OUT}")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GA portfolio runner")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: 3 runs, 50 generations, 10 periods",
    )
    parser.add_argument(
        "--clear-checkpoint",
        action="store_true",
        help="Delete existing checkpoint before running",
    )
    parser.add_argument("--n-runs",      type=int,   default=N_RUNS)
    parser.add_argument("--n-gens",      type=int,   default=ga_module.N_GENS)
    parser.add_argument("--max-periods", type=int,   default=None)
    parser.add_argument("--gamma",       type=float, default=TRANSACTION_COST)
    args = parser.parse_args()

    if args.debug:
        print("=== DEBUG MODE: 3 runs | 50 gens | 10 periods ===")
        results = run(
            n_runs=3,
            n_gens=50,
            max_periods=10,
            gamma=args.gamma,
            clear_checkpoint=args.clear_checkpoint,
        )
    else:
        results = run(
            n_runs=args.n_runs,
            n_gens=args.n_gens,
            max_periods=args.max_periods,
            gamma=args.gamma,
            clear_checkpoint=args.clear_checkpoint,
        )

    if len(results) > 0:
        print_results(results, "GA (MEDIAN 30 RUNS)")
