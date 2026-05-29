# Out-of-sample GA runner.
# Usage: python3 -m src.optimization.runner
# Debug: python3 -m src.optimization.runner --debug

import argparse
import multiprocessing as mp
from multiprocessing.pool import Pool
import os
import sys
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

N_RUNNERS = 8
BASE_SEED = 1000  # run i uses seed BASE_SEED + i
OUTPUT_DIR = "results/ga"
CHECKPOINT = os.path.join(OUTPUT_DIR, "checkpoint.parquet")
FINAL_OUT = os.path.join(OUTPUT_DIR, "ga_results.parquet")
WEIGHTS_CHECKPOINT = os.path.join(OUTPUT_DIR, "last_weights.npy")
PERMNOS_CHECKPOINT = os.path.join(OUTPUT_DIR, "last_permnos.npy")

# Fixed params from the initial Optuna run.
# Per-period tuning was too slow for the thesis run.
FIXED_PARAMS = {
    "pc": ga_module.PC,
    "pm": ga_module.PM,
    "sigma_m": ga_module.SIGMA_M,
    "lambda_": ga_module.LAMBDA,
}


def run_single_ga(args: tuple) -> np.ndarray:
    # args: (n_assets, mu, sigma, prev_weights, seed, n_gens, pc, pm, sigma_m, lambda_)
    n_assets, mu, sigma, prev_weights, seed, n_gens, pc, pm, sigma_m, lambda_ = args

    # Each worker has its own module copy.
    ga_module.N_GENS = n_gens
    ga_module.PC = pc
    ga_module.PM = pm
    ga_module.SIGMA_M = sigma_m
    ga_module.LAMBDA = lambda_

    rng = np.random.default_rng(seed)
    return run_ga(n_assets, mu, sigma, prev_weights, rng)


def set_affinity(core_id: int) -> None:
    try:
        allowed = sorted(os.sched_getaffinity(0))
        cpu = allowed[core_id % len(allowed)]
        os.sched_setaffinity(0, {cpu})
    except (AttributeError, OSError):
        pass


def run_ga_with_affinity(args: tuple) -> np.ndarray:
    core_id, actual_args = args
    set_affinity(core_id)
    return run_single_ga(actual_args)


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
    params: dict | None = None,
) -> tuple[dict, np.ndarray, list]:
    if params is None:
        params = {
            "pc": ga_module.PC,
            "pm": ga_module.PM,
            "sigma_m": ga_module.SIGMA_M,
            "lambda_": ga_module.LAMBDA,
        }

    eligible = universe[universe["date"] == t]["permno"].tolist()
    if len(eligible) == 0:
        return None, prev_weights, prev_permnos

    mu, sigma, valid_permnos = get_estimation_window(returns, eligible, t)
    if mu is None or len(valid_permnos) < 2:
        return None, prev_weights, prev_permnos

    n_assets = len(valid_permnos)

    if prev_weights is not None and prev_permnos is not None:
        # pw_raw: reindex without renormalizing, so exited stocks land at zero and
        # portfolio_turnover picks up the exit cost without a separate term
        pw_raw = (pd.Series(prev_weights, index=prev_permnos)
                  .reindex(valid_permnos, fill_value=0.0)
                  .values)
        # pw_aligned: renormalized version used as GA's starting-point prev_weights
        raw_total = pw_raw.sum()
        if raw_total > 0:
            pw_aligned = pw_raw / raw_total
        else:
            pw_aligned = np.ones(len(valid_permnos)) / len(valid_permnos)
    else:
        pw_raw = None
        pw_aligned = None

    ga_args_list = [
        (n_assets, mu, sigma, pw_aligned, BASE_SEED + i, n_gens,
         params["pc"], params["pm"], params["sigma_m"], params["lambda_"])
        for i in range(n_runs)
    ]
    indexed_args = [(i % N_RUNNERS, arg) for i, arg in enumerate(ga_args_list)]
    all_weights: list[np.ndarray] = pool.map(run_ga_with_affinity, indexed_args)

    month_ret = get_monthly_returns(returns, valid_permnos, apply_date)
    stock_rets = month_ret.values
    rf = get_rf_for_month(returns, apply_date)

    gross_returns = np.array([float(w @ stock_rets) for w in all_weights])

    # Pick the median in-sample run, not the best realised return.
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
        # pw_raw doesn't sum to 1 when stocks exited, so the half-sum formula
        # naturally includes the exit cost, no separate term needed
        turnover = portfolio_turnover(canon_w, pw_raw)
    else:
        turnover = 1.0  # first period, consistent with benchmarks

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
        # Spread across the independent GA runs
        "gross_ret_std": float(np.std(gross_returns)),
        "gross_ret_iqr": float(np.percentile(gross_returns, 75) -
                               np.percentile(gross_returns, 25)),
        "gross_ret_min": float(np.min(gross_returns)),
        "gross_ret_max": float(np.max(gross_returns)),
    }

    new_weights = compute_drift_weights(canon_w, stock_rets)
    new_permnos = valid_permnos

    return result, new_weights, new_permnos


def _save_checkpoint(completed_results: list[dict], prev_weights: np.ndarray,
                     prev_permnos: list) -> None:
    pd.DataFrame(completed_results).to_parquet(CHECKPOINT, index=False)
    np.save(WEIGHTS_CHECKPOINT, prev_weights)
    np.save(PERMNOS_CHECKPOINT, np.array(prev_permnos, dtype=object))


def _load_checkpoint() -> tuple[list[dict], set, np.ndarray | None, list | None]:
    # prev_weights and prev_permnos are None if weight files are missing.
    ckpt = pd.read_parquet(CHECKPOINT)
    completed_results = ckpt.to_dict("records")
    # Universe dates are start-of-month.
    # Checkpoint dates are apply dates, usually end-of-month.
    # Compare by (year, month) so the resume logic is not fooled by the day difference.
    completed_dates = {(pd.Timestamp(d).year, pd.Timestamp(d).month)
                       for d in ckpt["date"]}

    prev_weights = None
    prev_permnos = None
    if os.path.exists(WEIGHTS_CHECKPOINT) and os.path.exists(PERMNOS_CHECKPOINT):
        prev_weights = np.load(WEIGHTS_CHECKPOINT)
        prev_permnos = np.load(PERMNOS_CHECKPOINT, allow_pickle=True).tolist()

    return completed_results, completed_dates, prev_weights, prev_permnos


def _clear_checkpoint() -> None:
    # Delete all checkpoint files.
    # FINAL_OUT intentionally excluded: clearing state should not
    # destroy completed experiment results
    for path in [CHECKPOINT, WEIGHTS_CHECKPOINT, PERMNOS_CHECKPOINT]:
        if os.path.exists(path):
            os.remove(path)
    print("Checkpoint cleared.")


def run(n_runs: int = N_RUNNERS, n_gens: int = ga_module.N_GENS,
        max_periods: int | None = None, gamma: float = TRANSACTION_COST,
        clear_checkpoint: bool = False,
        lambda_val: float | None = None,
        output_path: str | None = None) -> pd.DataFrame:
    if lambda_val is not None:
        ga_module.LAMBDA = lambda_val
        FIXED_PARAMS["lambda_"] = lambda_val

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
            print("prev_weights found. Turnover will be accurate.")
        else:
            print("prev_weights not found. First resumed period uses turnover=1.0.")

    remaining = [t for t in rebalance_dates if (t.year, t.month) not in completed_dates]
    print(f"Periods remaining: {len(remaining)}")
    print(f"Runs: {n_runs}, max generations: {n_gens}")

    t0_total = time.time()

    with mp.Pool(processes=N_RUNNERS) as pool:
        for t in tqdm(remaining, desc="GA out-of-sample", position=0, leave=True):

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
                params=FIXED_PARAMS,
            )
            elapsed = time.time() - t0

            if result is None:
                continue

            completed_results.append(result)

            _save_checkpoint(completed_results, prev_weights, prev_permnos)

            tqdm.write(
                f"{apply_date.date()} | "
                f"ret={result['portfolio_ret']*100:+.2f}% "
                f"excess={result['excess_ret']*100:+.2f}% "
                f"turnover={result['turnover']*100:.1f}% "
                f"K={result['n_stocks']} "
                f"({elapsed:.1f}s)",
                file=sys.stderr,
            )

    total_elapsed = time.time() - t0_total
    print(f"\nTotal runtime: {total_elapsed/60:.1f} min")

    results_df = pd.DataFrame(completed_results)
    save_path = output_path if output_path is not None else FINAL_OUT
    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    results_df.to_parquet(save_path, index=False)
    print(f"Saved to {save_path}")

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
    parser.add_argument("--n-runs", type=int, default=N_RUNNERS)
    parser.add_argument("--n-gens", type=int, default=ga_module.N_GENS)
    parser.add_argument("--max-periods", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=TRANSACTION_COST)
    parser.add_argument("--lambda-val", type=float, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.debug:
        print("debug mode: 3 runs, 50 gens, 10 periods")
        results = run(
            n_runs=3,
            n_gens=50,
            max_periods=10,
            gamma=args.gamma,
            clear_checkpoint=args.clear_checkpoint,
            lambda_val=args.lambda_val,
            output_path=args.output,
        )
    else:
        results = run(
            n_runs=args.n_runs,
            n_gens=args.n_gens,
            max_periods=args.max_periods,
            gamma=args.gamma,
            clear_checkpoint=args.clear_checkpoint,
            lambda_val=args.lambda_val,
            output_path=args.output,
        )

    if len(results) > 0:
        n_runs_used = 3 if args.debug else args.n_runs
        print_results(results, f"GA (CANONICAL, {n_runs_used} RUNS)")
