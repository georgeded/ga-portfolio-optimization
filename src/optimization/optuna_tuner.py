"""
GA hyperparameter tuner (Optuna, TPE sampler).
Tuning period: Jan 2005 - Dec 2012 (96 periods).
Objective: maximise net Sharpe on the tuning period.
Parameters: PC -> [0.60, 0.95], PM -> [0.01, 0.30], SIGMA_M -> [0.01, 0.15], LAMBDA -> [0.00, 2.00].

python3 -m src.optimization.optuna_tuner (15 trials)
python3 -m src.optimization.optuna_tuner --debug  5 trials, 3 runs, 30 gens, 12 periods
Resume: SQLite storage auto-resumes.
"""

import argparse
import json
import multiprocessing as mp
from multiprocessing.pool import Pool
import os
import time

import numpy as np
import optuna
import pandas as pd
from tqdm import tqdm

import src.optimization.genetic_algorithm as ga_module
from src.optimization.genetic_algorithm import run_ga, fitness as ga_fitness
from src.evaluation.metrics import (
    portfolio_turnover,
    transaction_cost,
    sharpe_ratio,
    TRANSACTION_COST,
)
from src.utils.data import load_data
from src.utils.portfolio import (
    get_estimation_window,
    get_monthly_returns,
    get_rf_for_month,
    compute_drift_weights,
    align_drifted_weights,
)

TUNE_START  = "2005-01-01"
TUNE_END    = "2012-12-01"   # inclusive — 96 periods
N_TRIALS    = 15
N_RUNS_TUNE = 5              # fewer runs than full experiment to keep trials fast
N_GENS_TUNE = 100            # fewer gens than full experiment
BASE_SEED   = 2000           # different from runner.py (1000) to avoid overlap
OUTPUT_DIR  = "results/optuna"
STUDY_DB    = os.path.join(OUTPUT_DIR, "study.db")
BEST_PARAMS = os.path.join(OUTPUT_DIR, "best_params.json")


# module-level for pickling
def _run_single(args: tuple) -> np.ndarray:
    """Run one GA instance with trial-specific hyperparameters."""
    n_assets, mu, sigma, prev_weights, seed, n_gens, pc, pm, sigma_m, lambda_ = args

    ga_module.N_GENS  = n_gens
    ga_module.PC      = pc
    ga_module.PM      = pm
    ga_module.SIGMA_M = sigma_m
    ga_module.LAMBDA  = lambda_

    rng = np.random.default_rng(seed)
    return run_ga(n_assets, mu, sigma, prev_weights, rng)


def _eval_period(
    t: pd.Timestamp,
    apply_date: pd.Timestamp,
    universe: pd.DataFrame,
    returns: pd.DataFrame,
    prev_weights: np.ndarray | None,
    prev_permnos: list | None,
    n_runs: int,
    n_gens: int,
    params: dict,
    pool: Pool,
    gamma: float,
) -> tuple[dict | None, np.ndarray | None, list | None]:
    """Evaluate one period under given hyperparameters.
    Canonical = median in-sample fitness (mirrors runner.py). pw_aligned + lambda_ passed
    to ga_fitness so the turnover penalty matches what GA workers optimised against.
    """
    eligible = universe[universe["date"] == t]["permno"].tolist()
    if not eligible:
        return None, prev_weights, prev_permnos

    mu, sigma, valid_permnos = get_estimation_window(returns, eligible, t)
    if mu is None or len(valid_permnos) < 2:
        return None, prev_weights, prev_permnos

    n_assets = len(valid_permnos)

    if prev_weights is not None and prev_permnos is not None:
        prev_ret = get_monthly_returns(returns, prev_permnos, apply_date)
        drifted = compute_drift_weights(prev_weights, prev_ret.values)
        pw_aligned = align_drifted_weights(drifted, prev_permnos, valid_permnos)
    else:
        pw_aligned = None

    worker_args = [
        (n_assets, mu, sigma, pw_aligned,
         BASE_SEED + i, n_gens,
         params["pc"], params["pm"], params["sigma_m"], params["lambda_"])
        for i in range(n_runs)
    ]
    all_weights: list[np.ndarray] = pool.map(_run_single, worker_args)

    month_ret = get_monthly_returns(returns, valid_permnos, apply_date)
    stock_rets = month_ret.values
    rf = get_rf_for_month(returns, apply_date)

    gross_returns = np.array([float(w @ stock_rets) for w in all_weights])
    median_ret = float(np.median(gross_returns))

    # pw_aligned + lambda_ passed so penalty matches what run_ga optimised (critical for LAMBDA tuning)
    in_sample_fitnesses = np.array([
        ga_fitness(w, mu, sigma, pw_aligned, params["lambda_"])
        for w in all_weights
    ])
    median_fitness = float(np.median(in_sample_fitnesses))
    canonical_i = int(np.argmin(np.abs(in_sample_fitnesses - median_fitness)))
    canon_w = all_weights[canonical_i]

    portfolio_excess = median_ret - rf
    turnover = (
        portfolio_turnover(canon_w, pw_aligned)
        if pw_aligned is not None else 1.0
    )
    cost = transaction_cost(turnover, gamma)

    result = {
        "excess_ret": portfolio_excess,
        "rf": rf,
        "turnover": turnover,
        "cost": cost,
    }

    new_weights = compute_drift_weights(canon_w, stock_rets)
    return result, new_weights, valid_permnos


def _make_objective(
    universe: pd.DataFrame,
    returns: pd.DataFrame,
    tuning_dates: list,
    all_return_dates: list,
    n_runs: int,
    n_gens: int,
    n_workers: int,
    gamma: float,
):
    """Pool created once per trial to avoid 96 spawn/kill cycles."""
    def objective(trial: optuna.Trial) -> float:
        pc      = trial.suggest_float("pc",      0.60, 0.95)
        pm      = trial.suggest_float("pm",      0.01, 0.30)
        sigma_m = trial.suggest_float("sigma_m", 0.01, 0.15)
        lambda_ = trial.suggest_float("lambda_", 0.00, 2.00)

        trial_params = {
            "pc": pc, "pm": pm, "sigma_m": sigma_m, "lambda_": lambda_,
        }

        excess_rets = []
        turnovers_arr = []
        prev_weights = None
        prev_permnos = None

        with mp.Pool(processes=n_workers) as pool:
            for t in tuning_dates:
                apply_dates = [
                    d for d in all_return_dates
                    if d.year == t.year and d.month == t.month
                ]
                if not apply_dates:
                    continue

                result, prev_weights, prev_permnos = _eval_period(
                    t, apply_dates[0], universe, returns,
                    prev_weights, prev_permnos,
                    n_runs, n_gens,
                    trial_params,
                    pool, gamma,
                )
                if result is None:
                    continue

                excess_rets.append(result["excess_ret"])
                turnovers_arr.append(result["turnover"])

        if len(excess_rets) < 12:
            raise optuna.exceptions.TrialPruned()

        excess_arr = np.array(excess_rets)
        costs = np.array(turnovers_arr) * gamma
        net_excess = excess_arr - costs

        return sharpe_ratio(net_excess)

    return objective


def run_tuner(
    n_trials: int = N_TRIALS,
    n_runs: int = N_RUNS_TUNE,
    n_gens: int = N_GENS_TUNE,
    max_periods: int | None = None,
    gamma: float = TRANSACTION_COST,
) -> dict:
    """Run Optuna search and return best hyperparameter dict."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    universe, returns = load_data()

    all_dates = sorted(universe["date"].unique())
    all_return_dates = sorted(returns["date"].unique())

    tuning_dates = [
        t for t in all_dates
        if pd.Timestamp(TUNE_START) <= t <= pd.Timestamp(TUNE_END)
    ]
    if max_periods is not None:
        tuning_dates = tuning_dates[:max_periods]

    n_workers = min(n_runs, mp.cpu_count())

    print(f"Tuning period : {tuning_dates[0].date()} → {tuning_dates[-1].date()}")
    print(f"Periods       : {len(tuning_dates)}")
    print(f"Trials        : {n_trials}")
    print(f"Runs/period   : {n_runs}  |  Max gens: {n_gens}  |  Workers: {n_workers}")
    print(f"Storage       : {STUDY_DB}")
    print()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name="ga_hyperparameter_tuning",
        direction="maximize",
        storage=f"sqlite:///{STUDY_DB}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    already_done = len(study.trials)
    remaining = n_trials - already_done

    if remaining <= 0:
        print(f"Study already has {already_done} trials — nothing to do.")
        print("Delete the study DB to restart: rm results/optuna/study.db")
    else:
        if already_done > 0:
            print(f"Resuming: {already_done} trials done, {remaining} remaining.")

        objective = _make_objective(
            universe, returns, tuning_dates, all_return_dates,
            n_runs, n_gens, n_workers, gamma,
        )

        t0 = time.time()
        for trial_num in tqdm(range(remaining), desc="Optuna trials"):
            study.optimize(objective, n_trials=1, show_progress_bar=False)
            best = study.best_trial
            tqdm.write(
                f"  Trial {already_done + trial_num + 1:3d} | "
                f"net_sharpe={best.value:.4f} | "
                f"pc={best.params['pc']:.3f}  "
                f"pm={best.params['pm']:.3f}  "
                f"sigma_m={best.params['sigma_m']:.3f}  "
                f"lambda={best.params['lambda_']:.3f}"
            )
        print(f"\nTuning complete in {(time.time()-t0)/60:.1f} min")

    best_params = study.best_params
    best_value = study.best_value

    print("\n" + "=" * 50)
    print("BEST HYPERPARAMETERS")
    print("=" * 50)
    print(f"Net Sharpe (tuning period) : {best_value:.4f}")
    print(f"PC                         : {best_params['pc']:.4f}")
    print(f"PM                         : {best_params['pm']:.4f}")
    print(f"SIGMA_M                    : {best_params['sigma_m']:.4f}")
    print(f"LAMBDA                     : {best_params['lambda_']:.4f}")
    print("=" * 50)

    output = {
        "best_net_sharpe_tuning": round(best_value, 6),
        "pc":                     round(best_params["pc"],      4),
        "pm":                     round(best_params["pm"],      4),
        "sigma_m":                round(best_params["sigma_m"], 4),
        "lambda_":                round(best_params["lambda_"], 4),
        "tuning_period_start":    TUNE_START,
        "tuning_period_end":      TUNE_END,
        "n_trials":               len(study.trials),
        "n_runs_per_trial":       n_runs,
        "n_gens_per_trial":       n_gens,
    }
    with open(BEST_PARAMS, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {BEST_PARAMS}")
    print("\nNext: patch into src/optimization/genetic_algorithm.py:")
    print(f"  PC      = {output['pc']}")
    print(f"  PM      = {output['pm']}")
    print(f"  SIGMA_M = {output['sigma_m']}")
    print(f"  LAMBDA  = {output['lambda_']}")
    print("Then run: python3 -m src.optimization.runner")

    return best_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GA Optuna hyperparameter tuner")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: 5 trials, 3 runs, 30 gens, 12 periods",
    )
    parser.add_argument("--n-trials",    type=int,   default=N_TRIALS)
    parser.add_argument("--n-runs",      type=int,   default=N_RUNS_TUNE)
    parser.add_argument("--n-gens",      type=int,   default=N_GENS_TUNE)
    parser.add_argument("--max-periods", type=int,   default=None)
    parser.add_argument("--gamma",       type=float, default=TRANSACTION_COST)
    args = parser.parse_args()

    if args.debug:
        print("=== DEBUG MODE: 5 trials | 3 runs | 30 gens | 12 periods ===")
        run_tuner(n_trials=5, n_runs=3, n_gens=30, max_periods=12, gamma=args.gamma)
    else:
        run_tuner(
            n_trials=args.n_trials,
            n_runs=args.n_runs,
            n_gens=args.n_gens,
            max_periods=args.max_periods,
            gamma=args.gamma,
        )
