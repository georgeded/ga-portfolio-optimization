# GA Portfolio Optimization

BSc CS thesis project at VU Amsterdam. A genetic algorithm selects cardinality-constrained US equity portfolios and is evaluated out-of-sample against MVO and equal-weight benchmarks from January 2005 to December 2025 (252 monthly periods) on a universe of around 870 stocks. The GA underperforms all three benchmarks, with a Sharpe of 0.274 vs 0.581 for MVO.

## Results

| Strategy | Sharpe (net) | Ann. Return | Ann. Vol | Max Drawdown | Avg Turnover |
|---|---|---|---|---|---|
| **GA (adaptive K)** | **0.2741** | **5.5%** | **19.9%** | **-55.8%** | **22.2%** |
| Constrained MVO | 0.5810 | 8.3% | 14.3% | -43.1% | 17.1% |
| Unconstrained MVO | 0.5809 | 8.3% | 14.3% | -43.1% | 17.1% |
| 1/N (867 stocks) | 0.5307 | 9.1% | 17.2% | -51.3% | 4.5% |

Evaluation: January 2005 to December 2025, 252 monthly out-of-sample periods. All metrics are on net excess returns after transaction costs. Annualized return = mean monthly * 12, annualized vol = monthly std * sqrt(12), Sharpe = annualized return / annualized vol. Constrained MVO caps individual weights at 0.15.

The GA underperforms all three benchmarks. The Jobson-Korkie test shows the GA vs MVO Sharpe gap is statistically significant (p = 0.004). The GA vs 1/N gap is not significant (p = 0.132). Constrained and unconstrained MVO produce nearly identical results (Sharpe 0.5810 vs 0.5809). With around 62 stocks held on average, the implied average weight (1.6%) is well below the 15% cap. The underperformance is likely driven by in-sample fitness noise: with N/T around 14.5, the in-sample Sharpe is a poor proxy for out-of-sample returns even with Ledoit-Wolf regularisation.

Fixed-K experiments show K=25/30 outperform the adaptive version. See thesis for details. A lambda ablation confirms the turnover penalty reduces trading costs at the expense of Sharpe. See thesis.

![Cumulative net portfolio value](results/figures/F1_cumulative_returns.png)
Cumulative net portfolio value, 2005-2025.

![Rolling 12-month Sharpe ratio](results/figures/F2_rolling_sharpe.png)
Rolling 12-month Sharpe ratio. GA underperforms MVO across most of the evaluation period.

![Adaptive cardinality K](results/figures/F5_cardinality.png)
Adaptive cardinality K over time. The GA settles around K=16 on average, well below the upper bound of 30.

## Methodology

- **Data:** CRSP monthly stock file (CIZ format), Jan 2000-Dec 2025 (WRDS). Risk-free rate: FRED DTB3 (3-month T-bill, annual % converted to monthly decimal).
- **Universe:** NYSE/NASDAQ common stocks, market cap >= $2B (lagged 1 month). Around 870 eligible stocks per month. 60-month burn-in, first rebalancing January 2005. Covariance estimated with Ledoit-Wolf shrinkage (sklearn).
- **GA:** Real-valued weight vector, K in [10, 30] non-zero entries each in [0.02, 0.15] summing to 1. Fitness = monthly Sharpe - lambda * Turnover (lambda = 1.8437). Tournament selection, union-based crossover with arithmetic blend, Gaussian mutation, bisection repair onto bounded simplex. 8 independent runs per period. Population 100, max 200 generations, 20-generation early stop.
- **MVO:** Long-only Sharpe maximisation via SLSQP (3 random restarts). Constrained variant caps individual weights at 0.15. Same estimation window, universe, and cost model as the GA.
- **Evaluation:** 252 monthly OOS periods, rolling 60-month window. Transaction cost gamma = 0.3% per unit of turnover, deducted from all strategies. Significance: paired t-test and Jobson-Korkie test (Memmel 2003 correction).
- **Tuning:** Optuna TPE sampler, 15 trials on 2005-2012 (96 periods). Tuned parameters (pc=0.6054, pm=0.1370, sigma_m=0.1469, lambda=1.8437) fixed for the full 2005-2025 evaluation.

## Repository Structure

```
src/
  data/           loader.py, universe.py, returns.py, risk_free_rate.py
  benchmarks/     mvo.py, equal_weight.py
  optimization/   genetic_algorithm.py, runner.py, optuna_tuner.py,
                  k_sensitivity.py
  evaluation/     metrics.py, figures.py, significance.py, tables.py,
                  post_processing.py, convergence.py, frontier.py,
                  k_sensitivity_tables.py, k_sensitivity_figures.py
  ablation/       ablation_lambda.py
  utils/          portfolio.py, data.py
run_evaluation.sh
tests/
  test_metrics.py
  test_genetic_algorithm.py
  test_backtest_integrity.py
results/
  figures/         F1-F6, A1 convergence PNGs
  tables/          T1_performance through T8_k_behavior (CSV, LaTeX, PNG)
  k_sensitivity/   FK1, FK2, table (CSV, LaTeX, PNG)
  ablation/        lambda ablation (CSV, PNG)
  post_processing/ subperiod, TC sensitivity, K behavior (CSV, PNG)
```

## Reproduction

Run each step in order from the repository root. Raw CRSP data requires a WRDS subscription. Download the monthly stock file (CIZ format, 2000 to 2025) (WRDS path: CRSP Annual Update -> Stock Version 2 (CIZ) -> Monthly Stock File. Required columns include MthCalDt, MthRet, MthRetx, MthPrc, PrimaryExch, ShareType, SecurityType, SecuritySubType, USIncFlg, IssuerType, ShrOut) and the FRED DTB3 series, then place them in `data/raw/` as `crsp_returns.csv` and `risk_free_rate.csv`.

**1. Data loading**

```bash
python3 -m src.data.loader
python3 -m src.data.risk_free_rate
```

**2. Universe construction**

```bash
python3 -m src.data.universe
python3 -m src.data.returns
```

**3. MVO benchmarks**

```bash
python3 -m src.benchmarks.mvo
```

**4. Equal-weight benchmark**

```bash
python3 -m src.benchmarks.equal_weight
```

**5. Optuna tuning** (optional, tuned parameters are already in `genetic_algorithm.py`)

```bash
python3 -m src.optimization.optuna_tuner
```

**6. Full GA run** (35-45 minutes on c2-standard-16, 8 physical cores)

```bash
python3 -m src.optimization.runner
```

Resume from checkpoint by running the same command again. Use `--debug` for a fast 10-period smoke test (3 runs, 50 generations):

```bash
python3 -m src.optimization.runner --debug
```

**6b. K-sensitivity** (optional, runs each fixed K value independently)

```bash
python3 -m src.optimization.k_sensitivity
```

Or a single K value:

```bash
python3 -m src.optimization.k_sensitivity --k 25
```

**7. Lambda ablation** (about 30 minutes for the lambda=0 GA run)

```bash
python3 -m src.ablation.ablation_lambda
```

This runs the full 252-period GA with lambda=0, then compares with the main results parquet.

**8. Evaluation figures and tables**

```bash
./run_evaluation.sh
```

This runs the lambda ablation followed by all evaluation scripts in order. Individual scripts can also be run directly:

```bash
python3 -m src.evaluation.tables
python3 -m src.evaluation.significance
python3 -m src.evaluation.figures
python3 -m src.evaluation.post_processing
python3 -m src.evaluation.k_sensitivity_tables
python3 -m src.evaluation.k_sensitivity_figures
python3 -m src.evaluation.convergence
python3 -m src.evaluation.frontier
```

## Setup

Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m pytest
```

Key dependencies: `pandas`, `numpy`, `scipy`, `scikit-learn`, `pyarrow`, `statsmodels`, `optuna`, `matplotlib`, `seaborn`, `tqdm`.

## Limitations and reproducibility

Raw CRSP data requires a WRDS subscription and is not included. `REPRODUCIBILITY.md` gives the exact reproduction order from raw data to final figures. Integrity tests in `tests/` use synthetic data only and can be run without WRDS access.

## License

MIT © [Georgios Dedempilis](https://github.com/georgeded)
