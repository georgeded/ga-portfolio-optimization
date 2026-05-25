# GA Portfolio Optimization

BSc CS thesis project at VU Amsterdam. A genetic algorithm for cardinality-constrained portfolio optimization on US equities, evaluated out-of-sample from January 2005 to December 2025 and compared against mean-variance optimization (MVO) and equal-weight (1/N) benchmarks. The GA selects between 10 and 30 stocks per month, optimizes a Sharpe-minus-turnover fitness function with Optuna-tuned hyperparameters, and is tested across 252 monthly out-of-sample periods using a rolling 60-month estimation window and a transaction cost of γ = 0.3% per unit of turnover. The core question is whether an evolutionary algorithm that directly constrains portfolio size, weight concentration, and turnover can produce better out-of-sample risk-adjusted returns than MVO on a large universe where the sample covariance matrix is rank-deficient (T = 60, N ≈ 867).

## Results

| Strategy | Sharpe (net) | Ann. Return | Ann. Vol | Max Drawdown | Avg Turnover |
|---|---|---|---|---|---|
| **GA (adaptive K)** | **0.2355** | **3.99%** | **16.96%** | **-44.95%** | **21.34%** |
| Unconstrained MVO | 0.5605 | 7.62% | 13.59% | -38.67% | 23.64% |
| Constrained MVO | 0.5572 | 7.57% | 13.59% | -40.46% | 23.64% |
| 1/N (~867 stocks) | 0.5262 | 9.15% | 17.38% | -52.13% | 4.67% |

Evaluation: January 2005 to December 2025, 252 monthly out-of-sample periods. All metrics are on net excess returns after transaction costs. Annualized return = mean monthly × 12; annualized vol = monthly std × √12; Sharpe = annualized return / annualized vol. Constrained MVO caps individual weights at 0.15.

The GA underperforms all three benchmarks. The Optuna-tuned penalty (λ = 1.8437) kept turnover low but at the cost of return-seeking behaviour: at average monthly turnover of ~21%, the penalty subtracts roughly 0.39 Sharpe units per period, pulling the fitness function away from high-return portfolios. The GA converged toward low-turnover solutions rather than high-return ones.

Constrained and unconstrained MVO are nearly identical (Sharpe 0.5572 vs 0.5605). The 0.15 weight cap rarely binds when N ≈ 867, consistent with Jagannathan and Ma (2003).

The GA vs MVO Sharpe gap is statistically significant by the Jobson-Korkie test (p = 0.006 vs unconstrained, p = 0.006 vs constrained). The GA vs 1/N gap is not significant (p = 0.088).

## Figures

![Cumulative net portfolio value](results/figures/F1_cumulative_returns.png)
*Cumulative net portfolio value (2005 to 2025). All strategies show drawdowns during the 2008–09 GFC and the COVID-19 shock in 2020.*

![Rolling 12-month Sharpe ratio](results/figures/F2_rolling_sharpe.png)
*Rolling 12-month Sharpe ratio. GA underperforms MVO across most of the evaluation period.*

<details>
<summary>More figures</summary>

![Monthly portfolio turnover](results/figures/F3_turnover.png)
*Monthly portfolio turnover. GA turnover (21.3%) falls below MVO (23.6%) despite a much smaller portfolio, a direct consequence of the λ penalty.*

![HHI concentration](results/figures/F4_hhi.png)
*HHI concentration over time. GA is more concentrated by construction (avg K = 14). Theoretical minimum: GA ~1/14 = 0.071; MVO ~1/30 = 0.033; 1/N ~1/867 = 0.001.*

![Adaptive cardinality K](results/figures/F5_cardinality.png)
*Adaptive cardinality K over time. Lower bound (K=10) binds in ~21% of periods. Upper bound (K=30) never binds.*

![Mean-variance frontier](results/figures/F6_frontier.png)
*Mean-variance frontier at three representative dates.*

![GA convergence](results/figures/A1_convergence_comparison.png)
*GA convergence: Optuna-tuned vs default parameters.*

</details>

## λ Ablation

The turnover penalty was tested against λ = 0 (no penalty) over 20 out-of-sample periods to check whether the penalty itself was causing the underperformance. The ablation runs the full 252-period chain to keep the `prev_weights` state accurate; only the 20 sampled periods are reported.

| λ | Sharpe (net) | Ann. Return | Ann. Vol | Avg Turnover |
|---|---|---|---|---|
| 0.0 (no penalty) | 0.0366 | 0.50% | 13.69% | 35.72% |
| 1.8437 (tuned) | 0.7933 | 10.87% | 13.64% | 25.22% |

Without the penalty, the GA turns over 35.7% per month; transaction costs consume almost all returns and Sharpe collapses to 0.04. The penalty is necessary - not the cause of underperformance relative to MVO. Removing it makes things substantially worse. The gap between GA and MVO is a fitness-function design issue, not a penalty artifact: λ = 1.8437 keeps costs under control but tips the fitness function too far toward turnover minimization at the expense of returns.

## Repository Structure

```
ga-portfolio-optimization/
├── src/
│   ├── data/
│   │   ├── loader.py             # CRSP CSV to parquet
│   │   ├── universe.py           # Eligible universe construction per rebalancing date
│   │   ├── returns.py            # Excess return computation (ret - rf)
│   │   └── risk_free_rate.py     # FRED DTB3 processing
│   ├── benchmarks/
│   │   ├── mvo.py                # Unconstrained and constrained MVO (SLSQP)
│   │   └── equal_weight.py       # 1/N naive benchmark
│   ├── optimization/
│   │   ├── genetic_algorithm.py  # GA operators, fitness function, main loop
│   │   ├── runner.py             # Rolling OOS experiment with checkpointing
│   │   └── optuna_tuner.py       # Hyperparameter search (TPE, 15 trials)
│   ├── evaluation/
│   │   ├── metrics.py            # Sharpe, Sortino, drawdown, turnover, HHI
│   │   ├── figures.py            # F1 to F5 publication figures
│   │   ├── significance.py       # Paired t-test and Jobson-Korkie significance tests
│   │   ├── tables.py             # Performance and characteristics tables
│   │   ├── post_processing.py    # Subperiod robustness and cost sensitivity
│   │   ├── convergence.py        # Appendix A1 convergence plot
│   │   └── frontier.py           # Figure F6 efficient frontier
│   └── utils/
│       ├── portfolio.py          # Shared utilities (drift, alignment, estimation window)
│       └── data.py               # Data loading entry point
├── ablation_lambda.py            # λ = 0 ablation experiment
├── tests/
│   ├── test_genetic_algorithm.py
│   ├── test_metrics.py
│   └── test_additional.py
├── results/
│   ├── figures/                  # PNG outputs
│   ├── tables/                   # CSV, LaTeX, and PNG outputs
│   ├── ablation/                 # λ ablation results
│   └── post_processing/          # Robustness and sensitivity outputs
└── requirements.txt
```

## Methodology

**Data**
- CRSP monthly stock file (CIZ format), Jan 2000 to Dec 2025, sourced from WRDS
- NYSE and NASDAQ common stocks, market cap ≥ $2B (lagged 1 month to avoid look-ahead bias)
- Around 867 eligible stocks per month (range: 487 to 1,105)
- Risk-free rate: FRED DTB3 (3-month T-bill, annual % converted to monthly decimal)

**Universe construction**
- 60-month burn-in; first rebalancing date is January 2005
- Stock eligible if it has exactly 60 non-missing returns in the estimation window and market cap ≥ $2B at t−1
- Covariance regularized as Σ + 1e-4 × I (T=60 ≪ N≈867 makes the sample covariance rank-deficient)

**Genetic Algorithm**
- Chromosome: real-valued weight vector with K ∈ [10, 30] non-zero entries, each in [0.02, 0.15], summing to 1
- Fitness: monthly Sharpe minus λ × Turnover (λ = 1.8437, Optuna-tuned)
- Selection: tournament (k=3)
- Crossover: union-based asset sampling with arithmetic weight blend (pc = 0.6054)
- Mutation: Gaussian weight perturbation and asset swap (pm = 0.1370, σ_m = 0.1469)
- Repair: bisection projection onto bounded simplex; enforces cardinality, weight bounds, and budget constraint
- Local refinement: greedy pairwise weight-shift hill-climber on best elite, 5 iterations per generation
- Population: 100 | Max generations: 200 | Early stop: 20 stagnant generations
- 8 independent runs per period; canonical portfolio is the one closest to median in-sample fitness

**MVO benchmarks**
- Unconstrained: maximize Sharpe, long-only bounds [0, 1], SLSQP (maxiter=200, ftol=1e-6)
- Constrained: maximize Sharpe, long-only, max weight 0.15, SLSQP
- Both use the same estimation window, universe, and cost model as the GA

**Evaluation protocol**
- 252 monthly OOS periods, January 2005 to December 2025
- Rolling 60-month estimation window (not expanding)
- Transaction cost: γ = 0.3% per unit of turnover, deducted from all strategies
- Turnover computed against post-drift pre-rebalance weights
- Checkpointing after every period for resumable long-running experiments

**Statistical tests**
- Paired t-test: null = mean monthly net excess return difference is zero
- Jobson-Korkie test (Memmel 2003 correction): null = Sharpe ratio difference is zero
- Two-tailed, α = 0.05

**Hyperparameter tuning**
- Optuna TPE sampler, 15 trials, tuning period 2005 to 2012 (96 periods)
- Reduced settings per trial: 3 runs, 30 generations (vs 8 runs and 200 generations in main evaluation)
- Tuned parameters fixed for the full 2005 to 2025 evaluation

## Reproduction

Run each step in order from the repository root. Raw CRSP data requires a WRDS subscription - download the monthly stock file (CIZ format, 2000 to 2025) and the FRED DTB3 series, then place them in `data/raw/` as `crsp_returns.csv` and `risk_free_rate.csv`.

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

**5. Optuna tuning** *(optional — pre-tuned values are already in `genetic_algorithm.py`)*

```bash
python3 -m src.optimization.optuna_tuner
```

**6. Full GA run** (~59 hours on c2-standard-8)

```bash
python3 -m src.optimization.runner
```

Resume from checkpoint by running the same command again. Use `--debug` for a fast 10-period smoke test (3 runs, 50 generations):

```bash
python3 -m src.optimization.runner --debug
```

**7. λ ablation** (~2–3 hours on c2-standard-16)

```bash
python3 ablation_lambda.py
```

**8. Evaluation figures and tables**

```bash
python3 -m src.evaluation.figures
python3 -m src.evaluation.frontier
python3 -m src.evaluation.convergence
python3 -m src.evaluation.tables
python3 -m src.evaluation.significance
python3 -m src.evaluation.post_processing
```

## Requirements

Python 3.11+.

```bash
pip install -r requirements.txt
```

Key dependencies: `pandas`, `numpy`, `scipy`, `statsmodels`, `optuna`, `matplotlib`, `seaborn`, `tqdm`.

## License

MIT © [Georgios Dedempilis](https://github.com/georgeded)
