# GA Portfolio Optimization

Genetic algorithm for cardinality-constrained portfolio optimization on US equities. BSc CS thesis, VU Amsterdam.

![Python](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![Institution](https://img.shields.io/badge/thesis-VU%20Amsterdam-purple)

## Results

| Strategy | Sharpe (net) | Return (net) | Vol | Max DD | Avg Turnover |
|---|---|---|---|---|---|
| **GA (adaptive K)** | **0.2355** | **4.00%** | **16.96%** | **-44.95%** | **21.34%** |
| Unconstrained MVO | 0.5605 | 7.62% | 13.59% | -38.67% | 23.64% |
| Constrained MVO | 0.5572 | 7.57% | 13.59% | -40.46% | 23.64% |
| 1/N (~867 stocks) | 0.5262 | 9.15% | 17.38% | -52.13% | 4.67% |

Evaluation: January 2005 to December 2025, 252 monthly out-of-sample periods, transaction cost γ = 0.3% per unit of turnover.

The GA underperforms all three benchmarks. The Optuna-tuned turnover penalty (λ = 1.8437) dominated the fitness function during search: at average turnover of ~21%, the penalty subtracts ~0.39 Sharpe units per period, which suppresses return-seeking behaviour. The GA converged toward low-turnover portfolios rather than high-return ones. This illustrates a general sensitivity of fitness-function design in EA-based portfolio optimization that the prior literature, evaluated in-sample only, does not address.

Constrained and unconstrained MVO perform near-identically (Sharpe 0.557 vs 0.560). The 0.15 weight cap rarely binds when N is around 867, consistent with Jagannathan and Ma (2003).

## Overview

Mean-variance optimization breaks down when the estimation window is short relative to the number of assets. With T = 60 months and N around 867 stocks, the sample covariance matrix is nearly singular (rank at most 59 in an 867x867 matrix). The question is whether an evolutionary algorithm that directly constrains portfolio size, weight concentration, and turnover can produce better out-of-sample results than MVO and naive diversification.

The GA selects K in [10, 30] stocks each period and optimises a Sharpe-minus-turnover fitness function with Optuna-tuned parameters, evaluated over 252 monthly out-of-sample periods (January 2005 to December 2025). Against the same universe and cost model it achieves a net Sharpe of 0.2355, below all three benchmarks. The main finding is not that GAs are poor portfolio optimizers, but that λ is a critical design parameter. With λ = 1.8437 selected over 15 Optuna trials on a 96-period tuning window, the search traded off too much return for turnover stability.

## Figures

![Cumulative net portfolio value](results/figures/F1_cumulative_returns.png)
*Cumulative net portfolio value (2005 to 2025). All strategies show drawdowns during the 2008-09 GFC and the COVID-19 shock in 2020.*

![Rolling 12-month Sharpe ratio](results/figures/F2_rolling_sharpe.png)
*Rolling 12-month Sharpe ratio. GA underperforms MVO across most of the evaluation period.*

<details>
<summary>More figures</summary>

![Monthly portfolio turnover](results/figures/F3_turnover.png)
*Monthly portfolio turnover. GA turnover (21.3%) falls below MVO (23.6%) despite a much smaller portfolio, a direct consequence of the λ penalty.*

![HHI concentration](results/figures/F4_hhi.png)
*HHI concentration over time. GA is more concentrated by construction (avg K=14). Theoretical minimum differs across strategies: GA ~1/14 = 0.071; MVO ~1/30 = 0.033; 1/N ~1/867 = 0.001.*

![Adaptive cardinality K](results/figures/F5_cardinality.png)
*Adaptive cardinality K over time. Lower bound (K=10) binds in ~21% of periods. Upper bound (K=30) never binds.*

![Mean-variance frontier](results/figures/F6_frontier.png)
*Mean-variance frontier at three representative dates. GA position computed using 50 generations for illustration only.*

![GA convergence](results/figures/A1_convergence_comparison.png)
*GA convergence: Optuna-tuned vs default parameters. Tuned operators reach higher in-sample Sharpe in fewer generations.*

</details>

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
├── tests/
│   ├── test_genetic_algorithm.py # GA operator and feasibility tests
│   ├── test_metrics.py           # Evaluation metrics unit tests
│   └── test_additional.py        # Additional coverage
├── results/
│   ├── figures/                  # PNG outputs (tracked)
│   ├── tables/                   # CSV, LaTeX, and PNG outputs (tracked)
│   └── post_processing/          # Robustness and sensitivity outputs (tracked)
└── requirements.txt
```

## Methodology

**Data**
- CRSP monthly stock file (CIZ format), Jan 2000 to Dec 2025, sourced from WRDS
- NYSE and NASDAQ common stocks, market cap at least $2B (lagged 1 month to avoid look-ahead bias)
- Around 867 eligible stocks per month (range: 487 to 1,105)
- Risk-free rate: FRED DTB3 (3-month T-bill, annual % converted to monthly decimal)

**Universe construction**
- 60-month burn-in; first rebalancing date is January 2005
- Stock eligible if it has exactly 60 non-missing returns in the estimation window and market cap at least $2B at t-1
- Covariance regularized as Sigma + 1e-4 times I (T=60 much smaller than N=867 makes sample Sigma rank-deficient)

**Genetic Algorithm**
- Chromosome: real-valued weight vector with K in [10, 30] non-zero entries, each in [0.02, 0.15], summing to 1
- Fitness: monthly Sharpe minus lambda times Turnover (lambda = 1.8437, Optuna-tuned)
- Selection: tournament (k=3)
- Crossover: union-based asset sampling with arithmetic weight blend (pc = 0.6054)
- Mutation: Gaussian weight perturbation and asset swap (pm = 0.1370, sigma_m = 0.1469)
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
- Rolling 60-month estimation window (not expanding, to avoid regime contamination)
- Transaction cost: γ = 0.3% per unit of turnover, deducted from all strategies
- Turnover computed against post-drift pre-rebalance weights
- Checkpointing after every period for resumable long-running experiments

**Statistical tests**
- Paired t-test: null hypothesis is that mean monthly net excess return difference = 0
- Jobson-Korkie test (Memmel 2003 correction): null hypothesis is that Sharpe ratio difference = 0
- Alpha = 0.05, two-tailed

**Hyperparameter tuning**
- Optuna TPE sampler, 15 trials, tuning period 2005 to 2012 (96 periods)
- Reduced settings per trial: 3 runs, 30 generations (vs 8 runs and 200 generations in main evaluation)
- Tuned parameters fixed for the full 2005 to 2025 evaluation

## Installation

```bash
git clone https://github.com/georgeded/ga-portfolio-optimization.git
cd ga-portfolio-optimization
pip install -r requirements.txt
```

Raw CRSP data requires a WRDS subscription. Download the monthly stock file (CIZ format, 2000 to 2025) and the FRED DTB3 series, then place them in `data/raw/` as `crsp_returns.csv` and `risk_free_rate.csv`.

## Reproduction

Run each step in order from the repository root.

**1. Data pipeline**

```bash
python3 -m src.data.loader
python3 -m src.data.universe
python3 -m src.data.risk_free_rate
python3 -m src.data.returns
```

**2. Hyperparameter tuning** *(optional; pre-tuned values are already in `genetic_algorithm.py`)*

```bash
python3 -m src.optimization.optuna_tuner
```

**3. Benchmarks**

```bash
python3 -m src.benchmarks.mvo
python3 -m src.benchmarks.equal_weight
```

**4. GA out-of-sample experiment**

```bash
# Full run (around 59 hours on c2-standard-8)
python3 -m src.optimization.runner

# Debug mode (3 runs, 50 generations, 10 periods)
python3 -m src.optimization.runner --debug

# Resume from checkpoint
python3 -m src.optimization.runner
```

**5. Evaluation**

```bash
python3 -m src.evaluation.figures
python3 -m src.evaluation.frontier
python3 -m src.evaluation.convergence
python3 -m src.evaluation.tables
python3 -m src.evaluation.significance
python3 -m src.evaluation.post_processing
```

## License

MIT © [Georgios Dedempilis](https://github.com/georgeded)