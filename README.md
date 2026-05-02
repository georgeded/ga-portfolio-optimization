# GA Portfolio Optimization

A research-grade Python implementation of a **Genetic Algorithm (GA) for cardinality-constrained portfolio optimization**, developed as part of a BSc Computer Science thesis at Vrije Universiteit Amsterdam.

The project compares an evolutionary portfolio optimizer against classical mean-variance baselines and a naive diversification benchmark using a rolling-window, strictly out-of-sample evaluation on US equities.

---

## Overview

Mean-variance optimization is highly sensitive to estimation error, especially when the number of assets is large relative to the estimation window. This project studies whether a constraint-aware evolutionary algorithm can produce more robust out-of-sample portfolios by explicitly controlling portfolio size, asset weights, turnover, and concentration.

The GA optimizes a long-only portfolio under realistic investment constraints:

- cardinality constraint: `10 <= K <= 30`
- per-asset weight bounds: `0.02 <= w_i <= 0.15`
- full investment: `sum(w) = 1`
- no short selling
- optional turnover penalty in the fitness function
- monthly rolling-window rebalancing

The final evaluation compares:

1. Genetic Algorithm with adaptive cardinality
2. Constrained Mean-Variance Optimization
3. Long-only unconstrained Mean-Variance Optimization
4. Equal-weight `1/N` benchmark

---

## Research Design

### Data

The project uses monthly US equity data from CRSP and a monthly risk-free rate series.

Raw data is not included in the repository because CRSP data is proprietary. The expected local inputs are:

```text
data/raw/crsp_returns.csv
data/raw/risk_free_rate.csv
```

The pipeline converts these into processed parquet files used by the optimization and evaluation modules:

```text
data/raw/crsp_returns.parquet
data/processed/risk_free_rate.parquet
data/processed/universe.parquet
data/processed/returns.parquet
```

### Universe Construction

The eligible universe is built dynamically at every monthly rebalance date. The filters include:

- NYSE and NASDAQ listings only
- common stocks only
- lagged market capitalization of at least USD 2 billion
- at least 60 months of complete return history
- dynamic inclusion and exclusion to reduce survivorship bias

The first out-of-sample rebalance date is January 2005 after a 60-month estimation window.

### Optimization Setup

At each rebalance date `t`:

1. Estimate expected returns and covariance using months `t-60` to `t-1`
2. Optimize portfolio weights using each strategy
3. Apply the selected portfolio to the next realized monthly return
4. Record net out-of-sample excess return after transaction costs
5. Drift portfolio weights before computing turnover at the next rebalance

The evaluation period is January 2005 to December 2025.

---

## Genetic Algorithm

The chromosome is a real-valued weight vector of length `N`, where only `K` entries are non-zero.

### Core Components

- bounded-simplex repair operator for exact feasibility
- tournament selection
- two-child weight-biased crossover
- Gaussian weight mutation
- asset-swap mutation
- local pairwise weight-shift refinement
- elitism
- early stopping
- Optuna hyperparameter tuning

### Fitness Function

The GA maximizes penalized in-sample Sharpe ratio:

```text
fitness(w) = Sharpe(w) - lambda * turnover(w, previous_weights)
```

where turnover is measured against the drifted pre-rebalance portfolio when available.

---

## Repository Structure

```text
.
├── src/
│   ├── data/
│   │   ├── loader.py              # Load and validate raw CRSP data
│   │   ├── universe.py            # Build dynamic eligible universe
│   │   ├── risk_free_rate.py      # Process risk-free rate data
│   │   └── returns.py             # Compute excess returns
│   │
│   ├── optimization/
│   │   ├── genetic_algorithm.py   # Core GA operators and main loop
│   │   ├── optuna_tuner.py        # Hyperparameter tuning
│   │   └── runner.py              # Out-of-sample GA experiment runner
│   │
│   ├── benchmarks/
│   │   ├── equal_weight.py        # 1/N benchmark
│   │   └── mvo.py                 # MVO benchmarks
│   │
│   ├── evaluation/
│   │   ├── metrics.py             # Portfolio performance metrics
│   │   ├── tables.py              # Thesis result tables
│   │   ├── figures.py             # Main thesis figures
│   │   ├── significance.py        # Statistical significance tests
│   │   ├── convergence.py         # GA convergence diagnostics
│   │   └── frontier.py            # Efficient frontier visualization
│   │
│   └── utils/
│       ├── data.py                # Shared data loading helpers
│       └── portfolio.py           # Rolling-window and turnover utilities
│
├── tests/
│   ├── test_genetic_algorithm.py
│   └── test_metrics.py
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/georgeded/ga-portfolio-optimization.git
cd ga-portfolio-optimization
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Prepare Data

Load and validate raw CRSP data:

```bash
python3 -m src.data.loader
```

Process the risk-free rate:

```bash
python3 -m src.data.risk_free_rate
```

Build the eligible universe:

```bash
python3 -m src.data.universe
```

Compute monthly excess returns:

```bash
python3 -m src.data.returns
```

### 2. Run Benchmarks

Equal-weight benchmark:

```bash
python3 -m src.benchmarks.equal_weight
```

Mean-variance benchmarks:

```bash
python3 -m src.benchmarks.mvo
```

### 3. Tune GA Hyperparameters

Debug run:

```bash
python3 -m src.optimization.optuna_tuner --debug
```

Full tuning run:

```bash
python3 -m src.optimization.optuna_tuner
```

### 4. Run GA Experiment

Debug run:

```bash
python3 -m src.optimization.runner --debug
```

Full run:

```bash
python3 -m src.optimization.runner
```

The GA runner supports checkpointing and can resume interrupted cloud jobs.

### 5. Generate Tables and Figures

```bash
python3 -m src.evaluation.tables
python3 -m src.evaluation.figures
python3 -m src.evaluation.significance
python3 -m src.evaluation.convergence
python3 -m src.evaluation.frontier
```

Outputs are written to:

```text
results/benchmarks/
results/ga/
results/tables/
results/figures/
results/optuna/
```

These folders are ignored by Git by default because they may contain large generated files.

---

## Testing

Run the test suite with:

```bash
python3 -m pytest tests/ -v
```

The tests cover:

- bounded-simplex projection
- repair feasibility and idempotence
- population initialization
- tournament selection
- crossover and mutation operators
- local refinement
- full GA execution
- portfolio metric functions

---

## Evaluation Metrics

The project reports standard out-of-sample portfolio metrics:

- annualized net return
- annualized volatility
- Sharpe ratio
- Sortino ratio
- maximum drawdown
- monthly turnover
- transaction cost
- Herfindahl-Hirschman Index
- portfolio size / cardinality

Statistical significance is assessed using paired return tests and Sharpe-ratio comparison tests.

---

## Methodological Notes

- All reported performance is out-of-sample.
- Portfolio optimization uses only information available at the rebalance date.
- The risk-free rate is used for excess-return and Sharpe calculations, not as an investable GA asset.
- Covariance matrices are diagonally regularized for numerical stability in high-dimensional estimation windows.
- CRSP data and generated result files are excluded from version control.

---

## Tech Stack

- Python
- NumPy
- pandas
- SciPy
- Optuna
- Matplotlib
- pytest

---

## Disclaimer

This repository is for academic research purposes only. It is not financial advice and should not be used as a live trading or investment system.

---

## Author

**Georgios Dedempilis**  
BSc Computer Science, Vrije Universiteit Amsterdam  