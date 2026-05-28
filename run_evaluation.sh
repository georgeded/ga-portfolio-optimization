#!/bin/bash
set -e

# Guard against a nested results/results/ folder that some scripts can
# accidentally create when OUTPUT_DIR resolves relative to results/.
[ -d "results/results" ] && rm -rf results/results

echo "Running all evaluation scripts"

echo "[1/9] Ablation..."
python3 -m src.ablation.ablation_lambda

echo "[2/9] Tables..."
python3 -m src.evaluation.tables

echo "[3/9] Significance..."
python3 -m src.evaluation.significance

echo "[4/9] Figures..."
python3 -m src.evaluation.figures

echo "[5/9] Post processing..."
python3 -m src.evaluation.post_processing

echo "[6/9] K sensitivity tables..."
python3 -m src.evaluation.k_sensitivity_tables

echo "[7/9] K sensitivity figures..."
python3 -m src.evaluation.k_sensitivity_figures

echo "[8/9] Convergence..."
python3 -m src.evaluation.convergence

echo "[9/9] Frontier..."
python3 -m src.evaluation.frontier

echo ""
echo "ALL DONE"
echo "PNGs saved in:"
find results/ -name "*.png" | sort
