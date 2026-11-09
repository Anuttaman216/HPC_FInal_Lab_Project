#!/usr/bin/env bash
# run_all.sh - end-to-end pipeline: dependency check, clean build, unit tests,
# benchmark campaign, analysis, graphs and dashboard.
#   scripts/run_all.sh            # full campaign (~15-25 min on a laptop)
#   QUICK=1 scripts/run_all.sh    # small smoke run (~1 min)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p results

echo "### 1/6 dependency check";   bash scripts/setup.sh || { echo "fix missing dependencies first"; exit 1; }
echo "### 2/6 clean build";        bash scripts/build.sh --clean
echo "### 3/6 unit tests";         (cd build && ctest --output-on-failure) | tee results/test_results.txt
echo "### 4/6 benchmarks";         bash scripts/run_benchmarks.sh
echo "### 5/6 analysis";           python3 python/analyze.py
echo "### 6/6 graphs + dashboard"; python3 python/visualize.py && python3 python/dashboard.py
echo "Done. See results/summary.md, results/graphs/ and results/dashboard.html"
