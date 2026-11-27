#!/usr/bin/env bash
# Builds all figures that this deposit can build and runs the tests.
# A failing step does not stop the script; the summary lists every step.
cd "$(dirname "$0")"
mkdir -p figures
declare -A RC
step() { name=$1; shift; echo; echo "=== $name"; "$@"; RC[$name]=$?; }
step tests        python -m pytest -q -rxs tests
step fig2         python scripts/make_fig02.py
step fig6b        python scripts/make_fig06b.py
step fig7         python scripts/make_fig07.py
step comparison   python scripts/make_comparison.py
echo; echo "=== SUMMARY (exit codes)"
for k in tests fig2 fig6b fig7 comparison; do echo "  $k: ${RC[$k]}"; done
echo "  NOT built (no script/data in the deposit): fig1, fig3, fig4, fig5, fig6(a), Table VII benchmark, PINN"
