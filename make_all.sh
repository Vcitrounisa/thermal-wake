#!/usr/bin/env bash
# Run every study in dependency order and build the figures.
#
#   ./make_all.sh                  all studies except the long-time runs
#   ./make_all.sh --with-longrun   also run_longrun.py
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}

run () { echo; echo "== $1"; "$PY" "$1"; }

run run_selftest.py
run run_verification.py
run run_baseline.py
run run_baseline_diagnostics.py
run run_fullq_baseline.py
run run_fullq_convergence.py
run run_domain_length.py
run run_ic_sensitivity.py
if [[ "${1:-}" == "--with-longrun" ]]; then
  run run_longrun.py
fi
run make_figures.py
