#!/usr/bin/env bash
#
# Reproduce every figure and every quantitative claim of
#
#   I. Carlomagno, A. Sellitto, N. Geracitano, V. Citro,
#   "Thermal shadows in non-linear phonon hydrodynamics: Stokes-like
#    heat-flux wakes past a bluff body", Proc. R. Soc. A (2026).
#
# Usage:
#   ./make_all.sh                 # everything except the long-time study
#   ./make_all.sh --with-longrun  # also run_longrun.py (~1 h, see its docstring)
#
# The order below is the dependency order: make_new_figures.py consumes the
# output of five different run_*.py scripts and of the two baseline runs.

set -euo pipefail

# Always run from the archive directory: a couple of scripts write their
# output with a path relative to the current working directory.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PY=${PYTHON:-python3}
echo "python : $("$PY" -VV | head -1)"
echo "dir    : $PWD"
SECONDS=0
run () { echo; echo "=== $* ==="; "$PY" "$@"; }

# 0. implementation cross-check (numpy reference vs numba kernel)
run run_selftest.py

# 1. baseline switch-on wake  ->  gk_wake_results.npz
run gk_thermal_wake.py
run run_baseline_diagnostics.py          # -> baseline_coldstart_summary.json

# 2. full three-field run at the baseline  ->  fullQ_coldstart.npz
run run_fullQ_coldstart.py

# 3. paper Figs. 3, 4, 5
run make_paper_figures.py

# 4. verification and validation studies
run run_verification.py                  # -> verification_*
run run_fullQ.py                         # -> fullQ_*
run run_experiment.py                    # -> experiment_*
run run_ic_sensitivity.py                # -> ic_sensitivity_*
run run_wake_probe.py                    # -> wake_probe_*

# 5. paper Figs. 2, 6, 7 (+ the two response-letter figures)
run make_new_figures.py

# 6. long-time / fine-mesh convergence (expensive, opt in)
if [[ "${1:-}" == "--with-longrun" ]]; then
  run run_longrun.py                     # -> longrun_summary.json
else
  echo
  echo "=== skipping run_longrun.py (pass --with-longrun to include it) ==="
fi

echo
printf "All done in %dh %02dm %02ds.\n" $((SECONDS/3600)) $(((SECONDS%3600)/60)) $((SECONDS%60))
echo "Figures in ./figs, numbers in ./*_summary.json"
