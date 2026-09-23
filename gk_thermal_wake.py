"""
gk_thermal_wake.py
==================

Baseline run of the non-linear, non-local Guyer-Krumhansl (GK) heat-transport
solver for a 2D thin nanolayer with a square obstacle, driven by an imposed
temperature difference.  The initial state is defined as follows: the layer
is in equilibrium with the cold reservoir (T = 0, h = 0) and the hot wall is
switched on at t = 0; integration to t = 400 (about ten large-scale diffusive
times) reaches the quasi-stationary wake.  The numerical scheme and all model options live in
``gk_solver.py``; this script reproduces the simulation reported in the paper
and writes ``gk_wake_results.npz``.

Governing equations (non-dimensional):

    dT/dt + (Kn^2 / 3) div h = 0
    Phi(h) dh/dt + h + Phi(h) grad T - Kn^2 Phi(h) div[ Phi(h) grad h ] = 0
    Phi(h) = max(0, 1 - (Kn^2 / 6) |h|^2)        # truncated non-linear modulation

with T the temperature, h the heat flux, Kn the Knudsen number.

Boundary conditions: hot/cold Dirichlet T on left/right, impermeable
free-slip side walls, impermeable (adiabatic) no-slip square obstacle.

The snapshot list is denser than the four panels of Fig. 3 (which are
selected by nearest time in `make_paper_figures.py`) because the extra
snapshots feed the transient positivity diagnostics of
`run_baseline_diagnostics.py`.  Snapshot times do not affect the solution.

Run:  python gk_thermal_wake.py        # writes gk_wake_results.npz
"""

from pathlib import Path

import numpy as np

from gk_solver import solve, Q_from_closure

res = solve(
    Kn=0.7, Lx=8.0, Ly=4.0, Nx=200, Ny=100,
    obstacle=True, xObs=3.0, LObs=1.0,
    tEnd=400.0, CFL=0.20,
    tSnap=(2.0, 5.0, 10.0, 20.0, 30.0, 50.0, 80.0, 100.0, 150.0, 250.0, 400.0),
    ic="cold-rest", obstacle_fill="avg", sidewalls="freeslip",
    nonlinear=True,
)

# Q tensor reconstructed from the quasi-steady closure  Q = -Kn^2 Phi grad h
res.update(Q_from_closure(res))

out_dir = Path(__file__).resolve().parent
out_file = out_dir / "gk_wake_results.npz"
np.savez(out_file, **res)
print(f"Results written to {out_file}")
