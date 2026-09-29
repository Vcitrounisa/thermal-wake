"""
Baseline simulation: the layer starts in equilibrium with the cold reservoir
(T = 0, h = 0), the hot boundary is switched on at t = 0 and the system is
integrated to t = 400, well into the quasi-stationary regime.

Output: results/baseline.npz (fields, snapshots, history and the flux of the
heat flux from the closure, keys closure_Q*).
"""

import numpy as np

from gk_analysis import closure_Q
from gk_solver import solve
from settings import BASELINE, CASE, results_path

SNAPSHOTS = (2.0, 5.0, 10.0, 18.0, 20.0, 30.0, 50.0, 80.0, 100.0, 150.0, 250.0, 400.0)

res = solve(**CASE, **BASELINE, ic="cold", t_end=400.0, t_snap=SNAPSHOTS, verbose=True)
Q = closure_Q(res)
np.savez(results_path("baseline.npz"), **res, **{f"closure_{k}": v for k, v in Q.items()})
