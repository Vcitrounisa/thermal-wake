"""
Baseline simulation with the full three-field model (independent flux of the
heat flux Q, eps = tau_Q/tau_R = 0.02), for the comparison of the Q fields
with the closure.

Output: results/fullq_baseline.npz.
"""

import numpy as np

from gk_solver import solve
from settings import BASELINE, CASE, results_path

res = solve(**CASE, **BASELINE, ic="cold", t_end=400.0, full_q=True, eps=0.02,
            hist_every=500, verbose=True)
np.savez(results_path("fullq_baseline.npz"), **res)
