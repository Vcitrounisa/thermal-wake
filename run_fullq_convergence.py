"""
Full three-field model against the reduced model for a sequence of
relaxation-time ratios eps = tau_Q/tau_R, on the baseline grid.  Along the
transient the distance between the two solutions is expected to scale as
O(eps).

Output: results/fullq.npz, results/fullq.json.
"""

import json

import numpy as np

from gk_analysis import closure_Q
from gk_solver import solve
from settings import BASELINE, CASE, T_SHORT, results_path

TIMES = (2.0, 5.0, T_SHORT)
EPS = (0.02, 0.05, 0.1, 0.2)

common = dict(CASE, **BASELINE, ic="fourier", t_end=T_SHORT, t_snap=TIMES)
red = solve(**common)
fluid_T = ~red["in_obstacle"]
fluid_x, fluid_y = ~red["solid_hx"], ~red["solid_hy"]
Q_red = closure_Q(red)


def distance(a, b, k):
    """max over fluid nodes of the differences in T and in h at snapshot k."""
    dT = np.abs(a["snap_T"][k] - b["snap_T"][k])[fluid_T].max()
    dh = max(np.abs(a["snap_hx"][k] - b["snap_hx"][k])[fluid_x].max(),
             np.abs(a["snap_hy"][k] - b["snap_hy"][k])[fluid_y].max())
    return float(dT), float(dh)


err_T = np.zeros((len(TIMES), len(EPS)))
err_h = np.zeros((len(TIMES), len(EPS)))
err_Q = np.zeros(len(EPS))
for n, eps in enumerate(EPS):
    full = solve(full_q=True, eps=eps, **common)
    for k in range(len(TIMES)):
        err_T[k, n], err_h[k, n] = distance(full, red, k)
    err_Q[n] = np.abs(full["Qxx"] - Q_red["Qxx"])[fluid_T].max()
    print(f"eps = {eps}: |dh| = {np.array2string(err_h[:, n], precision=3)}, "
          f"|dQxx| = {err_Q[n]:.3e}")

slopes = np.diff(np.log(err_h), axis=1) / np.diff(np.log(EPS))
np.savez(results_path("fullq.npz"), eps=np.array(EPS), times=np.array(TIMES),
         err_T=err_T, err_h=err_h, err_Q=err_Q)
summary = {"eps": list(EPS), "times": list(TIMES),
           "err_h": err_h.tolist(), "err_T": err_T.tolist(), "err_Qxx": err_Q.tolist(),
           "slopes_h": slopes.tolist()}
results_path("fullq.json").write_text(json.dumps(summary, indent=2))
