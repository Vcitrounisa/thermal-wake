"""
Long-time independence of the steady state from the initial state: cold
start and Fourier start integrated to t = 1200 on the baseline grid and
compared.  Expensive; run with ./make_all.sh --with-longrun.  With
``--reuse`` the summary is recomputed from the saved runs.

Output: results/longrun_*.npz, results/longrun.json.
"""

import json
import sys

import numpy as np

from gk_analysis import load_result, obstacle_distance
from gk_solver import solve
from settings import BASELINE, CASE, results_path

T_LONG = 1200.0

runs = {}
for ic in ("fourier", "cold"):
    path = results_path(f"longrun_{ic}.npz")
    if "--reuse" in sys.argv and path.exists():
        runs[ic] = load_result(path)
        continue
    print(f"t = {T_LONG:g}, initial state {ic}")
    runs[ic] = solve(**CASE, **BASELINE, ic=ic, t_end=T_LONG, hist_every=2000)
    np.savez(path, **runs[ic])

a, b = runs["fourier"], runs["cold"]
fT, fx, fy = ~a["in_obstacle"], ~a["solid_hx"], ~a["solid_hy"]
X, _ = np.meshgrid(a["x_T"], a["y_T"])
Xu, Yu = np.meshgrid(a["x_hx"], a["y_T"])
dT = np.abs(a["T"] - b["T"])
first_ring = obstacle_distance(Xu, Yu, a) <= 1.001 * a["dx"]   # distance dx up to round-off
summary = {
    "t": T_LONG,
    "max_dT": float(dT[fT].max()),
    "max_dT_x_gt_1.5": float(dT[fT & (X > 1.5)].max()),
    "max_dh": float(max(np.abs(a["hx"] - b["hx"])[fx].max(), np.abs(a["hy"] - b["hy"])[fy].max())),
    "steady_min_hx": float(a["hx"][fx].min()),
    "steady_min_hx_outside_one_cell": float(a["hx"][fx & ~first_ring].min()),
    "final_residual": float(a["hist_res"][-1]),
}
results_path("longrun.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
