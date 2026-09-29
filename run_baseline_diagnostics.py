"""
Diagnostics of the baseline run: positivity of h_x (on its own nodes, with
and without the first node ring around the obstacle), size of the flux at
ignition, thermal shadow behind the obstacle and odd-even content of T.

Requires results/baseline.npz.  Output: results/baseline.json.
"""

import json

import numpy as np

from gk_analysis import checkerboard_index, load_result, obstacle_distance, probe
from gk_solver import flux_on_T_nodes
from settings import results_path

r = load_result(results_path("baseline.npz"))
Kn, Lx = r["Kn"], r["Lx"]
X, Y = np.meshgrid(r["x_hx"], r["y_T"])
fluid = ~r["solid_hx"]
dist = obstacle_distance(X, Y, r)
first_ring = dist <= 1.001 * r["dx"]              # distance dx up to round-off
halos = {"fluid": fluid, "outside_one_cell": fluid & ~first_ring,
         "outside_0.10": fluid & (dist > 0.10), "outside_0.15": fluid & (dist > 0.15)}


def snap(t):
    return int(np.argmin(np.abs(r["snap_t"] - t)))


def first_time_positive(mask):
    """First snapshot time from which min h_x stays positive in ``mask``."""
    mins = [r["snap_hx"][k][mask].min() for k in range(r["snap_t"].size)]
    for k in range(len(mins)):
        if all(m > 0 for m in mins[k:]):
            return float(r["snap_t"][k])
    return None


def max_flux(k):
    hxT, hyT = flux_on_T_nodes(r["snap_hx"][k], r["snap_hy"][k])
    return float(np.sqrt((hxT ** 2 + hyT ** 2).max()))


hxT, hyT = flux_on_T_nodes(r["hx"], r["hy"])
summary = {
    "grid": f"{r['Nx']}x{r['Ny']}",
    "t_end": r["t"],
    "max_h_final": float(np.hypot(hxT, hyT).max()),
    "max_h_t2": max_flux(snap(2.0)),
    "max_h_t5": max_flux(snap(5.0)),
    "min_phi_t2": 1.0 - (Kn ** 2 / 6.0) * max_flux(snap(2.0)) ** 2,
    "min_hx_final": {k: float(r["hx"][m].min()) for k, m in halos.items()},
    "min_hx_transient": float(min(r["snap_hx"][k][fluid].min() for k in range(r["snap_t"].size))),
    "hx_positive_from_t": {k: first_time_positive(m) for k, m in halos.items()},
    "hx_axis_x7": probe(r, "hx", 7.0, r["y_obs"]),
    "T_minus_TF_probe_4_2": probe(r, "T", 4.0, r["y_obs"]) - (1.0 - 4.0 / Lx),
    "checkerboard_index_T": checkerboard_index(r["T"], ~r["in_obstacle"]),
}
results_path("baseline.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print("\nmin h_x per snapshot (fluid | outside one cell | outside 0.10):")
for k, t in enumerate(r["snap_t"]):
    s = r["snap_hx"][k]
    print(f"  t = {t:6.1f}  {s[halos['fluid']].min():+.3e}  "
          f"{s[halos['outside_one_cell']].min():+.3e}  {s[halos['outside_0.10']].min():+.3e}")
