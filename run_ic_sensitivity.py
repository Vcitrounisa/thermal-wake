"""
run_ic_sensitivity.py
=====================

Two studies on the baseline wake configuration:

  (1) Initial-condition independence: the quasi-steady state is an attractor.
      Three initial states are integrated to t = 18:
        - "fourier-rest"  : T = Fourier profile, h = 0        (paper baseline)
        - "cold-rest"     : T = 0 everywhere,   h = 0         (equilibrium with
                            the cold reservoir; hot wall switched on at t = 0)
        - "fourier-flux"  : T = Fourier profile, h = G e_x    (Fourier-consistent
                            initial flux)
      and the pairwise differences of the final fields are reported.

  (2) Obstacle temperature-extension study:
        - "avg"    : neighbour-average fill (paper baseline)
        - "frozen" : no fill; T stays at its initial value inside the mask
                     (turns the obstacle into an isothermal cold spot ->
                     documents the artefact, referee 2 question 1)
        - "compat" : fill honouring the wall-compatibility relation
                     dT/dn = Kn^2 [div(Phi grad h)].n

Writes ic_sensitivity_results.npz + ic_sensitivity_summary.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gk_solver import solve

here = Path(__file__).resolve().parent
out, summary = {}, {}

# ---------------------------------------------------------------------------
# (1) initial-condition independence
# ---------------------------------------------------------------------------
runs = {}
for ic in ["fourier-rest", "cold-rest", "fourier-flux"]:
    print(f"--- IC = {ic}")
    runs[ic] = solve(ic=ic, tEnd=400.0, tSnap=(18.0, 400.0), verbose=False,
                     diag_every=200)

fluid = ~runs["fourier-rest"]["inObs"]
base = runs["fourier-rest"]
summary["ic_diffs"] = {}
for ic in ["cold-rest", "fourier-flux"]:
    r = runs[ic]
    d = {}
    for k, t in [(0, 18.0), (1, 400.0)]:
        dT = float(np.max(np.abs(r["snap_T"][k] - base["snap_T"][k])[fluid]))
        dh = float(np.max(np.hypot(r["snap_hx"][k] - base["snap_hx"][k],
                                   r["snap_hy"][k] - base["snap_hy"][k])[fluid]))
        d[f"t{t:g}"] = dict(max_dT=dT, max_dh=dh)
        print(f"    t={t:5.1f}: max|dT| = {dT:.3e},  max|dh| = {dh:.3e}")
    summary["ic_diffs"][ic] = d

jc = base["Ny"] // 2
out["x"] = base["xc"]
for ic, r in runs.items():
    out[f"T_mid_{ic}"] = r["T_final"][jc, :]
    out[f"hx_mid_{ic}"] = r["hx_final"][jc, :]
    out[f"diag_t_{ic}"] = r["diag_t"]
    out[f"diag_minhx_{ic}"] = r["diag_min_hx"]
    out[f"diag_maxh_{ic}"] = r["diag_max_h"]   # used by make_new_figures (fig_resp_ic)
out["inobs_x"] = (np.abs(base["xc"] - base["xObs"]) <= base["LObs"] / 2)

# ---------------------------------------------------------------------------
# (2) obstacle temperature-extension study
# ---------------------------------------------------------------------------
fills = {}
for fill in ["avg", "frozen", "compat"]:
    print(f"--- obstacle_fill = {fill}")
    fills[fill] = solve(obstacle_fill=fill, tSnap=(18.0,), verbose=False)

summary["fill_diffs"] = {}
for fill in ["frozen", "compat"]:
    r = fills[fill]
    dT = float(np.max(np.abs(r["T_final"] - fills["avg"]["T_final"])[fluid]))
    dh = float(np.max(np.hypot(r["hx_final"] - fills["avg"]["hx_final"],
                               r["hy_final"] - fills["avg"]["hy_final"])[fluid]))
    summary["fill_diffs"][fill] = dict(max_dT=dT, max_dh=dh)
    print(f"    {fill}: max|dT| = {dT:.3e},  max|dh| = {dh:.3e}")

for fill, r in fills.items():
    out[f"fill_T_{fill}"] = r["T_final"]
    out[f"fill_hx_{fill}"] = r["hx_final"]
    out[f"fill_hy_{fill}"] = r["hy_final"]
    out[f"fill_Tmid_{fill}"] = r["T_final"][jc, :]
out["Xc"], out["Yc"] = base["Xc"], base["Yc"]
out["inObs"] = base["inObs"]
out["xObs"], out["yObs"], out["LObs"] = base["xObs"], base["yObs"], base["LObs"]

# energy budget of the frozen run: net heat absorbed by the obstacle boundary
# (diagnosed by the flux divergence around the mask)
np.savez(here / "ic_sensitivity_results.npz", **out)
(here / "ic_sensitivity_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
