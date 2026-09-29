"""
Initial state: cold start (run_baseline.py) vs Fourier start with h = 0 and
with h = G e_x, at t = 18 and 400.  Enclosed T nodes: runs with their
initial value set to 0, T_F and 1.
"""

import json

import numpy as np

from gk_analysis import load_result
from gk_solver import solve
from settings import BASELINE, CASE, T_SHORT, results_path

common = dict(CASE, **BASELINE)


def snapshot(res, t):
    k = int(np.argmin(np.abs(res["snap_t"] - t)))
    return res["snap_T"][k], res["snap_hx"][k], res["snap_hy"][k]


def differences(a, b, masks):
    fT, fx, fy = masks
    return {"max_dT": float(np.abs(a[0] - b[0])[fT].max()),
            "max_dh": float(max(np.abs(a[1] - b[1])[fx].max(),
                                np.abs(a[2] - b[2])[fy].max()))}


# (1) initial state
runs = {"cold": load_result(results_path("baseline.npz"))}
for ic in ("fourier", "fourier-flux"):
    print(f"initial state: {ic}")
    runs[ic] = solve(**common, ic=ic, t_end=400.0, t_snap=(T_SHORT, 400.0), hist_every=200)
masks = (~runs["cold"]["in_obstacle"], ~runs["cold"]["solid_hx"], ~runs["cold"]["solid_hy"])

summary = {"initial_state": {}}
for ic in ("fourier", "fourier-flux"):
    summary["initial_state"][ic] = {
        f"t{t:g}": differences(snapshot(runs[ic], t), snapshot(runs["cold"], t), masks)
        for t in (T_SHORT, 400.0)}
    print(ic, summary["initial_state"][ic])
summary["initial_state"]["fourier-flux_vs_fourier"] = {
    f"t{t:g}": differences(snapshot(runs["fourier-flux"], t), snapshot(runs["fourier"], t), masks)
    for t in (T_SHORT, 400.0)}
print("fourier-flux vs fourier", summary["initial_state"]["fourier-flux_vs_fourier"])

out = {}
jc = BASELINE["Ny"] // 2
for ic, r in runs.items():
    out[f"T_mid_{ic}"] = r["T"][jc]
    out[f"hist_t_{ic}"], out[f"hist_max_h_{ic}"] = r["hist_t"], r["hist_max_h"]
out["x"] = runs["cold"]["x_T"]

# (2) enclosed T nodes
variants = {"zero": 0.0, "fourier": None, "one": 1.0}
enc = {}
for name, value in variants.items():
    print(f"enclosed T nodes: {name}")
    enc[name] = solve(**common, ic="fourier", t_end=T_SHORT, T_enclosed=value)
summary["enclosed_T"] = {
    name: differences((enc[name]["T"], enc[name]["hx"], enc[name]["hy"]),
                      (enc["zero"]["T"], enc["zero"]["hx"], enc["zero"]["hy"]), masks)
    for name in ("fourier", "one")}
print(summary["enclosed_T"])
for name, r in enc.items():
    out[f"enclosed_T_{name}"] = r["T"]
out["enclosed"] = enc["zero"]["enclosed_T"]
out["in_obstacle"] = enc["zero"]["in_obstacle"]
out["y"] = enc["zero"]["y_T"]

np.savez(results_path("ic_sensitivity.npz"), **out)
results_path("ic_sensitivity.json").write_text(json.dumps(summary, indent=2))
