"""
Position of the truncation sections.  The steady wake of the baseline domain
(L = 8, x_obs = 3) is compared with the ones obtained with the outlet moved
downstream by 4 and 8 (L = 12, 16) and with the inlet moved upstream by 4
(L = 12, x_obs = 7).  The obstacle adds a finite thermal resistance, so the
runs are compared at the same heat flux Q through the layer: h as h/Q and T
through the disturbance with respect to the conduction state with flux Q,

    theta = [T - Tm(x_obs - 2) + (Q/R) (x - x_obs + 2)] / Q    (Tm: mean in y)

on the nodes with -2 <= x - x_obs <= 3.

The longer domains start from the baseline state for 1 <= x - x_obs + 3 <= 6,
continued outside by the conduction state with the baseline flux; T_hot is
the value for which this state gives T = 0 at x = L.  All runs are integrated
until max |dh/dt| < 1e-8.

Requires results/baseline.npz.  Output: results/domain_length.npz,
results/domain_length.json.
"""

import json

import numpy as np

from gk_analysis import load_result, probe
from gk_solver import solve
from settings import CASE, results_path

DOMAINS = {"L8": (8.0, 3.0), "L12": (12.0, 3.0), "L16": (16.0, 3.0), "L12_inlet": (12.0, 7.0)}
PROBES = [("hx", 1.0, 2.0), ("hx", 1.5, 2.0), ("hx", 3.0, 1.0), ("theta", 1.0, 2.0), ("theta", 1.5, 2.0)]
TOL = 1e-8

base = load_result(results_path("baseline.npz"))
dx, R, Ny = base["dx"], base["Ly"], base["Ny"]


def flux(r):
    return float((r["hx"].sum(axis=0) * r["dy"]).mean())


def initial_state(L, x_obs):
    G = flux(base) / R
    s = int(round((x_obs - base["x_obs"]) / dx))
    lo, hi = int(round(1.0 / dx)), int(round(6.0 / dx))
    Nx = int(round(L / dx))
    x = np.arange(Nx + 1) * dx
    T = np.empty((Ny, Nx + 1))
    hx = np.full((Ny, Nx), G)
    hy = np.zeros((Ny + 1, Nx + 1))
    T[:, s + lo:s + hi + 1] = base["T"][:, lo:hi + 1]
    hx[:, s + lo:s + hi] = base["hx"][:, lo:hi]
    hy[:, s + lo:s + hi + 1] = base["hy"][:, lo:hi + 1]
    T_lo, T_hi = base["T"][:, lo].mean(), base["T"][:, hi].mean()
    shift = G * (L - x[s + hi]) - T_hi
    T[:, s + lo:s + hi + 1] += shift
    T[:, s + hi + 1:] = T_hi + shift - G * (x[s + hi + 1:] - x[s + hi])
    T[:, :s + lo] = T_lo + shift + G * (x[s + lo] - x[:s + lo])
    return float(T_lo + shift + G * x[s + lo]), (T, hx, hy)


def theta(r):
    Q = flux(r)
    i = int(round((r["x_obs"] - 2.0) / dx))
    return (r["T"] - r["T"][:, i].mean() + (Q / R) * (r["x_T"] - r["x_obs"] + 2.0)) / Q


def window(r, xi_lo):
    Q = flux(r)
    i0, i1 = (int(round((r["x_obs"] + xi) / dx)) for xi in (xi_lo, 3.0))
    return (r["hx"][:, i0:i1] / Q, ~r["solid_hx"][:, i0:i1],
            r["hy"][:, i0:i1 + 1] / Q, ~r["solid_hy"][:, i0:i1 + 1],
            theta(r)[:, i0:i1 + 1], ~r["in_obstacle"][:, i0:i1 + 1])


def differences(a, b, xi_lo=-2.0):
    """Largest differences on the nodes with x - x_obs >= xi_lo, relative to the maxima of a."""
    hxa, fx, hya, fy, tha, fT = window(a, xi_lo)
    hxb, _, hyb, _, thb, _ = window(b, xi_lo)
    dh = max(np.abs(hxa - hxb)[fx].max(), np.abs(hya - hyb)[fy].max())
    return {"max_dh": float(dh / np.abs(hxa).max()),
            "max_dtheta": float(np.abs(tha - thb)[fT].max() / np.abs(tha[fT]).max())}


def probes(r):
    Q, th = flux(r), theta(r)
    return {f"{k}({xi:g},{y:g})": (probe(r, "hx", r["x_obs"] + xi, y) / Q if k == "hx"
                                  else probe(dict(r, T=th), "T", r["x_obs"] + xi, y))
            for k, xi, y in PROBES}


runs = {}
for name, (L, x_obs) in DOMAINS.items():
    if name == "L8":
        T_hot, init = 1.0, (base["T"], base["hx"], base["hy"])
    else:
        T_hot, init = initial_state(L, x_obs)
    runs[name] = solve(**dict(CASE, Lx=L, Nx=int(round(L / dx)), x_obs=x_obs, T_hot=T_hot),
                       init=init, t_end=2000.0, steady_tol=TOL, hist_every=1000, verbose=True)

p0 = probes(runs["L8"])
summary = {}
for name, r in runs.items():
    Qx = r["hx"].sum(axis=0) * r["dy"]
    p = probes(r)
    summary[name] = {"L": r["Lx"], "x_obs": r["x_obs"], "T_hot": r["T_hot"], "t": r["t"],
                     "residual": float(r["hist_res"][-1]), "Q": flux(r),
                     "Q_variation": float(np.ptp(Qx) / Qx.mean()), "probes": p}
    if name != "L8":
        summary[name]["vs_L8"] = {
            "x-x_obs>=-2": differences(runs["L8"], r),
            "x-x_obs>=-1": differences(runs["L8"], r, -1.0),
            "x-x_obs>=0.5": differences(runs["L8"], r, 0.5),
            "probes": {k: abs(p[k] - p0[k]) / abs(p0[k]) for k in p}}
summary["L16"]["vs_L12"] = differences(runs["L12"], runs["L16"])
print(json.dumps(summary, indent=2))

jc = Ny // 2
np.savez(results_path("domain_length.npz"),
         **{f"xi_{n}": r["x_T"] - r["x_obs"] for n, r in runs.items()},
         **{f"theta_{n}": theta(r)[jc] for n, r in runs.items()},
         **{f"hx_{n}": r["hx"][jc] / flux(r) for n, r in runs.items()})
results_path("domain_length.json").write_text(json.dumps(summary, indent=2))
