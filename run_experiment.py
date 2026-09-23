"""
run_experiment.py
=================

Bridge between the GK channel model and published experiments on hydrodynamic
phonon transport in graphite ribbons.

For a strip of width w (reference length L_ref = w, so R = 1 and Kn = l/w),
the fully developed linear GK solution with the second-order slip boundary
condition of the paper [Eqs. (15)-(16)] gives

    kappa_eff / kappa_bulk = 1 - B (2 Kn) tanh(1/(2 Kn)),
    B = 1 / (1 + C tanh(1/(2Kn)) + alpha),

i.e. Eq. (25) of the paper, with C(Kn, v) and alpha(Kn) from the
accommodation model in Eq. (16).
This script:

  (1) validates the analytic formula against the 2D code (slip ghost cells)
      at several (Kn, v) pairs;
  (2) tabulates kappa_eff/kappa_bulk vs Kn for several accommodation
      coefficients v (plus the no-slip limit);

Writes experiment_results.npz + experiment_summary.json.
"""

import json
from pathlib import Path

import numpy as np

from gk_solver import solve, channel_kappa_eff, kappa_eff

here = Path(__file__).resolve().parent


def slip_coeffs(Kn, v):
    """Wall coefficients C(Kn, v), alpha(Kn) of Eq. (16), p = min(1/Kn, 1)."""
    p = min(1.0 / Kn, 1.0)
    C = (2.0 / 3.0) * ((3.0 - v * p ** 3) / v - 1.5 * (1.0 - p ** 2) / Kn)
    alpha = 0.25 * (p ** 4 + 2.0 * (1.0 - p ** 2) / Kn ** 2)
    return C, alpha


# ---------------------------------------------------------------------------
# (1) validate the analytic kappa_eff against the 2D code
# ---------------------------------------------------------------------------
val = []
for Kn_i, v in [(0.3, 0.9), (0.7, 0.9), (0.7, 0.3), (1.5, 0.6)]:
    C, alpha = slip_coeffs(Kn_i, v)
    r = solve(Kn=Kn_i, Lx=2.0, Ly=1.0, Nx=96, Ny=48, obstacle=False, ic="fourier-rest",
              tEnd=80.0, tSnap=(80.0,), sidewalls="slip",
              slip_C=C, slip_alpha=alpha, nonlinear=False,
              steady_tol=1.0e-9, diag_every=200, verbose=False)
    k_num = kappa_eff(r)
    k_ana = channel_kappa_eff(1.0, Kn_i, C, alpha)
    val.append(dict(Kn=Kn_i, v=v, C=C, alpha=alpha,
                    k_num=float(k_num), k_ana=float(k_ana),
                    rel_err=float(abs(k_num - k_ana) / k_ana)))
    print(f"(1) Kn={Kn_i:4.2f} v={v:3.1f}:  num={k_num:.5f}  ana={k_ana:.5f}"
          f"  rel.err={val[-1]['rel_err']:.2e}")

# no-slip validation point
r = solve(Kn=0.7, Lx=2.0, Ly=1.0, Nx=96, Ny=48, obstacle=False, ic="fourier-rest",
          tEnd=80.0, tSnap=(80.0,), sidewalls="noslip", nonlinear=False,
          steady_tol=1.0e-9, diag_every=200, verbose=False)
k_num = kappa_eff(r)
k_ana = channel_kappa_eff(1.0, 0.7)
val.append(dict(Kn=0.7, v=None, C=None, alpha=None, k_num=float(k_num),
                k_ana=float(k_ana), rel_err=float(abs(k_num - k_ana) / k_ana)))
print(f"(1) Kn=0.70 no-slip: num={k_num:.5f}  ana={k_ana:.5f}"
      f"  rel.err={val[-1]['rel_err']:.2e}")

# ---------------------------------------------------------------------------
# (2) kappa_eff/kappa_bulk vs Kn curves
# ---------------------------------------------------------------------------
Kn_grid = np.logspace(-2, 1.5, 200)
curves = {"noslip": np.array([channel_kappa_eff(1.0, k) for k in Kn_grid])}
for v in [0.1, 0.3, 0.6, 1.0]:
    ks = []
    for k in Kn_grid:
        C, alpha = slip_coeffs(k, v)
        ks.append(channel_kappa_eff(1.0, k, C, alpha))
    curves[f"v{v}"] = np.array(ks)

np.savez(here / "experiment_results.npz",
         Kn_grid=Kn_grid,
         **{f"curve_{k}": c for k, c in curves.items()},
         val_Kn=np.array([d["Kn"] for d in val]),
         val_num=np.array([d["k_num"] for d in val]),
         val_ana=np.array([d["k_ana"] for d in val]))
(here / "experiment_summary.json").write_text(json.dumps(dict(validation=val), indent=2))
print(json.dumps(dict(validation=val), indent=2))
