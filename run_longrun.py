"""
run_longrun.py
==============

Long-time convergence studies quoted in Sec. III (initial condition) and in
Sec. III A 1 c / Sec. IV A (fully converged steady state).

  (1) Initial-condition independence at long time.  The baseline wake is
      integrated to t = 1200 from two initial states,
      "fourier-rest" and "cold-rest", and the two final fields are compared.
      The temperature comparison is reported both globally and restricted to
      the wake region x > 1.5, because the residual difference at t = 1200 is
      concentrated in the inlet layer.

  (2) Fully converged steady state on the finest mesh.  The t = 1200 state of
      the 200 x 100 run is interpolated onto the 400 x 200 mesh and relaxed
      for a further t = 30.  The sign of min h_x on this state is the fully-converged
      positivity statement of the paper.



Writes ic_long_fourier-rest.npz, ic_long_cold-rest.npz, wake_fine_relaxed.npz
and longrun_summary.json.
"""

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from gk_solver import solve

here = Path(__file__).resolve().parent

T_LONG = 1200.0
T_RELAX = 30.0

# ---------------------------------------------------------------------------
# (1) initial-condition independence at t = 1200
# ---------------------------------------------------------------------------
long_runs = {}
for ic in ["fourier-rest", "cold-rest"]:
    print(f"--- long run, IC = {ic}, t_end = {T_LONG}")
    r = solve(ic=ic, tEnd=T_LONG, tSnap=(T_LONG,), diag_every=2000, verbose=False)
    long_runs[ic] = r
    np.savez(here / f"ic_long_{ic}.npz",
             T=r["T_final"], hx=r["hx_final"], hy=r["hy_final"],
             inObs=r["inObs"], xc=r["xc"], yc=r["yc"], Xc=r["Xc"], Yc=r["Yc"])

a, b = long_runs["fourier-rest"], long_runs["cold-rest"]
fluid = ~a["inObs"]
wake = fluid & (a["Xc"] > 1.5)

dT = np.abs(a["T_final"] - b["T_final"])
dh = np.hypot(a["hx_final"] - b["hx_final"], a["hy_final"] - b["hy_final"])

max_dT_global = float(dT[fluid].max())
max_dT_wake = float(dT[wake].max())
max_dh_global = float(dh[fluid].max())
min_hx_coarse = float(a["hx_final"][fluid].min())

print(f"    max|dT| global       = {max_dT_global:.4e}")
print(f"    max|dT| wake (x>1.5) = {max_dT_wake:.4e}")
print(f"    max|dh| global       = {max_dh_global:.4e}")
print(f"    min hx (200x100)     = {min_hx_coarse:+.4e}")

# ---------------------------------------------------------------------------
# (2) 400 x 200 steady state, relaxed from the interpolated coarse state
# ---------------------------------------------------------------------------
print(f"--- fine mesh 400x200, relaxed t = {T_RELAX} from the interpolated "
      f"200x100 state")
Nxf, Nyf = 400, 200
Lx, Ly = float(a["Lx"]), float(a["Ly"])
xf = (np.arange(Nxf) + 0.5) * (Lx / Nxf)
yf = (np.arange(Nyf) + 0.5) * (Ly / Nyf)
Xf, Yf = np.meshgrid(xf, yf)
pts = np.column_stack([Yf.ravel(), Xf.ravel()])


def lift(F):
    f = RegularGridInterpolator((a["yc"], a["xc"]), F,
                                bounds_error=False, fill_value=None)
    return f(pts).reshape(Xf.shape)


T0, hx0, hy0 = lift(a["T_final"]), lift(a["hx_final"]), lift(a["hy_final"])
inObs_f = ((np.abs(Xf - float(a["xObs"])) <= float(a["LObs"]) / 2)
           & (np.abs(Yf - float(a["yObs"])) <= float(a["LObs"]) / 2))
hx0[inObs_f] = 0.0
hy0[inObs_f] = 0.0

rf = solve(Nx=Nxf, Ny=Nyf, init=(T0, hx0, hy0), tEnd=T_RELAX,
           tSnap=(T_RELAX,), diag_every=2000, verbose=False)
np.savez(here / "wake_fine_relaxed.npz",
         T=rf["T_final"], hx=rf["hx_final"], hy=rf["hy_final"],
         inObs=rf["inObs"], xc=rf["xc"], yc=rf["yc"])
min_hx_fine = float(rf["hx_final"][~rf["inObs"]].min())
print(f"    min hx (400x200 relaxed) = {min_hx_fine:+.4e}")

# ---------------------------------------------------------------------------
summary = {
    "t_long": T_LONG,
    "ic_t1200_max_dT_wake_region_xgt1p5": max_dT_wake,
    "ic_t1200_max_dT_global": max_dT_global,
    "ic_t1200_max_dh_global": max_dh_global,
    "steady_min_hx_200x100": min_hx_coarse,
    "steady_min_hx_400x200_relaxed": min_hx_fine,
    "t_relax_fine": T_RELAX,
    "note": (f"200x100 steady states from t={T_LONG:g} runs (fourier-rest vs "
             f"cold-rest ICs); steady_min_hx_200x100 is the fourier-rest run. "
             f"400x200 steady state relaxed t={T_RELAX:g} from the "
             f"interpolated coarse steady state."),
}
(here / "longrun_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
