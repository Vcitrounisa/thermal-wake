"""
run_baseline_diagnostics.py
===========================

Positivity and thermal-shadow diagnostics of the baseline switch-on run
produced by ``gk_thermal_wake.py`` (cold start, t_end = 400, 200 x 100).

Two conventions are reported side by side for the near-wall exclusion, since
the distinction matters for the wording of the positivity statement:

  * ``halo = 0.10``  : 2.5 cells at the 200 x 100 resolution
  * ``halo = dx``    : a one-cell ring around the obstacle.

Writes baseline_coldstart_summary.json.
"""

import json
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
r = np.load(here / "gk_wake_results.npz")

Xc, Yc = r["Xc"], r["Yc"]
xc = r["xc"]
inObs = r["inObs"].astype(bool)
fluid = ~inObs
Kn = float(r["Kn"])
Lx, Ly = float(r["Lx"]), float(r["Ly"])
xObs, yObs, LObs = float(r["xObs"]), float(r["yObs"]), float(r["LObs"])
dx = Lx / int(r["Nx"])

T, hx, hy = r["T_final"], r["hx_final"], r["hy_final"]
st, sT, shx, shy = r["snap_t"], r["snap_T"], r["snap_hx"], r["snap_hy"]


def outside(halo):
    """Fluid cells farther than `halo` from the obstacle's faces."""
    near = ((np.abs(Xc - xObs) <= LObs / 2 + halo)
            & (np.abs(Yc - yObs) <= LObs / 2 + halo))
    return fluid & ~near


def positive_from(halo):
    """First snapshot time after which min hx stays > 0 outside `halo`."""
    m = outside(halo)
    vals = [float(shx[k][m].min()) for k in range(st.size)]
    for k in range(st.size):
        if all(v > 0.0 for v in vals[k:]):
            return float(st[k]), vals
    return None, vals


def smooth_121(F):
    G = F.astype(float).copy()
    G[:, 1:-1] = 0.25 * F[:, :-2] + 0.5 * F[:, 1:-1] + 0.25 * F[:, 2:]
    H = G.copy()
    H[1:-1, :] = 0.25 * G[:-2, :] + 0.5 * G[1:-1, :] + 0.25 * G[2:, :]
    return H


# -- quasi-stationary state -------------------------------------------------
mag = np.hypot(hx, hy)
k2 = int(np.argmin(np.abs(st - 2.0)))
k5 = int(np.argmin(np.abs(st - 5.0)))
mag2_t2 = shx[k2] ** 2 + shy[k2] ** 2

jc = int(np.argmin(np.abs(r["yc"] - yObs)))
i7 = int(np.argmin(np.abs(xc - 7.0)))
T_filt = smooth_121(T)
TF = 1.0 - xc / Lx


def probe_devT(x0, y0):
    """T - T_F at (x0, y0), bilinearly interpolated on the cell centres.
    """
    yc = r["yc"]
    i = int(np.clip(np.searchsorted(xc, x0) - 1, 0, xc.size - 2))
    j = int(np.clip(np.searchsorted(yc, y0) - 1, 0, yc.size - 2))
    wx = (x0 - xc[i]) / (xc[i + 1] - xc[i])
    wy = (y0 - yc[j]) / (yc[j + 1] - yc[j])
    val = ((1 - wx) * (1 - wy) * T_filt[j, i] + wx * (1 - wy) * T_filt[j, i + 1]
           + (1 - wx) * wy * T_filt[j + 1, i] + wx * wy * T_filt[j + 1, i + 1])
    return float(val - (1.0 - x0 / Lx))

t_pos_010, vals_010 = positive_from(0.10)
t_pos_1cell, vals_1c = positive_from(dx)

summary = {
    "protocol": f"cold start (switch-on), t_end={st[-1]:g}, "
                f"{int(r['Nx'])}x{int(r['Ny'])}, Kn={Kn:g}",
    "max_h_final": float(mag.max()),
    "max_h_ignition_t2": round(float(np.sqrt(mag2_t2.max())), 3),
    "max_h_t5": round(float(np.hypot(shx[k5], shy[k5]).max()), 3),
    "minPhi_t2": round(float(1.0 - (Kn ** 2 / 6.0) * mag2_t2.max()), 3),
    "min_hx_fluid_final": float(hx[fluid].min()),
    "min_hx_halo015_final": float(hx[outside(0.15)].min()),
    "axis_hx_x7_final": float(hx[jc, i7]),
    "devT_probe_4_2_filtered": round(probe_devT(4.0, yObs), 4),
    "transient_min_hx_bound": round(float(min(shx[k][fluid].min()
                                              for k in range(st.size))), 5),
    # positivity threshold under the two near-wall conventions
    "hx_positive_outside_halo010_from_t": t_pos_010,
    "hx_positive_outside_one_cell_from_t": t_pos_1cell,
    "one_cell_halo_dx": dx,
    "note": ("'halo' excludes fluid cells within the given distance of the "
             "obstacle faces."),
}

(here / "baseline_coldstart_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print("\nper-snapshot min hx (fluid | outside 0.10 | outside one cell):")
for k, t in enumerate(st):
    m010, m1c = outside(0.10), outside(dx)
    print(f"  t={t:7.2f}  {shx[k][fluid].min():+.3e}  "
          f"{shx[k][m010].min():+.3e}  {shx[k][m1c].min():+.3e}")
