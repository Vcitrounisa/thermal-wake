"""
run_wake_probe.py
=================

Grid- and domain-convergence of smooth pointwise wake diagnostics:

  hx at the centreline probes (4.0, R/2) and (4.5, R/2)  [thermal shadow]
  T - T_F  at the same points

for the wake on 100x50, 200x100, 400x200 grids and for the L=12 domain
(rescaled by the driving-gradient ratio), at the matched time t = 18 with
the Fourier-profile initialization.  Saves the full t = 18 fields of each
run for any future diagnostic.

Writes wake_probe_results.npz + wake_probe_summary.json.
"""

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from gk_solver import solve

here = Path(__file__).resolve().parent

probes = [(4.0, 2.0), (4.5, 2.0)]
out, summary = {}, {"probes": probes, "hx": {}, "devT": {}}

for tag, kw in [("100x50", dict(Nx=100, Ny=50)),
                ("200x100", dict(Nx=200, Ny=100)),
                ("400x200", dict(Nx=400, Ny=200)),
                ("L12", dict(Lx=12.0, Nx=300))]:
    print(f"--- wake {tag}")
    r = solve(ic="fourier-rest", tEnd=18.0, tSnap=(18.0,), verbose=False, **kw)
    G = 1.0 / r["Lx"]
    f_h = RegularGridInterpolator((r["yc"], r["xc"]), r["hx_final"])
    f_T = RegularGridInterpolator((r["yc"], r["xc"]), r["T_final"])
    hx_p = [float(f_h([y, x])[0]) for (x, y) in probes]
    dev_p = [float(f_T([y, x])[0] - (1 - x / r["Lx"])) for (x, y) in probes]
    # per-unit-driving-gradient values (G = T*/L changes with L)
    summary["hx"][tag] = [v / (G * 8.0) for v in hx_p]      # normalised to L=8 drive
    summary["devT"][tag] = [v / (G * 8.0) for v in dev_p]
    out[f"hx_{tag}"] = r["hx_final"]
    out[f"T_{tag}"] = r["T_final"]
    out[f"xc_{tag}"] = r["xc"]
    out[f"yc_{tag}"] = r["yc"]
    print(f"    hx(4,2) = {hx_p[0]:.5e}, hx(4.5,2) = {hx_p[1]:.5e}"
          f"  (per-unit-drive: {summary['hx'][tag][0]:.5e}, {summary['hx'][tag][1]:.5e})")

np.savez(here / "wake_probe_results.npz", **out)
(here / "wake_probe_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
