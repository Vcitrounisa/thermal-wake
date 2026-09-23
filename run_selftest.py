"""
run_selftest.py
===============

Cross-check of the two implementations of the time loop
(numpy reference vs numba kernel): short runs of every model variant must
agree to machine precision.
"""

import numpy as np

from gk_solver import solve

# The cross-check is only meaningful if the compiled kernel is actually
# importable: without numba, gk_solver.solve silently falls back to the numpy
# reference path and the test would compare numpy against numpy.
try:
    import gk_kernels
except Exception as exc:
    raise SystemExit(
        f"run_selftest.py needs the numba kernel to be importable, but "
        f"`import gk_kernels` failed with: {exc!r}\n"
        f"Install numba (`pip install numba`) and re-run; without it the "
        f"solver still works on the numpy path, but this cross-check is "
        f"not meaningful.")

CASES = [
    dict(),                                             
    dict(ic="cold-rest"),                                       # baseline wake
    dict(ic="fourier-flux"),
    dict(obstacle_fill="frozen"),
    dict(obstacle_fill="compat"),
    dict(sidewalls="noslip", obstacle=False, Lx=2.0, Ly=1.0, Nx=64, Ny=32),
    dict(sidewalls="slip", slip_C=2.0, slip_alpha=0.25,
         obstacle=False, Lx=2.0, Ly=1.0, Nx=64, Ny=32),
    dict(nonlinear=False),
    dict(full_Q=True, eps_Q=0.1),
    dict(ic="cold-rest", Thot_ramp=1.0),
]

# the `init=` restart path (used by run_longrun.py) is not reachable through a
# keyword in CASES, so it gets its own entry, built lazily below
_Ny, _Nx, _Ly, _Lx = 100, 200, 4.0, 8.0
_Xc, _Yc = np.meshgrid((np.arange(_Nx) + 0.5) * (_Lx / _Nx),
                       (np.arange(_Ny) + 0.5) * (_Ly / _Ny))
_rng = np.random.default_rng(0)
CASES.append(dict(init=(np.maximum(0.0, 1.0 - _Xc / _Lx),
                        0.1 * _rng.standard_normal((_Ny, _Nx)),
                        0.1 * _rng.standard_normal((_Ny, _Nx)))))

ok = True
for kw in CASES:
    kw = dict({"ic": "fourier-rest", **kw}, tEnd=0.5, tSnap=(0.5,), verbose=False,
              diag_every=25)
    a = solve(use_numba=False, **kw)
    b = solve(use_numba=True, **kw)
    dT = np.max(np.abs(a["T_final"] - b["T_final"]))
    dh = np.max(np.abs(a["hx_final"] - b["hx_final"]))
    dhy = np.max(np.abs(a["hy_final"] - b["hy_final"]))
    err = max(dT, dh, dhy)
    if "Qxx" in a:
        err = max(err, np.max(np.abs(a["Qxx"] - b["Qxx"])),
                  np.max(np.abs(a["Qyy"] - b["Qyy"])))
    status = "OK " if err < 1e-12 else "FAIL"
    if err >= 1e-12:
        ok = False
    print(f"{status} err = {err:.2e}  {kw}")

print("ALL OK" if ok else "SELF-TEST FAILED")
raise SystemExit(0 if ok else 1)
