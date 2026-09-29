"""
Check that the compiled kernel and the numpy reference implementation of the
time loop agree on every option of the solver.
"""

import sys

import numpy as np

from gk_solver import make_grid, solve

try:
    import gk_kernels  # noqa: F401
except ImportError as exc:
    sys.exit(f"numba kernel not available ({exc}); install numba to run this test")

CASES = [
    dict(ic="fourier"),
    dict(ic="cold"),
    dict(ic="fourier-flux"),
    dict(ic="fourier", T_enclosed=1.0),
    dict(ic="fourier", nonlinear=False),
    dict(ic="fourier", full_q=True, eps=0.1),
    dict(ic="cold", full_q=True, eps=0.1, walls="noslip"),
    dict(ic="fourier", walls="noslip", obstacle=False, Lx=2.0, Ly=1.0, Nx=64, Ny=32),
    dict(ic="cold", Nx=100, Ny=50),
]

# restart from a random state
Nx, Ny = 200, 100
g = make_grid(8.0, 4.0, Nx, Ny)
X, _ = np.meshgrid(g["x_T"], g["y_T"])
rng = np.random.default_rng(0)
hy0 = 0.1 * rng.standard_normal((Ny + 1, Nx + 1))
hy0[0, :] = hy0[-1, :] = hy0[:, 0] = hy0[:, -1] = 0.0
CASES.append(dict(init=(1.0 - X / 8.0, 0.1 * rng.standard_normal((Ny, Nx)), hy0)))

FIELDS = ("T", "hx", "hy", "hist_max_h", "hist_min_hx")

failed = 0
for case in CASES:
    kw = dict(Nx=Nx, Ny=Ny, t_end=0.2, t_snap=(0.2,), hist_every=25)
    kw.update(case)
    a = solve(use_numba=False, **kw)
    b = solve(use_numba=True, **kw)
    names = FIELDS + (("Qxx", "Qyy", "Qyx", "Qxy") if kw.get("full_q") else ())
    err = max(np.max(np.abs(a[k] - b[k])) for k in names)
    ok = err < 1e-12
    failed += not ok
    label = {k: ("<array>" if k == "init" else v) for k, v in case.items()}
    print(f"{'ok  ' if ok else 'FAIL'} {err:.1e}  {label}")

print("all tests passed" if not failed else f"{failed} test(s) failed")
sys.exit(1 if failed else 0)
