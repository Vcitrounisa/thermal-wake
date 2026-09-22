"""
run_verification.py
===================

Verification suite for the GK solver:

  (a) linear GK channel with no-slip walls vs the analytic fully-developed
      profile  hx(y) = G [1 - cosh((y-R/2)/Kn)/cosh(R/(2Kn))]  -- grid
      convergence of the error (expected second order);
  (b) non-linear GK channel vs an independent 1D collocation solution of the
      fully-developed ODE  Kn^2 Phi (Phi hx')' - hx + G Phi = 0;
  (c) self-convergence of the wake configuration under grid refinement;
  (d) robustness of the wake with respect to the domain length (outflow BC).

Writes verification_results.npz and prints a summary table.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_bvp

from gk_solver import solve, channel_noslip_profile

here = Path(__file__).resolve().parent
out = {}

# ---------------------------------------------------------------------------
# (a) linear channel vs analytic, grid convergence
# ---------------------------------------------------------------------------
R, Lch, KnC, Thot = 1.0, 2.0, 0.7, 1.0
G = Thot / Lch
grids = [8, 16, 32, 64]
errs = []
prof = {}
for Ny in grids:
    Nx = 2 * Ny
    r = solve(Kn=KnC, Lx=Lch, Ly=R, Nx=Nx, Ny=Ny, obstacle=False,
              tEnd=80.0, tSnap=(80.0,), Thot=Thot, sidewalls="noslip",
              nonlinear=False, steady_tol=1.0e-9, diag_every=200,
              verbose=False)
    j = Nx // 2
    hx_num = r["hx_final"][:, j]
    hx_ex = channel_noslip_profile(r["yc"], R, KnC, G)
    err = np.max(np.abs(hx_num - hx_ex)) / np.max(np.abs(hx_ex))
    errs.append(err)
    prof[Ny] = (r["yc"], hx_num, hx_ex)
    print(f"(a) linear channel Ny={Ny:4d}: rel. err = {err:.3e}")

orders = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
print(f"(a) observed orders: {[f'{o:.2f}' for o in orders]}")
out["a_grids"] = np.array(grids)
out["a_errs"] = np.array(errs)
Nyf = grids[-1]
out["a_y"], out["a_hx_num"], out["a_hx_ex"] = prof[Nyf]

# ---------------------------------------------------------------------------
# (b) non-linear channel vs independent 1D collocation
# ---------------------------------------------------------------------------
ThotNL = 20.0                     # strong drive: min Phi well below 1
GNL = ThotNL / Lch

def ode(y, Y):
    """Y = [hx, m] with m = Phi hx' ;  Kn^2 Phi m' - hx + G Phi = 0."""
    hx, m = Y
    Phi = 1.0 - (KnC ** 2 / 6.0) * hx ** 2
    dhx = m / Phi
    dm = (hx - GNL * Phi) / (KnC ** 2 * Phi)
    return np.vstack([dhx, dm])

def bc(Ya, Yb):
    return np.array([Ya[0], Yb[0]])

y_mesh = np.linspace(0.0, R, 801)
# continuation in the driving amplitude, warm-starting from the linear profile
sol_prev = np.vstack([channel_noslip_profile(y_mesh, R, KnC, 1.0),
                      np.gradient(channel_noslip_profile(y_mesh, R, KnC, 1.0), y_mesh)])
bvp = None
for G_step in [1.0, 3.0, 6.0, GNL]:
    def ode_step(y, Y, G_loc=G_step):
        hx, m = Y
        Phi = 1.0 - (KnC ** 2 / 6.0) * hx ** 2
        return np.vstack([m / Phi, (hx - G_loc * Phi) / (KnC ** 2 * Phi)])

    guess = sol_prev * (G_step / (1.0 if bvp is None else G_prev))
    bvp = solve_bvp(ode_step, bc, y_mesh, guess, tol=1.0e-9, max_nodes=400000)
    assert bvp.success, f"collocation BVP failed at G = {G_step}"
    sol_prev = bvp.sol(y_mesh)
    G_prev = G_step

r = solve(Kn=KnC, Lx=Lch, Ly=R, Nx=128, Ny=64, obstacle=False,
          tEnd=80.0, tSnap=(80.0,), Thot=ThotNL, sidewalls="noslip",
          nonlinear=True, steady_tol=1.0e-9, diag_every=200, verbose=False)
j = 64
hx_num = r["hx_final"][:, j]
hx_bvp = bvp.sol(r["yc"])[0]
err_nl = np.max(np.abs(hx_num - hx_bvp)) / np.max(np.abs(hx_bvp))
minPhi = 1.0 - (KnC ** 2 / 6.0) * np.max(hx_bvp) ** 2
print(f"(b) non-linear channel (min Phi = {minPhi:.3f}): rel. err vs collocation = {err_nl:.3e}")
out["b_y"], out["b_hx_num"], out["b_hx_bvp"] = r["yc"], hx_num, hx_bvp
out["b_err"], out["b_minPhi"] = err_nl, minPhi

# also compare against the linear profile to show the non-linear deviation
out["b_hx_lin"] = channel_noslip_profile(r["yc"], R, KnC, GNL)

# ---------------------------------------------------------------------------
# (c) wake self-convergence
# ---------------------------------------------------------------------------
def min_hx_halo(r, halo=0.15):
    """min hx over fluid cells at distance > halo from the obstacle.

    The raw fluid minimum sits in the first cell ring around the obstacle,
    where no-slip forces hx -> 0 at the wall, so it scales with the mesh
    size; at fixed distance from the body it converges to a positive value.
    """
    near = ((np.abs(r["Xc"] - r["xObs"]) <= r["LObs"] / 2 + halo)
            & (np.abs(r["Yc"] - r["yObs"]) <= r["LObs"] / 2 + halo))
    return float(np.min(r["hx_final"][~r["inObs"] & ~near]))

wake = {}
for (Nx, Ny) in [(100, 50), (200, 100), (400, 200)]:
    r = solve(Nx=Nx, Ny=Ny, tSnap=(18.0,), verbose=False)
    wake[(Nx, Ny)] = r
    print(f"(c) wake {Nx}x{Ny}: min hx (fluid) = {np.min(r['hx_final'][~r['inObs']]):.4e}"
          f", min hx (halo 0.15) = {min_hx_halo(r):.4e}")

# sample steady fields of all runs on the coarse-grid cell centres
def sample(r, X, Y):
    from scipy.interpolate import RegularGridInterpolator
    f_T = RegularGridInterpolator((r["yc"], r["xc"]), r["T_final"],
                                  bounds_error=False, fill_value=None)
    f_h = RegularGridInterpolator((r["yc"], r["xc"]), r["hx_final"],
                                  bounds_error=False, fill_value=None)
    pts = np.column_stack([Y.ravel(), X.ravel()])
    return f_T(pts).reshape(X.shape), f_h(pts).reshape(X.shape)

rc = wake[(100, 50)]
Xs, Ys = rc["Xc"], rc["Yc"]
mask = ~rc["inObs"]
# exclude a two-cell halo around the obstacle from the error norm
from scipy.ndimage import binary_dilation
mask &= ~binary_dilation(rc["inObs"], iterations=2)

T1, h1 = sample(wake[(100, 50)], Xs, Ys)
T2, h2 = sample(wake[(200, 100)], Xs, Ys)
T3, h3 = sample(wake[(400, 200)], Xs, Ys)
eT12 = np.max(np.abs(T1 - T2)[mask]); eT23 = np.max(np.abs(T2 - T3)[mask])
eh12 = np.max(np.abs(h1 - h2)[mask]); eh23 = np.max(np.abs(h2 - h3)[mask])
print(f"(c) |T_100-T_200| = {eT12:.3e}, |T_200-T_400| = {eT23:.3e}  (order ~ {np.log2(eT12/eT23):.2f})")
print(f"(c) |h_100-h_200| = {eh12:.3e}, |h_200-h_400| = {eh23:.3e}  (order ~ {np.log2(eh12/eh23):.2f})")
out["c_eT"] = np.array([eT12, eT23]); out["c_eh"] = np.array([eh12, eh23])
out["c_minhx"] = np.array([float(np.min(wake[g]["hx_final"][~wake[g]["inObs"]]))
                           for g in [(100, 50), (200, 100), (400, 200)]])
out["c_minhx_halo"] = np.array([min_hx_halo(wake[g])
                                for g in [(100, 50), (200, 100), (400, 200)]])

# ---------------------------------------------------------------------------
# (d) domain-length robustness (outflow condition placement)
# ---------------------------------------------------------------------------
r8 = wake[(200, 100)]
r12 = solve(Lx=12.0, Nx=300, tSnap=(18.0,), verbose=False)
# compare in the shared window x < 7 (temperature rescaled by the local
# Fourier reference to remove the trivial change of the driving gradient)
jc = r8["Ny"] // 2
x8, x12 = r8["xc"], r12["xc"]
TF8 = 1.0 - x8 / 8.0
TF12 = 1.0 - x12 / 12.0
dev8 = r8["T_final"][jc, :] - TF8          # non-Fourier deviation, L = 8
dev12 = r12["T_final"][jc, :] - TF12       # non-Fourier deviation, L = 12
out["d_x8"], out["d_dev8"] = x8, dev8
out["d_x12"], out["d_dev12"] = x12, dev12
out["d_minhx8"] = float(np.min(r8["hx_final"][~r8["inObs"]]))
out["d_minhx12"] = float(np.min(r12["hx_final"][~r12["inObs"]]))
out["d_minhx8_halo"] = min_hx_halo(r8)
out["d_minhx12_halo"] = min_hx_halo(r12)
print(f"(d) min hx: L=8 -> {out['d_minhx8']:.4e},  L=12 -> {out['d_minhx12']:.4e}")
print(f"(d) min hx (halo): L=8 -> {out['d_minhx8_halo']:.4e},  L=12 -> {out['d_minhx12_halo']:.4e}")

np.savez(here / "verification_results.npz", **out)
summary = dict(
    linear_orders=[float(o) for o in orders],
    linear_finest_err=float(errs[-1]),
    nonlinear_err=float(err_nl), nonlinear_minPhi=float(minPhi),
    wake_orders=dict(T=float(np.log2(eT12 / eT23)), h=float(np.log2(eh12 / eh23))),
    minhx_by_grid=[float(v) for v in out["c_minhx"]],
    minhx_halo_by_grid=[float(v) for v in out["c_minhx_halo"]],
    minhx_L8=out["d_minhx8"], minhx_L12=out["d_minhx12"],
    minhx_halo_L8=out["d_minhx8_halo"], minhx_halo_L12=out["d_minhx12_halo"],
    # max-norm self-convergence errors quoted in Sec. III A 1 of the paper
    wake_err_T=[float(eT12), float(eT23)], wake_err_h=[float(eh12), float(eh23)],
)
(here / "verification_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
