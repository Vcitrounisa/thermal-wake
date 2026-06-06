"""
gk_thermal_wake.py
==================

Non-linear, non-local Guyer-Krumhansl (GK) heat-transport solver for a 2D
thin nanolayer with a square obstacle, driven by an imposed temperature
gradient. Explicit finite differences on a cell-centred grid with ghost cells.

Governing equations:

    dT/dt + (Kn^2 / 3) div h = 0
    Phi(h) dh/dt + h + Phi(h) grad T - Kn^2 Phi(h) div[ Phi(h) grad h ] = 0
    Phi(h) = max(0, 1 - (Kn^2 / 6) |h|^2)        # truncated non-linear modulation

with T the temperature, h the heat flux, Kn the Knudsen number.

Boundary conditions: hot/cold Dirichlet T on left/right, adiabatic free-slip
side walls, adiabatic no-slip square obstacle.

Run:  python gk_thermal_wake.py        # writes gk_wake_results.npz
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy.io import savemat


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
Kn      = 0.7                                       # Knudsen number (hydrodynamic regime, Kn ~ O(1))
Lx      = 8.0                                       # domain length (x)
Ly      = 4.0                                       # domain width  (y); aspect ratio Lx/Ly = 2
Nx      = 200                                       # grid cells along x
Ny      = 100                                       # grid cells along y

xObs    = 3.0                                       # obstacle centre, x
yObs    = Ly / 2.0                                  # obstacle centre, y (mid-channel)
LObs    = 1.0                                       # obstacle side length

tEnd    = 18.0                                      # final non-dimensional time
CFL     = 0.20                                      # CFL number
tSnap   = np.array([0.5, 2.0, 5.0, 10.0, 18.0])    # times at which snapshots are stored

ghostFloor = 1.0e-3                                 # floor on Phi to avoid division by zero


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
dx = Lx / Nx
dy = Ly / Ny
xc = (np.arange(Nx) + 0.5) * dx          # cell-centre coordinates, x
yc = (np.arange(Ny) + 0.5) * dy          # cell-centre coordinates, y
Xc, Yc = np.meshgrid(xc, yc)             # shape (Ny, Nx)

inObs = (np.abs(Xc - xObs) <= LObs / 2) & (np.abs(Yc - yObs) <= LObs / 2)  # obstacle mask


# ---------------------------------------------------------------------------
# Initial fields
# ---------------------------------------------------------------------------
T  = np.maximum(0.0, 1.0 - Xc / Lx)      # linear temperature gradient at t = 0
hx = np.zeros((Ny, Nx))                  # heat flux starts at rest
hy = np.zeros((Ny, Nx))
T[inObs]  = 0.0


# ---------------------------------------------------------------------------
# Time-step from CFL
# ---------------------------------------------------------------------------
cWave = Kn / np.sqrt(3)                          # second-sound (wave) speed
dt_w  = CFL * min(dx, dy) / cWave                # hyperbolic (wave) CFL limit
dt_d  = CFL * min(dx, dy) ** 2 / (2 * Kn ** 2)   # parabolic (diffusion) CFL limit
dt    = min(dt_w, dt_d)                           # take the more restrictive one
Nstep = int(np.ceil(tEnd / dt))
dt    = tEnd / Nstep                              # adjust dt to land exactly on tEnd

print(f"==== GK thermal-wake solver (python) ====")
print(f"  Kn = {Kn:.3f},  Lx = {Lx:.2f},  Ly = {Ly:.2f},"
      f"  Nx x Ny = {Nx} x {Ny}")
print(f"  dx = {dx:.4f},  dy = {dy:.4f},  dt = {dt:.4e},  Nstep = {Nstep}")
print(f"  Obstacle: square of side {LObs:.2f} at ({xObs:.2f}, {yObs:.2f})")


# ---------------------------------------------------------------------------
# Snapshot bookkeeping
# ---------------------------------------------------------------------------
snapStep = np.maximum(1, np.round(tSnap / dt).astype(int))
snap_t   = np.zeros_like(tSnap)
snap_T   = np.zeros((tSnap.size, Ny, Nx))
snap_hx  = np.zeros_like(snap_T)
snap_hy  = np.zeros_like(snap_T)


# ---------------------------------------------------------------------------
# Helpers: ghost-cell padding
# ---------------------------------------------------------------------------
def pad_T(T):
    """Dirichlet (hot=1 / cold=0) on x, Neumann (adiabatic) on y."""
    g = np.zeros((Ny + 2, Nx + 2))
    g[1:-1, 1:-1] = T
    g[0,   1:-1]  = T[0,   :]   # bottom: dT/dy = 0
    g[-1,  1:-1]  = T[-1,  :]   # top:    dT/dy = 0
    g[1:-1, 0]    = 1.0         # left:   T = 1
    g[1:-1, -1]   = 0.0         # right:  T = 0
    return g


def pad_hx(hx):
    """Zero-gradient (free-slip / free-flow) on all sides."""
    g = np.zeros((Ny + 2, Nx + 2))
    g[1:-1, 1:-1] = hx
    g[0,   1:-1]  = hx[0,   :]    # free slip on bottom
    g[-1,  1:-1]  = hx[-1,  :]    # free slip on top
    g[1:-1, 0]    = hx[:, 0]      # free flow on left
    g[1:-1, -1]   = hx[:, -1]     # free flow on right
    return g


def pad_hy(hy):
    """No-penetration on y (ghost = -interior), zero-gradient on x."""
    g = np.zeros((Ny + 2, Nx + 2))
    g[1:-1, 1:-1] = hy
    g[0,   1:-1]  = -hy[0,   :]   # no-penetration => hy ghost = -hy
    g[-1,  1:-1]  = -hy[-1,  :]
    g[1:-1, 0]    = hy[:, 0]
    g[1:-1, -1]   = hy[:, -1]
    return g


def pad_zero_grad(P):
    """Zero-gradient ghost layer on all sides (for a scalar field)."""
    g = np.zeros((Ny + 2, Nx + 2))
    g[1:-1, 1:-1] = P
    g[0,   1:-1]  = P[0,   :]
    g[-1,  1:-1]  = P[-1,  :]
    g[1:-1, 0]    = P[:, 0]
    g[1:-1, -1]   = P[:, -1]
    return g


# ---------------------------------------------------------------------------
# Main time loop
# ---------------------------------------------------------------------------
tic = time.time()
for n in range(1, Nstep + 1):          # explicit forward-Euler time stepping

    # 1. Adiabatic obstacle: copy T from non-obstacle neighbours -----------
    Tl = np.concatenate((T[:, :1],  T[:, :-1]), axis=1)
    Tr = np.concatenate((T[:, 1:], T[:, -1:]),  axis=1)
    Td = np.concatenate((T[:1, :],  T[:-1, :]), axis=0)
    Tu = np.concatenate((T[1:, :], T[-1:, :]),  axis=0)
    obs_l = np.concatenate((inObs[:, :1],  inObs[:, :-1]), axis=1)
    obs_r = np.concatenate((inObs[:, 1:], inObs[:, -1:]),  axis=1)
    obs_d = np.concatenate((inObs[:1, :],  inObs[:-1, :]), axis=0)
    obs_u = np.concatenate((inObs[1:, :], inObs[-1:, :]),  axis=0)
    nbSum = (Tl * (~obs_l) + Tr * (~obs_r) +
             Td * (~obs_d) + Tu * (~obs_u))
    nbCnt = ((~obs_l).astype(int) + (~obs_r).astype(int)
           + (~obs_d).astype(int) + (~obs_u).astype(int))
    T_avg = nbSum / np.maximum(nbCnt, 1)
    T[inObs] = T_avg[inObs]

    # 2. Pad ghost cells ---------------------------------------------------
    Tg  = pad_T(T)
    hxg = pad_hx(hx)
    hyg = pad_hy(hy)

    # 3. Central differences -----------------------------------------------
    div_h = ((hxg[1:-1, 2:] - hxg[1:-1, :-2]) / (2 * dx)
           + (hyg[2:,  1:-1] - hyg[:-2, 1:-1]) / (2 * dy))

    gradT_x = (Tg[1:-1, 2:] - Tg[1:-1, :-2]) / (2 * dx)
    gradT_y = (Tg[2:,  1:-1] - Tg[:-2, 1:-1]) / (2 * dy)

    # 4. Non-linear modulation Phi+ on cell centres, face-averaged for diffusion
    mag2 = hx ** 2 + hy ** 2
    Phi  = np.maximum(0.0, 1.0 - (Kn ** 2 / 6.0) * mag2)   # clipped at 0 (positivity)
    Phig = pad_zero_grad(Phi)
    Phi_e = 0.5 * (Phig[1:-1, 1:-1] + Phig[1:-1, 2:])
    Phi_w = 0.5 * (Phig[1:-1, 1:-1] + Phig[1:-1, :-2])
    Phi_n = 0.5 * (Phig[1:-1, 1:-1] + Phig[2:, 1:-1])
    Phi_s = 0.5 * (Phig[1:-1, 1:-1] + Phig[:-2, 1:-1])

    # 5. Variable-coefficient Laplacian -----------------------------------
    diff_hx = ((Phi_e * (hxg[1:-1, 2:]    - hxg[1:-1, 1:-1])
              - Phi_w * (hxg[1:-1, 1:-1] - hxg[1:-1, :-2])) / dx ** 2
             + (Phi_n * (hxg[2:,  1:-1] - hxg[1:-1, 1:-1])
              - Phi_s * (hxg[1:-1, 1:-1] - hxg[:-2, 1:-1])) / dy ** 2)

    diff_hy = ((Phi_e * (hyg[1:-1, 2:]    - hyg[1:-1, 1:-1])
              - Phi_w * (hyg[1:-1, 1:-1] - hyg[1:-1, :-2])) / dx ** 2
             + (Phi_n * (hyg[2:,  1:-1] - hyg[1:-1, 1:-1])
              - Phi_s * (hyg[1:-1, 1:-1] - hyg[:-2, 1:-1])) / dy ** 2)

    # 6. RHS ---------------------------------------------------------------
    Phi_safe = np.maximum(Phi, ghostFloor)
    dT  = -(Kn ** 2 / 3.0) * div_h
    dhx = -hx / Phi_safe - gradT_x + Kn ** 2 * diff_hx
    dhy = -hy / Phi_safe - gradT_y + Kn ** 2 * diff_hy

    # 7. Update outside the obstacle --------------------------------------
    upd = ~inObs
    T[upd]  += dt * dT[upd]
    hx[upd] += dt * dhx[upd]
    hy[upd] += dt * dhy[upd]

    # 8. Hard-zero h inside the obstacle ----------------------------------
    hx[inObs] = 0.0
    hy[inObs] = 0.0

    # 9. Snapshots ---------------------------------------------------------
    if n in snapStep:
        k = int(np.where(snapStep == n)[0][0])
        snap_t[k]    = n * dt
        snap_T[k]    = T.copy()
        snap_hx[k]   = hx.copy()
        snap_hy[k]   = hy.copy()

    if n % max(1, Nstep // 20) == 0:
        print(f"  step {n}/{Nstep}  (t = {n*dt:.2f})  max|h| = {np.max(np.abs(hx)):.3f}")

elapsed = time.time() - tic
print(f"Solver finished in {elapsed:.1f} s.")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir  = Path(__file__).resolve().parent
out_file = out_dir / "gk_wake_results.npz"
np.savez(                                 # snapshots, final fields, grid and parameters
    out_file,
    Kn=Kn, Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
    xObs=xObs, yObs=yObs, LObs=LObs,
    Xc=Xc, Yc=Yc, xc=xc, yc=yc,
    inObs=inObs,
    T_final=T, hx_final=hx, hy_final=hy,
    snap_t=snap_t, snap_T=snap_T, snap_hx=snap_hx, snap_hy=snap_hy,
    tSnap=tSnap, dt=dt, Nstep=Nstep,
)
print(f"Results written to {out_file}")


