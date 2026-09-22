"""
gk_kernels.py
=============

Numba-accelerated time loops for gk_solver.py.  The kernels reproduce the
numpy reference implementation operation-by-operation (same stencils, same
update order); gk_solver.solve() falls back to the numpy path when numba is
not available, and the two paths agree to machine precision (see
run_selftest.py).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

# sidewall modes
FREESLIP, NOSLIP, SLIP = 0, 1, 2
# obstacle fill modes
FILL_AVG, FILL_FROZEN, FILL_COMPAT = 0, 1, 2


@njit(cache=True)
def _pads(T, hx, hy, Tg, hxg, hyg, Thot, sidewall, a1, a2):
    Ny, Nx = T.shape
    for j in range(Ny):
        for i in range(Nx):
            Tg[j + 1, i + 1] = T[j, i]
            hxg[j + 1, i + 1] = hx[j, i]
            hyg[j + 1, i + 1] = hy[j, i]
    for i in range(Nx):
        Tg[0, i + 1] = T[0, i]
        Tg[Ny + 1, i + 1] = T[Ny - 1, i]
        if sidewall == FREESLIP:
            hxg[0, i + 1] = hx[0, i]
            hxg[Ny + 1, i + 1] = hx[Ny - 1, i]
        elif sidewall == NOSLIP:
            hxg[0, i + 1] = -hx[0, i]
            hxg[Ny + 1, i + 1] = -hx[Ny - 1, i]
        else:
            hxg[0, i + 1] = a1 * hx[0, i] + a2 * hx[1, i]
            hxg[Ny + 1, i + 1] = a1 * hx[Ny - 1, i] + a2 * hx[Ny - 2, i]
        hyg[0, i + 1] = -hy[0, i]
        hyg[Ny + 1, i + 1] = -hy[Ny - 1, i]
    for j in range(Ny):
        Tg[j + 1, 0] = 2.0 * Thot - T[j, 0]
        Tg[j + 1, Nx + 1] = -T[j, Nx - 1]
        hxg[j + 1, 0] = hx[j, 0]
        hxg[j + 1, Nx + 1] = hx[j, Nx - 1]
        hyg[j + 1, 0] = hy[j, 0]
        hyg[j + 1, Nx + 1] = hy[j, Nx - 1]


@njit(cache=True)
def _pad_scalar(P, Pg):
    Ny, Nx = P.shape
    for j in range(Ny):
        for i in range(Nx):
            Pg[j + 1, i + 1] = P[j, i]
    for i in range(Nx):
        Pg[0, i + 1] = P[0, i]
        Pg[Ny + 1, i + 1] = P[Ny - 1, i]
    for j in range(Ny):
        Pg[j + 1, 0] = P[j, 0]
        Pg[j + 1, Nx + 1] = P[j, Nx - 1]


@njit(cache=True)
def _obstacle_fill(T, inObs, obs_j, obs_i, fill, Dx, Dy, dx, dy, Kn2):
    Ny, Nx = T.shape
    for k in range(obs_j.size):
        j, i = obs_j[k], obs_i[k]
        s = 0.0
        c = 0
        # left neighbour (fluid -> obstacle normal = +x)
        if i > 0 and not inObs[j, i - 1]:
            v = T[j, i - 1]
            if fill == FILL_COMPAT:
                v += dx * Kn2 * Dx[j, i - 1]
            s += v; c += 1
        if i < Nx - 1 and not inObs[j, i + 1]:
            v = T[j, i + 1]
            if fill == FILL_COMPAT:
                v -= dx * Kn2 * Dx[j, i + 1]
            s += v; c += 1
        if j > 0 and not inObs[j - 1, i]:
            v = T[j - 1, i]
            if fill == FILL_COMPAT:
                v += dy * Kn2 * Dy[j - 1, i]
            s += v; c += 1
        if j < Ny - 1 and not inObs[j + 1, i]:
            v = T[j + 1, i]
            if fill == FILL_COMPAT:
                v -= dy * Kn2 * Dy[j + 1, i]
            s += v; c += 1
        if c > 0:
            T[j, i] = s / c


@njit(cache=True)
def run_loop(T, hx, hy, inObs, obs_j, obs_i,
             Kn, dx, dy, dt, Nstep, Thot, Thot_ramp,
             sidewall, a1, a2, fill, nonlinear,
             fullQ, eps_Q, Qxx, Qxy, Qyx, Qyy,
             ghostFloor, steady_tol,
             snap_steps, snap_t, snap_T, snap_hx, snap_hy,
             diag_every, diag_t, diag_min_hx, diag_max_h, diag_res):
    """Advance the GK system Nstep steps.  Returns (n_stop, n_diag)."""
    Ny, Nx = T.shape
    Kn2 = Kn * Kn
    Tg = np.zeros((Ny + 2, Nx + 2))
    hxg = np.zeros((Ny + 2, Nx + 2))
    hyg = np.zeros((Ny + 2, Nx + 2))
    Phi = np.ones((Ny, Nx))
    Phig = np.zeros((Ny + 2, Nx + 2))
    dhx = np.zeros((Ny, Nx))
    dhy = np.zeros((Ny, Nx))
    dTt = np.zeros((Ny, Nx))
    Dx = np.zeros((Ny, Nx))
    Dy = np.zeros((Ny, Nx))
    Qxxg = np.zeros((Ny + 2, Nx + 2))
    Qxyg = np.zeros((Ny + 2, Nx + 2))
    Qyxg = np.zeros((Ny + 2, Nx + 2))
    Qyyg = np.zeros((Ny + 2, Nx + 2))
    n_diag = 0
    n_stop = Nstep

    for n in range(1, Nstep + 1):
        # 1. obstacle temperature fill
        if obs_j.size > 0 and fill != FILL_FROZEN:
            use_compat = (fill == FILL_COMPAT) and (n > 1)
            _obstacle_fill(T, inObs, obs_j, obs_i,
                           FILL_COMPAT if use_compat else FILL_AVG,
                           Dx, Dy, dx, dy, Kn2)

        # 2. ghost cells (hot wall optionally raised smoothly over Thot_ramp)
        if Thot_ramp > 0.0:
            Thot_n = Thot * (1.0 - math.exp(-(n * dt) / Thot_ramp))
        else:
            Thot_n = Thot
        _pads(T, hx, hy, Tg, hxg, hyg, Thot_n, sidewall, a1, a2)

        # 3. Phi on cell centres
        if nonlinear:
            for j in range(Ny):
                for i in range(Nx):
                    m2 = hx[j, i] ** 2 + hy[j, i] ** 2
                    p = 1.0 - (Kn2 / 6.0) * m2
                    Phi[j, i] = p if p > 0.0 else 0.0
        _pad_scalar(Phi, Phig)

        # 4. RHS
        if not fullQ:
            for j in range(Ny):
                for i in range(Nx):
                    jp, ip = j + 1, i + 1
                    div_h = ((hxg[jp, ip + 1] - hxg[jp, ip - 1]) / (2 * dx)
                             + (hyg[jp + 1, ip] - hyg[jp - 1, ip]) / (2 * dy))
                    gTx = (Tg[jp, ip + 1] - Tg[jp, ip - 1]) / (2 * dx)
                    gTy = (Tg[jp + 1, ip] - Tg[jp - 1, ip]) / (2 * dy)
                    p0 = Phig[jp, ip]
                    pe = 0.5 * (p0 + Phig[jp, ip + 1])
                    pw = 0.5 * (p0 + Phig[jp, ip - 1])
                    pn = 0.5 * (p0 + Phig[jp + 1, ip])
                    ps = 0.5 * (p0 + Phig[jp - 1, ip])
                    dfx = ((pe * (hxg[jp, ip + 1] - hxg[jp, ip])
                            - pw * (hxg[jp, ip] - hxg[jp, ip - 1])) / dx ** 2
                           + (pn * (hxg[jp + 1, ip] - hxg[jp, ip])
                              - ps * (hxg[jp, ip] - hxg[jp - 1, ip])) / dy ** 2)
                    dfy = ((pe * (hyg[jp, ip + 1] - hyg[jp, ip])
                            - pw * (hyg[jp, ip] - hyg[jp, ip - 1])) / dx ** 2
                           + (pn * (hyg[jp + 1, ip] - hyg[jp, ip])
                              - ps * (hyg[jp, ip] - hyg[jp - 1, ip])) / dy ** 2)
                    Dx[j, i] = dfx
                    Dy[j, i] = dfy
                    ps_safe = Phi[j, i] if Phi[j, i] > ghostFloor else ghostFloor
                    dTt[j, i] = -(Kn2 / 3.0) * div_h
                    dhx[j, i] = -hx[j, i] / ps_safe - gTx + Kn2 * dfx
                    dhy[j, i] = -hy[j, i] / ps_safe - gTy + Kn2 * dfy
        else:
            # 4'. evolve the face-centred Q with exact exponential relaxation.
            # Q lives on cell faces so that div Q reduces, in the limit
            # eps -> 0, to exactly the conservative variable-coefficient
            # Laplacian of the reduced scheme.
            #   Qxx[j,i], Qxy[j,i]: x-face between ghost cols i and i+1, i=0..Nx
            #   Qyx[j,i], Qyy[j,i]: y-face between ghost rows j and j+1, j=0..Ny
            for j in range(Ny):
                for i in range(Nx + 1):
                    jp = j + 1
                    pf = 0.5 * (Phig[jp, i] + Phig[jp, i + 1])
                    pf_safe = pf if pf > ghostFloor else ghostFloor
                    fac = math.exp(-dt / (eps_Q * pf_safe))
                    Sxx = -Kn2 * pf * (hxg[jp, i + 1] - hxg[jp, i]) / dx
                    Sxy = -Kn2 * pf * (hyg[jp, i + 1] - hyg[jp, i]) / dx
                    Qxx[j, i] = Sxx + (Qxx[j, i] - Sxx) * fac
                    Qxy[j, i] = Sxy + (Qxy[j, i] - Sxy) * fac
            for j in range(Ny + 1):
                for i in range(Nx):
                    ip = i + 1
                    pf = 0.5 * (Phig[j, ip] + Phig[j + 1, ip])
                    pf_safe = pf if pf > ghostFloor else ghostFloor
                    fac = math.exp(-dt / (eps_Q * pf_safe))
                    Syx = -Kn2 * pf * (hxg[j + 1, ip] - hxg[j, ip]) / dy
                    Syy = -Kn2 * pf * (hyg[j + 1, ip] - hyg[j, ip]) / dy
                    Qyx[j, i] = Syx + (Qyx[j, i] - Syx) * fac
                    Qyy[j, i] = Syy + (Qyy[j, i] - Syy) * fac
            for j in range(Ny):
                for i in range(Nx):
                    jp, ip = j + 1, i + 1
                    div_h = ((hxg[jp, ip + 1] - hxg[jp, ip - 1]) / (2 * dx)
                             + (hyg[jp + 1, ip] - hyg[jp - 1, ip]) / (2 * dy))
                    gTx = (Tg[jp, ip + 1] - Tg[jp, ip - 1]) / (2 * dx)
                    gTy = (Tg[jp + 1, ip] - Tg[jp - 1, ip]) / (2 * dy)
                    divQ_x = ((Qxx[j, i + 1] - Qxx[j, i]) / dx
                              + (Qyx[j + 1, i] - Qyx[j, i]) / dy)
                    divQ_y = ((Qxy[j, i + 1] - Qxy[j, i]) / dx
                              + (Qyy[j + 1, i] - Qyy[j, i]) / dy)
                    ps_safe = Phi[j, i] if Phi[j, i] > ghostFloor else ghostFloor
                    dTt[j, i] = -(Kn2 / 3.0) * div_h
                    dhx[j, i] = -hx[j, i] / ps_safe - gTx - divQ_x
                    dhy[j, i] = -hy[j, i] / ps_safe - gTy - divQ_y
                    Dx[j, i] = -divQ_x / Kn2
                    Dy[j, i] = -divQ_y / Kn2

        # 5. update outside the obstacle, hard-zero h inside
        for j in range(Ny):
            for i in range(Nx):
                if not inObs[j, i]:
                    T[j, i] += dt * dTt[j, i]
                    hx[j, i] += dt * dhx[j, i]
                    hy[j, i] += dt * dhy[j, i]
                else:
                    hx[j, i] = 0.0
                    hy[j, i] = 0.0

        # 6. snapshots
        for k in range(snap_steps.size):
            if snap_steps[k] == n:
                snap_t[k] = n * dt
                for j in range(Ny):
                    for i in range(Nx):
                        snap_T[k, j, i] = T[j, i]
                        snap_hx[k, j, i] = hx[j, i]
                        snap_hy[k, j, i] = hy[j, i]

        # 7. diagnostics / early exit
        if n % diag_every == 0 or n == Nstep:
            mn = 1.0e300
            mx = 0.0
            res = 0.0
            for j in range(Ny):
                for i in range(Nx):
                    if not inObs[j, i]:
                        if hx[j, i] < mn:
                            mn = hx[j, i]
                        m2 = hx[j, i] ** 2 + hy[j, i] ** 2
                        if m2 > mx:
                            mx = m2
                        r1 = abs(dhx[j, i])
                        if r1 > res:
                            res = r1
                        r2 = abs(dhy[j, i])
                        if r2 > res:
                            res = r2
            diag_t[n_diag] = n * dt
            diag_min_hx[n_diag] = mn
            diag_max_h[n_diag] = math.sqrt(mx)
            diag_res[n_diag] = res
            n_diag += 1
            if steady_tol > 0.0 and res < steady_tol:
                n_stop = n
                break

    return n_stop, n_diag
