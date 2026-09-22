"""
gk_solver.py
============

Library version of the non-linear, non-local Guyer-Krumhansl (GK) solver for
a 2D thin nanolayer, driven by an imposed temperature difference between the
two transversal walls.  Explicit finite differences on a cell-centred grid
with ghost cells.

Reduced (closed) model
----------------------
    dT/dt + (Kn^2 / 3) div h = 0
    Phi(h) dh/dt + h + Phi(h) grad T - Kn^2 Phi(h) div[ Phi(h) grad h ] = 0
    Phi(h) = max(0, 1 - (Kn^2 / 6) |h|^2)

Full three-field model (option ``full_Q=True``)
-----------------------------------------------
    dT/dt + (Kn^2 / 3) div h = 0
    Phi dh/dt + h + Phi grad T + Phi div Q = 0
    eps Phi dQ/dt + Q + Kn^2 Phi grad h = 0 ,   eps = tau_Q / tau_R

with Q the (rank-two) flux of the heat flux, stored componentwise as
Q_ji ~ -Kn^2 Phi d_j h_i so that (div Q)_i = d_j Q_ji.  The stiff local
relaxation of Q is integrated with an exact exponential (integrating-factor)
update, so the time step is not limited by eps.

The default options reproduce the simulation reported in the paper.
"""

from __future__ import annotations

import time

import numpy as np


# ---------------------------------------------------------------------------
# Ghost-cell padding helpers
# ---------------------------------------------------------------------------
def _pad_T(T, Thot, Ny, Nx):
    """Dirichlet (hot/cold) on x, even reflection (symmetry) on y.

    The Dirichlet values are imposed at the wall faces to second order by
    linear extrapolation into the ghost cell: ghost = 2*Twall - interior.
    """
    g = np.zeros((Ny + 2, Nx + 2))
    g[1:-1, 1:-1] = T
    g[0,   1:-1]  = T[0,   :]            # bottom: even reflection
    g[-1,  1:-1]  = T[-1,  :]            # top:    even reflection
    g[1:-1, 0]    = 2.0 * Thot - T[:, 0]  # left:  T = Thot at the face
    g[1:-1, -1]   = -T[:, -1]             # right: T = 0    at the face
    return g


def _pad_h(hx, hy, sidewalls, Ny, Nx, slip_g_coeffs=None):
    """Ghost layers for both heat-flux components.

    x = 0, L : zero streamwise gradient (fully-developed) for both components.
    y = 0, R : depends on ``sidewalls``:
        'freeslip' : hx even (d hx/dy = 0), hy odd (hy = 0 at the face)
        'noslip'   : hx odd, hy odd (both components vanish at the face)
        'slip'     : second-order slip condition on hx (see _slip_ghost),
                     hy odd
    """
    gx = np.zeros((Ny + 2, Nx + 2))
    gy = np.zeros((Ny + 2, Nx + 2))
    gx[1:-1, 1:-1] = hx
    gy[1:-1, 1:-1] = hy
    # left / right: zero-gradient
    gx[1:-1, 0] = hx[:, 0]
    gx[1:-1, -1] = hx[:, -1]
    gy[1:-1, 0] = hy[:, 0]
    gy[1:-1, -1] = hy[:, -1]
    # bottom / top walls
    if sidewalls == "freeslip":
        gx[0, 1:-1] = hx[0, :]
        gx[-1, 1:-1] = hx[-1, :]
    elif sidewalls == "noslip":
        gx[0, 1:-1] = -hx[0, :]
        gx[-1, 1:-1] = -hx[-1, :]
    elif sidewalls == "slip":
        a1, a2 = slip_g_coeffs
        gx[0, 1:-1] = a1 * hx[0, :] + a2 * hx[1, :]
        gx[-1, 1:-1] = a1 * hx[-1, :] + a2 * hx[-2, :]
    else:
        raise ValueError(f"unknown sidewalls option: {sidewalls!r}")
    gy[0, 1:-1] = -hy[0, :]
    gy[-1, 1:-1] = -hy[-1, :]
    return gx, gy


def _pad_zero_grad(P, Ny, Nx):
    g = np.zeros((Ny + 2, Nx + 2))
    g[1:-1, 1:-1] = P
    g[0, 1:-1] = P[0, :]
    g[-1, 1:-1] = P[-1, :]
    g[1:-1, 0] = P[:, 0]
    g[1:-1, -1] = P[:, -1]
    return g


def slip_ghost_coeffs(C, alpha, Kn, dy):
    """Ghost coefficients for the second-order slip condition on h_t.

    Wall condition (outward normal n):
        h_t + C Kn dh_t/dn + alpha Kn^2 d2h_t/dn2 = 0
    discretised at the wall face with the ghost value g and the first two
    interior values u1, u2:
        (g + u1)/2 + C Kn (g - u1)/dy + alpha Kn^2 (g - 2 u1 + u2)/dy^2 = 0
    solved for g = a1 * u1 + a2 * u2.
    """
    A = 0.5 + C * Kn / dy + alpha * Kn ** 2 / dy ** 2
    a1 = -(0.5 - C * Kn / dy - 2.0 * alpha * Kn ** 2 / dy ** 2) / A
    a2 = -(alpha * Kn ** 2 / dy ** 2) / A
    return a1, a2


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------
def solve(
    Kn=0.7, Lx=8.0, Ly=4.0, Nx=200, Ny=100,
    obstacle=True, xObs=3.0, yObs=None, LObs=1.0,
    tEnd=18.0, CFL=0.20, tSnap=(0.5, 2.0, 5.0, 10.0, 18.0),
    Thot=1.0, Thot_ramp=0.0,     # >0: hot wall raised as Thot*(1-exp(-t/Thot_ramp))
    ic="fourier-rest",           # "fourier-rest" | "cold-rest" | "fourier-flux"
    init=None,                   # (T0, hx0, hy0) arrays: restart state, overrides `ic`
    obstacle_fill="avg",         # "avg" | "frozen" | "compat"
    sidewalls="freeslip",        # "freeslip" | "noslip" | "slip"
    slip_C=0.0, slip_alpha=0.0,  # coefficients for sidewalls="slip"
    nonlinear=True,
    full_Q=False, eps_Q=0.1,
    ghostFloor=1.0e-3,
    steady_tol=None,             # early exit when max|dh/dt| < steady_tol
    diag_every=50,
    verbose=True,
    use_numba=True,              # use the numba kernel when available
):
    """Integrate the GK model; return a dict with fields and diagnostics."""

    log = print if verbose else (lambda *a, **k: None)
    if yObs is None:
        yObs = Ly / 2.0
    tSnap = np.atleast_1d(np.asarray(tSnap, dtype=float))

    # -- grid ---------------------------------------------------------------
    dx = Lx / Nx
    dy = Ly / Ny
    xc = (np.arange(Nx) + 0.5) * dx
    yc = (np.arange(Ny) + 0.5) * dy
    Xc, Yc = np.meshgrid(xc, yc)

    if obstacle:
        inObs = (np.abs(Xc - xObs) <= LObs / 2) & (np.abs(Yc - yObs) <= LObs / 2)
    else:
        inObs = np.zeros((Ny, Nx), dtype=bool)
    fluid = ~inObs

    # -- initial fields -----------------------------------------------------
    # `init` restarts from a given state (long-run continuation and
    # fine-mesh relaxation studies); it overrides `ic`.
    if init is not None:
        T = np.array(init[0], dtype=float, copy=True)
        hx = np.array(init[1], dtype=float, copy=True)
        hy = np.array(init[2], dtype=float, copy=True)
        if T.shape != (Ny, Nx) or hx.shape != (Ny, Nx) or hy.shape != (Ny, Nx):
            raise ValueError(f"init fields must have shape {(Ny, Nx)}")
    elif ic == "fourier-rest":
        T = np.maximum(0.0, Thot * (1.0 - Xc / Lx))
        hx = np.zeros((Ny, Nx))
        hy = np.zeros((Ny, Nx))
    elif ic == "cold-rest":
        T = np.zeros((Ny, Nx))
        hx = np.zeros((Ny, Nx))
        hy = np.zeros((Ny, Nx))
    elif ic == "fourier-flux":
        T = np.maximum(0.0, Thot * (1.0 - Xc / Lx))
        hx = np.full((Ny, Nx), Thot / Lx)
        hy = np.zeros((Ny, Nx))
    else:
        raise ValueError(f"unknown ic option: {ic!r}")
    T[inObs] = 0.0
    hx[inObs] = 0.0
    hy[inObs] = 0.0

    # -- time step from CFL -------------------------------------------------
    cWave = Kn / np.sqrt(3.0)                        # second-sound speed
    dt_w = CFL * min(dx, dy) / cWave                 # hyperbolic limit
    dt_d = CFL * min(dx, dy) ** 2 / (2.0 * Kn ** 2)  # diffusive limit
    dt = min(dt_w, dt_d)
    if full_Q:                                       # ballistic h-Q wave, speed Kn/sqrt(eps)
        dt = min(dt, CFL * min(dx, dy) * np.sqrt(eps_Q) / Kn)
    Nstep = int(np.ceil(tEnd / dt))
    dt = tEnd / Nstep

    slip_g = slip_ghost_coeffs(slip_C, slip_alpha, Kn, dy) if sidewalls == "slip" else None

    log("==== GK thermal-wake solver ====")
    log(f"  Kn = {Kn:.3f},  Lx = {Lx:.2f},  Ly = {Ly:.2f},  Nx x Ny = {Nx} x {Ny}")
    log(f"  dx = {dx:.4f},  dy = {dy:.4f},  dt = {dt:.4e},  Nstep = {Nstep}")
    log(f"  ic = {'restart (init)' if init is not None else ic},"
        f" sidewalls = {sidewalls}, obstacle_fill = {obstacle_fill},"
        f" nonlinear = {nonlinear}, full_Q = {full_Q}"
        + (f" (eps = {eps_Q})" if full_Q else ""))
    if obstacle:
        log(f"  Obstacle: square of side {LObs:.2f} at ({xObs:.2f}, {yObs:.2f})")

    # -- snapshot bookkeeping ----------------------------------------------
    snapStep = np.maximum(1, np.round(tSnap / dt).astype(int))
    snap_t = np.zeros_like(tSnap)
    snap_T = np.zeros((tSnap.size, Ny, Nx))
    snap_hx = np.zeros_like(snap_T)
    snap_hy = np.zeros_like(snap_T)

    # -- Q fields (full model only), stored on cell faces so that div Q
    #    reduces, for eps -> 0, to exactly the conservative operator of the
    #    reduced scheme ------------------------------------------------------
    if full_Q:
        Qxx = np.zeros((Ny, Nx + 1))   # x-faces, target: -Kn^2 Phi dx hx
        Qxy = np.zeros((Ny, Nx + 1))   # x-faces, target: -Kn^2 Phi dx hy
        Qyx = np.zeros((Ny + 1, Nx))   # y-faces, target: -Kn^2 Phi dy hx
        Qyy = np.zeros((Ny + 1, Nx))   # y-faces, target: -Kn^2 Phi dy hy

    # -- diagnostics --------------------------------------------------------
    diag_t, diag_min_hx, diag_max_h, diag_res = [], [], [], []

    tic = time.time()
    n_stop = Nstep

    # -- fast path: numba kernel -------------------------------------------
    K = None
    if use_numba:
        try:
            import gk_kernels as K
        except Exception:
            K = None
    if K is not None:
        sw = {"freeslip": K.FREESLIP, "noslip": K.NOSLIP, "slip": K.SLIP}[sidewalls]
        fm = {"avg": K.FILL_AVG, "frozen": K.FILL_FROZEN,
              "compat": K.FILL_COMPAT}[obstacle_fill]
        a1, a2 = slip_g if slip_g is not None else (0.0, 0.0)
        obs_j, obs_i = np.where(inObs)
        Qxx = np.zeros((Ny, Nx + 1)); Qxy = np.zeros((Ny, Nx + 1))
        Qyx = np.zeros((Ny + 1, Nx)); Qyy = np.zeros((Ny + 1, Nx))
        nd_max = Nstep // diag_every + 2
        dg_t = np.zeros(nd_max); dg_mn = np.zeros(nd_max)
        dg_mx = np.zeros(nd_max); dg_res = np.zeros(nd_max)
        n_stop, n_diag = K.run_loop(
            T, hx, hy, inObs, obs_j.astype(np.int64), obs_i.astype(np.int64),
            Kn, dx, dy, dt, Nstep, Thot, Thot_ramp,
            sw, a1, a2, fm, nonlinear,
            full_Q, eps_Q, Qxx, Qxy, Qyx, Qyy,
            ghostFloor, 0.0 if steady_tol is None else steady_tol,
            snapStep.astype(np.int64), snap_t, snap_T, snap_hx, snap_hy,
            diag_every, dg_t, dg_mn, dg_mx, dg_res)
        elapsed = time.time() - tic
        log(f"Solver finished in {elapsed:.1f} s (numba kernel, {n_stop} steps).")
        out = dict(
            Kn=Kn, Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
            xObs=xObs, yObs=yObs, LObs=LObs,
            Xc=Xc, Yc=Yc, xc=xc, yc=yc,
            inObs=inObs,
            T_final=T, hx_final=hx, hy_final=hy,
            snap_t=snap_t, snap_T=snap_T, snap_hx=snap_hx, snap_hy=snap_hy,
            tSnap=tSnap, dt=dt, Nstep=n_stop,
            diag_t=dg_t[:n_diag], diag_min_hx=dg_mn[:n_diag],
            diag_max_h=dg_mx[:n_diag], diag_res=dg_res[:n_diag],
        )
        if full_Q:
            out.update(_Q_faces_to_centres(Qxx, Qxy, Qyx, Qyy), eps_Q=eps_Q)
        return out

    for n in range(1, Nstep + 1):

        # 1. obstacle temperature fill --------------------------------------
        # NOTE: the shifted arrays below are edge-clamped, so an obstacle cell
        # touching the domain border would take itself as a neighbour, whereas
        # the numba kernel skips out-of-range neighbours.  Irrelevant for every
        # configuration used in the paper (the body is strictly interior), but
        # the two paths would differ for a wall-mounted obstacle.
        if obstacle and obstacle_fill != "frozen":
            Tl = np.concatenate((T[:, :1], T[:, :-1]), axis=1)
            Tr = np.concatenate((T[:, 1:], T[:, -1:]), axis=1)
            Td = np.concatenate((T[:1, :], T[:-1, :]), axis=0)
            Tu = np.concatenate((T[1:, :], T[-1:, :]), axis=0)
            obs_l = np.concatenate((inObs[:, :1], inObs[:, :-1]), axis=1)
            obs_r = np.concatenate((inObs[:, 1:], inObs[:, -1:]), axis=1)
            obs_d = np.concatenate((inObs[:1, :], inObs[:-1, :]), axis=0)
            obs_u = np.concatenate((inObs[1:, :], inObs[-1:, :]), axis=0)
            if obstacle_fill == "compat" and n > 1:
                # extend T into the obstacle honouring the wall compatibility
                # relation dT/dn = Kn^2 [div(Phi grad h)] . n  (n: fluid->obstacle)
                Dxl = np.concatenate((Dx[:, :1], Dx[:, :-1]), axis=1)
                Dxr = np.concatenate((Dx[:, 1:], Dx[:, -1:]), axis=1)
                Dyd = np.concatenate((Dy[:1, :], Dy[:-1, :]), axis=0)
                Dyu = np.concatenate((Dy[1:, :], Dy[-1:, :]), axis=0)
                Tl = Tl + dx * Kn ** 2 * Dxl   # left nb: n = +x
                Tr = Tr - dx * Kn ** 2 * Dxr   # right nb: n = -x
                Td = Td + dy * Kn ** 2 * Dyd   # lower nb: n = +y
                Tu = Tu - dy * Kn ** 2 * Dyu   # upper nb: n = -y
            nbSum = (Tl * (~obs_l) + Tr * (~obs_r)
                     + Td * (~obs_d) + Tu * (~obs_u))
            nbCnt = ((~obs_l).astype(int) + (~obs_r).astype(int)
                     + (~obs_d).astype(int) + (~obs_u).astype(int))
            T_avg = nbSum / np.maximum(nbCnt, 1)
            T[inObs] = T_avg[inObs]

        # 2. ghost cells ----------------------------------------------------
        Thot_n = Thot if Thot_ramp <= 0.0 else Thot * (1.0 - np.exp(-(n * dt) / Thot_ramp))
        Tg = _pad_T(T, Thot_n, Ny, Nx)
        hxg, hyg = _pad_h(hx, hy, sidewalls, Ny, Nx, slip_g)

        # 3. first derivatives ----------------------------------------------
        div_h = ((hxg[1:-1, 2:] - hxg[1:-1, :-2]) / (2 * dx)
                 + (hyg[2:, 1:-1] - hyg[:-2, 1:-1]) / (2 * dy))
        gradT_x = (Tg[1:-1, 2:] - Tg[1:-1, :-2]) / (2 * dx)
        gradT_y = (Tg[2:, 1:-1] - Tg[:-2, 1:-1]) / (2 * dy)

        # 4. non-linear modulation ------------------------------------------
        if nonlinear:
            mag2 = hx ** 2 + hy ** 2
            Phi = np.maximum(0.0, 1.0 - (Kn ** 2 / 6.0) * mag2)
        else:
            Phi = np.ones((Ny, Nx))
        Phi_safe = np.maximum(Phi, ghostFloor)

        if not full_Q:
            # 5. variable-coefficient Laplacian  div(Phi grad h) ------------
            Phig = _pad_zero_grad(Phi, Ny, Nx)
            Phi_e = 0.5 * (Phig[1:-1, 1:-1] + Phig[1:-1, 2:])
            Phi_w = 0.5 * (Phig[1:-1, 1:-1] + Phig[1:-1, :-2])
            Phi_n = 0.5 * (Phig[1:-1, 1:-1] + Phig[2:, 1:-1])
            Phi_s = 0.5 * (Phig[1:-1, 1:-1] + Phig[:-2, 1:-1])
            diff_hx = ((Phi_e * (hxg[1:-1, 2:] - hxg[1:-1, 1:-1])
                        - Phi_w * (hxg[1:-1, 1:-1] - hxg[1:-1, :-2])) / dx ** 2
                       + (Phi_n * (hxg[2:, 1:-1] - hxg[1:-1, 1:-1])
                          - Phi_s * (hxg[1:-1, 1:-1] - hxg[:-2, 1:-1])) / dy ** 2)
            diff_hy = ((Phi_e * (hyg[1:-1, 2:] - hyg[1:-1, 1:-1])
                        - Phi_w * (hyg[1:-1, 1:-1] - hyg[1:-1, :-2])) / dx ** 2
                       + (Phi_n * (hyg[2:, 1:-1] - hyg[1:-1, 1:-1])
                          - Phi_s * (hyg[1:-1, 1:-1] - hyg[:-2, 1:-1])) / dy ** 2)
            Dx, Dy = diff_hx, diff_hy      # wall-compatibility diagnostics

            # 6. RHS and update ---------------------------------------------
            dT = -(Kn ** 2 / 3.0) * div_h
            dhx = -hx / Phi_safe - gradT_x + Kn ** 2 * diff_hx
            dhy = -hy / Phi_safe - gradT_y + Kn ** 2 * diff_hy
        else:
            # 5'. evolve the face-centred Q with exact exponential relaxation
            Phig = _pad_zero_grad(Phi, Ny, Nx)
            # x-faces, shape (Ny, Nx+1)
            Phf_x = 0.5 * (Phig[1:-1, :-1] + Phig[1:-1, 1:])
            fac_x = np.exp(-dt / (eps_Q * np.maximum(Phf_x, ghostFloor)))
            Sxx = -Kn ** 2 * Phf_x * (hxg[1:-1, 1:] - hxg[1:-1, :-1]) / dx
            Sxy = -Kn ** 2 * Phf_x * (hyg[1:-1, 1:] - hyg[1:-1, :-1]) / dx
            Qxx = Sxx + (Qxx - Sxx) * fac_x
            Qxy = Sxy + (Qxy - Sxy) * fac_x
            # y-faces, shape (Ny+1, Nx)
            Phf_y = 0.5 * (Phig[:-1, 1:-1] + Phig[1:, 1:-1])
            fac_y = np.exp(-dt / (eps_Q * np.maximum(Phf_y, ghostFloor)))
            Syx = -Kn ** 2 * Phf_y * (hxg[1:, 1:-1] - hxg[:-1, 1:-1]) / dy
            Syy = -Kn ** 2 * Phf_y * (hyg[1:, 1:-1] - hyg[:-1, 1:-1]) / dy
            Qyx = Syx + (Qyx - Syx) * fac_y
            Qyy = Syy + (Qyy - Syy) * fac_y

            # 6'. div Q and RHS ---------------------------------------------
            divQ_x = ((Qxx[:, 1:] - Qxx[:, :-1]) / dx
                      + (Qyx[1:, :] - Qyx[:-1, :]) / dy)
            divQ_y = ((Qxy[:, 1:] - Qxy[:, :-1]) / dx
                      + (Qyy[1:, :] - Qyy[:-1, :]) / dy)
            dT = -(Kn ** 2 / 3.0) * div_h
            dhx = -hx / Phi_safe - gradT_x - divQ_x
            dhy = -hy / Phi_safe - gradT_y - divQ_y
            # wall-compatibility diagnostics: Kn^2 div(Phi grad h) ~ -div Q
            Dx, Dy = -divQ_x / Kn ** 2, -divQ_y / Kn ** 2

        # 7. update outside the obstacle ------------------------------------
        T[fluid] += dt * dT[fluid]
        hx[fluid] += dt * dhx[fluid]
        hy[fluid] += dt * dhy[fluid]

        # 8. hard-zero h inside the obstacle --------------------------------
        hx[inObs] = 0.0
        hy[inObs] = 0.0

        # 9. snapshots / diagnostics ----------------------------------------
        if n in snapStep:
            for k in np.where(snapStep == n)[0]:
                snap_t[k] = n * dt
                snap_T[k] = T.copy()
                snap_hx[k] = hx.copy()
                snap_hy[k] = hy.copy()

        if n % diag_every == 0 or n == Nstep:
            res = max(np.max(np.abs(dhx[fluid])), np.max(np.abs(dhy[fluid])))
            diag_t.append(n * dt)
            diag_min_hx.append(float(np.min(hx[fluid])))
            diag_max_h.append(float(np.sqrt(np.max(hx ** 2 + hy ** 2))))
            diag_res.append(float(res))
            if steady_tol is not None and res < steady_tol:
                n_stop = n
                log(f"  steady state reached at t = {n*dt:.3f} (residual {res:.2e})")
                break

        if verbose and n % max(1, Nstep // 20) == 0:
            log(f"  step {n}/{Nstep}  (t = {n*dt:.2f})  max|h| = {np.max(np.abs(hx)):.3f}")

    elapsed = time.time() - tic
    log(f"Solver finished in {elapsed:.1f} s.")

    out = dict(
        Kn=Kn, Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
        xObs=xObs, yObs=yObs, LObs=LObs,
        Xc=Xc, Yc=Yc, xc=xc, yc=yc,
        inObs=inObs,
        T_final=T, hx_final=hx, hy_final=hy,
        snap_t=snap_t, snap_T=snap_T, snap_hx=snap_hx, snap_hy=snap_hy,
        tSnap=tSnap, dt=dt, Nstep=n_stop,
        diag_t=np.array(diag_t), diag_min_hx=np.array(diag_min_hx),
        diag_max_h=np.array(diag_max_h), diag_res=np.array(diag_res),
    )
    if full_Q:
        out.update(_Q_faces_to_centres(Qxx, Qxy, Qyx, Qyy), eps_Q=eps_Q)
    return out


def _Q_faces_to_centres(Qxx, Qxy, Qyx, Qyy):
    """Average the face-centred Q components back to the cell centres."""
    return dict(
        Qxx=0.5 * (Qxx[:, :-1] + Qxx[:, 1:]),
        Qxy=0.5 * (Qxy[:, :-1] + Qxy[:, 1:]),
        Qyx=0.5 * (Qyx[:-1, :] + Qyx[1:, :]),
        Qyy=0.5 * (Qyy[:-1, :] + Qyy[1:, :]),
    )


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------
def Q_from_closure(res):
    """Q tensor from the quasi-steady closure Q = -Kn^2 Phi grad h.

    Components returned in the storage convention Q_ji ~ -Kn^2 Phi d_j h_i,
    computed with the same central differences and ghost cells as the solver
    (free-slip side walls assumed).
    """
    Kn, Nx, Ny = res["Kn"], res["Nx"], res["Ny"]
    dx, dy = res["Lx"] / Nx, res["Ly"] / Ny
    hx, hy = res["hx_final"], res["hy_final"]
    hxg, hyg = _pad_h(hx, hy, "freeslip", Ny, Nx)
    mag2 = hx ** 2 + hy ** 2
    Phi = np.maximum(0.0, 1.0 - (Kn ** 2 / 6.0) * mag2)
    dhx_dx = (hxg[1:-1, 2:] - hxg[1:-1, :-2]) / (2 * dx)
    dhy_dx = (hyg[1:-1, 2:] - hyg[1:-1, :-2]) / (2 * dx)
    dhx_dy = (hxg[2:, 1:-1] - hxg[:-2, 1:-1]) / (2 * dy)
    dhy_dy = (hyg[2:, 1:-1] - hyg[:-2, 1:-1]) / (2 * dy)
    return dict(
        Qxx=-Kn ** 2 * Phi * dhx_dx,
        Qxy=-Kn ** 2 * Phi * dhy_dx,
        Qyx=-Kn ** 2 * Phi * dhx_dy,
        Qyy=-Kn ** 2 * Phi * dhy_dy,
    )


def kappa_eff(res, x_probe=None):
    """Effective conductivity kappa_eff/kappa_bulk = <hx>/G of a channel run."""
    G = 1.0 / res["Lx"]
    xc, hx = res["xc"], res["hx_final"]
    if x_probe is None:
        x_probe = res["Lx"] / 2.0
    j = int(np.argmin(np.abs(xc - x_probe)))
    return float(np.mean(hx[:, j]) / G)


# ---------------------------------------------------------------------------
# Analytic reference solutions (linear GK channel, fully developed)
# ---------------------------------------------------------------------------
def channel_noslip_profile(y, R, Kn, G):
    """hx(y) = G [1 - cosh((y-R/2)/Kn)/cosh(R/(2Kn))]  (no-slip walls)."""
    return G * (1.0 - np.cosh((y - R / 2) / Kn) / np.cosh(R / (2 * Kn)))


def channel_slip_profile(y, R, Kn, G, C, alpha):
    """Fully developed linear GK profile with the second-order slip BC.

    hx(y) = G [1 - B cosh((y-R/2)/Kn) / cosh(R/(2Kn))], with B fixed by
        hx + C Kn dhx/dn + alpha Kn^2 d2hx/dn2 = 0  at the walls (n outward).
    """
    th = np.tanh(R / (2 * Kn))
    B = 1.0 / (1.0 + C * th + alpha)
    return G * (1.0 - B * np.cosh((y - R / 2) / Kn) / np.cosh(R / (2 * Kn)))


def channel_kappa_eff(R, Kn, C=None, alpha=None):
    """kappa_eff/kappa_bulk = 1 - B (2 Kn / R) tanh(R/(2 Kn))."""
    th = np.tanh(R / (2.0 * Kn))
    B = 1.0 if C is None else 1.0 / (1.0 + C * th + alpha)
    return 1.0 - B * (2.0 * Kn / R) * th
