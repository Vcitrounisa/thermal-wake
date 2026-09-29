"""
Numba version of the time loop of gk_solver.solve, written node by node.
"""

import math

import numpy as np
from numba import njit


@njit(cache=True)
def _phi(m2, Kn2, nonlinear):
    if not nonlinear:
        return 1.0
    p = 1.0 - (Kn2 / 6.0) * m2
    return p if p > 0.0 else 0.0


@njit(cache=True)
def run(T, hx, hy, Qxx, Qyy, Qyx, Qxy, solid_hx, fluid_hy, noslip,
        Kn, dx, dy, dt, nsteps, nonlinear, full_q, eps, phi_floor, steady_tol,
        snap_step, snap_t, snap_T, snap_hx, snap_hy, hist_every, hist):
    """Advance the solution by ``nsteps`` explicit steps (in place).

    Returns the number of steps taken and the number of history entries.
    """
    Ny, Nx = hx.shape
    Kn2 = Kn * Kn
    gB = np.zeros(Nx)
    gT = np.zeros(Nx)
    phiT = np.ones((Ny, Nx + 1))
    phiC = np.ones((Ny + 1, Nx))
    Fxx = np.zeros((Ny, Nx + 1))
    Fyy = np.zeros((Ny, Nx + 1))
    Fyx = np.zeros((Ny + 1, Nx))
    Fxy = np.zeros((Ny + 1, Nx))
    dT = np.zeros((Ny, Nx + 1))
    dhx = np.zeros((Ny, Nx))
    dhy = np.zeros((Ny + 1, Nx + 1))
    sgn = -1.0 if noslip else 1.0
    n_hist = 0
    n_last = nsteps

    for n in range(1, nsteps + 1):
        # wall ghosts of h_x
        for i in range(Nx):
            gB[i] = sgn * hx[0, i]
            gT[i] = sgn * hx[Ny - 1, i]

        # T nodes: Phi, Fxx, Fyy (Fxx = 0 on the boundary columns)
        for j in range(Ny):
            for i in range(Nx + 1):
                if i == 0:
                    a = hx[j, 0]
                    dxhx = 0.0
                elif i == Nx:
                    a = hx[j, Nx - 1]
                    dxhx = 0.0
                else:
                    a = 0.5 * (hx[j, i - 1] + hx[j, i])
                    dxhx = (hx[j, i] - hx[j, i - 1]) / dx
                b = 0.5 * (hy[j, i] + hy[j + 1, i])
                p = _phi(a ** 2 + b ** 2, Kn2, nonlinear)
                phiT[j, i] = p
                Fxx[j, i] = p * dxhx
                Fyy[j, i] = p * ((hy[j + 1, i] - hy[j, i]) / dy)

        # corners: Phi, Fyx, Fxy
        for j in range(Ny + 1):
            for i in range(Nx):
                if j == 0:
                    a = 0.5 * (gB[i] + hx[0, i])
                    dyhx = (hx[0, i] - gB[i]) / dy
                elif j == Ny:
                    a = 0.5 * (gT[i] + hx[Ny - 1, i])
                    dyhx = (gT[i] - hx[Ny - 1, i]) / dy
                else:
                    a = 0.5 * (hx[j - 1, i] + hx[j, i])
                    dyhx = (hx[j, i] - hx[j - 1, i]) / dy
                b = 0.5 * (hy[j, i] + hy[j, i + 1])
                p = _phi(a ** 2 + b ** 2, Kn2, nonlinear)
                phiC[j, i] = p
                Fyx[j, i] = p * dyhx
                Fxy[j, i] = p * ((hy[j, i + 1] - hy[j, i]) / dx)

        # full model: exact exponential relaxation of Q towards -Kn^2 F
        if full_q:
            for j in range(Ny):
                for i in range(Nx + 1):
                    p = phiT[j, i] if phiT[j, i] > phi_floor else phi_floor
                    f = math.exp(-dt / (eps * p))
                    s = -Kn2 * Fxx[j, i]
                    Qxx[j, i] = s + (Qxx[j, i] - s) * f
                    s = -Kn2 * Fyy[j, i]
                    Qyy[j, i] = s + (Qyy[j, i] - s) * f
            for j in range(Ny + 1):
                for i in range(Nx):
                    p = phiC[j, i] if phiC[j, i] > phi_floor else phi_floor
                    f = math.exp(-dt / (eps * p))
                    s = -Kn2 * Fyx[j, i]
                    Qyx[j, i] = s + (Qyx[j, i] - s) * f
                    s = -Kn2 * Fxy[j, i]
                    Qxy[j, i] = s + (Qxy[j, i] - s) * f

        # energy equation on the interior T nodes
        for j in range(Ny):
            for i in range(1, Nx):
                dT[j, i] = -(Kn2 / 3.0) * ((hx[j, i] - hx[j, i - 1]) / dx
                                           + (hy[j + 1, i] - hy[j, i]) / dy)

        # h_x equation
        for j in range(Ny):
            for i in range(Nx):
                if solid_hx[j, i]:
                    dhx[j, i] = 0.0
                    continue
                b = 0.25 * (hy[j, i] + hy[j, i + 1] + hy[j + 1, i] + hy[j + 1, i + 1])
                p = _phi(hx[j, i] ** 2 + b ** 2, Kn2, nonlinear)
                p = p if p > phi_floor else phi_floor
                if full_q:
                    v = -((Qxx[j, i + 1] - Qxx[j, i]) / dx
                          + (Qyx[j + 1, i] - Qyx[j, i]) / dy)
                else:
                    v = Kn2 * ((Fxx[j, i + 1] - Fxx[j, i]) / dx
                               + (Fyx[j + 1, i] - Fyx[j, i]) / dy)
                dhx[j, i] = -hx[j, i] / p - (T[j, i + 1] - T[j, i]) / dx + v

        # h_y equation on the interior nodes
        for j in range(1, Ny):
            for i in range(1, Nx):
                if not fluid_hy[j, i]:
                    dhy[j, i] = 0.0
                    continue
                a = 0.25 * (hx[j - 1, i - 1] + hx[j - 1, i] + hx[j, i - 1] + hx[j, i])
                p = _phi(hy[j, i] ** 2 + a ** 2, Kn2, nonlinear)
                p = p if p > phi_floor else phi_floor
                if full_q:
                    v = -((Qxy[j, i] - Qxy[j, i - 1]) / dx
                          + (Qyy[j, i] - Qyy[j - 1, i]) / dy)
                else:
                    v = Kn2 * ((Fxy[j, i] - Fxy[j, i - 1]) / dx
                               + (Fyy[j, i] - Fyy[j - 1, i]) / dy)
                dhy[j, i] = -hy[j, i] / p - (T[j, i] - T[j - 1, i]) / dy + v

        # update
        for j in range(Ny):
            for i in range(1, Nx):
                T[j, i] += dt * dT[j, i]
            for i in range(Nx):
                hx[j, i] += dt * dhx[j, i]
        for j in range(1, Ny):
            for i in range(1, Nx):
                hy[j, i] += dt * dhy[j, i]

        for k in range(snap_step.size):
            if snap_step[k] == n:
                snap_t[k] = n * dt
                snap_T[k, :, :] = T
                snap_hx[k, :, :] = hx
                snap_hy[k, :, :] = hy

        # history: min h_x (fluid), max |h| on the T nodes, residual
        if n % hist_every == 0 or n == nsteps:
            mn = 1e300
            res = 0.0
            for j in range(Ny):
                for i in range(Nx):
                    if not solid_hx[j, i] and hx[j, i] < mn:
                        mn = hx[j, i]
                    if abs(dhx[j, i]) > res:
                        res = abs(dhx[j, i])
            for j in range(1, Ny):
                for i in range(1, Nx):
                    if abs(dhy[j, i]) > res:
                        res = abs(dhy[j, i])
            mx = 0.0
            for j in range(Ny):
                for i in range(Nx + 1):
                    if i == 0:
                        a = hx[j, 0]
                    elif i == Nx:
                        a = hx[j, Nx - 1]
                    else:
                        a = 0.5 * (hx[j, i - 1] + hx[j, i])
                    b = 0.5 * (hy[j, i] + hy[j + 1, i])
                    if a ** 2 + b ** 2 > mx:
                        mx = a ** 2 + b ** 2
            hist[0, n_hist] = n * dt
            hist[1, n_hist] = mn
            hist[2, n_hist] = math.sqrt(mx)
            hist[3, n_hist] = res
            n_hist += 1
            if steady_tol > 0.0 and res < steady_tol:
                n_last = n
                break

    return n_last, n_hist
