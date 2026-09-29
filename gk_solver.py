"""
Staggered-grid solver for the non-linear Guyer-Krumhansl equations in a 2-D
layer with a square obstacle.

Model (non-dimensional)::

    dT/dt + (Kn^2/3) div h = 0
    Phi dh/dt + h + Phi grad T - Kn^2 Phi div(Phi grad h) = 0
    Phi = max(0, 1 - Kn^2 |h|^2 / 6)

With ``full_q=True`` the last term is replaced by ``Phi div Q`` and the flux
of the heat flux obeys ``eps Phi dQ/dt + Q + Kn^2 Phi grad h = 0``.

Boundary conditions::

    x = 0     T = T_hot, h_y = 0
    x = L     T = 0,     h_y = 0
    y = 0, R  h_y = 0 and either dh_x/dy = 0 ("freeslip") or h_x = 0 ("noslip")
    obstacle  h = 0

Grid (arrays indexed [j, i])::

    T    (Ny,   Nx+1)   at (i dx,       (j+1/2) dy)
    h_x  (Ny,   Nx)     at ((i+1/2) dx, (j+1/2) dy)
    h_y  (Ny+1, Nx+1)   at (i dx,       j dy)
    corners (Ny+1, Nx)  at ((i+1/2) dx, j dy)

T is stored at the centres of the energy control volumes and the heat-flux
components on their faces, as the pressure and the velocity of a MAC scheme.
All boundary data sit on grid nodes.  No condition is imposed on T at the
walls or in the obstacle, and none on h_x at x = 0, L: there the energy
equation with fixed T and h_y = 0 gives dh_x/dx = 0, which is used as the
normal flux of the viscous term.  See NUMERICS.md for the full description.
"""

import time

import numpy as np

WALLS = ("freeslip", "noslip")
INITIAL_CONDITIONS = ("cold", "fourier", "fourier-flux")


def make_grid(Lx, Ly, Nx, Ny):
    """Node coordinates of the staggered grid."""
    dx, dy = Lx / Nx, Ly / Ny
    return {
        "dx": dx,
        "dy": dy,
        "x_T": np.arange(Nx + 1) * dx,        # T and h_y columns
        "y_T": (np.arange(Ny) + 0.5) * dy,    # T and h_x rows
        "x_hx": (np.arange(Nx) + 0.5) * dx,   # h_x columns
        "y_hy": np.arange(Ny + 1) * dy,       # h_y rows
    }


def obstacle_masks(grid, x_obs, y_obs, side):
    """Masks describing the obstacle on the staggered grid.

    Returns
    -------
    solid_hx, solid_hy : bool arrays
        Heat-flux nodes inside the closed square (h = 0, never updated).
    enclosed_T : bool array
        T nodes whose four faces are solid.
    in_obstacle : bool array
        T nodes inside the closed square (used for plotting only).
    """
    tol = 1e-9 * min(grid["dx"], grid["dy"])
    half = 0.5 * side + tol

    def inside(x, y):
        X, Y = np.meshgrid(x, y)
        return (np.abs(X - x_obs) <= half) & (np.abs(Y - y_obs) <= half)

    solid_hx = inside(grid["x_hx"], grid["y_T"])
    solid_hy = inside(grid["x_T"], grid["y_hy"])
    in_obstacle = inside(grid["x_T"], grid["y_T"])
    enclosed_T = np.zeros_like(in_obstacle)
    enclosed_T[:, 1:-1] = (solid_hx[:, :-1] & solid_hx[:, 1:]
                           & solid_hy[:-1, 1:-1] & solid_hy[1:, 1:-1])
    return solid_hx, solid_hy, enclosed_T, in_obstacle


def phi(m2, Kn, nonlinear=True):
    """Non-linear modulation factor as a function of |h|^2."""
    if not nonlinear:
        return np.ones_like(m2)
    return np.maximum(0.0, 1.0 - (Kn ** 2 / 6.0) * m2)


def wall_ghosts(hx, walls):
    """Ghost rows of h_x below y = 0 and above y = R."""
    if walls == "freeslip":
        return hx[0].copy(), hx[-1].copy()
    if walls == "noslip":
        return -hx[0], -hx[-1]
    raise ValueError(f"unknown wall condition {walls!r}")


def flux_on_T_nodes(hx, hy):
    """Average both heat-flux components onto the T nodes."""
    hxT = np.empty((hx.shape[0], hx.shape[1] + 1))
    hxT[:, 1:-1] = 0.5 * (hx[:, :-1] + hx[:, 1:])
    hxT[:, 0], hxT[:, -1] = hx[:, 0], hx[:, -1]
    return hxT, 0.5 * (hy[:-1] + hy[1:])


def operators(hx, hy, walls, Kn, dx, dy, nonlinear=True):
    """Phi and the fluxes F = Phi grad h of the viscous operator div(Phi grad h).

    Fxx, Fyy live on the T nodes, Fyx, Fxy on the corners.  At x = 0, L the
    normal flux Fxx is zero (dh_x/dx = 0 from the energy equation).
    """
    Ny, Nx = hx.shape
    gB, gT = wall_ghosts(hx, walls)

    # heat flux reconstructed on the T nodes and on the corners
    hxT, hyT = flux_on_T_nodes(hx, hy)
    hxC = np.empty((Ny + 1, Nx))
    hxC[1:-1] = 0.5 * (hx[:-1] + hx[1:])
    hxC[0], hxC[-1] = 0.5 * (gB + hx[0]), 0.5 * (gT + hx[-1])
    hyC = 0.5 * (hy[:, :-1] + hy[:, 1:])

    # transverse component at the h_x nodes and at the interior h_y nodes
    hy_on_x = 0.25 * (hy[:-1, :-1] + hy[:-1, 1:] + hy[1:, :-1] + hy[1:, 1:])
    hx_on_y = 0.25 * (hx[:-1, :-1] + hx[:-1, 1:] + hx[1:, :-1] + hx[1:, 1:])

    phi_T = phi(hxT ** 2 + hyT ** 2, Kn, nonlinear)
    phi_C = phi(hxC ** 2 + hyC ** 2, Kn, nonlinear)
    phi_x = phi(hx ** 2 + hy_on_x ** 2, Kn, nonlinear)
    phi_y = phi(hy[1:-1, 1:-1] ** 2 + hx_on_y ** 2, Kn, nonlinear)

    dxhx = np.zeros((Ny, Nx + 1))
    dxhx[:, 1:-1] = (hx[:, 1:] - hx[:, :-1]) / dx
    dyhx = np.empty((Ny + 1, Nx))
    dyhx[1:-1] = (hx[1:] - hx[:-1]) / dy
    dyhx[0], dyhx[-1] = (hx[0] - gB) / dy, (gT - hx[-1]) / dy

    return {
        "phi_x": phi_x, "phi_y": phi_y, "phi_T": phi_T, "phi_C": phi_C,
        "Fxx": phi_T * dxhx,
        "Fyy": phi_T * (hy[1:] - hy[:-1]) / dy,
        "Fyx": phi_C * dyhx,
        "Fxy": phi_C * (hy[:, 1:] - hy[:, :-1]) / dx,
    }


def div_x(Axx, Ayx, dx, dy):
    """x component of the divergence of a flux pair, at the h_x nodes."""
    return (Axx[:, 1:] - Axx[:, :-1]) / dx + (Ayx[1:] - Ayx[:-1]) / dy


def div_y(Axy, Ayy, dx, dy):
    """y component of the divergence of a flux pair, at the interior h_y nodes."""
    return ((Axy[1:-1, 1:] - Axy[1:-1, :-1]) / dx
            + (Ayy[1:, 1:-1] - Ayy[:-1, 1:-1]) / dy)


def time_step(Kn, dx, dy, cfl=0.2, full_q=False, eps=0.1):
    """Explicit time step: second-sound, diffusive and (full model) h-Q limits."""
    d = min(dx, dy)
    dt = min(cfl * d * np.sqrt(3.0) / Kn, cfl * d ** 2 / (2.0 * Kn ** 2))
    if full_q:
        dt = min(dt, cfl * d * np.sqrt(eps) / Kn)
    return dt


def _initial_state(grid, ic, T_hot, Lx):
    Ny, Nx = grid["y_T"].size, grid["x_hx"].size
    X, _ = np.meshgrid(grid["x_T"], grid["y_T"])
    hx = np.zeros((Ny, Nx))
    hy = np.zeros((Ny + 1, Nx + 1))
    if ic == "cold":
        T = np.zeros((Ny, Nx + 1))
    else:
        T = T_hot * (1.0 - X / Lx)
        if ic == "fourier-flux":
            hx[:] = T_hot / Lx
    return T, hx, hy


def _step_numpy(T, hx, hy, Q, solid_hx, fluid_hy, walls, Kn, dx, dy, dt,
                nonlinear, full_q, eps, phi_floor):
    """One explicit step of the reference (numpy) implementation.

    Updates T, hx, hy (and Q) in place and returns (dhx, dhy) for the
    residual.
    """
    Kn2 = Kn ** 2
    op = operators(hx, hy, walls, Kn, dx, dy, nonlinear)

    if full_q:
        f_T = np.exp(-dt / (eps * np.maximum(op["phi_T"], phi_floor)))
        f_C = np.exp(-dt / (eps * np.maximum(op["phi_C"], phi_floor)))
        for name, f in (("Fxx", f_T), ("Fyy", f_T), ("Fyx", f_C), ("Fxy", f_C)):
            key = "Q" + name[1:]
            target = -Kn2 * op[name]
            Q[key] = target + (Q[key] - target) * f
        visc_x = -div_x(Q["Qxx"], Q["Qyx"], dx, dy)
        visc_y = -div_y(Q["Qxy"], Q["Qyy"], dx, dy)
    else:
        visc_x = Kn2 * div_x(op["Fxx"], op["Fyx"], dx, dy)
        visc_y = Kn2 * div_y(op["Fxy"], op["Fyy"], dx, dy)

    dT = -(Kn2 / 3.0) * ((hx[:, 1:] - hx[:, :-1]) / dx
                         + (hy[1:, 1:-1] - hy[:-1, 1:-1]) / dy)
    dhx = (-hx / np.maximum(op["phi_x"], phi_floor)
           - (T[:, 1:] - T[:, :-1]) / dx + visc_x)
    dhy = (-hy[1:-1, 1:-1] / np.maximum(op["phi_y"], phi_floor)
           - (T[1:, 1:-1] - T[:-1, 1:-1]) / dy + visc_y)
    dhx[solid_hx] = 0.0
    dhy[~fluid_hy[1:-1, 1:-1]] = 0.0

    T[:, 1:-1] += dt * dT
    hx += dt * dhx
    hy[1:-1, 1:-1] += dt * dhy
    return dhx, dhy


def solve(Kn=0.7, Lx=8.0, Ly=4.0, Nx=200, Ny=100,
          obstacle=True, x_obs=3.0, y_obs=None, side=1.0,
          t_end=400.0, cfl=0.2, t_snap=(),
          T_hot=1.0, ic="cold", init=None, T_enclosed=0.0,
          walls="freeslip", nonlinear=True, full_q=False, eps=0.1,
          phi_floor=1e-3, steady_tol=None, hist_every=50,
          use_numba=True, verbose=False):
    """Integrate the model up to ``t_end`` (or to a steady state).

    Parameters
    ----------
    ic : {"cold", "fourier", "fourier-flux"}
        Initial state: T = 0, h = 0; Fourier profile with h = 0; Fourier
        profile with the Fourier flux.
    init : tuple of arrays, optional
        Restart state (T, hx, hy) on the staggered grid; overrides ic.
    T_enclosed : float or None
        Initial value of the T nodes enclosed by the obstacle.
        None keeps the value of the initial condition.
    walls : {"freeslip", "noslip"}
        Condition on h_x at y = 0, R (h_y = 0 in both cases).
    steady_tol : float, optional
        Stop when max |dh/dt| falls below this value.

    Returns
    -------
    dict
        Parameters, grid coordinates, masks, final fields (T, hx, hy
        and, for the full model, Qxx, Qyy on the T nodes and Qyx, Qxy on
        the corners), snapshots and the history of min h_x, max |h| and
        of the residual.
    """
    if y_obs is None:
        y_obs = 0.5 * Ly
    if walls not in WALLS:
        raise ValueError(f"unknown wall condition {walls!r}")
    if init is None and ic not in INITIAL_CONDITIONS:
        raise ValueError(f"unknown initial condition {ic!r}")
    log = print if verbose else (lambda *a, **k: None)

    grid = make_grid(Lx, Ly, Nx, Ny)
    dx, dy = grid["dx"], grid["dy"]
    if obstacle:
        solid_hx, solid_hy, enclosed_T, in_obstacle = obstacle_masks(grid, x_obs, y_obs, side)
    else:
        solid_hx = np.zeros((Ny, Nx), bool)
        solid_hy = np.zeros((Ny + 1, Nx + 1), bool)
        enclosed_T = np.zeros((Ny, Nx + 1), bool)
        in_obstacle = enclosed_T.copy()
    fluid_hy = ~solid_hy
    fluid_hy[0, :] = fluid_hy[-1, :] = fluid_hy[:, 0] = fluid_hy[:, -1] = False

    if init is not None:
        T, hx, hy = (np.array(a, dtype=float) for a in init)
        shapes = ((Ny, Nx + 1), (Ny, Nx), (Ny + 1, Nx + 1))
        if (T.shape, hx.shape, hy.shape) != shapes:
            raise ValueError(f"init arrays must have shapes {shapes}")
    else:
        T, hx, hy = _initial_state(grid, ic, T_hot, Lx)
        if T_enclosed is not None:
            T[enclosed_T] = T_enclosed
    T[:, 0], T[:, -1] = T_hot, 0.0
    hx[solid_hx] = 0.0
    hy[~fluid_hy] = 0.0

    dt = time_step(Kn, dx, dy, cfl, full_q, eps)
    nsteps = int(np.ceil(t_end / dt))
    dt = t_end / nsteps

    t_snap = np.atleast_1d(np.asarray(t_snap, dtype=float))
    snap_step = np.maximum(1, np.round(t_snap / dt)).astype(np.int64)
    snap_t = np.zeros(t_snap.size)
    snap_T = np.zeros((t_snap.size, Ny, Nx + 1))
    snap_hx = np.zeros((t_snap.size, Ny, Nx))
    snap_hy = np.zeros((t_snap.size, Ny + 1, Nx + 1))

    Q = {"Qxx": np.zeros((Ny, Nx + 1)), "Qyy": np.zeros((Ny, Nx + 1)),
         "Qyx": np.zeros((Ny + 1, Nx)), "Qxy": np.zeros((Ny + 1, Nx))}

    log(f"GK solver: Kn={Kn}, {Nx}x{Ny}, dt={dt:.4e}, {nsteps} steps, "
        f"ic={'restart' if init is not None else ic}, walls={walls}, "
        f"full_q={full_q}" + (f" (eps={eps})" if full_q else ""))

    kernels = None
    if use_numba:
        try:
            import gk_kernels as kernels
        except ImportError:
            kernels = None

    tic = time.time()
    if kernels is not None:
        nh = nsteps // hist_every + 2
        hist = np.zeros((4, nh))
        n_last, n_hist = kernels.run(
            T, hx, hy, Q["Qxx"], Q["Qyy"], Q["Qyx"], Q["Qxy"],
            solid_hx, fluid_hy, walls == "noslip",
            Kn, dx, dy, dt, nsteps, nonlinear, full_q, eps, phi_floor,
            0.0 if steady_tol is None else steady_tol,
            snap_step, snap_t, snap_T, snap_hx, snap_hy, hist_every, hist)
        hist = hist[:, :n_hist]
    else:
        rows = []
        n_last = nsteps
        for n in range(1, nsteps + 1):
            dhx, dhy = _step_numpy(T, hx, hy, Q, solid_hx, fluid_hy, walls, Kn,
                                   dx, dy, dt, nonlinear, full_q, eps, phi_floor)
            for k in np.flatnonzero(snap_step == n):
                snap_t[k] = n * dt
                snap_T[k], snap_hx[k], snap_hy[k] = T, hx, hy
            if n % hist_every == 0 or n == nsteps:
                res = max(np.abs(dhx).max(), np.abs(dhy).max())
                hxT, hyT = flux_on_T_nodes(hx, hy)
                rows.append((n * dt, hx[~solid_hx].min(),
                             np.sqrt((hxT ** 2 + hyT ** 2).max()), res))
                if steady_tol is not None and res < steady_tol:
                    n_last = n
                    break
        hist = np.array(rows).T
    log(f"done in {time.time() - tic:.1f} s ({n_last} steps)")

    out = {
        "Kn": Kn, "Lx": Lx, "Ly": Ly, "Nx": Nx, "Ny": Ny, "dx": dx, "dy": dy,
        "x_obs": x_obs, "y_obs": y_obs, "side": side, "obstacle": obstacle,
        "walls": walls, "nonlinear": nonlinear, "full_q": full_q, "eps": eps,
        "T_hot": T_hot, "dt": dt, "nsteps": n_last, "t": n_last * dt,
        **{k: grid[k] for k in ("x_T", "y_T", "x_hx", "y_hy")},
        "solid_hx": solid_hx, "solid_hy": solid_hy,
        "enclosed_T": enclosed_T, "in_obstacle": in_obstacle,
        "T": T, "hx": hx, "hy": hy,
        "snap_t": snap_t, "snap_T": snap_T, "snap_hx": snap_hx, "snap_hy": snap_hy,
        "hist_t": hist[0], "hist_min_hx": hist[1], "hist_max_h": hist[2],
        "hist_res": hist[3],
    }
    if full_q:
        out.update(Q)
    return out
