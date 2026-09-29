"""
Post-processing utilities for the results of gk_solver.solve.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from gk_solver import operators


def load_result(path):
    """Load a .npz result into a dictionary, 0-d arrays turned into scalars."""
    with np.load(path, allow_pickle=False) as data:
        return {k: (data[k].item() if data[k].ndim == 0 else data[k]) for k in data.files}


def corners_on_T_nodes(A):
    """Average a corner field (Ny+1, Nx) onto the T nodes (Ny, Nx+1)."""
    Ay = 0.5 * (A[:-1] + A[1:])
    B = np.empty((Ay.shape[0], Ay.shape[1] + 1))
    B[:, 1:-1] = 0.5 * (Ay[:, :-1] + Ay[:, 1:])
    B[:, 0], B[:, -1] = Ay[:, 0], Ay[:, -1]
    return B


def closure_Q(res):
    """Flux of the heat flux from the closure Q = -Kn^2 Phi grad h.

    Returns Qxx, Qyy on the T nodes and Qyx, Qxy on the corners.
    """
    op = operators(res["hx"], res["hy"], res["walls"], res["Kn"],
                   res["dx"], res["dy"], res["nonlinear"])
    Kn2 = res["Kn"] ** 2
    return {"Qxx": -Kn2 * op["Fxx"], "Qyy": -Kn2 * op["Fyy"],
            "Qyx": -Kn2 * op["Fyx"], "Qxy": -Kn2 * op["Fxy"]}


def interpolate(F, y, x, Y, X):
    """Linear interpolation of a nodal field onto the points (Y, X)."""
    f = RegularGridInterpolator((y, x), F, bounds_error=False, fill_value=None)
    return f(np.column_stack([np.ravel(Y), np.ravel(X)])).reshape(np.shape(X))


def probe(res, name, x, y):
    """Value of T or hx at the point (x, y), from its own staggered nodes."""
    xs = res["x_T"] if name == "T" else res["x_hx"]
    return float(interpolate(res[name], res["y_T"], xs, np.array([y]), np.array([x]))[0])


def obstacle_distance(X, Y, res):
    """Chebyshev distance of the points (X, Y) from the square obstacle."""
    return (np.maximum(np.abs(X - res["x_obs"]), np.abs(Y - res["y_obs"]))
            - 0.5 * res["side"])


def checkerboard_index(F, mask=None):
    """Odd-even content of a nodal field, max |F - S(F)| / range(F).

    S is the 1-2-1 filter in both directions.  Only nodes whose 3x3 stencil
    lies in ``mask`` are considered.
    """
    G = F.astype(float).copy()
    G[:, 1:-1] = 0.25 * F[:, :-2] + 0.5 * F[:, 1:-1] + 0.25 * F[:, 2:]
    H = G.copy()
    H[1:-1, :] = 0.25 * G[:-2, :] + 0.5 * G[1:-1, :] + 0.25 * G[2:, :]
    D = np.abs(F - H)[1:-1, 1:-1]
    if mask is not None:
        m = np.ones_like(D, dtype=bool)
        for a in range(3):
            for b in range(3):
                m &= mask[a:a + D.shape[0], b:b + D.shape[1]]
        D = D[m]
    return float(D.max() / (F.max() - F.min()))


def observed_order(e_coarse, e_fine, ratio=2.0):
    """Order from the norms of two successive grid differences."""
    return float(np.log(e_coarse / e_fine) / np.log(ratio))


def channel_noslip_profile(y, R, Kn, G):
    """Fully developed linear channel flow with no-slip walls:
    h_x(y) = G [1 - cosh((y - R/2)/Kn) / cosh(R/(2 Kn))]."""
    return G * (1.0 - np.cosh((y - 0.5 * R) / Kn) / np.cosh(0.5 * R / Kn))
