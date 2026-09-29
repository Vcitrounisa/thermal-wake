"""
Verification against reference solutions of the channel problem (no
obstacle, no-slip walls, fully developed state):

  (a) linear model vs the analytic profile, on grids refined by factors of
      two (Ny = 8 ... 64);
  (b) non-linear model with a strong drive (min Phi ~ 0.6) vs a collocation
      solution of the fully developed equation
          Kn^2 Phi (Phi h_x')' - h_x + G Phi = 0.

Output: results/verification.npz, results/verification.json.
"""

import json

import numpy as np
from scipy.integrate import solve_bvp

from gk_analysis import channel_noslip_profile, observed_order
from gk_solver import solve
from settings import results_path

Kn, R, L = 0.7, 1.0, 2.0
CHANNEL = dict(Kn=Kn, Lx=L, Ly=R, obstacle=False, walls="noslip", ic="fourier",
               t_end=80.0, steady_tol=1e-9, hist_every=200)


def centre_column(res):
    return res["hx"][:, res["Nx"] // 2]


# (a) linear channel
G = 1.0 / L
grids = [8, 16, 32, 64]
errors = []
for Ny in grids:
    res = solve(Nx=2 * Ny, Ny=Ny, nonlinear=False, **CHANNEL)
    exact = channel_noslip_profile(res["y_T"], R, Kn, G)
    errors.append(float(np.max(np.abs(centre_column(res) - exact)) / exact.max()))
    print(f"linear channel, Ny = {Ny:3d}: relative error {errors[-1]:.3e}")
orders = [observed_order(errors[k], errors[k + 1]) for k in range(len(errors) - 1)]
print("observed orders:", np.round(orders, 3))
y_lin, h_lin, h_exact = res["y_T"], centre_column(res), exact

# (b) non-linear channel
T_hot = 20.0
G_nl = T_hot / L


def rhs(y, Y, G):
    h, m = Y                                  # m = Phi h'
    p = 1.0 - (Kn ** 2 / 6.0) * h ** 2
    return np.vstack([m / p, (h - G * p) / (Kn ** 2 * p)])


y = np.linspace(0.0, R, 801)
guess = np.vstack([channel_noslip_profile(y, R, Kn, 1.0),
                   np.gradient(channel_noslip_profile(y, R, Kn, 1.0), y)])
G_prev = 1.0
for G_step in (1.0, 3.0, 6.0, G_nl):         # continuation in the drive
    sol = solve_bvp(lambda s, Y: rhs(s, Y, G_step), lambda a, b: np.array([a[0], b[0]]),
                    y, guess * (G_step / G_prev), tol=1e-9, max_nodes=400000)
    if not sol.success:
        raise RuntimeError(f"collocation failed at G = {G_step}")
    guess, G_prev = sol.sol(y), G_step

res = solve(Nx=128, Ny=64, T_hot=T_hot, **CHANNEL)
h_nl = centre_column(res)
h_bvp = sol.sol(res["y_T"])[0]
err_nl = float(np.max(np.abs(h_nl - h_bvp)) / h_bvp.max())
min_phi = float(1.0 - (Kn ** 2 / 6.0) * h_bvp.max() ** 2)
print(f"non-linear channel (min Phi = {min_phi:.3f}): relative error {err_nl:.3e}")

np.savez(results_path("verification.npz"),
         grids=np.array(grids), errors=np.array(errors),
         y_lin=y_lin, h_lin=h_lin, h_exact=h_exact,
         y_nl=res["y_T"], h_nl=h_nl, h_bvp=h_bvp,
         G_lin=G, G_nl=G_nl)
summary = {"linear_channel": {"Ny": grids, "relative_error": errors, "orders": orders},
           "nonlinear_channel": {"relative_error": err_nl, "min_phi": min_phi}}
results_path("verification.json").write_text(json.dumps(summary, indent=2))
