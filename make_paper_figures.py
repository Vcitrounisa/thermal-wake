"""
make_paper_figures.py
=====================

Loads the simulation produced by `gk_thermal_wake.py` and writes three
publication-ready figures into ../figs/ :

   fig_evolution.pdf/png        time evolution of |h| (4 snapshots)
   fig_steady.pdf/png           steady-state T field + |h| with streamlines
   fig_zoom_centreline.pdf/png  zoom on the obstacle + centreline diagnostics
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


# -- I/O paths ---------------------------------------------------------------
here    = Path(__file__).resolve().parent
in_file = here / "gk_wake_results.npz"
fig_dir = here / "figs"
fig_dir.mkdir(parents=True, exist_ok=True)

data = np.load(in_file, allow_pickle=False)
Xc    = data["Xc"]
Yc    = data["Yc"]
xc    = data["xc"]
yc    = data["yc"]
inObs = data["inObs"].astype(bool)
xObs  = float(data["xObs"])
yObs  = float(data["yObs"])
LObs  = float(data["LObs"])
Lx    = float(data["Lx"])
Ly    = float(data["Ly"])
Kn    = float(data["Kn"])

T_final  = data["T_final"]
hx_final = data["hx_final"]
hy_final = data["hy_final"]
mag_final = np.sqrt(hx_final ** 2 + hy_final ** 2)

snap_t  = data["snap_t"]
snap_T  = data["snap_T"]
snap_hx = data["snap_hx"]
snap_hy = data["snap_hy"]


# -- Style -------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize":  10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 110,
})


def add_obstacle(ax, *, fc=(0.30, 0.30, 0.30), ec="k", lw=0.9):
    ax.add_patch(Rectangle(
        (xObs - LObs / 2, yObs - LObs / 2), LObs, LObs,
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=5))


def mask_obstacle(F):
    G = F.astype(float).copy()
    G[inObs] = np.nan
    return G


def left_seed_points(n_seed, y_lo=0.10, y_hi=None):
    """Seeds for streamlines: a vertical comb on the left wall.
    Produces parallel inflow streamlines that deflect smoothly around
    the obstacle without crossing each other.
    """
    if y_hi is None:
        y_hi = Ly - 0.10
    ys = np.linspace(y_lo, y_hi, n_seed)
    xs = np.full_like(ys, 0.05)
    return np.column_stack([xs, ys])


# ---------------------------------------------------------------------------
#  Figure 1 (paper Fig. 2):  time evolution of |h| with streamlines
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6), constrained_layout=True)
panels = [0, 1, 2, len(snap_t) - 1]
vmax = float(np.nanmax(mask_obstacle(np.sqrt(snap_hx[-1] ** 2
                                          +  snap_hy[-1] ** 2))))
for k, ax in zip(panels, axes.flat):
    hx_k = snap_hx[k]
    hy_k = snap_hy[k]
    mag = mask_obstacle(np.sqrt(hx_k ** 2 + hy_k ** 2))
    pc = ax.pcolormesh(Xc, Yc, mag, shading="gouraud",
                       cmap="viridis", vmin=0, vmax=vmax, rasterized=True)
    seeds = left_seed_points(13)
    ax.streamplot(xc, yc, hx_k, hy_k, color="white",
                  density=2.0, linewidth=0.55, arrowsize=0.7,
                  broken_streamlines=False, start_points=seeds)
    add_obstacle(ax)
    ax.set_aspect("equal")
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_title(rf"$\mathsf{{t}} = {snap_t[k]:.2f}$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
cbar = fig.colorbar(pc, ax=axes, location="right", shrink=0.85, pad=0.02)
cbar.set_label(r"$|\mathbf{h}|$")
fig.savefig(fig_dir / "fig_evolution.pdf", bbox_inches="tight")
fig.savefig(fig_dir / "fig_evolution.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------------------
#  Figure 2 (paper Fig. 3):  quasi-steady state composite
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(6.8, 4.5), constrained_layout=True,
                         sharex=True)

# (a) Temperature isotherms
ax = axes[0]
levels = np.linspace(0, 1, 25)
T_plot = mask_obstacle(T_final)
cf = ax.contourf(Xc, Yc, T_plot, levels=levels, cmap="magma")
cs = ax.contour(Xc, Yc, T_plot, levels=levels[::3],
                colors="white", linewidths=0.4, alpha=0.7)
add_obstacle(ax)
ax.set_aspect("equal")
ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
ax.set_ylabel(r"$y$")
ax.set_title(r"(a) Quasi-stationary temperature field $\mathsf{T}$")
cbar1 = fig.colorbar(cf, ax=ax, shrink=0.9, pad=0.02)
cbar1.set_label(r"$\mathsf{T}$")

# (b) heat-flux magnitude with streamlines
ax = axes[1]
mag_plot = mask_obstacle(mag_final)
pc = ax.pcolormesh(Xc, Yc, mag_plot, shading="gouraud",
                   cmap="viridis", rasterized=True)
seeds = left_seed_points(15)
ax.streamplot(xc, yc, hx_final, hy_final, color="white",
              density=2.0, linewidth=0.55, arrowsize=0.7,
              broken_streamlines=False, start_points=seeds)
add_obstacle(ax)
ax.set_aspect("equal")
ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$")
ax.set_title(r"(b) Quasi-stationary heat-flux magnitude $|\mathbf{h}|$ "
             r"with streamlines")
cbar2 = fig.colorbar(pc, ax=ax, shrink=0.9, pad=0.02)
cbar2.set_label(r"$|\mathbf{h}|$")

fig.savefig(fig_dir / "fig_steady.pdf", bbox_inches="tight")
fig.savefig(fig_dir / "fig_steady.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------------------
#  Figure 3 (paper Fig. 4):  zoom around the obstacle + centre-line cuts
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), constrained_layout=True,
                         gridspec_kw={"width_ratios": [1.05, 1.0]})

# (a) zoom : T contours + heat-flux quiver
ax = axes[0]
T_plot = mask_obstacle(T_final)
cf = ax.contourf(Xc, Yc, T_plot, levels=np.linspace(0, 1, 25),
                 cmap="magma", zorder=0)
cs = ax.contour(Xc, Yc, T_plot, levels=np.linspace(0, 1, 13),
                colors="white", linewidths=0.4, alpha=0.55)
# Quiver field: subsample regularly for clean, non-overlapping arrows
step = 7
qx = Xc[::step, ::step]
qy = Yc[::step, ::step]
qhx = hx_final[::step, ::step].copy()
qhy = hy_final[::step, ::step].copy()
qmask = inObs[::step, ::step]
qhx[qmask] = np.nan
qhy[qmask] = np.nan
ax.quiver(qx, qy, qhx, qhy, scale=1.4, scale_units="xy",
          width=0.0035, headwidth=4, headlength=4.5, color="white",
          alpha=0.95, zorder=4)
add_obstacle(ax)
ax.set_aspect("equal")
ax.set_xlim(xObs - 1.2, xObs + 3.0)
ax.set_ylim(yObs - 1.5, yObs + 1.5)
ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$")
ax.set_title(r"(a) Heat-flux deflection and isotherms")
cb = fig.colorbar(cf, ax=ax, shrink=0.9, pad=0.02)
cb.set_label(r"$\mathsf{T}$")

# (b) centre-line cuts of T and h_x ; off-axis cut for reference
ax = axes[1]
jc  = int(np.argmin(np.abs(yc - yObs)))                          # y = R/2 (through the obstacle)
joa = int(np.argmin(np.abs(yc - (yObs + LObs / 2 + 0.10))))      # just above the obstacle
T_line   = T_final[jc, :].copy()
hx_line  = hx_final[jc, :].copy()
T_off    = T_final[joa, :].copy()                                # uninterrupted cut

# Mask the obstacle interval for the on-axis cut
in_obs_x = (xc >= xObs - LObs / 2) & (xc <= xObs + LObs / 2)
T_line[in_obs_x]  = np.nan
hx_line[in_obs_x] = np.nan

# Reference Fourier-like profile T_F(x) = 1 - x/Lx
Tref = 1.0 - xc / Lx

# Obstacle band
ax.axvspan(xObs - LObs / 2, xObs + LObs / 2,
           color="0.88", lw=0, zorder=0)

# Curves
ax.plot(xc, Tref, "--", color="0.45", linewidth=1.0,
        label=r"linear (Fourier) reference $\mathsf{T}_F$", zorder=2)
ax.plot(xc, T_off, "-.", color="#1f4e9a", linewidth=1.2, alpha=0.65,
        label=r"$\mathsf{T}(x, y\!=\!y_c\!+\!0.6)$", zorder=3)
ax.plot(xc, T_line, "-", color="#1f4e9a", linewidth=2.0,
        label=r"$\mathsf{T}(x, y_c\!=\!L_y/2)$", zorder=4)

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\mathsf{T}$", color="#1f4e9a")
ax.tick_params(axis="y", labelcolor="#1f4e9a")
ax.set_ylim(-0.05, 1.10)
ax.set_xlim(0, Lx)
ax.legend(loc="upper right", frameon=False, fontsize=7.5)
ax.set_title(r"(b) Centre-line profiles: T and $h_x$")

# h_x on twin axis -- emphasise that it remains positive everywhere
ax2 = ax.twinx()
ax2.axhline(0.0, color="0.7", lw=0.6, zorder=1)
ax2.plot(xc, hx_line, "-", color="#c0392b", linewidth=1.4)
ax2.set_ylabel(r"$h_x$", color="#c0392b")
ax2.tick_params(axis="y", labelcolor="#c0392b")
ax2.set_ylim(-0.025, 0.14)

fig.savefig(fig_dir / "fig_zoom_centreline.pdf", bbox_inches="tight")
fig.savefig(fig_dir / "fig_zoom_centreline.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Figures written to {fig_dir}")
