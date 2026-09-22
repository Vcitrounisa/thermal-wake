"""
make_new_figures.py
===================

Builds the figures added in the revision:

  figs/fig_verification.pdf/png  paper Fig. 2  channel benchmarks, grid
                                 convergence, full-vs-reduced O(eps) validation
  figs/fig_Q.pdf/png             paper Fig. 6  flux-of-heat-flux tensor
  figs/fig_experiment.pdf/png    paper Fig. 7  effective conductivity of a
                                 strip vs the graphite-ribbon literature
  figs/fig_resp_ic.pdf/png       (response letter) IC independence
  figs/fig_resp_fill.pdf/png     (response letter) obstacle-fill artefact

Inputs (all produced by the run_*.py scripts, see README for the order):
  verification_results.npz   run_verification.py
  fullQ_results.npz          run_fullQ.py
  experiment_results.npz     run_experiment.py
  ic_sensitivity_results.npz run_ic_sensitivity.py
  gk_wake_results.npz        gk_thermal_wake.py       (closure Q, cold start)
  fullQ_coldstart.npz        run_fullQ_coldstart.py   (full three-field Q)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

here = Path(__file__).resolve().parent
fig_dir = here / "figs"
fig_dir.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 110,
})

BLUE, RED, GREY = "#1f4e9a", "#c0392b", "0.45"


def save(fig, name):
    fig.savefig(fig_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}")


# ===========================================================================
# fig_verification
# ===========================================================================
v = np.load(here / "verification_results.npz")
q = np.load(here / "fullQ_results.npz")          # verification runs (Fourier-profile IC)
b = np.load(here / "gk_wake_results.npz")        # cold-start baseline (closure Q fields)
qf = np.load(here / "fullQ_coldstart.npz")       # full three-field run, cold start

fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.6), constrained_layout=True)

# (a) channel profiles, normalised by G
ax = axes[0]
Gl, Gnl = 0.5, 10.0
ax.plot(v["a_y"], v["a_hx_ex"] / Gl, "-", color=BLUE, lw=1.4,
        label="linear, exact")
ax.plot(v["a_y"][::4], v["a_hx_num"][::4] / Gl, "o", ms=3.2, mfc="none",
        color=BLUE, label="linear, 2D code")
ax.plot(v["b_y"], v["b_hx_bvp"] / Gnl, "-", color=RED, lw=1.4,
        label="non-linear, collocation")
ax.plot(v["b_y"][::4], v["b_hx_num"][::4] / Gnl, "s", ms=3.2, mfc="none",
        color=RED, label="non-linear, 2D code")
ax.set_xlabel(r"$y$")
ax.set_ylabel(r"$h_x/\mathsf{G}$")
ax.set_title("(a) channel benchmarks", fontsize=9)
ax.legend(frameon=False, fontsize=6.5, loc="lower center")

# (b) grid convergence
ax = axes[1]
ax.loglog(v["a_grids"], v["a_errs"], "o-", color=BLUE, lw=1.2, ms=4,
          label="linear channel")
ref = v["a_errs"][0] * (v["a_grids"][0] / v["a_grids"].astype(float)) ** 2
ax.loglog(v["a_grids"], ref, "--", color=GREY, lw=1.0,
          label=r"slope $-2$")
ax.set_xticks([8, 16, 32, 64])
ax.set_xticklabels(["8", "16", "32", "64"])
ax.minorticks_off()
ax.set_xlabel(r"$N_y$")
ax.set_ylabel("max. relative error")
ax.set_title("(b) grid convergence", fontsize=9)
ax.legend(frameon=False)

# (c) full-vs-reduced O(eps)
ax = axes[2]
eps = q["eps"]
tS = q["tS"]
mk = ["o", "s"]
for k, t in enumerate(tS[:2]):
    ax.loglog(eps, q["err_h"][k], mk[k] + "-", ms=4, lw=1.2,
              color=[BLUE, RED][k], label=rf"$\mathsf{{t}}={t:g}$")
ref = q["err_h"][0][-1] * (eps / eps[-1])
ax.loglog(eps, ref, "--", color=GREY, lw=1.0, label=r"slope $+1$")
ax.set_xlabel(r"$\varepsilon=\tau_Q/\tau_R$")
ax.set_ylabel(r"$\max|\mathbf{h}_{\rm full}-\mathbf{h}_{\rm red}|$")
ax.set_title("(c) validation of the reduction", fontsize=9)
ax.legend(frameon=False)

save(fig, "fig_verification")

# ===========================================================================
# fig_Q
# ===========================================================================
Xc, Yc, inObs = b["Xc"], b["Yc"], b["inObs"].astype(bool)
xObs, yObs, LObs = float(b["xObs"]), float(b["yObs"]), float(b["LObs"])
Lx, Ly = float(b["Lx"]), float(b["Ly"])


def mask(F):
    G = F.copy()
    G[inObs] = np.nan
    return G


fig = plt.figure(figsize=(7.6, 2.9), constrained_layout=True)
gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0])

for r, (name, comp) in enumerate([(r"$\mathcal{Q}_{xx}$", "Qxx"),
                                  (r"$\mathcal{Q}_{yx}$", "Qyx")]):
    ax = fig.add_subplot(gs[r, 0])
    F = mask(b[comp])
    vmax = np.nanmax(np.abs(F))
    pc = ax.pcolormesh(Xc, Yc, F, shading="gouraud", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, rasterized=True)
    ax.add_patch(Rectangle((xObs - LObs / 2, yObs - LObs / 2), LObs, LObs,
                           facecolor=(0.3, 0.3, 0.3), edgecolor="k", lw=0.8,
                           zorder=5))
    ax.set_aspect("equal")
    ax.set_xlim(0.8, 7.2)
    ax.set_ylim(0, Ly)
    ax.set_ylabel(r"$y$")
    if r == 1:
        ax.set_xlabel(r"$x$")
    ax.set_title(("(a) " if r == 0 else "(b) ") + name, fontsize=9)
    cb = fig.colorbar(pc, ax=ax, shrink=0.9, pad=0.02)
    cb.ax.tick_params(labelsize=7)

ax = fig.add_subplot(gs[:, 1])
x = b["xc"]
Ny_b = b["Qxx"].shape[0]
jc = Ny_b // 2
joff = int(jc + Ny_b * 0.15)
ax.plot(x, b["Qxx"][jc, :], "-", color=BLUE, lw=1.4,
        label=r"$\mathcal{Q}_{xx}\,(y=\mathsf{R}/2)$, closure")
ax.plot(x[::5], qf["Qxx"][jc, ::5], "o", ms=3, mfc="none", color=BLUE,
        label=r"full system, $\varepsilon=0.02$")
ax.plot(x, b["Qyx"][joff, :], "-", color=RED, lw=1.4,
        label=r"$\mathcal{Q}_{yx}\,(y=\mathsf{R}/2+0.6)$, closure")
ax.plot(x[::5], qf["Qyx"][joff, ::5], "s", ms=3, mfc="none", color=RED,
        label=r"full system, $\varepsilon=0.02$")
ax.axvspan(xObs - LObs / 2, xObs + LObs / 2, color="0.90", lw=0, zorder=0)
ax.axhline(0, color="0.75", lw=0.6)
ax.set_xlim(0, Lx)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\mathcal{Q}$")
ax.set_title("(c) closure vs full system", fontsize=9)
ax.legend(frameon=False, fontsize=6.5)

save(fig, "fig_Q")

# ===========================================================================
# fig_experiment
# ===========================================================================
e = np.load(here / "experiment_results.npz")
Kn = e["Kn_grid"]

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)

ax = axes[0]
ax.semilogx(Kn, e["curve_noslip"], "-", color="k", lw=1.6, label="no-slip")
cmap = plt.cm.viridis
for i, vv in enumerate([0.1, 0.3, 0.6, 1.0]):
    ax.semilogx(Kn, e[f"curve_v{vv}"], "-", lw=1.3, color=cmap(0.15 + 0.75 * i / 3),
                label=rf"slip, $\upsilon={vv}$")
ax.plot(e["val_Kn"], e["val_num"], "o", ms=4.5, mfc="none", color=RED,
        zorder=5, label="2D code")
ax.axvline(0.7, color="0.6", lw=0.8, ls=":")
ax.text(0.78, 0.90, r"$\operatorname{Kn}=0.7$", fontsize=7, color="0.35")
ax.set_xlabel(r"$\operatorname{Kn}=\ell/w$")
ax.set_ylabel(r"$\kappa_{\rm eff}/\kappa$")
ax.set_ylim(0, 1.05)
ax.set_title("(a) effective conductivity of a strip", fontsize=9)
ax.legend(frameon=False, fontsize=6.5)

ax = axes[1]
x = 1.0 / Kn                      # w / ell
kbar = Kn * e["curve_noslip"]
i = np.argsort(x)
ax.loglog(x[i], kbar[i], "-", color="k", lw=1.6, label="no-slip GK")
xa = np.logspace(-1.1, 0.65, 50)
ax.loglog(xa, xa / 12, "--", color=BLUE, lw=1.0,
          label=r"Poiseuille $\;\bar{k}=w/(12\ell)$")
xb = np.logspace(0.6, 2.0, 50)
ax.loglog(xb, 1.0 / xb, "--", color=RED, lw=1.0,
          label=r"Fourier $\;\bar{k}=\ell/w$")
ax.axvspan(1.5 / 0.9, 2.8 / 0.9, color="#f2d5a0", alpha=0.55, lw=0,
           label=r"Poiseuille window, graphite 90 K")
ax.axvline(1 / 0.7, color="0.6", lw=0.8, ls=":")
ax.set_xlim(0.08, 100)
ax.set_ylim(6e-3, 0.35)
ax.set_xlabel(r"$w/\ell$")
ax.set_ylabel(r"$\bar{k}=\operatorname{Kn}\,\kappa_{\rm eff}/\kappa$")
ax.set_title("(b) conductance per width vs ribbon width", fontsize=9)
secax = ax.secondary_xaxis("top", functions=(lambda z: z * 0.9, lambda z: z / 0.9))
secax.set_xlabel(r"$w\ \left[\mu\mathrm{m}\right]$ (graphite, $90$ K, $\ell=0.9\,\mu$m)",
                 fontsize=8)
secax.tick_params(labelsize=7)
ax.legend(frameon=False, fontsize=6.5, loc="lower left")

save(fig, "fig_experiment")

# ===========================================================================
# fig_resp_ic  (response letter)
# ===========================================================================
s = np.load(here / "ic_sensitivity_results.npz")
x = s["x"]
obx = s["inobs_x"].astype(bool)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), constrained_layout=True)
labels = {"cold-rest": r"$\mathsf{T}=0$, $\mathbf{h}=0$ (paper baseline)",
          "fourier-rest": r"$\mathsf{T}=\mathsf{T}_F$, $\mathbf{h}=0$ (verification runs)",
          "fourier-flux": r"$\mathsf{T}=\mathsf{T}_F$, $\mathbf{h}=G\hat{\mathbf{e}}_x$"}
styles = {"cold-rest": ("-", BLUE, 2.2), "fourier-rest": ("--", RED, 1.4),
          "fourier-flux": (":", "k", 1.4)}
ax = axes[0]
for ic in labels:
    Tm = s[f"T_mid_{ic}"].copy()
    if ic == "cold-rest":
        # remove the neutral odd-even (checkerboard) component left over from
        # the discontinuous switch-on (see response text)
        Tm[1:-1] = 0.25 * Tm[:-2] + 0.5 * Tm[1:-1] + 0.25 * Tm[2:]
    Tm[obx] = np.nan
    st = styles[ic]
    ax.plot(x, Tm, st[0], color=st[1], lw=st[2], label=labels[ic])
ax.plot(x, 1 - x / 8, "-", color="0.75", lw=0.8, label=r"Fourier $\mathsf{T}_F$")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\mathsf{T}(x,\mathsf{R}/2)$")
ax.set_title("(a) centreline temperature, three initial states", fontsize=9)
ax.legend(frameon=False, fontsize=6.5)

ax = axes[1]
for ic in labels:
    st = styles[ic]
    ax.plot(s[f"diag_t_{ic}"], s[f"diag_maxh_{ic}"],
            st[0], color=st[1], lw=st[2], label=labels[ic])
ax.set_xlabel(r"$\mathsf{t}$")
ax.set_ylabel(r"$\max|\mathbf{h}|$")
ax.set_xlim(0, 200)
ax.set_title("(b) approach to the attractor", fontsize=9)
ax.legend(frameon=False, fontsize=6.5, loc="lower right")

save(fig, "fig_resp_ic")

# ===========================================================================
# fig_resp_fill  (response letter)
# ===========================================================================
Xc2, Yc2, inObs2 = s["Xc"], s["Yc"], s["inObs"].astype(bool)
xo, yo, Lo = float(s["xObs"]), float(s["yObs"]), float(s["LObs"])

fig = plt.figure(figsize=(7.4, 2.7), constrained_layout=True)
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.15])
vmin, vmax = 0.0, 1.0
for k, (fill, title) in enumerate([("avg", "(a) neighbour-average fill (paper)"),
                                   ("frozen", "(b) no fill (frozen mask)")]):
    ax = fig.add_subplot(gs[0, k])
    F = s[f"fill_T_{fill}"].copy()
    F[inObs2] = np.nan
    pc = ax.pcolormesh(Xc2, Yc2, F, shading="gouraud", cmap="magma",
                       vmin=vmin, vmax=vmax, rasterized=True)
    ax.contour(Xc2, Yc2, F, levels=np.linspace(0, 1, 13), colors="w",
               linewidths=0.35, alpha=0.6)
    ax.add_patch(Rectangle((xo - Lo / 2, yo - Lo / 2), Lo, Lo,
                           facecolor=(0.3, 0.3, 0.3), edgecolor="k", lw=0.8,
                           zorder=5))
    ax.set_aspect("equal")
    ax.set_xlim(1.2, 5.5)
    ax.set_ylim(0.5, 3.5)
    ax.set_title(title, fontsize=8.5)
    ax.set_xlabel(r"$x$")
    if k == 0:
        ax.set_ylabel(r"$y$")
fig.colorbar(pc, ax=fig.axes[:2], shrink=0.85, pad=0.02, label=r"$\mathsf{T}$")

ax = fig.add_subplot(gs[0, 2])
for fill, st, lab in [("avg", ("-", BLUE, 2.0), "neighbour average (paper)"),
                      ("compat", ("--", "k", 1.2), "compatibility-corrected fill"),
                      ("frozen", ("-", RED, 1.4), "frozen mask (artefact)")]:
    Tm = s[f"fill_Tmid_{fill}"].copy()
    Tm[obx] = np.nan
    ax.plot(x, Tm, st[0], color=st[1], lw=st[2], label=lab)
ax.plot(x, 1 - x / 8, "-", color="0.75", lw=0.8, label=r"Fourier $\mathsf{T}_F$")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\mathsf{T}(x,\mathsf{R}/2)$")
ax.set_title("(c) centreline temperature", fontsize=8.5)
ax.legend(frameon=False, fontsize=6.5)

save(fig, "fig_resp_fill")
