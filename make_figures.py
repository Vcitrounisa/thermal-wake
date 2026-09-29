"""
Figures from the results in ./results, written to ./figs as PDF and PNG.

  fig_evolution        build-up of the wake (|h| and streamlines)
  fig_steady           quasi-stationary T and |h|
  fig_zoom_centreline  zoom around the obstacle and centre-line profiles
  fig_verification     channel benchmarks and reduction of the full model
  fig_Q                flux of the heat flux: closure vs full model
  fig_initial_state    independence of the initial state
  fig_enclosed_T       the T nodes inside the obstacle are inert
  fig_obstacle_nodes   staggered nodes around a corner of the obstacle

Each figure is skipped if the results it needs are missing.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt

from gk_analysis import corners_on_T_nodes, load_result
from gk_solver import flux_on_T_nodes, make_grid, obstacle_masks
from settings import BASELINE, CASE, FIGURES, RESULTS

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 110,
})
BLUE, RED, GREY = "#1f4e9a", "#c0392b", "0.45"
FIGURES.mkdir(exist_ok=True)


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}")


def available(*names):
    missing = [n for n in names if not (RESULTS / n).exists()]
    if missing:
        print(f"skipping: missing {', '.join(missing)}")
    return not missing


def add_obstacle(ax, r, fc=(0.30, 0.30, 0.30)):
    a = 0.5 * r["side"]
    ax.add_patch(Rectangle((r["x_obs"] - a, r["y_obs"] - a), r["side"], r["side"],
                           facecolor=fc, edgecolor="k", linewidth=0.9, zorder=5))


def masked(F, mask):
    G = np.array(F, dtype=float)
    G[mask] = np.nan
    return G


def fill_enclosed(F, enclosed):
    
    """Fill the enclosed nodes with the nearest value (plotting only)."""

    idx = distance_transform_edt(enclosed, return_distances=False, return_indices=True)
    return np.asarray(F)[tuple(idx)]


def seeds(r, n):
    y = np.linspace(0.10, r["Ly"] - 0.10, n)
    return np.column_stack([np.full_like(y, 0.05), y])


def flux_lines(r, hx, hy, starts, ds=None, max_length=None):
    """Field lines of h through the points ``starts``.

    Each component is interpolated bilinearly on its own staggered nodes and
    the unit vector h/|h| is integrated with RK4, forward and backward.  A
    line ends on the boundary of the domain or on the surface of the obstacle
    (the last step is cut there by bisection), or where h vanishes or turns
    back.  Returns a list of (n, 2) arrays ordered along h.
    """
    ds = 0.25 * min(r["dx"], r["dy"]) if ds is None else ds
    max_length = 3.0 * (r["Lx"] + r["Ly"]) if max_length is None else max_length
    fx = RegularGridInterpolator((r["y_T"], r["x_hx"]), hx)
    fy = RegularGridInterpolator((r["y_hy"], r["x_T"]), hy)
    lo = np.array([r["x_hx"][0], r["y_T"][0]])
    hi = np.array([r["x_hx"][-1], r["y_T"][-1]])
    a = 0.5 * r["side"]
    tiny = 1e-9 * max(np.abs(hx).max(), np.abs(hy).max())

    def outside(P):
        in_box = np.all((P >= 0.0) & (P <= [r["Lx"], r["Ly"]]), axis=1)
        in_obs = (np.abs(P[:, 0] - r["x_obs"]) <= a) & (np.abs(P[:, 1] - r["y_obs"]) <= a)
        return ~in_box | in_obs

    def direction(P):
        Q = np.clip(P, lo, hi)[:, ::-1]
        h = np.column_stack([fx(Q), fy(np.clip(P, 0.0, [r["Lx"], r["Ly"]])[:, ::-1])])
        s = np.hypot(h[:, 0], h[:, 1])
        return h / np.where(s > tiny, s, np.inf)[:, None]

    def trace(sign):
        P = np.array(starts, dtype=float)
        paths = [[p.copy()] for p in P]
        alive = np.ones(len(P), bool)
        u_old = sign * direction(P)
        for _ in range(int(max_length / ds)):
            if not alive.any():
                break
            idx = np.flatnonzero(alive)
            p = P[idx]
            k1 = sign * direction(p)
            k2 = sign * direction(p + 0.5 * ds * k1)
            k3 = sign * direction(p + 0.5 * ds * k2)
            k4 = sign * direction(p + ds * k3)
            step = ds * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            q = p + step
            stop = (np.hypot(*k1.T) == 0) | (np.einsum("ij,ij->i", k1, u_old[idx]) < 0)
            hit = outside(q) & ~stop
            if hit.any():
                s0, s1 = np.zeros(hit.sum()), np.ones(hit.sum())
                for _ in range(30):
                    sm = 0.5 * (s0 + s1)
                    bad = outside(p[hit] + sm[:, None] * step[hit])
                    s1 = np.where(bad, sm, s1)
                    s0 = np.where(bad, s0, sm)
                q[hit] = p[hit] + s0[:, None] * step[hit]
            for n, i in enumerate(idx):
                if not stop[n]:
                    paths[i].append(q[n].copy())
            P[idx], u_old[idx] = q, k1
            alive[idx[stop | hit]] = False
        return [np.array(pth) for pth in paths]

    forward, backward = trace(+1.0), trace(-1.0)
    return [np.vstack([bw[::-1], fw[1:]]) for fw, bw in zip(forward, backward)]


def draw_flux_lines(ax, lines, color="white", lw=0.55, arrowsize=0.7):
    """Lines with one arrow half way along each."""
    for L in lines:
        if len(L) < 3:
            continue
        ax.plot(L[:, 0], L[:, 1], color=color, lw=lw, zorder=4, solid_capstyle="butt")
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(L, axis=0).T))])
        m = min(int(np.searchsorted(s, 0.5 * s[-1])), len(L) - 2)
        ax.add_patch(FancyArrowPatch(L[m], L[m + 1], arrowstyle="-|>", color=color, lw=lw,
                                     mutation_scale=10 * arrowsize, zorder=4))


def frame(ax, r, xlabel=True):
    ax.set_aspect("equal")
    ax.set_xlim(0, r["Lx"])
    ax.set_ylim(0, r["Ly"])
    ax.set_ylabel(r"$y$")
    if xlabel:
        ax.set_xlabel(r"$x$")


# baseline figures
if available("baseline.npz"):
    b = load_result(RESULTS / "baseline.npz")
    X, Y = np.meshgrid(b["x_T"], b["y_T"])
    x, y = b["x_T"], b["y_T"]
    obs, enc = b["in_obstacle"], b["enclosed_T"]
    hx, hy = flux_on_T_nodes(b["hx"], b["hy"])
    T = b["T"]

    # build-up of the wake
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6), constrained_layout=True)
    panels = [int(np.argmin(np.abs(b["snap_t"] - t))) for t in (5.0, 30.0, 100.0, 400.0)]
    vmax = np.hypot(hx, hy).max()
    for k, ax in zip(panels, axes.flat):
        sx, sy = flux_on_T_nodes(b["snap_hx"][k], b["snap_hy"][k])
        pc = ax.pcolormesh(X, Y, np.hypot(sx, sy), shading="gouraud",
                           cmap="viridis", vmin=0, vmax=vmax, rasterized=True)
        draw_flux_lines(ax, flux_lines(b, b["snap_hx"][k], b["snap_hy"][k], seeds(b, 13)))
        add_obstacle(ax, b)
        frame(ax, b)
        ax.set_title(rf"$t = {b['snap_t'][k]:.0f}$")
    fig.colorbar(pc, ax=axes, location="right", shrink=0.85, pad=0.02).set_label(r"$|\mathbf{h}|$")
    save(fig, "fig_evolution")

    # quasi-stationary state
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 4.5), constrained_layout=True, sharex=True)
    levels = np.linspace(0, 1, 25)
    Tm = fill_enclosed(T, enc)
    ax = axes[0]
    cf = ax.contourf(X, Y, Tm, levels=levels, cmap="magma", extend="max")
    ax.contour(X, Y, Tm, levels=levels[::3], colors="white", linewidths=0.4, alpha=0.7)
    add_obstacle(ax, b)
    frame(ax, b, xlabel=False)
    ax.set_title(r"(a) temperature $T$")
    fig.colorbar(cf, ax=ax, shrink=0.9, pad=0.02).set_label(r"$T$")
    ax = axes[1]
    pc = ax.pcolormesh(X, Y, np.hypot(hx, hy), shading="gouraud",
                       cmap="viridis", rasterized=True)
    draw_flux_lines(ax, flux_lines(b, b["hx"], b["hy"], seeds(b, 15)))
    add_obstacle(ax, b)
    frame(ax, b)
    ax.set_title(r"(b) heat-flux magnitude $|\mathbf{h}|$ and streamlines")
    fig.colorbar(pc, ax=ax, shrink=0.9, pad=0.02).set_label(r"$|\mathbf{h}|$")
    save(fig, "fig_steady")

    # zoom and centre-line profiles
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.05, 1.0]})
    ax = axes[0]
    cf = ax.contourf(X, Y, Tm, levels=np.linspace(0, 1, 25), cmap="magma", extend="max",
                     zorder=0)
    ax.contour(X, Y, Tm, levels=np.linspace(0, 1, 13), colors="white", linewidths=0.4, alpha=0.55)
    step = 7
    qx, qy = masked(hx, obs)[::step, ::step], masked(hy, obs)[::step, ::step]
    ax.quiver(X[::step, ::step], Y[::step, ::step], qx, qy, scale=1.4, scale_units="xy",
              width=0.0035, headwidth=4, headlength=4.5, color="white", alpha=0.95, zorder=4)
    add_obstacle(ax, b)
    ax.set_aspect("equal")
    ax.set_xlim(b["x_obs"] - 1.2, b["x_obs"] + 3.0)
    ax.set_ylim(b["y_obs"] - 1.5, b["y_obs"] + 1.5)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title("(a) heat-flux deflection and isotherms")
    fig.colorbar(cf, ax=ax, shrink=0.9, pad=0.02).set_label(r"$T$")

    ax = axes[1]
    jc = int(np.argmin(np.abs(y - b["y_obs"])))
    jo = int(np.argmin(np.abs(y - (b["y_obs"] + 0.5 * b["side"] + 0.10))))
    inside = np.abs(x - b["x_obs"]) <= 0.5 * b["side"]
    T_line, hx_line = T[jc].copy(), hx[jc].copy()
    T_line[inside] = hx_line[inside] = np.nan
    ax.axvspan(b["x_obs"] - 0.5 * b["side"], b["x_obs"] + 0.5 * b["side"], color="0.88", lw=0)
    ax.plot(x, 1.0 - x / b["Lx"], "--", color=GREY, lw=1.0, label=r"Fourier profile $T_F$")
    ax.plot(x, T[jo], "-.", color=BLUE, lw=1.2, alpha=0.65, label=rf"$T(x, y={y[jo]:.2f})$")
    ax.plot(x, T_line, "-", color=BLUE, lw=2.0, label=rf"$T(x, y={y[jc]:.2f})$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$T$", color=BLUE)
    ax.tick_params(axis="y", labelcolor=BLUE)
    ax.set_ylim(-0.05, 1.10)
    ax.set_xlim(0, b["Lx"])
    ax.legend(loc="upper right", frameon=False, fontsize=7.5)
    ax.set_title(r"(b) centre-line profiles of $T$ and $h_x$")
    ax2 = ax.twinx()
    ax2.axhline(0.0, color="0.7", lw=0.6)
    ax2.plot(x, hx_line, "-", color=RED, lw=1.4)
    ax2.set_ylabel(r"$h_x$", color=RED)
    ax2.tick_params(axis="y", labelcolor=RED)
    ax2.set_ylim(-0.2 * np.nanmax(hx_line), 1.15 * np.nanmax(hx_line))
    save(fig, "fig_zoom_centreline")

# verification and reduction of the full model
if available("verification.npz", "fullq.npz"):
    v = np.load(RESULTS / "verification.npz")
    q = np.load(RESULTS / "fullq.npz")
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.6), constrained_layout=True)
    ax = axes[0]
    ax.plot(v["y_lin"], v["h_exact"] / v["G_lin"], "-", color=BLUE, lw=1.4, label="linear, exact")
    ax.plot(v["y_lin"][::4], v["h_lin"][::4] / v["G_lin"], "o", ms=3.2, mfc="none",
            color=BLUE, label="linear, 2-D solver")
    ax.plot(v["y_nl"], v["h_bvp"] / v["G_nl"], "-", color=RED, lw=1.4,
            label="non-linear, collocation")
    ax.plot(v["y_nl"][::4], v["h_nl"][::4] / v["G_nl"], "s", ms=3.2, mfc="none",
            color=RED, label="non-linear, 2-D solver")
    ax.set_xlabel(r"$y$")
    ax.set_ylabel(r"$h_x/G$")
    ax.set_title("(a) channel benchmarks", fontsize=9)
    ax.legend(frameon=False, fontsize=6.5, loc="lower center")

    ax = axes[1]
    n = v["grids"].astype(float)
    ax.loglog(n, v["errors"], "o-", color=BLUE, lw=1.2, ms=4, label="linear channel")
    ax.loglog(n, v["errors"][0] * (n[0] / n) ** 2, "--", color=GREY, lw=1.0, label="slope $-2$")
    ax.set_xticks(n)
    ax.set_xticklabels([f"{int(k)}" for k in n])
    ax.minorticks_off()
    ax.set_xlabel(r"$N_y$")
    ax.set_ylabel("max. relative error")
    ax.set_title(r"(b) grid convergence, $\Phi = 1$", fontsize=9)
    ax.legend(frameon=False)

    ax = axes[2]
    eps = q["eps"]
    for k, (mk, c) in enumerate((("o", BLUE), ("s", RED))):
        ax.loglog(eps, q["err_h"][k], mk + "-", ms=4, lw=1.2, color=c,
                  label=rf"$t = {q['times'][k]:g}$")
    ax.loglog(eps, q["err_h"][0][-1] * eps / eps[-1], "--", color=GREY, lw=1.0, label="slope $+1$")
    ax.set_xlabel(r"$\varepsilon = \tau_Q/\tau_R$")
    ax.set_ylabel(r"$\max|\mathbf{h}_{\rm full} - \mathbf{h}_{\rm red}|$")
    ax.set_title("(c) reduction of the full model", fontsize=9)
    ax.legend(frameon=False)
    save(fig, "fig_verification")

# flux of the heat flux
if available("baseline.npz", "fullq_baseline.npz"):
    b = load_result(RESULTS / "baseline.npz")
    f = load_result(RESULTS / "fullq_baseline.npz")
    X, Y = np.meshgrid(b["x_T"], b["y_T"])
    enc = b["enclosed_T"]
    closure ={"Qxx": b["closure_Qxx"], "Qyx": corners_on_T_nodes(b["closure_Qyx"])}
    full = {"Qxx": f["Qxx"], "Qyx": corners_on_T_nodes(f["Qyx"])}

    fig = plt.figure(figsize=(7.6, 2.9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0])
    components = ((r"$\mathcal{Q}_{xx}$", "Qxx"), (r"$\mathcal{Q}_{yx}$", "Qyx"))
    for row, (label, key) in enumerate(components):
        ax = fig.add_subplot(gs[row, 0])
        F = fill_enclosed(closure[key], enc)
        vmax = np.abs(F).max()
        pc = ax.pcolormesh(X, Y, F, shading="gouraud", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                           rasterized=True)
        add_obstacle(ax, b)
        ax.set_aspect("equal")
        ax.set_xlim(0.8, 7.2)
        ax.set_ylim(0, b["Ly"])
        ax.set_ylabel(r"$y$")
        if row == 1:
            ax.set_xlabel(r"$x$")
        ax.set_title(("(a) " if row == 0 else "(b) ") + label, fontsize=9)
        fig.colorbar(pc, ax=ax, shrink=0.9, pad=0.02).ax.tick_params(labelsize=7)

    ax = fig.add_subplot(gs[:, 1])
    x = b["x_T"]
    jc = b["Ny"] // 2
    jo = int(jc + 0.15 * b["Ny"])
    every = max(1, b["Nx"] // 40)
    ax.plot(x, closure["Qxx"][jc], "-", color=BLUE, lw=1.4,
            label=rf"$\mathcal{{Q}}_{{xx}}(y={b['y_T'][jc]:.2f})$, closure")
    ax.plot(x[::every], full["Qxx"][jc, ::every], "o", ms=3, mfc="none", color=BLUE,
            label=r"full model, $\varepsilon = 0.02$")
    ax.plot(x, closure["Qyx"][jo], "-", color=RED, lw=1.4,
            label=rf"$\mathcal{{Q}}_{{yx}}(y={b['y_T'][jo]:.2f})$, closure")
    ax.plot(x[::every], full["Qyx"][jo, ::every], "s", ms=3, mfc="none", color=RED,
            label=r"full model, $\varepsilon = 0.02$")
    ax.axvspan(b["x_obs"] - 0.5 * b["side"], b["x_obs"] + 0.5 * b["side"],
               color="0.90", lw=0, zorder=0)
    ax.axhline(0, color="0.75", lw=0.6)
    ax.set_xlim(0, b["Lx"])
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\mathcal{Q}$")
    ax.set_title("(c) closure vs full model", fontsize=9)
    ax.legend(frameon=False, fontsize=6.5)
    save(fig, "fig_Q")

# initial state and enclosed nodes
if available("ic_sensitivity.npz"):
    s = np.load(RESULTS / "ic_sensitivity.npz")
    x = s["x"]
    inside = np.abs(x - CASE["x_obs"]) <= 0.5 * CASE["side"]
    labels = {"cold": r"$T = 0$, $\mathbf{h} = 0$ (baseline)",
              "fourier": r"$T = T_F$, $\mathbf{h} = 0$",
              "fourier-flux": r"$T = T_F$, $\mathbf{h} = G\,\hat{\mathbf{e}}_x$"}
    styles = {"cold": ("-", BLUE, 2.2), "fourier": ("--", RED, 1.4),
              "fourier-flux": (":", "k", 1.4)}

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), constrained_layout=True)
    for ic, lab in labels.items():
        ls, col, lw = styles[ic]
        Tm = s[f"T_mid_{ic}"].copy()
        Tm[inside] = np.nan
        axes[0].plot(x, Tm, ls, color=col, lw=lw, label=lab)
        axes[1].plot(s[f"hist_t_{ic}"], s[f"hist_max_h_{ic}"], ls, color=col, lw=lw, label=lab)
    axes[0].plot(x, 1.0 - x / CASE["Lx"], "-", color="0.75", lw=0.8, label=r"$T_F$")
    axes[0].set_xlabel(r"$x$")
    axes[0].set_ylabel(r"$T(x, R/2)$")
    axes[0].set_title("(a) centre-line temperature at $t = 400$", fontsize=9)
    axes[0].legend(frameon=False, fontsize=6.5)
    axes[1].set_xlim(0, 200)
    axes[1].set_xlabel(r"$t$")
    axes[1].set_ylabel(r"$\max|\mathbf{h}|$")
    axes[1].set_title("(b) approach to the quasi-stationary state", fontsize=9)
    axes[1].legend(frameon=False, fontsize=6.5, loc="upper right")
    save(fig, "fig_initial_state")

    X, Y = np.meshgrid(x, s["y"])
    obs, enc = s["in_obstacle"].astype(bool), s["enclosed"].astype(bool)
    a = 0.5 * CASE["side"]
    fig = plt.figure(figsize=(7.4, 2.7), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.15])
    for k, (name, title) in enumerate((("zero", r"(a) enclosed nodes $T = 0$"),
                                       ("one", r"(b) enclosed nodes $T = 1$"))):
        ax = fig.add_subplot(gs[0, k])
        F = s[f"enclosed_T_{name}"]
        pc = ax.pcolormesh(X, Y, masked(F, enc), shading="gouraud", cmap="magma",
                           vmin=0, vmax=1, rasterized=True)
        ax.contour(X, Y, masked(F, enc), levels=np.linspace(0, 1, 13), colors="w",
                   linewidths=0.35, alpha=0.6)
        jj, ii = np.indices(F.shape)
        dots = enc & (ii % 4 == 0) & (jj % 4 == 0)
        ax.scatter(X[dots], Y[dots], c=F[dots], cmap="magma", vmin=0, vmax=1,
                   s=4, zorder=6, edgecolors="none")
        corner = (CASE["x_obs"] - a, 0.5 * CASE["Ly"] - a)
        ax.add_patch(Rectangle(corner, CASE["side"], CASE["side"],
                               facecolor="none", edgecolor="k", lw=0.8, zorder=7))
        ax.set_aspect("equal")
        ax.set_xlim(1.2, 5.5)
        ax.set_ylim(0.5, 3.5)
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel(r"$x$")
        if k == 0:
            ax.set_ylabel(r"$y$")
    fig.colorbar(pc, ax=fig.axes[:2], shrink=0.85, pad=0.02, label=r"$T$")
    ax = fig.add_subplot(gs[0, 2])
    jc = s["y"].size // 2
    for name, (ls, col, lw), lab in (("zero", ("-", BLUE, 2.4), r"enclosed $T = 0$"),
                                     ("fourier", ("--", "k", 1.3), r"enclosed $T = T_F$"),
                                     ("one", (":", RED, 1.6), r"enclosed $T = 1$")):
        Tm = s[f"enclosed_T_{name}"][jc].copy()
        Tm[inside] = np.nan
        ax.plot(x, Tm, ls, color=col, lw=lw, label=lab)
    diff = max(np.abs(s[f"enclosed_T_{n}"] - s["enclosed_T_zero"])[~obs].max()
               for n in ("fourier", "one"))
    ax.text(0.03, 0.05, rf"max fluid $|\Delta T| = {diff:.1e}$", transform=ax.transAxes, fontsize=7)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$T(x, R/2)$")
    ax.set_title("(c) centre-line temperature", fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.5)
    save(fig, "fig_enclosed_T")

# staggered nodes around the lower-left corner of the obstacle
g = make_grid(CASE["Lx"], CASE["Ly"], BASELINE["Nx"], BASELINE["Ny"])
y_obs = 0.5 * CASE["Ly"]
sx, sy, enc, ins = obstacle_masks(g, CASE["x_obs"], y_obs, CASE["side"])
open_faces = np.zeros(enc.shape, int)
open_faces[:, 1:-1] = ((~sx[:, :-1]).astype(int) + (~sx[:, 1:])
                       + (~sy[:-1, 1:-1]) + (~sy[1:, 1:-1]))
pocket = ins & (open_faces == 1) & ~enc
x_edge, y_edge = CASE["x_obs"] - 0.5 * CASE["side"], y_obs - 0.5 * CASE["side"]
d = 4.3 * g["dx"]
x0, x1, y0, y1 = x_edge - d, x_edge + d, y_edge - d, y_edge + d
ink, soft, solid = "#0b0b0b", "#9a9994", "#3a3a38"

fig, ax = plt.subplots(figsize=(7.0, 5.2))
ax.add_patch(Rectangle((x_edge, y_edge), CASE["side"], CASE["side"],
                       facecolor="#e6e5e0", edgecolor=ink, lw=1.6, zorder=1))
for xf in g["x_hx"]:
    ax.axvline(xf, color="#d6d5cf", lw=0.6, zorder=0)
for yf in g["y_hy"]:
    ax.axhline(yf, color="#d6d5cf", lw=0.6, zorder=0)


def window(Xn, Yn):
    return (Xn > x0) & (Xn < x1) & (Yn > y0) & (Yn < y1)


XT, YT = np.meshgrid(g["x_T"], g["y_T"])
Xu, Yu = np.meshgrid(g["x_hx"], g["y_T"])
Xv, Yv = np.meshgrid(g["x_T"], g["y_hy"])
m = window(XT, YT)
ax.scatter(XT[m & ~enc & ~pocket], YT[m & ~enc & ~pocket], s=70, facecolor="white",
           edgecolor=ink, lw=1.1, zorder=4)
ax.scatter(XT[m & enc], YT[m & enc], s=70, facecolor=soft, edgecolor=soft, zorder=4)
ax.scatter(XT[m & pocket], YT[m & pocket], s=80, facecolor="#eb6834", edgecolor=ink,
           lw=0.8, zorder=4)
for Xn, Yn, S, mk in ((Xu, Yu, sx, ">"), (Xv, Yv, sy, "^")):
    m = window(Xn, Yn)
    ax.scatter(Xn[m & ~S], Yn[m & ~S], s=60, marker=mk, facecolor="white", edgecolor=ink,
               lw=1.0, zorder=5)
    ax.scatter(Xn[m & S], Yn[m & S], s=60, marker=mk, facecolor=solid, edgecolor=solid, zorder=5)
# where each component effectively meets the obstacle
tol = 1e-9 * g["dx"]
walls = []
x_hy = g["x_T"][sy.any(axis=0)].min()          # tangential h_y on the vertical edge
if x_hy - x_edge > tol:
    ax.plot([x_hy] * 2, [y_edge, y1], ls=(0, (4, 2)), color="#2a78d6", lw=1.6, zorder=3)
    walls.append(Line2D([], [], color="#2a78d6", ls=(0, (4, 2)), lw=1.6,
                        label=r"effective no-slip wall of $h_y$"))
y_hx = g["y_T"][sx.any(axis=1)].min()          # tangential h_x on the horizontal edge
if y_hx - y_edge > tol:
    ax.plot([x_edge, x1], [y_hx] * 2, ls=(0, (4, 2)), color="#2a78d6", lw=1.6, zorder=3)
    walls.append(Line2D([], [], color="#2a78d6", ls=(0, (4, 2)), lw=1.6,
                        label=r"effective no-slip wall of $h_x$"))
if np.any(np.abs(YT[pocket] - y_edge) < tol):  # pockets along the horizontal edge
    ax.plot([x_edge, x1], [y_edge - 0.5 * g["dy"]] * 2, ls=(0, (1, 1.5)), color="#eb6834",
            lw=1.8, zorder=3)
    walls.append(Line2D([], [], color="#eb6834", ls=(0, (1, 1.5)), lw=1.8,
                        label="effective impermeable wall of $h_y$\n(steady state)"))
if np.any(np.abs(XT[pocket] - x_edge) < tol):  # pockets along the vertical edge
    ax.plot([x_edge - 0.5 * g["dx"]] * 2, [y_edge, y1], ls=(0, (1, 1.5)), color="#eb6834",
            lw=1.8, zorder=3)
    walls.append(Line2D([], [], color="#eb6834", ls=(0, (1, 1.5)), lw=1.8,
                        label="effective impermeable wall of $h_x$\n(steady state)"))
ax.set_xlim(x0, x1)
ax.set_ylim(y0, y1)
ax.set_aspect("equal")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_title(f"Lower-left corner of the obstacle, {BASELINE['Nx']}x{BASELINE['Ny']} grid",
             fontsize=10)
handles = [
    Line2D([], [], marker="o", ls="", mfc="white", mec=ink, ms=8, label="T node"),
    Line2D([], [], marker="o", ls="", mfc="#eb6834", mec=ink, ms=8,
           label="T node with one open face"),
    Line2D([], [], marker="o", ls="", mfc=soft, mec=soft, ms=8,
           label="enclosed T node"),
    Line2D([], [], marker=">", ls="", mfc="white", mec=ink, ms=8, label=r"$h_x$ node"),
    Line2D([], [], marker=">", ls="", mfc=solid, mec=solid, ms=8, label=r"solid $h_x$ node ($=0$)"),
    Line2D([], [], marker="^", ls="", mfc="white", mec=ink, ms=8, label=r"$h_y$ node"),
    Line2D([], [], marker="^", ls="", mfc=solid, mec=solid, ms=8, label=r"solid $h_y$ node ($=0$)"),
    Rectangle((0, 0), 1, 1, facecolor="#e6e5e0", edgecolor=ink, label="obstacle"),
    Line2D([], [], color="#d6d5cf", lw=0.8, label="faces of the energy volumes"),
] + walls
ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8)
save(fig, "fig_obstacle_nodes")
