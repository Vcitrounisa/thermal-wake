# Guyer–Krumhansl thermal-wake solver

Code and data accompanying:

> I. Carlomagno, A. Sellitto, N. Geracitano, V. Citro,
> *Thermal shadows in non-linear phonon hydrodynamics: Stokes-like heat-flux
> wakes past a bluff body*, submitted to Proc. R. Soc. A (2026).

Non-linear, weakly non-local Guyer–Krumhansl heat-transport solver for a 2-D
thin nanolayer with a square obstacle, together with the scripts that
regenerate every figure and every quantitative claim in the paper.

Everything here is deterministic: the scheme is explicit, there is no random
input, and re-running the pipeline reproduces the distributed `.npz`/`.json`
files bit for bit on the same platform.

## Quick start

```bash
pip install -r requirements.txt
./make_all.sh                  # add --with-longrun for the t = 1200 study
```

`make_all.sh` runs the scripts in dependency order.

The archive ships the pre-computed `*_results.npz`, so both
figure scripts run without repeating any
integration:

```bash
python make_paper_figures.py   # Figs. 3, 4, 5  -> ./figs
python make_new_figures.py     # Figs. 2, 6, 7  -> ./figs
```

Delete the `.npz` files (and run `make_all.sh`) to regenerate them from
scratch instead.

## Files

### Solver

| file | contents |
|---|---|
| `gk_solver.py` | library: reduced GK model and full three-field model (independent flux-of-heat-flux tensor `Q`, face-centred, exact exponential relaxation update); initial-condition, side-wall (free-slip / no-slip / second-order slip) and obstacle-fill options; analytic channel solutions |
| `gk_kernels.py` | numba-compiled time loop (optional; the solver falls back to the numpy reference implementation when numba is unavailable) |

The two implementations agree to machine precision on
every model variant used in this work.

### Runs

Each writes a `*_results.npz` and, where numbers are quoted in the paper, a
`*_summary.json`.

| script | what it does | outputs |
|---|---|---|
| `gk_thermal_wake.py` | baseline switch-on wake of the paper (cold start, `t_end = 400`, 200×100) | `gk_wake_results.npz` |
| `run_baseline_diagnostics.py` | post-processes the baseline run: positivity and thermal-shadow diagnostics | `baseline_coldstart_summary.json` |
| `run_fullQ_coldstart.py` | full three-field run at the baseline, `eps = 0.02` (symbols of Fig. 6c) | `fullQ_coldstart.npz` |
| `run_selftest.py` | numpy vs numba cross-check (machine precision) on all model variants | stdout only |
| `run_verification.py` | analytic linear-channel benchmark + grid convergence, non-linear channel vs collocation, wake self-convergence, domain-length robustness | `verification_*` |
| `run_ic_sensitivity.py` | initial-condition independence; obstacle temperature-extension study (average / frozen / compatibility fill) | `ic_sensitivity_*` |
| `run_fullQ.py` | full three-field system vs reduced model: `O(tau_Q/tau_R)` transient convergence, `Q` fields | `fullQ_*` |
| `run_experiment.py` | effective conductivity of a strip with slip walls: closed-form `kappa_eff` validation and curves vs `Kn` (graphite-ribbon comparison) | `experiment_*` |
| `run_wake_probe.py` | grid- and domain-convergence of smooth pointwise wake diagnostics | `wake_probe_*` |
| `run_longrun.py` | long-time IC independence (`t = 1200`) and fully converged 400×200 steady state | `longrun_summary.json`, `ic_long_*.npz`, `wake_fine_relaxed.npz` |

### Figures

| script | paper figures |
|---|---|
| `make_new_figures.py` | **Fig. 2** (`fig_verification`), **Fig. 6** (`fig_Q`), **Fig. 7** (`fig_experiment`), plus the two response-letter figures `fig_resp_ic`, `fig_resp_fill` |
| `make_paper_figures.py` | **Fig. 3** (`fig_evolution`), **Fig. 4** (`fig_steady`), **Fig. 5** (`fig_zoom_centreline`) |

All figures are written to `./figs/` as both `.pdf` and `.png`.


## Requirements

Python 3.9+ with `numpy`, `scipy`, `matplotlib`, and optionally `numba`
(see `requirements.txt`).

## License

MIT - © 2026 I. Carlomagno, A. Sellitto, N. Geracitano, V. Citro. See `LICENSE`.
