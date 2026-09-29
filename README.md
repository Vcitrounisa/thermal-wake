# Gk-thermal-wake

Solver and studies for the heat-flux wake past a square obstacle in a thin
layer described by the non-linear, weakly non-local Guyer–Krumhansl
equations, accompanying

> I. Carlomagno, A. Sellitto, N. Geracitano, V. Citro,
> *Thermal shadows in non-linear phonon hydrodynamics: Stokes-like heat-flux
> wakes past a bluff body*, submitted to Proc. R. Soc. A (2026).

The equations are discretised with explicit finite differences on a
staggered (MAC) grid: the temperature sits at the centres of the energy
control volumes and the heat-flux components on their faces.  Only the
physical boundary data are imposed (temperatures at the two ends,
adiabatic free-slip walls, `h = 0` on the obstacle); there is no boundary
condition on the temperature at the walls or inside the obstacle, and no
odd–even mode.  The method is described in [NUMERICS.md](NUMERICS.md).

## Quick start

```bash
pip install -r requirements.txt
./make_all.sh                  # add --with-longrun for the t = 1200 study (about 1 h)
```

`results/` contains the output of every study, so the figures can be
rebuilt without running the solver:

```bash
python make_figures.py         # -> figs/
```

## Contents

| file | |
|---|---|
| `gk_solver.py` | grid, obstacle masks, discrete operators, time integration (reduced and full three-field model) |
| `gk_kernels.py` | numba version of the time loop (optional; the solver falls back to numpy) |
| `gk_analysis.py` | post-processing: I/O, reconstruction on the T nodes, closure Q, probes, diagnostics |
| `settings.py` | physical case and grid |

| script | study | output in `results/` |
|---|---|---|
| `run_selftest.py` | numba kernel vs numpy reference on every option | – |
| `run_verification.py` | linear channel vs analytic solution (Ny = 8…64), non-linear channel vs collocation | `verification.*` |
| `run_baseline.py` | baseline wake: cold start, t = 400 | `baseline.npz` |
| `run_baseline_diagnostics.py` | positivity of h_x, ignition, thermal shadow, odd–even content | `baseline.json` |
| `run_fullq_baseline.py` | baseline with the full three-field model, ε = 0.02 | `fullq_baseline.npz` |
| `run_fullq_convergence.py` | full vs reduced model for ε = 0.02…0.2 | `fullq.*` |
| `run_domain_length.py` | steady wake with the outlet moved downstream by 4 and 8 and the inlet upstream by 4, at the same heat flux | `domain_length.*` |
| `run_ic_sensitivity.py` | independence of the initial state; inert T nodes inside the obstacle | `ic_sensitivity.*` |
| `run_longrun.py` | independence of the initial state at t = 1200 (opt-in) | `longrun*` |
| `make_figures.py` | all figures | `figs/` |

## Figures

| file | content |
|---|---|
| `fig_evolution` | build-up of the wake: \|h\| and streamlines at t = 5, 30, 100, 400 |
| `fig_steady` | quasi-stationary temperature and heat flux |
| `fig_zoom_centreline` | isotherms and flux around the obstacle, centre-line profiles |
| `fig_verification` | channel benchmarks, grid convergence of the channel, reduction of the full model |
| `fig_Q` | flux of the heat flux: closure vs full model |
| `fig_initial_state` | independence of the initial state |
| `fig_enclosed_T` | the T nodes inside the obstacle do not affect the solution |
| `fig_obstacle_nodes` | staggered nodes around a corner of the obstacle |

## Main results

| quantity | value | file |
|---|---|---|
| linear channel, observed orders (Ny = 8→64) | 1.95, 1.97, 1.99 | `verification.json` |
| non-linear channel vs collocation (min Φ = 0.61) | 2.8e-04 | `verification.json` |
| full vs reduced model, slope in ε (t = 2, 5, 18) | 0.99–1.11 | `fullq.json` |
| baseline max \|h\| at t = 400 | 0.1512 | `baseline.json` |
| baseline min h_x at t = 400 (fluid) | +2.27e-04 | `baseline.json` |
| odd–even content of T (largest at the obstacle corners) | 9.8e-03 | `baseline.json` |
| cold vs Fourier start, max \|ΔT\| at t = 400 / 1200 | 9.9e-05 / 1.4e-11 | `ic_sensitivity.json`, `longrun.json` |
| Fourier start with h = 0 vs h = G e_x, max \|ΔT\| / max \|Δh\| at t = 400 | 3.6e-07 / 1.2e-07 | `ic_sensitivity.json` |
| enclosed T nodes set to 0, T_F or 1: max fluid \|ΔT\| | 0.0 | `ic_sensitivity.json` |
| outlet moved downstream by 4 / 8, largest change of h/Q around the obstacle | 5.9e-05 / 6.3e-05 | `domain_length.json` |
| inlet moved upstream by 4, largest change of h/Q in the wake | 1.5e-05 | `domain_length.json` |

## Grid

All obstacle runs use a 200×100 grid (Δx = Δy = 0.04).  The discretisation
is verified on the channel problem, where the linear solution converges at
second order under grid refinement.

## Requirements

Python ≥ 3.9 with numpy, scipy, matplotlib; numba is optional but strongly
recommended (the numpy path is several times slower).  The distributed
results were produced with Python 3.11, numpy 2.4, scipy 1.17,
matplotlib 3.10 and numba 0.67 on Linux x86-64.

## License

MIT, see [LICENSE](LICENSE).
