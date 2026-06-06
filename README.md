# Guyer–Krumhansl thermal-wake solver

Code accompanying:

> I. Carlomagno, A. Sellitto, V. Citro,
> *Thermal shadows in non-linear phonon hydrodynamics: Stokes-like heat-flux
> wakes past a bluff body*, Submitted to Proc. R. Soc. A (2026).

Non-linear, non-local Guyer–Krumhansl heat-transport solver for a 2-D thin
nanolayer with a square obstacle, plus the script that reproduces the figures
in the paper.

## Files
- `gk_thermal_wake.py` — solver; writes `gk_wake_results.npz`
- `make_paper_figures.py` — reads the results and writes the figures

## Requirements
Python 3.9+, with `numpy`, `scipy`, `matplotlib`:
```bash
pip install numpy scipy matplotlib
```

## Run
```bash
python gk_thermal_wake.py       # ~35 s; writes gk_wake_results.npz
python make_paper_figures.py    # writes the figure files
```
Model parameters (`Kn`, grid, obstacle, `tEnd`, …) are at the top of
`gk_thermal_wake.py`.

## License
MIT — © 2026 I. Carlomagno, A. Sellitto, V. Citro.
