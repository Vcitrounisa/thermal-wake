"""
Common settings of the numerical studies.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figs"

# physical configuration
CASE = dict(Kn=0.7, Lx=8.0, Ly=4.0, x_obs=3.0, side=1.0, T_hot=1.0)

# grid of all obstacle runs
BASELINE = dict(Nx=200, Ny=100)

# end time of the short (transient) comparison runs
T_SHORT = 18.0


def results_path(name):
    RESULTS.mkdir(exist_ok=True)
    return RESULTS / name
