"""
run_fullQ_coldstart.py
======================

Full three-field run (independent Q tensor, eps = tau_Q/tau_R = 0.02) from
the cold-start initial state, integrated to t = 400,
for the closure-vs-full overlay of the Q figure at the paper baseline.

Writes fullQ_coldstart.npz.
"""

import numpy as np

from gk_solver import solve

r = solve(ic="cold-rest", full_Q=True, eps_Q=0.02, tEnd=400.0,
          tSnap=(400.0,), diag_every=500, verbose=False)
np.savez("fullQ_coldstart.npz",
         T=r["T_final"], hx=r["hx_final"], hy=r["hy_final"],
         Qxx=r["Qxx"], Qxy=r["Qxy"], Qyx=r["Qyx"], Qyy=r["Qyy"],
         inObs=r["inObs"], xc=r["xc"], yc=r["yc"])
print("done")
