"""
run_fullQ.py
============

Validation of the quasi-steady elimination of the flux-of-heat-flux tensor Q

Integrates the FULL three-field system

    dT/dt + (Kn^2/3) div h = 0
    Phi dh/dt + h + Phi grad T + Phi div Q = 0
    eps Phi dQ/dt + Q + Kn^2 Phi grad h = 0        eps = tau_Q / tau_R

for a sequence of relaxation-time ratios eps, and compares it with the
reduced (closed) GK model used in the paper (formally eps -> 0).


Along the TRANSIENT (t = 2, 5) the distance between full and reduced
solutions scales as O(eps) -- the singular-perturbation estimate

Also stores the Q fields of the full model and of the closure
Q = -Kn^2 Phi grad h for the paper figure.

Writes fullQ_results.npz + fullQ_summary.json.
"""

import json
from pathlib import Path

import numpy as np

from gk_solver import solve, Q_from_closure

here = Path(__file__).resolve().parent
tS = (2.0, 5.0, 18.0)

print("--- reduced (closed) model")
red = solve(ic="fourier-rest", tEnd=18.0, tSnap=tS, verbose=False)
fluid = ~red["inObs"]
Qc = Q_from_closure(red)

eps_list = [0.02, 0.05, 0.1, 0.2]
err_h = {t: [] for t in tS}
err_T = {t: [] for t in tS}
errs_Q = []
full_runs = {}
for eps in eps_list:
    print(f"--- full model, eps = {eps}")
    r = solve(ic="fourier-rest", tEnd=18.0, full_Q=True, eps_Q=eps, tSnap=tS, verbose=False)
    full_runs[eps] = r
    for k, t in enumerate(tS):
        eh = float(np.max(np.hypot(r["snap_hx"][k] - red["snap_hx"][k],
                                   r["snap_hy"][k] - red["snap_hy"][k])[fluid]))
        eT = float(np.max(np.abs(r["snap_T"][k] - red["snap_T"][k])[fluid]))
        err_h[t].append(eh)
        err_T[t].append(eT)
    eQ = float(np.max(np.abs(r["Qxx"] - Qc["Qxx"])[fluid]))
    errs_Q.append(eQ)
    print("    " + "  ".join(f"errh(t={t}) = {err_h[t][-1]:.3e}" for t in tS)
          + f"  errQxx(final) = {eQ:.3e}")

href = float(np.max(np.hypot(red["hx_final"], red["hy_final"])))
out = dict(
    eps=np.array(eps_list),
    tS=np.array(tS),
    err_h=np.array([err_h[t] for t in tS]),   # shape (n_times, n_eps)
    err_T=np.array([err_T[t] for t in tS]),
    err_Q=np.array(errs_Q),
    href=href,
    Xc=red["Xc"], Yc=red["Yc"], inObs=red["inObs"],
    xObs=red["xObs"], yObs=red["yObs"], LObs=red["LObs"],
    Lx=red["Lx"], Ly=red["Ly"], Kn=red["Kn"],
    Qxx_closure=Qc["Qxx"], Qxy_closure=Qc["Qxy"],
    Qyx_closure=Qc["Qyx"], Qyy_closure=Qc["Qyy"],
    Qxx_full=full_runs[eps_list[0]]["Qxx"],
    Qxy_full=full_runs[eps_list[0]]["Qxy"],
    Qyx_full=full_runs[eps_list[0]]["Qyx"],
    Qyy_full=full_runs[eps_list[0]]["Qyy"],
    hx_red=red["hx_final"], hy_red=red["hy_final"], T_red=red["T_final"],
)
jc = red["Ny"] // 2
joff = int(jc + red["Ny"] * 0.15)
out["x"] = red["xc"]
out["Qxx_closure_mid"] = Qc["Qxx"][jc, :]
out["Qxx_full_mid"] = full_runs[eps_list[0]]["Qxx"][jc, :]
out["Qyx_closure_off"] = Qc["Qyx"][joff, :]
out["Qyx_full_off"] = full_runs[eps_list[0]]["Qyx"][joff, :]

np.savez(here / "fullQ_results.npz", **out)

def slope(errs):
    return [float(np.log(errs[i + 1] / errs[i]) / np.log(eps_list[i + 1] / eps_list[i]))
            for i in range(len(eps_list) - 1)]

summary = dict(eps=eps_list,
               err_h={str(t): err_h[t] for t in tS},
               err_T={str(t): err_T[t] for t in tS},
               err_Q=errs_Q, href=href,
               slopes_h={str(t): slope(err_h[t]) for t in tS})
(here / "fullQ_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
