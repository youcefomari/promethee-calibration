"""Side effects of calibrating for the reversal rate on the other conflict measures: incomparability
and cyclicity of the elicited model, the knee and the best point within a deviation of 0.02 of
every reversal-rate front, and over all points of the fronts with a deviation of at most 0.05.
Usage: python -m experiments.run_sideeffects"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from experiments.run_calibration import DATASETS, RES, ROOT, SCENARIOS, _cal
from gpii.calibration import knee_point
from gpii.conflict import conflict_measures
from gpii.core import promethee, relations_promethee_i

OUT = ROOT / "results" / "revision" / "sideeffects.csv"


def measures(X, model):
    pr = promethee(X, model)
    c = conflict_measures(pr.P, which=("fragility", "event_rate", "cyclicity"))
    _, _, R = relations_promethee_i(pr)
    m = len(X)
    c["incomparability"] = float(np.triu(R, 1).sum() / (m * (m - 1) / 2))
    return c


def main():
    Z = np.load(RES / "final_fronts.npz")
    rows = []
    for ds in DATASETS:
        for sc in SCENARIOS:
            d, cal = _cal(ds, "event_rate", sc)
            X, F = Z[f"{ds}|event_rate|{sc}|X"], Z[f"{ds}|event_rate|{sc}|F"]
            base = measures(d.X, d.model)
            k = knee_point(F)
            ok = np.nonzero(F[:, 1] <= 0.02 + 1e-12)[0]
            b = ok[np.argmin(F[ok, 0])]
            pts = {"knee": k, "best@0.02": b}
            for eps in (0.05, 0.1, 0.2, 0.5):
                okk = np.nonzero(F[:, 1] <= eps + 1e-12)[0]
                pts[f"best@{eps}"] = okk[np.argmin(F[okk, 0])]
            pts["min RR"] = int(np.argmin(F[:, 0] + 1e-9 * F[:, 1]))
            for lab, i in pts.items():
                c = measures(d.X, cal.decode(X[i]))
                tau = __import__("scipy.stats", fromlist=["kendalltau"]).kendalltau(
                    promethee(d.X, d.model).phi, promethee(d.X, cal.decode(X[i])).phi)[0]
                rows.append(dict(dataset=ds, scenario=sc, point=lab, D=F[i, 1], tau=tau,
                                 RR0=base["event_rate"], RR=c["event_rate"],
                                 inc0=base["incomparability"], inc=c["incomparability"],
                                 K0=base["cyclicity"], K=c["cyclicity"]))
            near = np.nonzero(F[:, 1] <= 0.05 + 1e-12)[0]
            cs = [measures(d.X, cal.decode(X[i])) for i in near]
            inc = np.array([c["incomparability"] for c in cs]); K = np.array([c["cyclicity"] for c in cs])
            rows.append(dict(dataset=ds, scenario=sc, point="front<=0.05", D=np.nan, RR0=base["event_rate"], RR=np.nan,
                             inc0=base["incomparability"], inc=float(np.max(inc)), K0=base["cyclicity"], K=float(np.max(K)),
                             n=len(near), inc_min=float(np.min(inc)), K_min=float(np.min(K))))
            print(ds, sc, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    df["d_inc_pp"] = 100 * (df.inc - df.inc0)
    df["K_rel"] = df.K / df.K0
    print(df.round(4).to_string())


if __name__ == "__main__":
    main()
