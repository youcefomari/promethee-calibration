"""Literature-based conflict measures before and after calibration: the impact (share of alternatives
whose deletion causes at least one rank reversal, the quantity estimated by simulation studies) and the
two-cycle index of Coquelet et al. (2025), for the knee and the best point within 0.02 of the
reversal-rate fronts. Usage: python -m experiments.run_litmeasures"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from experiments.run_calibration import DATASETS, RES, ROOT, _cal
from gpii.calibration import knee_point
from gpii.conflict import conflict_measures
from gpii.core import promethee
from gpii.cycles import two_cycle_index

OUT = ROOT / "results" / "revision" / "litmeasures.csv"


def lit(X, model):
    pr = promethee(X, model)
    c = conflict_measures(pr.P, which=("impact",))
    return c["impact"], two_cycle_index(pr.pi)


def main():
    Z = np.load(RES / "final_fronts.npz")
    rows = []
    for ds in DATASETS:
        for sc in ("W", "WT"):
            d, cal = _cal(ds, "event_rate", sc)
            X, F = Z[f"{ds}|event_rate|{sc}|X"], Z[f"{ds}|event_rate|{sc}|F"]
            imp0, i20 = lit(d.X, d.model)
            ok = np.nonzero(F[:, 1] <= 0.02 + 1e-12)[0]
            for lab, i in (("knee", knee_point(F)), ("best@0.02", ok[np.argmin(F[ok, 0])])):
                imp, i2 = lit(d.X, cal.decode(X[i]))
                rows.append(dict(dataset=ds, scenario=sc, point=lab, D=F[i, 1], impact0=imp0, impact=imp,
                                 I20=i20, I2=i2))
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(df.round(4).to_string())


if __name__ == "__main__":
    main()
