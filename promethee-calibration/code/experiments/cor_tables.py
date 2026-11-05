"""Tables specific to the C&OR manuscript (main_results.tex, compare_summary_main.tex), computed
from the cached results of run_calibration.py. Usage: python -m experiments.cor_tables ../paper/tables"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from experiments.calibration_tables import NAMES
from experiments.run_calibration import RES

ORDER = ["HDI", "WHR24", "BLI", "ASEM", "ELECTRIC"]
ALGS = ["NSGA-II", "SMS-EMOA", "MOEA/D", "SPEA2", "Random"]


def red(x):
    return str(int(round(100 * (1 - min(1.0, x)))) + 0)


def main(tab):
    tab = Path(tab)
    c = pd.read_csv(RES / "calibrate.csv").set_index(["dataset", "measure", "scenario"])
    L = []
    for ds in ORDER:
        for k, sc in enumerate(["W", "WT", "F"]):
            r, q = c.loc[(ds, "event_rate", sc)], c.loc[(ds, "cyclicity", sc)]
            L.append(f"{NAMES[ds] if k == 0 else ''} & {sc} & {red(r['best@0.01'])} & {red(r['best@0.02'])} & "
                     f"{red(r['best@0.05'])} & {red(r.knee_C)} & {r.knee_D:.3f} & {r.knee_tau:.3f} & "
                     f"{'yes' if r.knee_top1 else 'no'} & {red(q['best@0.05'])} & {red(q['best@0.1'])} & "
                     f"{red(q.knee_C)} & {q.knee_D:.3f} & {q.knee_tau:.3f} \\\\")
        if ds != ORDER[-1]:
            L.append("\\addlinespace")
    (tab / "main_results.tex").write_text("\n".join(L) + "\n")
    m = pd.read_csv(RES / "compare_metrics.csv")
    mean = m.groupby(["dataset", "measure", "scenario", "algorithm"]).hv.mean().unstack()
    best = mean.eq(mean.max(axis=1), axis=0).sum()
    t = m.groupby("algorithm").time.mean()
    (tab / "compare_summary_main.tex").write_text(
        "Best mean HV (cases) & " + " & ".join(str(int(best[a])) for a in ALGS) + " \\\\\n"
        + "Mean time (s) & " + " & ".join(f"{t[a]:.1f}" for a in ALGS) + " \\\\\n")


if __name__ == "__main__":
    main(sys.argv[1])
