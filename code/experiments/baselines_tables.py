"""LaTeX tables and number macros for the comparison with literature baselines (run_baselines.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.calibration_tables import NAMES, macro
from experiments.run_baselines import RES

METHODS = ["calibration", "max-margin", "most-stable", "smaa-mean"]


def load():
    df = pd.read_csv(RES / "baselines.csv")
    el = df[df.method == "elicited"].set_index("dataset")
    d = df[df.method != "elicited"].copy()
    d["ins_r"] = [r.ins / el.loc[r.dataset, "ins"] for r in d.itertuples()]
    d["sub_r"] = [r.sub / el.loc[r.dataset, "sub"] for r in d.itertuples()]
    return d


def fmt(x, best):
    s = f"{x:.2f}"
    return f"\\textbf{{{s}}}" if best else s


def make_all(tab_dir: Path):
    d = load()
    order = ["HDI", "WHR24", "BLI", "ASEM", "ELECTRIC"]
    main, full = [], []
    for ds in order:
        for k, e in enumerate([0.01, 0.02, 0.05]):
            g = d[(d.dataset == ds) & (d.eps == e)].set_index("method")
            rr_best = g.RR.min()
            ins_best = g.ins_r.min()
            cells = [fmt(g.loc[m, "RR"], abs(g.loc[m, "RR"] - rr_best) < 1e-9) for m in METHODS]
            cells += [fmt(g.loc[m, "ins_r"], abs(g.loc[m, "ins_r"] - ins_best) < 1e-9) for m in METHODS]
            main.append(f"{NAMES[ds] if k == 0 else ''} & {e:g} & " + " & ".join(cells) + " \\\\")
            for m in METHODS:
                r = g.loc[m]
                full.append(f"{NAMES[ds]} & {e:g} & {m} & {r.dev:.4f} & {r.RR:.3f} & {r.Fr:.3f} & {r.tau:.3f} & "
                            f"{'yes' if r.top1 else 'no'} & {r.ins_r:.3f} & {r.sub_r:.3f} \\\\")
        if ds != order[-1]:
            main.append("\\addlinespace")
    (tab_dir / "baselines_main.tex").write_text("\n".join(main) + "\n")
    (tab_dir / "baselines_full.tex").write_text("\n".join(full) + "\n")
    N = {}
    for e in [0.01, 0.02, 0.05]:
        g = d[d.eps == e].groupby("method")
        for m in METHODS:
            key = m.replace("-", "")
            N[f"bl{key}RR{e}"] = str(int(round(100 * (1 - g.RR.median()[m]))) + 0)
            N[f"bl{key}ins{e}"] = str(int(round(100 * (1 - g.ins_r.median()[m]))) + 0)
            N[f"bl{key}sub{e}"] = str(int(round(100 * (1 - g.sub_r.median()[m]))) + 0)
    wins = 0
    for _, g in d.groupby(["dataset", "eps"]):
        if g[g.method == "calibration"].RR.iloc[0] <= g.RR.min() + 1e-12:
            wins += 1
    N["blcalwins"] = str(wins)
    N["blcases"] = str(d.groupby(["dataset", "eps"]).ngroups)
    for m in METHODS:
        g = d[d.method == m]
        key = m.replace("-", "")
        N[f"bl{key}insbetter"] = str(int((g.ins_r < 1).sum()))
        N[f"bl{key}subbetter"] = str(int((g.sub_r < 1).sum()))
    mm = d[d.method == "max-margin"]
    N["blmmmaxdevsmall"] = f"{100 * mm[mm.dataset.isin(['HDI', 'WHR24', 'ASEM'])].dev.max():.2f}"
    ms = d[d.method == "most-stable"]
    N["blmsminshare"] = f"{100 * ms[ms.dataset == 'WHR24'].mode_share.max():.2f}"
    with open(tab_dir / "numbers_bl.tex", "w") as f:
        for k, v in sorted(N.items()):
            f.write(f"\\newcommand{{\\{macro(k)}}}{{{v}}}\n")
    print(pd.Series(N).to_string())


if __name__ == "__main__":
    make_all(Path(sys.argv[1]))
