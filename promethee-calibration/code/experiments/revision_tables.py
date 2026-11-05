"""Tables and number macros of the revision analyses (run_revision.py, revision_stats.py,
run_synthetic.py). Usage: python -m experiments.revision_tables ../paper/tables"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.calibration_tables import NAMES, macro

ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / "results" / "revision"
SYN = ROOT / "results" / "synthetic"
CAL = ROOT / "results" / "calibration"


def pct(x, nd=0):
    if x is None or not np.isfinite(x):
        return "--"
    v = round(100 * x, nd)
    return f"{v:.{nd}f}" if nd else str(int(v) + 0)


def f3(x):
    v = f"{x:.3f}"
    return "0.000" if v == "-0.000" else v


def f4(x):
    v = f"{x:.4f}"
    return "0.0000" if v == "-0.0000" else v


def write(tab, name, lines):
    (tab / name).write_text("\n".join(lines) + "\n")


def ties(tab, N):
    d = pd.read_csv(REV / "ties.csv")
    p = d[d.point != "front"]
    for tol in (1e-10, 1e-6, 1e-4):
        g = p[p.tol == tol]
        N[f"tiesat{tol:.0e}"] = str(int(g.ties.sum()))
        N[f"strictat{tol:.0e}"] = str(int(g.strict.sum()))
    f = d[d.point == "front"]
    N["frontpoints"] = f"{int(f.npoints.sum()):,}"
    N["frontties"] = str(int(f.ties.sum()))
    N["frontevents"] = f"{int(f.strict.sum() + f.ties.sum()):,}"
    g4 = p[p.tol == 1e-4]
    N["nearevents"] = f"{int(g4.strict.sum() + g4.ties.sum()):,}"
    N["tiesbroken"] = str(int(p.ties_broken.fillna(0).sum()))
    L = []
    for (ds, sc, pt), g in p.groupby(["dataset", "scenario", "point"], sort=False):
        r = g.set_index("tol")
        L.append(f"{NAMES[ds]} & {sc} & {pt} & {int(r.loc[1e-10, 'pairs'])} & {int(r.loc[1e-10, 'strict'])} & "
                 f"{int(r.loc[1e-10, 'ties'])} & {int(r.loc[1e-10, 'ties_broken'])} & {int(r.loc[1e-6, 'strict'])} & "
                 f"{int(r.loc[1e-6, 'ties'])} & {int(r.loc[1e-4, 'strict'])} & {int(r.loc[1e-4, 'ties'])} \\\\")
    write(tab, "rev_ties.tex", L)


def milp(tab, N):
    d = pd.read_csv(REV / "milp.csv")
    N["milpruns"] = str(len(d))
    N["milpoptimal"] = str(int((d.status == 0).sum()))
    N["milpgapmax"] = f"{np.nanmax(np.abs(d.milp_value * 0 + (d.bound - d.strict)) ) :.0f}"
    N["milptimemax"] = f"{d.milp_time.max():.0f}"
    N["milpbuildmax"] = f"{d.build_time.max():.2f}"
    base = d[np.isclose(d.eta, 1e-5)].set_index(["dataset", "eps"])
    big = d[np.isclose(d.eta, 1e-3)].set_index(["dataset", "eps"])
    small = d[np.isclose(d.eta, 1e-6)].set_index(["dataset", "eps"])
    e4 = d[np.isclose(d.eta, 1e-4)].set_index(["dataset", "eps"])
    N["milpsameetasmall"] = str(int(sum(np.isclose(small.RR_rel, base.RR_rel)) + sum(np.isclose(e4.RR_rel, base.RR_rel))))
    N["milpcompsmall"] = str(2 * len(base))
    N["milpworseetabig"] = str(int(sum(big.RR_rel > base.RR_rel + 1e-12)))
    N["milpcompbig"] = str(len(base))
    L = []
    for ds in ["ELECTRIC", "BLI", "ASEM"]:
        for k, e in enumerate(sorted(d.eps.unique())):
            r, rb = base.loc[(ds, e)], big.loc[(ds, e)]
            t = [d[(d.dataset == ds) & np.isclose(d.eps, e) & np.isclose(d.eta, x)].milp_time.iloc[0]
                 for x in (1e-6, 1e-5, 1e-4, 1e-3)]
            L.append(f"{NAMES[ds] if k == 0 else ''} & {e:g} & {int(r.candidate_pairs)} & {int(r.binaries)} & {int(r.variables)} & "
                     f"{int(r['rows'])} & {r.build_time:.2f} & {min(t):.1f}--{max(t):.1f} & opt. & "
                     f"{r.RR_rel:.3f} & {rb.RR_rel:.3f} \\\\")
        if ds != "ASEM":
            L.append("\\addlinespace")
    write(tab, "rev_milp.tex", L)
    meta = pd.read_csv(REV / "milp_meta.csv", index_col=0).iloc[:, 0]
    N["highsversion"] = str(meta["highs"])
    N["scipyversion"] = str(meta["scipy"])
    N["pythonversion"] = str(meta["python"])


def stats(tab, N):
    s = pd.read_csv(REV / "stats_summary.csv", index_col=0)
    t = pd.read_csv(REV / "stats_times.csv", index_col=0)
    order = ["SMS-EMOA", "MOEA/D", "SPEA2", "Random"]
    L = []
    for a in order:
        r = s.loc[a]
        p = r.p_holm
        ps = "<0.001" if p < 1e-3 else f"{p:.3f}"
        L.append(f"{a} & {int(r.better)} & {int(r.equivalent)} & {int(r.worse)} & {r.A12_median:.2f} & "
                 f"{int(r.wins)}/{int(r.losses)} & {f4(r.median_diff)} [{f4(r.ci_lo)}, {f4(r.ci_hi)}] & {ps} & {t.loc[a, 'mean']:.1f} \\\\")
        key = a.replace("-", "").replace("/", "").replace("2", "two")
        N[f"st{key}better"] = str(int(r.better)); N[f"st{key}equiv"] = str(int(r.equivalent))
        N[f"st{key}worse"] = str(int(r.worse)); N[f"st{key}wins"] = str(int(r.wins))
        N[f"st{key}phol"] = ps; N[f"st{key}diff"] = f3(r.median_diff)
        N[f"st{key}lo"] = f3(r.ci_lo); N[f"st{key}hi"] = f3(r.ci_hi); N[f"st{key}Aonetwo"] = f"{r.A12_median:.2f}"
    write(tab, "rev_stats.tex", L)
    c = pd.read_csv(REV / "stats_cases.csv")
    L = []
    mname = {"event_rate": "RR", "incomparability": "$\\iota$"}
    for (ds, me, sc), g in c.groupby(["dataset", "measure", "scenario"], sort=False):
        g = g.set_index("other")
        cells = []
        for a in order:
            r = g.loc[a]
            mark = "" if r.p_holm >= 0.05 else ("$^{+}$" if r.med_diff > 0 else "$^{-}$")
            cells.append(f"{r.A12:.2f}{mark}")
        L.append(f"{NAMES[ds]} & {mname.get(me, me)} & {sc} & " + " & ".join(cells) + " \\\\")
    write(tab, "rev_stats_cases.tex", L)
    N["stNSGAIItime"] = f"{t.loc['NSGA-II', 'mean']:.1f}"


def blsens(tab, N):
    d = pd.read_csv(REV / "blsens.csv")
    ms = d[d.method == "most-stable"]
    g = ms.groupby(["dataset", "eps", "N"]).RR.agg(["mean", "std"]).reset_index()
    cal = pd.read_csv(ROOT / "results" / "baselines" / "baselines.csv")
    cal = cal[cal.method == "calibration"].set_index(["dataset", "eps"]).RR
    # how often the calibration beats the mean of the most-stable weights with 16,000 samples
    g16 = g[g.N == 16000].set_index(["dataset", "eps"])["mean"]
    N["blsenswins"] = str(int(sum(cal.loc[i] <= g16.loc[i] + 1e-12 for i in g16.index)))
    N["blsenscases"] = str(len(g16))
    w = g[(g.dataset == "WHR24") & np.isclose(g.eps, 0.01)].set_index("N")["mean"]
    N["whrmsone"] = f"{w.loc[1000]:.2f}"; N["whrmsfour"] = f"{w.loc[4000]:.2f}"; N["whrmssixteen"] = f"{w.loc[16000]:.2f}"
    N["whrcalone"] = f"{cal.loc[('WHR24', 0.01)]:.2f}"
    tm = d.groupby(["method", "N"]).time.mean()
    N["tmaxmargin"] = f"{tm.loc[('max-margin', 0)]:.2f}"
    N["tmostsixteen"] = f"{tm.loc[('most-stable', 16000)]:.0f}"
    N["tmostfour"] = f"{tm.loc[('most-stable', 4000)]:.0f}"
    L = []
    for (ds, e), h in g.groupby(["dataset", "eps"]):
        vals = " & ".join(f"{h[h.N == n]['mean'].iloc[0]:.3f} ({h[h.N == n]['std'].iloc[0]:.3f})" for n in (1000, 4000, 16000))
        L.append(f"{NAMES[ds]} & {e:g} & {cal.loc[(ds, e)]:.3f} & {vals} \\\\")
    write(tab, "rev_blsens.tex", L)


def oos(tab, N):
    d = pd.read_csv(REV / "oos.csv")
    for gname, g in d.groupby("generator"):
        N[f"oos{gname}red"] = pct(1 - g.ratio.median())
        N[f"oos{gname}better"] = str(int((g.ratio < 1).sum()))
    N["ooscases"] = str(d.groupby(["dataset", "scenario"]).ngroups)
    L = []
    for (ds, sc), g in d.groupby(["dataset", "scenario"], sort=False):
        r = g.set_index("generator").ratio
        L.append(f"{NAMES[ds]} & {sc} & {r['independent']:.2f} & {r['bootstrap']:.2f} & {r['copula']:.2f} \\\\")
    write(tab, "rev_oos.tex", L)


def roc(tab, N):
    d = pd.read_csv(REV / "roc.csv")
    L = []
    for _, r in d.iterrows():
        g = lambda c: 100 * (1 - min(1, r[c]))
        L.append(f"{NAMES[r.dataset]} & {r.tv_roc_pub:.3f} & " + " & ".join(
            f"{g(f'pub@{b}'):.0f} & {g(f'roc@{b}'):.0f}" for b in ("0.01", "0.02", "0.05")) + " \\\\")
    write(tab, "rev_roc.tex", L)
    N["roctvmin"] = f"{100 * d.tv_roc_pub.min():.0f}"; N["roctvmax"] = f"{100 * d.tv_roc_pub.max():.0f}"


def dev(tab, N):
    d = pd.read_csv(REV / "dev.csv")
    order = ["alpha=0.25", "alpha=0.5 (paper)", "alpha=0.75", "max over criteria",
             "weights only, |nu|<=0.25", "weights only, |nu|<=0.5"]
    lab = {"alpha=0.25": "$\\alpha=0.25$", "alpha=0.5 (paper)": "$\\alpha=0.5$ (main text)",
           "alpha=0.75": "$\\alpha=0.75$", "max over criteria": "maximum over criteria",
           "weights only, |nu|<=0.25": "separate, $|\\nu_j|\\le0.25$", "weights only, |nu|<=0.5": "separate, $|\\nu_j|\\le0.5$"}
    L = []
    for v in order:
        g = d[d.variant == v].set_index("dataset")
        cells = " & ".join(f"{100 * (1 - min(1, g.loc[ds, 'best@0.02'])):.0f}" for ds in ["HDI", "WHR24", "BLI", "ASEM", "ELECTRIC"])
        L.append(f"{lab[v]} & {cells} \\\\")
    write(tab, "rev_dev.tex", L)
    s = d[d.variant == "weights only, |nu|<=0.25"]["best@0.02"]
    N["sepquarterredmin"] = pct(1 - s.max()); N["sepquarterredmax"] = pct(1 - s.min())
    cal = pd.read_csv(CAL / "calibrate.csv")
    w = cal[(cal.measure == "event_rate") & (cal.scenario == "W")]["best@0.02"]
    N["wonlyredmed"] = pct(1 - w.median())
    N["sepquarterredmed"] = pct(1 - s.median())


def synthetic(tab, N):
    f = SYN / "synthetic.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    N["syninstances"] = str(len(d))
    pos = d[d.RR0 > 0]
    N["synpos"] = str(len(pos))
    N["synzero"] = str(len(d) - len(pos))
    for b in (0.01, 0.02, 0.05):
        N[f"synWred{b}"] = pct(pos[f"W_red@{b}"].median())
        N[f"synWTred{b}"] = pct(pos[f"WT_red@{b}"].median())
    N["synkneetaumin"] = f"{pos.W_knee_tau.min():.3f}"
    N["synkneetaumed"] = f"{pos.W_knee_tau.median():.3f}"
    N["syntopone"] = pct(pos.W_knee_top1.mean())
    ex = pos.dropna(subset=["ex_status"])
    N["synexatt"] = str(len(ex)); N["synexopt"] = str(int((ex.ex_status == 0).sum()))
    N["synexlimit"] = str(int((ex.ex_status == 1).sum()))
    hit = ex[(ex.ex_status == 0) & (ex.ties == "continuous")].dropna(subset=["ex_red"])
    N["synevohit"] = str(int((hit.evo_red >= hit.ex_red - 1e-9).sum())); N["synexsolved"] = str(len(hit))
    rnd = ex[(ex.ex_status == 0) & (ex.ties == "rounded")].dropna(subset=["ex_red"])
    N["synexrounded"] = str(len(rnd))
    N["synexroundedbetter"] = str(int((rnd.ex_red > rnd.evo_red + 1e-9).sum()))
    N["synexroundedworse"] = str(int((rnd.ex_red < rnd.evo_red - 1e-9).sum()))
    for st, nm in ((0, "opt"), (1, "limit"), (2, "infeas")):
        for tv in ("continuous", "rounded"):
            N[f"synex{nm}{tv}"] = str(int(((ex.ex_status == st) & (ex.ties == tv)).sum()))
    N["synexattcontinuous"] = str(int((ex.ties == "continuous").sum()))
    N["synexattrounded"] = str(int((ex.ties == "rounded").sum()))
    N["synevogapmed"] = pct((hit.ex_red - hit.evo_red).median(), 1)
    N["synevogapmax"] = pct((hit.ex_red - hit.evo_red).max())
    N["synevogapmean"] = pct((hit.ex_red - hit.evo_red).mean(), 1)
    N["synbigmodels"] = str(int((pos.bin_002 > 5000).sum()))
    # speed-up and candidate share by m
    for m, g in d.groupby("m"):
        N[f"synspeed{m}"] = f"{(g.t_rerank_ms / g.t_eval_ms).median():.1f}"
        N[f"syncand{m}"] = pct(g.cand_share.median())
        N[f"synevalms{m}"] = f"{g.t_eval_ms.median():.2f}"
        N[f"synruntime{m}"] = f"{g.W_time.median():.1f}"
        N[f"synbin{m}"] = str(int(g.bin_002.median())) if g.bin_002.notna().any() else "--"
    L = []
    for fac, levels in (("m", [20, 50, 100]), ("n", [4, 8, 12]), ("structure", ["concordant", "independent", "conflicting"]),
                        ("ties", ["continuous", "rounded"]), ("weights", ["order", "equal"])):
        for lv in levels:
            g = d[d[fac] == lv]
            gp = g[g.RR0 > 0]
            name = {"m": f"$m={lv}$", "n": f"$n={lv}$"}.get(fac, str(lv).replace("order", "strict order").replace("equal", "equal weights"))
            exg = gp.dropna(subset=["ex_status"])
            L.append(f"{name} & {len(g)} & {100 * g.RR0.mean():.3f} & {100 * g.Fr0.mean():.2f} & {g.K0.mean():.3f} & "
                     f"{pct(gp['W_red@0.02'].median())} & {pct(gp['WT_red@0.02'].median())} & {gp.W_knee_tau.median():.3f} & "
                     f"{100 * g.cand_share.median():.1f} & {(g.t_rerank_ms / g.t_eval_ms).median():.1f} & "
                     f"{int(gp.bin_002.median()) if gp.bin_002.notna().any() else 0} & {int((exg.ex_status == 0).sum())}/{len(exg)} \\\\")
        L.append("\\addlinespace")
    write(tab, "rev_synthetic.tex", L[:-1])
    C = []
    for (m, st, wt), g in d.groupby(["m", "structure", "weights"], sort=False):
        gp = g[g.RR0 > 0]
        exg = gp.dropna(subset=["ex_status"])
        sol = exg[exg.ex_status == 0].dropna(subset=["ex_red"])
        gap = pct((sol.ex_red - sol.evo_red).median()) if len(sol) else "--"
        name = "strict order" if wt == "order" else "equal"
        C.append(f"{m} & {st} & {name} & {len(gp)}/{len(g)} & " + " & ".join(
            pct(gp[f"W_red@{b}"].median()) for b in (0.01, 0.02, 0.05)) +
            f" & {pct(gp['WT_red@0.02'].median())} & {gp.W_knee_tau.median():.3f} & {pct(gp.W_knee_top1.mean())} & "
            f"{int((exg.ex_status == 0).sum())}/{len(exg)} & {gap} \\\\")
    write(tab, "rev_synthetic_cells.tex", C)


def timing(tab, N):
    f = REV / "timing.csv"
    if not f.exists():
        return
    d = pd.read_csv(f).set_index("dataset")
    L = []
    for ds in ["HDI", "WHR24", "BLI", "ASEM", "ELECTRIC"]:
        r = d.loc[ds]
        ex = lambda v: f"{v:.1f}" if v > 0 else "--"
        L.append(f"{NAMES[ds]} & {int(r.m)} & {int(r.n)} & {1e3 * r.diagnosis:.1f} & {r.dm_search:.1f} & {ex(r.dm_exact)} & "
                 f"{r.dm_total:.0f} & {r.an_search:.0f} & {ex(r.an_exact)} & {r.an_validation:.2f} & "
                 f"{r.an_total / 60:.1f} \\\\")
    write(tab, "rev_timing.tex", L)
    N["tdmmin"] = f"{d.dm_total.min():.0f}"; N["tdmmax"] = f"{d.dm_total.max():.0f}"
    N["tanmin"] = f"{d.an_total.min() / 60:.1f}"; N["tanmax"] = f"{d.an_total.max() / 60:.1f}"
    N["tdiagmax"] = f"{1e3 * d.diagnosis.max():.0f}"


def scaling(tab, N):
    f = REV / "scaling_synthetic.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    L = []
    for r in d.itertuples():
        rr = f"{1e3 * r.t_rerank:.1f}" if np.isfinite(r.t_rerank) else "--"
        sp = f"{r.t_rerank / r.t_eval:.0f}" if np.isfinite(r.t_rerank) else "--"
        L.append(f"{r.n} & {r.m:,} & {r.Q:,} & {1e3 * r.t_Pj:.2f} & {1e3 * r.t_agg:.3f} & {1e3 * r.t_eval:.2f} & {rr} & {sp} \\\\")
    write(tab, "rev_scaling.tex", L)
    big = d[d.m >= 200]
    sl = {}
    for c in ("t_Pj", "t_agg", "t_eval", "t_rerank", "Q"):
        v = [np.polyfit(np.log(g.m), np.log(g[c]), 1)[0] for _, g in big.dropna(subset=[c]).groupby("n")]
        sl[c] = (min(v), max(v))
        N[f"slope{c.replace('t_', '')}lo"] = f"{min(v):.1f}"; N[f"slope{c.replace('t_', '')}hi"] = f"{max(v):.1f}"
    s2 = d[d.m == 200]
    N["speedtwolo"] = f"{(s2.t_rerank / s2.t_eval).min():.0f}"; N["speedtwohi"] = f"{(s2.t_rerank / s2.t_eval).max():.0f}"
    s8 = d[d.m == 800]
    N["speedeightlo"] = f"{(s8.t_rerank / s8.t_eval).min():.0f}"; N["speedeighthi"] = f"{(s8.t_rerank / s8.t_eval).max():.0f}"
    g = d[d.m == 1600].set_index("n")
    N["pjnratio"] = f"{g.loc[16, 't_Pj'] / g.loc[4, 't_Pj']:.1f}"
    b = pd.read_csv(REV / "scaling_benchmarks.csv")
    N["evalWlo"] = f"{1e3 * b.t_eval_W.min():.1f}"; N["evalWhi"] = f"{1e3 * b.t_eval_W.max():.1f}"
    N["evalWTlo"] = f"{1e3 * b.t_eval_WT.min():.1f}"; N["evalWThi"] = f"{1e3 * b.t_eval_WT.max():.1f}"
    N["wtratiolo"] = f"{(b.t_eval_WT / b.t_eval_W).min():.1f}"; N["wtratiohi"] = f"{(b.t_eval_WT / b.t_eval_W).max():.1f}"
    ov = pd.concat([b.overhead_W, b.overhead_WT]).clip(lower=0)
    N["overheadmax"] = f"{ov.max():.1f}"
    B = []
    for r in b.itertuples():
        B.append(f"{NAMES[r.dataset]} & {r.m} & {r.n} & {1e3 * r.t_eval_W:.2f} & {r.t_run_W:.1f} & {1e3 * r.t_eval_WT:.2f} & {r.t_run_WT:.1f} \\\\")
    write(tab, "rev_scaling_bench.tex", B)


def sideeffects(tab, N):
    f = REV / "sideeffects.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    d["dinc"] = 100 * (d.inc - d.inc0)
    d["krel"] = d.K / d.K0
    d["rrrel"] = d.RR / d.RR0
    # larger budgets
    b10 = d[d.point == "best@0.1"]
    N["sebigtaumin"] = f"{b10.tau.min():.2f}"
    N["sebigkrelmin"] = f"{b10.krel.min():.2f}"; N["sebigkrelmax"] = f"{b10.krel.max():.2f}"
    N["sebigdincmin"] = f"{b10.dinc.min():.1f}"; N["sebigdincmax"] = f"{b10.dinc.max():.1f}"
    mn = d[d.point == "min RR"]
    N["seminDmax"] = f"{mn.D.max():.2f}"
    N["semintaumin"] = f"{mn.tau.min():.2f}"; N["seminkrelmax"] = f"{mn.krel.max():.1f}"
    N["semindincmin"] = f"{mn.dinc.min():.1f}"
    small = mn[mn.dataset.isin(["BLI", "ASEM", "ELECTRIC"])]
    N["sesmallDmax"] = f"{small.D.max():.2f}"
    lab = {"best@0.02": "0.02", "best@0.05": "0.05", "best@0.1": "0.1", "best@0.2": "0.2", "min RR": "min"}
    B = []
    for (ds, sc), g in d[d.scenario.isin(["W", "WT"]) & d.point.isin(list(lab))].groupby(["dataset", "scenario"], sort=False):
        for k, (_, r) in enumerate(g.set_index("point").loc[list(lab)].reset_index().iterrows()):
            B.append(f"{NAMES[ds] if k == 0 else ''} & {sc if k == 0 else ''} & {lab[r.point]} & {r.D:.3f} & {r.rrrel:.2f} & "
                     f"{r.tau:.3f} & {f'{r.dinc:+.1f}'.replace('-0.0', '0.0').replace('+0.0', '0.0')} & {r.krel:.2f} \\\\")
        B.append("\\addlinespace")
    write(tab, "rev_sidebudget.tex", B[:-1])
    p = d[d.point.isin(["knee", "best@0.02"])].copy()
    N["sepoints"] = str(len(p))
    N["sekrelmin"] = f"{p.krel.min():.2f}"; N["sekrelmax"] = f"{p.krel.max():.2f}"
    N["sedincmin"] = f"{p.dinc.min():.1f}"; N["sedincmax"] = f"{p.dinc.max():.1f}"
    N["sedincmed"] = f"{p.dinc.median():.1f}"
    N["sedecreaseorsame"] = str(int((p.dinc <= 1e-9).sum()))
    other = p[p.dataset != "ELECTRIC"]
    N["sedincmaxother"] = f"{other.dinc.max():.1f}"
    L = []
    for (ds, sc), g in p.groupby(["dataset", "scenario"], sort=False):
        g = g.set_index("point")
        k, b = g.loc["knee"], g.loc["best@0.02"]
        L.append(f"{NAMES[ds]} & {sc} & {100 * k.inc0:.1f} & {100 * k.inc:.1f} & {100 * b.inc:.1f} & "
                 f"{k.K0:.3f} & {k.K:.3f} & {b.K:.3f} \\\\")
    write(tab, "rev_sideeffects.tex", L)


def litmeasures(tab, N):
    f = REV / "litmeasures.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    L = []
    for ds in ["HDI", "WHR24", "BLI", "ASEM", "ELECTRIC"]:
        g = d[d.dataset == ds].set_index(["scenario", "point"])
        r0 = g.iloc[0]
        cells = [100 * r0.impact0] + [100 * g.loc[(sc, pt), "impact"] for sc, pt in (("W", "best@0.02"), ("W", "knee"), ("WT", "knee"))]
        cells2 = [r0.I20] + [g.loc[(sc, pt), "I2"] for sc, pt in (("W", "best@0.02"), ("W", "knee"), ("WT", "knee"))]
        L.append(f"{NAMES[ds]} & " + " & ".join(f"{v:.0f}" for v in cells) + " & " + " & ".join(f"{v:.3f}" for v in cells2) + " \\\\")
    write(tab, "rev_lit.tex", L)
    e = d.drop_duplicates("dataset")
    N["litimpzero"] = pct(e.impact0.median())
    k = d[(d.scenario == "W") & (d.point == "best@0.02")]
    N["litimpbest"] = pct(k.impact.median())
    kw = d[(d.scenario == "WT") & (d.point == "knee")]
    N["litimpwt"] = pct(kw.impact.median())
    N["litimpzeromin"] = pct(e.impact0.min()); N["litimpzeromax"] = pct(e.impact0.max())
    N["litItwochange"] = f"{(d.I2 - d.I20).abs().max():.3f}"


def main(tab):
    tab = Path(tab)
    N = {}
    for f in (ties, milp, stats, blsens, oos, roc, dev, synthetic, timing, scaling, sideeffects, litmeasures):
        f(tab, N)
    with open(tab / "numbers_rev.tex", "w") as fh:
        for k, v in sorted(N.items()):
            fh.write(f"\\newcommand{{\\{macro(k)}}}{{{v}}}\n")
    print(pd.Series(N).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
