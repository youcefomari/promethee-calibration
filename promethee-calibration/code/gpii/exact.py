"""Exact epsilon-constraint MILP for weight calibration, with certified pruning.

For fixed thresholds and shapes P(w) = sum_j w_j P^j is linear in the weights, hence so are
D_ab(w) = sum_s [P(a,s) - P(b,s)] and E_abx(w) = D_ab(w) - P(a,x) - P(x,b) (Theorem 4).
A deletion event (a, b, x) breaks the order of the pair iff sign(E_abx) != sign(D_ab).

Sign-stability radius (Lemma 5 of the manuscript). For f(w) = sum_j w_j f^j and two weight
vectors of the simplex, |f(w) - f(w')| <= TV(w, w') * (max_j f^j - min_j f^j). Hence if
|f(w0)| > eps * range_j(f^j), the sign of f is the same for every w with TV(w, w0) <= eps and
needs no binary variable. With binaries only for the remaining (uncertain) signs,

    min  sum_e f_e              (event rate)    or   sum_ab g_ab   (fragility)
    s.t. sign constraints  f(w) >= eta - M (1 - z),  f(w) <= -eta + M z   (uncertain signs)
         f_e >= |z_ab - u_abx|  (linearised),  g_ab >= f_e
         w = V lam,  lam in simplex,  1/2 ||w - w0||_1 <= eps.

Events whose two signs are certified contribute a constant. Pairs with |D_ab(w)| > 2 cannot be
broken by one deletion (|delta| <= 2), so only pairs with |D_ab(w0)| <= 2 + eps range(D^j) are
examined. Functions that vanish at w0 (ties of the elicited model) get a third state, the tie
f(w) = 0, with a second binary; functions that vanish on the whole simplex are constant ties. A
tied pair is not strictly ordered and has no events, and an event whose function is a tie breaks
its pair, as in the tie-aware measure. The solution is exact up to the margin eta; it is
re-evaluated with the tie-aware measure.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .calibration import Calibration
from .model import unicriterion

TOL = 1e-9


def _structures(cal: Calibration, model=None):
    model = model or cal.model0
    Pj = cal._Pj0 if (cal.scenario == "W" and model is cal.model0) else unicriterion(model, cal.X, cal.X)
    Pa = Pj - Pj.transpose(0, 2, 1)
    return Pa, Pa.sum(2)


def model_size(cal: Calibration, eps: float):
    """Numbers of pairs, events and binaries of the pruned model."""
    Pa, rj = _structures(cal)
    n, m, _ = Pa.shape
    w0 = cal.model0.w
    iu = np.triu_indices(m, 1)
    Dj = rj[:, iu[0]] - rj[:, iu[1]]
    D0 = w0 @ Dj
    rD = np.ptp(Dj, 0)
    cand = np.abs(D0) <= 2 + eps * rD + TOL

    def nbin(f0, rng_):
        unc = np.abs(f0) <= eps * rng_ + TOL
        zero = np.abs(f0) <= TOL
        return int(np.sum(unc & ~zero) + 2 * np.sum(unc & zero & (rng_ > TOL)))

    nb_pair = nbin(D0[cand], rD[cand])
    nb_ev = 0
    for i in np.nonzero(cand)[0]:
        a, b = iu[0][i], iu[1][i]
        xs = np.setdiff1d(np.arange(m), [a, b])
        Ej = Dj[:, i][:, None] - Pa[:, a, xs] - Pa[:, xs, b]
        nb_ev += nbin(w0 @ Ej, np.ptp(Ej, 0))
    return {"pairs": len(D0), "candidate_pairs": int(cand.sum()), "binaries": nb_pair + nb_ev,
            "full_binaries": len(D0) * (m - 1)}


def solve_eps(cal: Calibration, eps: float, eta: float = 1e-5, time_limit: float = 600.0,
              prune: bool = True):
    if cal.measure not in ("event_rate", "fragility"):
        raise ValueError("exact model available for event_rate and fragility")
    Pa, rj = _structures(cal)
    n, m, _ = Pa.shape
    V = cal.ws.V
    k = V.shape[1]
    w0 = cal.model0.w
    frag = cal.measure == "fragility"
    M = 2.0 * m + 1.0
    iu = np.triu_indices(m, 1)
    Dj_all = rj[:, iu[0]] - rj[:, iu[1]]
    D0_all = w0 @ Dj_all
    n_strict0 = None
    if prune:
        cand_pairs = np.nonzero(np.abs(D0_all) <= 2 + eps * np.ptp(Dj_all, 0) + TOL)[0]
    else:
        cand_pairs = np.arange(len(D0_all))

    # variable registry ------------------------------------------------------
    nv = k + n
    var_lo, var_hi, integ = [0.0] * k + [0.0] * n, [1.0] * k + [np.inf] * n, [0] * (k + n)

    def new_var(binary=False, ub=1.0):
        nonlocal nv
        var_lo.append(0.0); var_hi.append(ub); integ.append(1 if binary else 0)
        nv += 1
        return nv - 1

    rows, cols, vals, lo, hi = [], [], [], [], []
    nrow = 0

    def add(coefs: dict, l, h):
        nonlocal nrow
        for c, v in coefs.items():
            if v != 0:
                rows.append(nrow); cols.append(c); vals.append(float(v))
        lo.append(l); hi.append(h)
        nrow += 1

    def lam_coefs(cw):
        return {i: v for i, v in enumerate(cw @ V)}

    def sign_var(cw, f0):
        """Sign of f(w) = cw . w as a pair (s, t) of positive and tie indicators, each a constant
        or a binary variable index: ('const', s, t) if certified within eps, else ('var', s, t)."""
        rng_ = np.ptp(cw)
        if rng_ <= TOL and abs(f0) <= TOL:                 # f = 0 on the whole simplex
            return ("const", 0, 1)
        if (prune or rng_ <= TOL) and abs(f0) > eps * rng_ + TOL:
            return ("const", 1 if f0 > 0 else 0, 0)
        c = lam_coefs(cw)
        Mf = float(np.max(np.abs(cw))) + 10 * eta        # |f(w)| <= max_j |f^j| on the simplex
        zs = new_var(binary=True)
        if abs(f0) > TOL:                                  # strict at w0: two states
            add({**c, zs: -Mf}, eta - Mf, np.inf)
            add({**c, zs: -Mf}, -np.inf, -eta)
            return ("var", zs, 0)
        zt = new_var(binary=True)                          # tie at w0: positive, negative or tie
        add({zs: 1.0, zt: 1.0}, -np.inf, 1.0)
        add({**c, zs: -Mf}, eta - Mf, np.inf)              # s = 1 -> f >= eta
        add({**c, zs: -Mf, zt: -Mf}, -np.inf, -eta)        # s = t = 0 -> f <= -eta
        add({**c, zt: Mf}, -np.inf, Mf)                    # t = 1 -> f <= 0
        add({**c, zt: -Mf}, -Mf, np.inf)                   # t = 1 -> f >= 0
        return ("var", zs, zt)

    add({i: 1.0 for i in range(k)}, 1.0, 1.0)
    for j in range(n):
        e = np.zeros(n); e[j] = 1
        c = lam_coefs(e)
        add({**c, k + j: -1.0}, -np.inf, w0[j])
        add({**{i: -v for i, v in c.items()}, k + j: -1.0}, -np.inf, -w0[j])
    add({k + j: 0.5 for j in range(n)}, -np.inf, eps)

    obj = {}
    const = 0.0
    def lin(rep, which):
        """(coefficients, constant) of the s (which=1) or t (which=2) indicator."""
        v = rep[which]
        if rep[0] == "var" and not (which == 2 and v == 0):
            return {v: 1.0}, 0.0
        return {}, float(v)

    def comb(*terms):
        out, k0 = {}, 0.0
        for sign, (cf, kc) in terms:
            for kk, v in cf.items():
                out[kk] = out.get(kk, 0.0) + sign * v
            k0 += sign * kc
        return out, k0

    for i in cand_pairs:
        a, b = iu[0][i], iu[1][i]
        dz = sign_var(Dj_all[:, i], D0_all[i])
        if dz[0] == "const" and dz[2] == 1:              # tied for every weight: no events
            continue
        xs = np.setdiff1d(np.arange(m), [a, b])
        Ej = Dj_all[:, i][:, None] - Pa[:, a, xs] - Pa[:, xs, b]
        E0 = w0 @ Ej
        g = new_var() if frag else None
        pair_const = 0.0
        sD, tD = lin(dz, 1), lin(dz, 2)
        for t in range(len(xs)):
            ez = sign_var(Ej[:, t], E0[t])
            if dz[0] == "const" and ez[0] == "const":
                if dz[1] != ez[1] or ez[2] == 1:
                    pair_const = 1.0
                    if not frag:
                        const += 1.0
                continue
            sE, tE = lin(ez, 1), lin(ez, 2)
            f = new_var()
            # broken = (1 - tD) * max(|sD - sE|, tE):  f >= sD - sE - tD, f >= sE - sD - tD, f >= tE - tD
            for expr in (comb((1, sD), (-1, sE), (-1, tD)), comb((-1, sD), (1, sE), (-1, tD)),
                         comb((1, tE), (-1, tD))):
                cf, k0 = expr
                add({f: 1.0, **{kk: -v for kk, v in cf.items()}}, k0, np.inf)
            if frag:
                add({g: 1.0, f: -1.0}, 0.0, np.inf)
            else:
                obj[f] = 1.0
        if frag:
            if pair_const:
                add({g: 1.0}, 1.0, np.inf)
            obj[g] = 1.0
    cvec = np.zeros(nv)
    for kk, v in obj.items():
        cvec[kk] = v
    A = coo_matrix((vals, (rows, cols)), shape=(nrow, nv)).tocsr()
    t0 = time.time()
    res = milp(cvec, constraints=LinearConstraint(A, lo, hi), integrality=np.array(integ),
               bounds=Bounds(np.array(var_lo), np.array(var_hi)),
               options={"time_limit": time_limit, "disp": False, "mip_rel_gap": 0.0})
    out = {"eps": eps, "status": res.status, "time": time.time() - t0,
           "n_binaries": int(np.sum(integ)), "n_vars": nv, "n_rows": nrow,
           "candidate_pairs": len(cand_pairs)}
    if res.x is not None:
        lam = cal.repair_lam(res.x[:k])
        x = cal.x0.copy()
        x[:k] = lam
        out.update(lam=lam, x=x, F=cal.evaluate(x))
        denom = len(D0_all) if frag else len(D0_all) * (m - 2)
        out["milp_value"] = (float(res.fun) + const) / denom / (cal.C0 if cal.C0 > 0 else 1.0)
        bound = getattr(res, "mip_dual_bound", None)
        if bound is not None and np.isfinite(bound):
            out["dual_bound"] = float(bound) + const
    return out


def refine(cal: Calibration, x: np.ndarray, time_limit: float = 120.0):
    """Hybrid step: keep the thresholds of an evolutionary solution x and re-optimise its
    weights exactly with the remaining deviation budget (rank-reversal measures only).

    In scenarios WT and F, Delta = (Delta_w + Delta_f)/2, so the weight budget that keeps
    Delta(x) is 2 Delta(x) - Delta_f(x); the preference functions of x are kept. Returns (x_new, F_new, info); x_new = x if no improvement.
    """
    x = cal.repair(x)
    F = cal.evaluate(x)
    dw, dp = cal.deviation_parts(x)
    model = cal.decode(x)
    sub = Calibration(cal.X, cal.model0.with_(p=model.p, q=model.q, s=model.s, types=model.types),
                      cal.measure, "W", cal.ws.mode, ratio_range=cal.R)
    eps = 2 * F[1] - dp if cal.scenario in ("WT", "F") else F[1]
    eps = max(eps, 0.0)
    o = solve_eps(sub, eps, time_limit=time_limit)
    info = {"eps_w": eps, "status": o["status"], "time": o["time"], "binaries": o["n_binaries"]}
    if "lam" not in o:
        return x, F, info
    xn = x.copy()
    xn[:cal.ws.k] = o["lam"]
    Fn = cal.evaluate(xn)
    if (Fn[0] < F[0] - 1e-12 and Fn[1] <= F[1] + 1e-9) or (Fn[0] <= F[0] + 1e-12 and Fn[1] < F[1] - 1e-9):
        return xn, Fn, info
    return x, F, info
