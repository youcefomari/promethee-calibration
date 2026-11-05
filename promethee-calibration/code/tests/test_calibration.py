"""Checks of the calibration paper's statements (conflict measures, degeneracy, structure,
feasible encoding, exact model)."""
import itertools

import numpy as np
import pytest

from gpii.calibration import Calibration, WeightSpace, knee_point, nondominated
from gpii.conflict import conflict_measures, deletion_events
from gpii.core import promethee, relations_promethee_i
from gpii.cycles import cyclic_residual, cyclicity_index
from gpii.model import Model, unicriterion
from gpii.moea import hypervolume, igd_plus, run
from gpii.reversal import loo_reversal_rate, single_change_reversible, tsign


def rand_problem(rng, m=12, n=4, ratio=0.5):
    X = rng.normal(size=(m, n))
    w = rng.dirichlet(np.ones(n))
    return X, Model(np.ones(n, bool), w, np.zeros(n), ratio * X.std(0))


def brute_measures(X, model):
    """Fragility and event rate by re-running PROMETHEE II on every A minus {x}."""
    m = X.shape[0]
    phi = promethee(X, model).phi
    s0 = tsign(phi[:, None] - phi[None, :])
    broken = np.zeros((m, m), bool)
    ev = tot = 0
    for x in range(m):
        idx = np.delete(np.arange(m), x)
        ph = promethee(X[idx], model).phi
        s = tsign(ph[:, None] - ph[None, :])
        b = s0[np.ix_(idx, idx)]
        st = np.triu(b != 0, 1)
        fl = st & (s != b)
        ev += fl.sum(); tot += st.sum()
        broken[np.ix_(idx, idx)] |= fl
    strict = np.triu(s0 != 0, 1)
    return broken[strict].mean(), ev / tot


@pytest.mark.parametrize("seed", range(6))
def test_conflict_measures_exact(seed):
    rng = np.random.default_rng(seed)
    X, model = rand_problem(rng, m=14, n=3 + seed % 3)
    P = promethee(X, model).P
    c = conflict_measures(P)
    frag, ev = brute_measures(X, model)
    assert c["fragility"] == pytest.approx(frag, abs=1e-12)
    assert c["event_rate"] == pytest.approx(ev, abs=1e-12)
    assert c["event_rate"] == pytest.approx(loo_reversal_rate(P), abs=1e-12)
    g = single_change_reversible(P)
    assert c["fragility"] == pytest.approx(g["deletion"].sum() / g["pairs"].sum())


def test_candidate_pairs_suffice():
    """Pairs with |D| > 2 are never broken by one deletion (|delta| <= 2)."""
    rng = np.random.default_rng(3)
    X, model = rand_problem(rng, m=20, n=4, ratio=0.3)
    P = promethee(X, model).P
    r = P.sum(1)
    D = r[:, None] - r[None, :]
    for x in range(20):
        dx = P[:, x][:, None] + P[x, :][None, :]
        assert np.abs(dx).max() <= 2 + 1e-12
    a, b, flips, _ = deletion_events(P)
    assert np.all(np.abs(D[a, b]) <= 2 + 1e-10)


def test_degeneracy_single_criterion():
    """Theorem (i): one positive weight -> no incomparability, no strict reversal anywhere."""
    rng = np.random.default_rng(5)
    for code in ["usual", "level", "linear", "vshape", "gauss"]:
        X = rng.normal(size=(15, 3))
        model = Model(np.ones(3, bool), np.array([1.0, 0, 0]), np.full(3, 0.2), np.full(3, 0.8),
                      types=(code,) * 3)
        pr = promethee(X, model)
        _, _, R = relations_promethee_i(pr)
        assert not R.any()
        s0 = np.sign(pr.phi[:, None] - pr.phi[None, :])
        for B in itertools.islice(itertools.combinations(range(15), 6), 300):
            B = np.array(B)
            ph = pr.subproblem(B).phi
            s = np.sign(ph[:, None] - ph[None, :])
            assert not np.any(s * s0[np.ix_(B, B)] < 0)       # never inverted


def test_degeneracy_unsaturated_linear():
    """Theorem (ii): unsaturated linear functions -> K = 0 and identical order in every subproblem."""
    rng = np.random.default_rng(6)
    X = rng.normal(size=(15, 4))
    model = Model(np.ones(4, bool), rng.dirichlet(np.ones(4)), np.zeros(4), np.ptp(X, 0))
    pr = promethee(X, model)
    assert cyclicity_index(pr.P) < 1e-12
    c = conflict_measures(pr.P)
    assert c["fragility"] == 0 and c["event_rate"] == 0


def test_cyclicity_rayleigh_quotient():
    """Proposition: K(w) = w'Aw / w'Bw with A <= B (Loewner) for fixed thresholds."""
    rng = np.random.default_rng(7)
    X, model = rand_problem(rng, m=12, n=4, ratio=0.4)
    Pj = unicriterion(model, X, X)
    Pa = Pj - Pj.transpose(0, 2, 1)
    R = np.array([cyclic_residual(p) for p in Pa])
    A = np.einsum("iab,jab->ij", R, R)
    B = np.einsum("iab,jab->ij", Pa, Pa)
    assert np.linalg.eigvalsh(B - A).min() > -1e-10
    from scipy.linalg import eigh
    lam = eigh(A, B, eigvals_only=True)
    for _ in range(20):
        w = rng.dirichlet(np.ones(4))
        K = cyclicity_index(np.tensordot(w, Pa, 1))
        assert K == pytest.approx(w @ A @ w / (w @ B @ w), rel=1e-10)
        assert lam[0] - 1e-12 <= K <= lam[-1] + 1e-12


@pytest.mark.parametrize("w0", [[0.3, 0.3, 0.2, 0.1, 0.1], [0.5, 0.2, 0.2, 0.1, 0.0],
                                [0.25, 0.25, 0.25, 0.25], [0.4, 0.3, 0.2, 0.1]])
def test_feasible_encoding(w0):
    """Proposition: lam -> V lam is a bijection from the simplex onto the admissible weights."""
    w0 = np.array(w0)
    ws = WeightSpace.build(w0, "order")
    assert np.allclose(ws.weights(ws.lam0), w0 / w0.sum())
    rng = np.random.default_rng(0)
    L = rng.dirichlet(np.ones(ws.k), size=200)
    for lam in L:
        w = ws.weights(lam)
        assert ws.contains(w)
        lam2, *_ = np.linalg.lstsq(ws.V, w, rcond=None)
        assert np.allclose(lam2, lam, atol=1e-10)                   # injective
        # deviation contraction: TV(w, w0) <= TV(lam, lam0)
        assert 0.5 * np.abs(w - ws.weights(ws.lam0)).sum() <= 0.5 * np.abs(lam - ws.lam0).sum() + 1e-12
    assert np.linalg.matrix_rank(ws.V) == ws.k


def test_feasible_encoding_surjective():
    """Any weight vector respecting the weak order has coordinates in the simplex."""
    ws = WeightSpace.build(np.array([0.3, 0.3, 0.2, 0.1, 0.1]), "order")
    w = np.array([0.35, 0.35, 0.2, 0.05, 0.05])
    lam, *_ = np.linalg.lstsq(ws.V, w, rcond=None)
    assert np.all(lam >= -1e-12) and lam.sum() == pytest.approx(1)
    assert np.allclose(ws.V @ lam, w)


def test_calibration_problem_and_moea():
    rng = np.random.default_rng(2)
    X, model = rand_problem(rng, m=15, n=4, ratio=0.4)
    cal = Calibration(X, model, "event_rate", "WT", "order")
    f0 = cal.evaluate(cal.x0)
    assert f0[0] == pytest.approx(1.0) or cal.C0 == 0
    assert f0[1] == pytest.approx(0.0, abs=1e-12)
    S = cal.sample(50, rng)
    for x in S:
        assert cal.ws.contains(cal.decode(x).w)
    out = run(cal, "NSGA-II", budget=400, pop=20, seed=0)
    F = out["F"]
    assert nondominated(F).all()
    assert hypervolume(F) >= hypervolume(f0[None, :]) - 1e-12
    assert igd_plus(F, F) == 0.0
    assert 0 <= knee_point(F) < len(F)


def test_hypervolume_exact():
    F = np.array([[0.2, 0.6], [0.5, 0.3], [0.8, 0.1]])
    # union of rectangles to ref (1, 1)
    assert hypervolume(F) == pytest.approx(0.8 * 0.4 + 0.5 * 0.3 + 0.2 * 0.2)


def test_exact_milp_small():
    from gpii.exact import solve_eps
    rng = np.random.default_rng(11)
    X, model = rand_problem(rng, m=8, n=3, ratio=0.3)
    cal = Calibration(X, model, "event_rate", "W", "free")
    o = solve_eps(cal, 0.2, time_limit=60)
    assert o["status"] == 0
    # exhaustive grid search over the simplex cannot beat the MILP optimum (up to the margin)
    best = np.inf
    for a in np.linspace(0, 1, 41):
        for b in np.linspace(0, 1 - a, max(2, int(round((1 - a) * 40)) + 1)):
            x = np.array([a, b, 1 - a - b])
            if cal.deviation(x) <= 0.2:
                best = min(best, cal.evaluate(x)[0])
    assert o["F"][0] <= best + 1e-9


def test_full_scenario_encoding():
    """Scenario F: theta0 is reproduced exactly, deviations are in [0, 1], shapes switch."""
    rng = np.random.default_rng(21)
    X, model = rand_problem(rng, m=14, n=4, ratio=0.5)
    cal = Calibration(X, model, "event_rate", "F", "order")
    m0 = cal.decode(cal.x0)
    assert np.allclose(unicriterion(m0, X, X), unicriterion(model, X, X))
    assert cal.evaluate(cal.x0)[1] == pytest.approx(0.0, abs=1e-12) and cal.shape_changes(cal.x0) == 0
    S = cal.sample(200, rng)
    for x in S:
        dw, df = cal.deviation_parts(x)
        assert 0 <= dw <= 1 and 0 <= df <= 1
        mdl = cal.decode(x)
        assert cal.ws.contains(mdl.w)
        assert np.all(mdl.q <= 0.5 * mdl.p + 1e-12)
    x = cal.x0.copy()
    x[cal.ws.k + 2 * cal.n] = 0.9                        # criterion 0 -> Gaussian
    assert cal.decode(x).types[0] == "gauss" and cal.shape_changes(x) == 1
    assert cal.function_deviation(x)[0] == 1.0
    # WT is nested in F: same thresholds, same deviation
    wt = Calibration(X, model, "event_rate", "WT", "order")
    xw = wt.x0.copy(); xw[wt.ws.k:] = 0.3
    xf = cal.x0.copy(); xf[cal.ws.k:cal.ws.k + cal.n] = 0.3
    assert np.allclose(wt.decode(xw).p, cal.decode(xf).p)
    assert wt.deviation(xw) == pytest.approx(cal.deviation(xf))


def test_full_scenario_refine():
    from gpii.exact import refine
    rng = np.random.default_rng(22)
    X, model = rand_problem(rng, m=10, n=3, ratio=0.4)
    cal = Calibration(X, model, "event_rate", "F", "free")
    x = cal.sample(5, rng)[3]
    xn, fn, info = refine(cal, x, time_limit=60)
    f = cal.evaluate(x)
    assert fn[0] <= f[0] + 1e-12 and fn[1] <= f[1] + 1e-9
    assert np.allclose(cal.decode(xn).p, cal.decode(x).p)


def test_wt_embeds_in_f_with_elicited_gaussian():
    """Scenario WT is the part of F where types and indifference ratios are kept, also when some
    elicited criteria are Gaussian (main threshold = s for these, conventions of Sec. 4.1)."""
    import numpy as np
    from gpii.calibration import Calibration
    from gpii.model import Model
    rng = np.random.default_rng(3)
    X = rng.normal(size=(12, 4))
    s0 = np.array([0.4, 0.6, 0.5, 0.7])
    model = Model(np.ones(4, bool), np.full(4, 0.25), np.array([0.0, 0.0, 0.1, 0.2]),
                  np.array([0.8, 1.2, 0.9, 1.0]), types=("gauss", "gauss", "linear", "linear"), s=s0)
    model = model.with_(p=np.where(np.array([1, 1, 0, 0], bool), 2 * s0, model.p))
    for me in ("event_rate", "cyclicity"):
        T = Calibration(X, model, me, "WT", "free")
        F = Calibration(X, model, me, "F", "free")
        assert np.isclose(T.C0, F.C0)
        for _ in range(20):
            x = T.sample(2, rng)[1]
            y = F.x0.copy()
            y[:T.ws.k + T.n] = x
            assert np.allclose(T.evaluate(x), F.evaluate(y))
