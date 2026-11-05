"""Numerical verification of every statement of the manuscript (current numbering).

Each test checks one statement against an independent computation that does not use it
(re-running PROMETHEE II on subsets, a generic least-squares solver, enumeration, ...),
on random instances that include criteria to minimise, zero weights, ties and several
preference-function types. The mapping statement -> test is listed in VERIFICATION.md.
"""
import itertools

import numpy as np
import pytest

from gpii.calibration import Calibration, WeightSpace
from gpii.conflict import conflict_measures
from gpii.core import promethee, relations_promethee_i
from gpii.cycles import cyclic_residual, cyclicity_index
from gpii.model import Model, unicriterion

TYPES = ["usual", "ushape", "vshape", "level", "linear", "gauss"]
SEEDS = range(8)


def instance(rng, m=9, n=4, types=None, ties=False, minimise=True, zero_weight=False):
    X = rng.normal(size=(m, n))
    if ties:
        X = np.round(X, 0)                      # many equal evaluations
    w = rng.dirichlet(np.ones(n))
    if zero_weight:
        w[0] = 0.0
        w = w / w.sum()
    maxi = np.ones(n, bool)
    if minimise:
        maxi[::2] = False
    p = rng.uniform(0.3, 1.5, n)
    q = p * rng.uniform(0, 0.5, n)
    types = types or tuple(rng.choice(TYPES, n))
    return X, Model(maxi, w, q, p, types=types, s=(q + p) / 2)


def Pi_of(X, model):
    return promethee(X, model).P


def D_on(Pi, B, a, b):
    return sum(Pi[a, s] - Pi[b, s] for s in B)


def all_variants(seed):
    rng = np.random.default_rng(seed)
    return [instance(rng), instance(rng, ties=True), instance(rng, zero_weight=True),
            instance(rng, types=("linear",) * 4), instance(rng, types=("gauss",) * 4)]


# ---------------------------------------------------------------------------
# Lemma 1 (basic facts)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_lemma1_basic_facts(seed):
    for X, model in all_variants(seed):
        pr = promethee(X, model)
        Pi, m = pr.P, X.shape[0]
        # (i) |Pi| <= 1 and pi(a,b) + pi(b,a) <= 1
        assert np.abs(Pi).max() <= 1 + 1e-12
        assert (pr.pi + pr.pi.T).max() <= 1 + 1e-12
        # (ii) phi = row sums of Pi / (m-1)
        assert np.allclose(pr.phi, Pi.sum(1) / (m - 1))
        # (iii) D_B = (|B|-1)(phi_B(a) - phi_B(b)) for random subsets, using PROMETHEE on B
        rng = np.random.default_rng(seed)
        for _ in range(10):
            B = np.sort(rng.choice(m, rng.integers(2, m + 1), replace=False))
            phiB = promethee(X[B], model).phi
            for i, j in itertools.combinations(range(len(B)), 2):
                a, b = B[i], B[j]
                assert D_on(Pi, B, a, b) == pytest.approx((len(B) - 1) * (phiB[i] - phiB[j]), abs=1e-10)


# ---------------------------------------------------------------------------
# Theorem 1 (deletion and insertion) and Example 1
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_theorem1_deletion_insertion(seed):
    rng = np.random.default_rng(100 + seed)
    for X, model in all_variants(seed):
        m = X.shape[0]
        Y = rng.normal(size=(3, X.shape[1]))               # alternatives outside A
        XA = np.vstack([X, Y])
        Pi = Pi_of(XA, model)                               # preferences on A plus the new ones
        A = list(range(m))
        for a, b in itertools.combinations(A, 2):
            for x in A:
                if x in (a, b):
                    continue
                delta = Pi[a, x] - Pi[b, x]
                Bx = [s for s in A if s != x]
                assert D_on(Pi, Bx, a, b) == pytest.approx(D_on(Pi, A, a, b) - delta, abs=1e-12)
                C = Pi[a, b] + Pi[b, x] + Pi[x, a]
                assert delta == pytest.approx(Pi[a, b] - C, abs=1e-12)
                assert abs(delta) <= 2 + 1e-12
            for y in range(m, m + 3):
                assert D_on(Pi, A + [y], a, b) == pytest.approx(D_on(Pi, A, a, b) + Pi[a, y] - Pi[b, y], abs=1e-12)


def test_example1_numbers():
    X = np.array([[1.0, 0.0], [0.0, 2.0], [2.0, 1.0]])           # a, b, c
    model = Model(np.ones(2, bool), np.array([0.6, 0.4]), np.zeros(2), np.ones(2), types=("linear",) * 2)
    pr = promethee(X, model)
    Pi = pr.P
    assert Pi[0, 1] == pytest.approx(0.2) and Pi[1, 2] == pytest.approx(-0.2) and Pi[2, 0] == pytest.approx(1.0)
    assert np.allclose(pr.phi, [-0.4, -0.2, 0.6])
    assert 2 * (pr.phi[1] - pr.phi[0]) == pytest.approx(0.4)                  # D(b,a)
    C = Pi[1, 0] + Pi[0, 2] + Pi[2, 1]
    assert C == pytest.approx(-1.0)
    assert Pi[1, 0] - C == pytest.approx(0.8)                                  # delta_c(b,a)
    phi_ab = promethee(X[:2], model).phi                                       # delete c
    assert phi_ab[0] > phi_ab[1]                                               # a above b
    assert cyclicity_index(Pi) == pytest.approx(25 / 81)
    R = cyclic_residual(Pi)
    assert R[0, 1] == pytest.approx(1 / 3) and R[1, 2] == pytest.approx(1 / 3) and R[2, 0] == pytest.approx(1 / 3)


def test_example1_transitive_but_not_additive():
    """Review F1: C != 0 does not mean an ordinal cycle; Example 1 is transitive."""
    X = np.array([[1.0, 0.0], [0.0, 2.0], [2.0, 1.0]])
    model = Model(np.ones(2, bool), np.array([0.6, 0.4]), np.zeros(2), np.ones(2), types=("linear",) * 2)
    Pi = Pi_of(X, model)
    assert Pi[2, 0] > 0 and Pi[2, 1] > 0 and Pi[0, 1] > 0          # c > a, c > b, a > b: transitive
    assert abs(Pi[0, 1] + Pi[1, 2] + Pi[2, 0]) > 0.5                  # but the circulation is not zero


@pytest.mark.parametrize("seed", SEEDS)
def test_breaking_condition_after_theorem1(seed):
    """Review F2: (a,b) in S is broken by x iff D(a,b) <= Pi(a,b) - C(a,b,x)."""
    for X, model in all_variants(seed):
        Pi = Pi_of(X, model)
        m = X.shape[0]
        for a, b in itertools.permutations(range(m), 2):
            D = D_on(Pi, range(m), a, b)
            if D <= 1e-12:
                continue
            for x in range(m):
                if x in (a, b):
                    continue
                Cx = Pi[a, b] + Pi[b, x] + Pi[x, a]
                broken = D_on(Pi, [s for s in range(m) if s != x], a, b) <= 1e-12
                assert broken == (D <= Pi[a, b] - Cx + 1e-12)


def test_breaking_with_zero_circulation():
    """Review F2: a pair can be broken by x although C(a,b,x) = 0 (no cycle is needed)."""
    # two criteria, Type III (linear, q = 0), p = 1, equal weights
    X = np.array([[0.7, 1.3], [0.3, 1.8], [2.3, 2.6], [1.0, 0.3], [1.2, 1.8]])
    model = Model(np.ones(2, bool), np.array([0.5, 0.5]), np.zeros(2), np.ones(2), types=("linear",) * 2)
    Pi = Pi_of(X, model)
    a, b, x = 1, 0, 4
    assert abs(Pi[a, b] + Pi[b, x] + Pi[x, a]) < 1e-12                # C(a,b,x) = 0
    D = D_on(Pi, range(5), a, b)
    assert D > 0 and D_on(Pi, [0, 1, 2, 3], a, b) <= 1e-12            # broken by deleting x


# ---------------------------------------------------------------------------
# Definition 1 and Proposition 1 (exact evaluation)
# ---------------------------------------------------------------------------
def brute_force_measures(X, model):
    """Fr and RR by re-running PROMETHEE II on every A minus {x} (no use of Theorem 1)."""
    m = X.shape[0]
    phi = promethee(X, model).phi
    S = [(a, b) for a in range(m) for b in range(m) if phi[a] - phi[b] > 1e-10]
    broken, events = set(), 0
    for x in range(m):
        idx = [i for i in range(m) if i != x]
        ph = dict(zip(idx, promethee(X[idx], model).phi))
        for a, b in S:
            if x in (a, b):
                continue
            if ph[a] - ph[b] <= 1e-10:
                events += 1
                broken.add((a, b))
    return len(broken) / len(S), events / ((m - 2) * len(S)), S, phi


@pytest.mark.parametrize("seed", SEEDS)
def test_prop1_exact_evaluation(seed):
    for X, model in all_variants(seed):
        fr, rr, S, phi = brute_force_measures(X, model)
        c = conflict_measures(Pi_of(X, model))
        assert c["fragility"] == pytest.approx(fr, abs=1e-12)
        assert c["event_rate"] == pytest.approx(rr, abs=1e-12)
        # only pairs with D <= 2 can be broken
        m = X.shape[0]
        Pi = Pi_of(X, model)
        for a, b in S:
            D = (m - 1) * (phi[a] - phi[b])
            if D > 2 + 1e-9:
                for x in range(m):
                    if x not in (a, b):
                        assert D - (Pi[a, x] - Pi[b, x]) > 0


# ---------------------------------------------------------------------------
# Theorem 2 (least-squares score), Definition 2, Theorem 3
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_theorem2_least_squares(seed):
    for X, model in all_variants(seed):
        Pi = Pi_of(X, model)
        m = Pi.shape[0]
        # generic least squares: unknown u in R^m, one equation per ordered pair
        rows, rhs = [], []
        for a in range(m):
            for b in range(m):
                r = np.zeros(m); r[a] += 1; r[b] -= 1
                rows.append(r); rhs.append(Pi[a, b])
        u, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
        v = Pi.mean(1)
        assert np.allclose(u - u.mean(), v - v.mean(), atol=1e-10)                 # (i)
        G = v[:, None] - v[None, :]
        R = Pi - G
        assert (Pi ** 2).sum() == pytest.approx((G ** 2).sum() + (R ** 2).sum())  # (ii)
        Cmean = np.array([[np.mean([Pi[a, b] + Pi[b, x] + Pi[x, a] for x in range(m)]) for b in range(m)]
                          for a in range(m)])
        assert np.allclose(R, Cmean, atol=1e-12)                                    # (iii)
        K = cyclicity_index(Pi)
        assert 0 <= K <= 1 + 1e-12                                                  # Definition 2


def test_theorem3_cycle_free():
    rng = np.random.default_rng(3)
    for _ in range(20):
        m = 8
        u = rng.normal(size=m)
        Pi = u[:, None] - u[None, :]
        assert cyclicity_index(Pi) < 1e-12
        assert np.allclose([Pi[a, b] + Pi[b, x] + Pi[x, a] for a in range(m) for b in range(m) for x in range(m)], 0)
        for _ in range(20):
            B = rng.choice(m, rng.integers(2, m + 1), replace=False)
            for a, b in itertools.combinations(B, 2):
                assert D_on(Pi, B, a, b) == pytest.approx(len(B) * (u[a] - u[b]))
    # converse direction on real PROMETHEE matrices: K = 0 <=> all circulations vanish
    for seed in SEEDS:
        for X, model in all_variants(seed):
            Pi = Pi_of(X, model)
            m = Pi.shape[0]
            maxC = max(abs(Pi[a, b] + Pi[b, x] + Pi[x, a]) for a in range(m) for b in range(m) for x in range(m))
            assert (cyclicity_index(Pi) < 1e-12) == (maxC < 1e-9)


# ---------------------------------------------------------------------------
# Theorem 4 (degenerate models)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("code", TYPES)
@pytest.mark.parametrize("sigma", [True, False])
def test_theorem4_single_criterion(code, sigma):
    rng = np.random.default_rng(4)
    X = np.round(rng.normal(size=(10, 3)), 1)
    Y = np.round(rng.normal(size=(4, 3)), 1)                    # alternatives that may be inserted
    model = Model(np.array([sigma, True, True]), np.array([1.0, 0.0, 0.0]), np.full(3, 0.2),
                  np.full(3, 0.9), types=(code,) * 3, s=np.full(3, 0.5))
    pr = promethee(X, model)
    _, _, J = relations_promethee_i(pr)
    assert not J.any()                                            # Inc = 0
    XY = np.vstack([X, Y])
    Pi = Pi_of(XY, model)
    m = X.shape[0]
    D = Pi[:m, :m].sum(1)[:, None] - Pi[:m, :m].sum(1)[None, :]
    for B in itertools.islice(itertools.chain.from_iterable(
            itertools.combinations(range(m + 4), r) for r in range(2, 8)), 2000):
        for a, b in itertools.combinations([i for i in B if i < m], 2):
            if abs(D[a, b]) > 1e-12:
                s = np.sign(D[a, b])
                assert s * D_on(Pi, B, a, b) >= -1e-12            # never inverted (ties allowed)


def test_theorem4_scope_of_case_i():
    """Review F11: one criterion removes Inc and strict reversals, but not ties nor cycles."""
    X = np.array([[0.3], [0.0], [1.0], [2.5]])                       # Type V, q = 0.5, p = 1
    model = Model(np.ones(1, bool), np.ones(1), np.full(1, 0.5), np.ones(1), types=("linear",))
    Pi = Pi_of(X, model)
    c = conflict_measures(Pi)
    assert c["fragility"] > 0 and c["event_rate"] > 0                 # a tie is created
    assert c["cyclicity"] > 1e-6                                      # saturation leaves cycles
    _, _, J = relations_promethee_i(promethee(X, model))
    assert not J.any()


def test_theorem4_scope_of_case_ii():
    """Review F11: an unsaturated linear model can still have PROMETHEE I incomparabilities."""
    found = False
    for seed in range(200):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(8, 3))
        p = np.ptp(X, 0) * 1.01
        model = Model(np.ones(3, bool), rng.dirichlet(np.ones(3)), np.zeros(3), p, types=("linear",) * 3)
        Pi = Pi_of(X, model)
        assert conflict_measures(Pi)["event_rate"] == 0
        _, _, J = relations_promethee_i(promethee(X, model))
        if J.any():
            found = True
            break
    assert found


def test_theorem4_unsaturated_linear():
    rng = np.random.default_rng(5)
    for _ in range(10):
        X = rng.normal(size=(9, 4))
        maxi = rng.random(4) < 0.5
        p = np.ptp(X, 0) * rng.uniform(1.0, 2.0, 4)
        w = rng.dirichlet(np.ones(4))
        model = Model(maxi, w, np.zeros(4), p, types=("linear",) * 4)
        Pi = Pi_of(X, model)
        sig = np.where(maxi, 1.0, -1.0)
        u = (X * sig * w / p).sum(1)
        assert np.allclose(Pi, u[:, None] - u[None, :])
        c = conflict_measures(Pi)
        assert c["cyclicity"] < 1e-12 and c["fragility"] == 0 and c["event_rate"] == 0


# ---------------------------------------------------------------------------
# Proposition 2 (step functions; cyclicity bound)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_prop2_step_functions(seed):
    """Perturbations of w that keep all signs of w'd_ab, w'e_abx (and of the flow differences)
    keep Fr, RR (and Inc) unchanged."""
    rng = np.random.default_rng(seed)
    X, model = instance(rng, m=8, n=4)
    Pj = unicriterion(model, X, X)
    Pa = Pj - Pj.transpose(0, 2, 1)
    m = X.shape[0]

    def forms(w):
        Pi = np.tensordot(w, Pa, 1)
        r = Pi.sum(1)
        D = r[:, None] - r[None, :]
        E = D[:, :, None] - (Pi[:, None, :] - Pi[None, :, :])        # E[a,b,x] = D - delta_x
        pi = np.tensordot(w, Pj, 1)
        fp, fm = pi.sum(1), pi.sum(0)
        return np.concatenate([np.sign(np.round(D, 12)).ravel(), np.sign(np.round(E, 12)).ravel(),
                               np.sign(np.round(fp[:, None] - fp[None, :], 12)).ravel(),
                               np.sign(np.round(fm[:, None] - fm[None, :], 12)).ravel()]), Pi, pi

    w0 = model.w
    s0, Pi0, _ = forms(w0)
    c0 = conflict_measures(Pi0)
    inc0 = relations_promethee_i(promethee(X, model))[2].sum()
    checked = 0
    for _ in range(300):
        d = rng.normal(size=4); d -= d.mean()
        w = np.clip(w0 + 1e-4 * d, 0, None); w /= w.sum()
        s, Pi, _ = forms(w)
        if np.array_equal(s, s0):
            checked += 1
            c = conflict_measures(Pi)
            assert c["fragility"] == c0["fragility"] and c["event_rate"] == c0["event_rate"]
            assert relations_promethee_i(promethee(X, model.with_(w=w)))[2].sum() == inc0
    assert checked > 50


@pytest.mark.parametrize("seed", SEEDS)
def test_prop2_cyclicity_ratio_and_bound(seed):
    from scipy.linalg import eigh
    rng = np.random.default_rng(seed)
    X, model = instance(rng, m=10, n=4)
    Pj = unicriterion(model, X, X)
    Pa = Pj - Pj.transpose(0, 2, 1)
    R = np.array([cyclic_residual(p) for p in Pa])
    A = np.einsum("iab,jab->ij", R, R)
    B = np.einsum("iab,jab->ij", Pa, Pa)
    for mode in ["free", "order"]:
        ws = WeightSpace.build(model.w, mode)
        V = ws.V
        Kmin = eigh(V.T @ A @ V, V.T @ B @ V, eigvals_only=True)[0]
        for lam in rng.dirichlet(np.ones(ws.k), size=200):
            w = V @ lam
            K = cyclicity_index(np.tensordot(w, Pa, 1))
            assert K == pytest.approx(w @ A @ w / (w @ B @ w), rel=1e-9)
            assert K >= Kmin - 1e-10


# ---------------------------------------------------------------------------
# Lemma 2 (stability bound) and Corollary 1 (stability radius)
# ---------------------------------------------------------------------------
def test_lemma2_bound_and_tightness():
    rng = np.random.default_rng(6)
    for _ in range(2000):
        n = rng.integers(2, 8)
        c = rng.normal(size=n)
        w, w2 = rng.dirichlet(np.ones(n)), rng.dirichlet(np.ones(n))
        tv = 0.5 * np.abs(w - w2).sum()
        assert abs(w @ c - w2 @ c) <= tv * np.ptp(c) + 1e-12
    # tightness: move mass t from argmin c to argmax c
    for _ in range(200):
        n = rng.integers(2, 8)
        c = rng.normal(size=n)
        w = rng.dirichlet(np.ones(n))
        i, j = np.argmin(c), np.argmax(c)
        t = w[i] * rng.uniform(0, 1)
        w2 = w.copy(); w2[i] -= t; w2[j] += t
        assert abs(w2 @ c - w @ c) == pytest.approx(t * np.ptp(c))
        assert 0.5 * np.abs(w2 - w).sum() == pytest.approx(t)
    # sign statement
    for _ in range(2000):
        n = rng.integers(2, 8)
        c = rng.normal(size=n)
        w0 = rng.dirichlet(np.ones(n))
        r = abs(w0 @ c) / np.ptp(c)
        w = rng.dirichlet(np.ones(n))
        if 0.5 * np.abs(w - w0).sum() < r:
            assert np.sign(w @ c) == np.sign(w0 @ c)


@pytest.mark.parametrize("seed", SEEDS)
def test_corollary1_stability_radius(seed):
    rng = np.random.default_rng(seed)
    X, model = instance(rng, m=9, n=4)
    Pj = unicriterion(model, X, X)
    Pa = Pj - Pj.transpose(0, 2, 1)
    rj = Pa.sum(2)
    w0 = model.w
    r0 = w0 @ rj
    m = X.shape[0]
    pairs = [(a, b) for a in range(m) for b in range(m) if r0[a] - r0[b] > 1e-10]
    radius = {}
    for a, b in pairs:
        d = rj[:, a] - rj[:, b]
        radius[(a, b)] = (r0[a] - r0[b]) / np.ptp(d) if np.ptp(d) > 0 else np.inf
    rmin = min(radius.values())
    for _ in range(500):
        w = rng.dirichlet(np.ones(4))
        t = 0.5 * np.abs(w - w0).sum()
        r = w @ rj
        for (a, b), rad in radius.items():
            if t < rad:
                assert r[a] - r[b] > 0                                  # order kept
        if t < rmin:
            assert np.array_equal(np.argsort(-r, kind="stable"), np.argsort(-r0, kind="stable"))
        # event status
        Pi = np.tensordot(w, Pa, 1)
        Pi0 = np.tensordot(w0, Pa, 1)
        for (a, b), rad in list(radius.items())[:15]:
            for x in range(m):
                if x in (a, b):
                    continue
                e = (rj[:, a] - rj[:, b]) - Pa[:, a, x] + Pa[:, b, x]
                e0 = w0 @ e
                if np.ptp(e) > 0 and t < min(rad, abs(e0) / np.ptp(e)):
                    assert np.sign(w @ e) == np.sign(e0)


# ---------------------------------------------------------------------------
# Proposition 3 (encoding)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("w0", [[0.5, 0.3, 0.2], [0.4, 0.2, 0.2, 0.1, 0.1], [0.3, 0.3, 0.2, 0.2]])
def test_prop3_encoding(w0):
    rng = np.random.default_rng(7)
    w0 = np.array(w0)
    ws = WeightSpace.build(w0, "order")
    V, k = ws.V, ws.k
    # one-to-one: explicit inverse of the proof (Step 2)
    for lam in rng.dirichlet(np.ones(k), size=200):
        w = V @ lam
        assert ws.contains(w)
        c = np.array([w[L[0]] for L in ws.levels] + [0.0])
        T = np.cumsum([len(L) for L in ws.levels])
        lam2 = T * (c[:-1] - c[1:])
        assert np.allclose(lam2, lam) and lam2.sum() == pytest.approx(1)
    # (i) uniformity: compare with rejection sampling from the uniform simplex
    n = len(w0)
    U = rng.dirichlet(np.ones(n), size=400000)
    acc = np.array([ws.contains(u, tol=1e-12) for u in U[:200000]]) if k < n else None
    if k == n:
        acc = np.all(np.diff(U[:, np.argsort(-w0, kind="stable")], axis=1) <= 0, axis=1)
        ref = U[acc]
        enc = (V @ rng.dirichlet(np.ones(k), size=len(ref)).T).T
        assert np.allclose(ref.mean(0), enc.mean(0), atol=4e-3)
        assert np.allclose(np.cov(ref.T), np.cov(enc.T), atol=2e-3)
    # (iii) contraction of the total variation
    for _ in range(500):
        l1, l2 = rng.dirichlet(np.ones(k)), rng.dirichlet(np.ones(k))
        assert 0.5 * np.abs(V @ (l1 - l2)).sum() <= 0.5 * np.abs(l1 - l2).sum() + 1e-12
    # (iv) rank-order centroid for strict orders
    if k == n:
        cen = V @ np.full(k, 1 / k)
        roc = np.array([sum(1 / l for l in range(i, n + 1)) / n for i in range(1, n + 1)])
        assert np.allclose(np.sort(cen)[::-1], roc)


def test_prop3_worked_example():
    ws = WeightSpace.build(np.array([0.5, 0.3, 0.2]), "order")
    assert np.allclose(ws.V, [[1, 1 / 2, 1 / 3], [0, 1 / 2, 1 / 3], [0, 0, 1 / 3]])
    assert np.allclose(ws.lam0, [0.2, 0.2, 0.6])


# ---------------------------------------------------------------------------
# Proposition 4 (exact model) against exact enumeration of the cells (one free coordinate)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("measure", ["event_rate", "fragility"])
@pytest.mark.parametrize("seed", range(6))
def test_prop4_exact_model_by_enumeration(seed, measure):
    from gpii.exact import solve_eps
    rng = np.random.default_rng(200 + seed)
    m = 9
    X = rng.normal(size=(m, 2))
    model = Model(np.ones(2, bool), rng.dirichlet(np.ones(2)), np.zeros(2), 0.4 * X.std(0),
                  types=("linear",) * 2)
    cal = Calibration(X, model, measure, "W", "free")
    Pj = unicriterion(model, X, X)
    Pa = Pj - Pj.transpose(0, 2, 1)
    rj = Pa.sum(2)
    # every linear form f(w) = w'c with w = (t, 1 - t): roots in t
    forms = []                                                         # (c, pair coefficients)
    for a, b in itertools.combinations(range(m), 2):
        d = rj[:, a] - rj[:, b]
        forms.append((d, d))
        for x in range(m):
            if x not in (a, b):
                forms.append((d - Pa[:, a, x] + Pa[:, b, x], d))
    w0 = model.w
    roots = []
    for c, _ in forms:
        if abs(c[0] - c[1]) > 1e-14:
            roots.append(-c[1] / (c[0] - c[1]))
    for eps in [0.02, 0.05, 0.1, 0.3]:
        t0 = model.w[0]
        lo, hi = max(0.0, t0 - eps), min(1.0, t0 + eps)               # TV((t,1-t), w0) = |t - t0|
        pts = np.sort([lo, hi] + [r for r in roots if lo < r < hi])
        cand = np.concatenate([pts, (pts[:-1] + pts[1:]) / 2])
        # uncertified forms of the candidate pairs (the margin of Prop. 4 applies to these only)
        unc = [c for c, d in forms
               if abs(w0 @ d) <= 2 + eps * np.ptp(d) and abs(w0 @ c) <= eps * np.ptp(c)]
        best = np.inf
        for t in cand:
            w = np.array([t, 1 - t])
            if unc and np.min(np.abs([w @ c for c in unc])) < 1e-5:     # outside the set W of Prop. 4
                continue
            # on W every pair is strictly ordered (Step 1 of the proof)
            assert np.min(np.abs([w @ d for _, d in forms])) > 0
            best = min(best, conflict_measures(np.tensordot(w, Pa, 1))[measure])
        o = solve_eps(cal, eps, time_limit=120)
        assert (o["status"] == 0) == np.isfinite(best)                  # feasible iff W non-empty
        assert o["F"][0] * cal.C0 == pytest.approx(best, abs=1e-12)
        assert o["milp_value"] * cal.C0 == pytest.approx(best, abs=1e-12)   # objective = true count
        w = cal.ws.V @ o["lam"]                                          # the optimum lies in W
        assert abs(w[0] - t0) <= eps + 1e-9
        assert not unc or np.min(np.abs([w @ c for c in unc])) >= 1e-5 - 1e-9
