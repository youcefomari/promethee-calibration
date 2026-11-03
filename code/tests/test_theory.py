"""Numerical verification of every formal statement of the manuscript.

Each test names the statement it checks. Random instances use a seeded generator and mix
the six preference-function shapes, so the checks are not specific to one model.
"""
import itertools

import numpy as np
import pytest

from gpii import core
from gpii.core import promethee, relations
from gpii.cycles import cyclic_residual, cyclicity_index, mean_triangle, potential
from gpii.model import Model, preference_index
from gpii.preference import CODES, preference_degree
from gpii.reversal import D_matrix, deletion_radius, insertion_worst_case

RNG = np.random.default_rng(12345)
ATOL = 1e-12


def random_instance(m=8, n=4, rng=RNG, types=None):
    X = rng.uniform(0, 10, size=(m, n))
    q = rng.uniform(0, 1.5, n)
    p = q + rng.uniform(0.5, 4, n)
    types = types or tuple(rng.choice(CODES, n))
    model = Model(rng.random(n) < 0.6, rng.uniform(0.1, 1, n), q, p, types)
    return X, model


# ----------------------------------------------------------------------------- preference functions
def test_preference_shapes():
    d = np.array([-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    assert np.allclose(preference_degree(d, "usual"), [0, 0, 1, 1, 1, 1, 1])
    assert np.allclose(preference_degree(d, "ushape", q=1.0), [0, 0, 0, 0, 1, 1, 1])
    assert np.allclose(preference_degree(d, "vshape", p=2.0), [0, 0, .25, .5, .75, 1, 1])
    assert np.allclose(preference_degree(d, "level", q=1.0, p=2.0), [0, 0, 0, 0, .5, .5, 1])
    assert np.allclose(preference_degree(d, "linear", q=1.0, p=2.0), [0, 0, 0, 0, .5, 1, 1])
    g = preference_degree(d, "gauss", s=1.0)
    assert g[0] == 0 and g[1] == 0 and np.all(np.diff(g[1:]) > 0) and g[-1] < 1


def test_thresholds_and_types_affect_pi():
    """Regression for the notebook bug: changing q, p or the type must change pi."""
    X, model = random_instance(types=("linear",) * 4)
    pi0 = preference_index(model, X)
    assert not np.allclose(pi0, preference_index(model.with_(p=model.p * 3), X))
    assert not np.allclose(pi0, preference_index(model.with_(types=("usual",) * 4), X))


# ----------------------------------------------------------------------------- Section 3
def test_lemma1_pair_decomposition():
    """Lemma 1: pi(a,b) + pi(b,a) = |P(a,b)| + 2 CI(a,b), with CI <= 1/2, |P| <= 1."""
    X, model = random_instance()
    pr = promethee(X, model)
    assert np.allclose(pr.pi + pr.pi.T, np.abs(pr.P) + 2 * pr.ci)
    assert pr.ci.max() <= 0.5 + ATOL and np.abs(pr.P).max() <= 1 + ATOL


def test_theorem1_direct_cyclic_identity_any_reference_set():
    """Theorem 1: phi_S(a) - phi_S(b) = P(a,b) - mean_{s in S} C(a,b,s) for any multiset S."""
    X, model = random_instance(m=7)
    Y = RNG.uniform(0, 10, size=(5, X.shape[1]))
    U = np.vstack([X, Y])                         # universe: alternatives + external points
    pi = preference_index(model, U)
    P = pi - pi.T
    S = [0, 2, 2, 7, 9, 11]                       # a multiset containing external points
    for a, b in itertools.permutations(range(len(U)), 2):
        lhs = np.mean([P[a, s] for s in S]) - np.mean([P[b, s] for s in S])
        C = [P[a, b] + P[b, s] + P[s, a] for s in S]
        assert abs(lhs - (P[a, b] - np.mean(C))) < 1e-12


def test_theorem1_promethee_form():
    """(m-1)[phi(a) - phi(b)] = m P(a,b) - sum_{x in A} C(a,b,x)."""
    X, model = random_instance(m=9)
    pr = promethee(X, model)
    P, m = pr.P, pr.m
    for a, b in itertools.permutations(range(m), 2):
        cyc = sum(P[a, b] + P[b, x] + P[x, a] for x in range(m))
        assert abs((m - 1) * (pr.phi[a] - pr.phi[b]) - (m * P[a, b] - cyc)) < 1e-12


def test_theorem2_least_squares_and_residual():
    """Theorem 2: phi_A is the LS potential; residual = mean triangle; orthogonality."""
    X, model = random_instance(m=8)
    pr = promethee(X, model)
    P, m = pr.P, pr.m
    rows, rhs = [], []
    for a, b in itertools.permutations(range(m), 2):
        r = np.zeros(m); r[a], r[b] = 1, -1
        rows.append(r); rhs.append(P[a, b])
    v_ls = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]
    v_ls -= v_ls.mean()
    assert np.allclose(v_ls, potential(P))
    assert np.allclose(potential(P), (m - 1) / m * pr.phi)
    Rc = cyclic_residual(P)
    assert np.allclose(Rc, mean_triangle(P))
    u = RNG.normal(size=m)
    assert abs((Rc * (u[:, None] - u[None, :])).sum()) < 1e-10      # Rc orthogonal to gradients
    G = P - Rc
    assert abs((P ** 2).sum() - (G ** 2).sum() - (Rc ** 2).sum()) < 1e-10  # Pythagoras


def gradient_model(X, rng=RNG):
    """Unsaturated linear functions: q = 0 and p_j >= range_j (Proposition 2(i))."""
    n = X.shape[1]
    return Model(rng.random(n) < 0.5, rng.uniform(0.1, 1, n), np.zeros(n),
                 np.ptp(X, 0) * rng.uniform(1.0, 2.0, n), ("linear",) * n)


def test_theorem3_cycle_free_equivalences_and_no_rank_reversal():
    """Theorem 3: C = 0 <=> P gradient <=> K = 0, and then every subproblem orders as sign P."""
    X = RNG.uniform(0, 10, size=(7, 3))
    model = gradient_model(X)
    pr = promethee(X, model)
    P = pr.P
    m = pr.m
    for a, b, x in itertools.permutations(range(m), 3):
        assert abs(P[a, b] + P[b, x] + P[x, a]) < 1e-12
    assert cyclicity_index(P) < 1e-20
    for size in range(2, m + 1):
        for B in itertools.combinations(range(m), size):
            sub = pr.subproblem(list(B))
            for i, j in itertools.combinations(range(size), 2):
                a, b = B[i], B[j]
                assert np.sign(round(sub.phi[i] - sub.phi[j], 12)) == np.sign(round(P[a, b], 12))


def test_theorem3_converse_cyclic_instance_has_K_positive():
    X, model = random_instance(m=8)
    P = promethee(X, model).P
    has_cycle = any(abs(P[a, b] + P[b, x] + P[x, a]) > 1e-9
                    for a, b, x in itertools.combinations(range(8), 3))
    assert has_cycle == (cyclicity_index(P) > 1e-15)


def test_proposition_triple_reversal():
    """Proposition 1: in {a,b,c} the order of (a,b) is opposite to sign P(a,b) iff
    C(a,b,c) P(a,b) > 0 and |C(a,b,c)| > 3|P(a,b)|."""
    for _ in range(20):
        X, model = random_instance(m=6)
        P = promethee(X, model).P
        for a, b, c in itertools.permutations(range(6), 3):
            Cabc = P[a, b] + P[b, c] + P[c, a]
            diff3 = 3 * P[a, b] - Cabc            # = 3 (phi_B(a) - phi_B(b)) for B = {a,b,c}
            reversed_ = np.sign(diff3) == -np.sign(P[a, b]) != 0
            pred = (Cabc * P[a, b] > 0) and (abs(Cabc) > 3 * abs(P[a, b]))
            assert reversed_ == pred


def test_single_criterion_with_thresholds_can_cycle():
    """Proposition 2(ii): one criterion with a saturating function is not cycle-free."""
    X = np.array([[0.0], [1.0], [2.0]])
    model = Model([True], [1.0], [0.0], [1.0], ("linear",))
    P = promethee(X, model).P
    assert abs(P[0, 1] + P[1, 2] + P[2, 0]) > 0.5


# ----------------------------------------------------------------------------- Section 4
def test_theorem4_exact_deletion_and_insertion():
    """Theorem 4: D_{B\\x} = D_B - delta_x and D_{B+y} = D_B + delta_y, delta in [-2, 2]."""
    X, model = random_instance(m=9)
    Y = RNG.uniform(0, 10, size=(1, X.shape[1]))
    U = np.vstack([X, Y])
    P = preference_index(model, U); P = P - P.T
    A = list(range(9))
    D = lambda B: D_matrix(P[np.ix_(B, B)])
    DA = D(A)
    for x in A:
        B = [i for i in A if i != x]
        dl = (P[:, x][:, None] + P[x, :][None, :])[np.ix_(B, B)]
        assert np.allclose(D(B), DA[np.ix_(B, B)] - dl)
        assert np.abs(dl).max() <= 2 + ATOL
    Bp = A + [9]
    dy = (P[:, 9][:, None] + P[9, :][None, :])[np.ix_(A, A)]
    assert np.allclose(D(Bp)[np.ix_(range(9), range(9))], DA + dy)


def test_corollary_universal_certificate():
    """Corollary 2: |D_A(a,b)| > 2k => order kept after any k deletions."""
    for _ in range(5):
        X, model = random_instance(m=9)
        P = promethee(X, model).P
        DA = D_matrix(P)
        for k in (1, 2):
            for Xdel in itertools.combinations(range(9), k):
                B = [i for i in range(9) if i not in Xdel]
                DB = D_matrix(P[np.ix_(B, B)])
                for i, j in itertools.combinations(range(len(B)), 2):
                    a, b = B[i], B[j]
                    if abs(DA[a, b]) > 2 * k:
                        assert np.sign(DB[i, j]) == np.sign(DA[a, b])


def test_corollary_exact_deletion_radius_bruteforce():
    """Corollary 3: the sorted-prefix radius equals the brute-force minimum."""
    X, model = random_instance(m=8)
    P = promethee(X, model).P
    DA = D_matrix(P)
    for a, b in itertools.permutations(range(8), 2):
        if DA[a, b] <= 0:
            continue
        r = deletion_radius(P, a, b)
        others = [x for x in range(8) if x not in (a, b)]
        best = np.inf
        for k in range(1, len(others) + 1):
            if any(DA[a, b] - sum(P[a, x] + P[x, b] for x in S) <= 1e-12
                   for S in itertools.combinations(others, k)):
                best = k
                break
        assert r == best


def test_proposition_insertion_worst_case_is_a_lower_bound_and_tight():
    """Proposition 3: min_y delta_y over a box is separable and attained at breakpoints."""
    for types in [("linear",) * 3, ("vshape", "level", "linear"), ("usual", "ushape", "linear")]:
        X, model = random_instance(m=6, n=3, types=types)
        lo, hi = X.min(0), X.max(0)
        Ysamp = RNG.uniform(lo, hi, size=(20000, 3))
        for a, b in [(0, 1), (2, 3), (4, 5)]:
            bound = insertion_worst_case(model, X[a], X[b], lo, hi)
            pa = preference_index(model, X[[a]], Ysamp) - preference_index(model, Ysamp, X[[a]]).T
            pb = preference_index(model, Ysamp, X[[b]]).T - preference_index(model, X[[b]], Ysamp)
            dvals = (pa + pb).ravel()
            assert dvals.min() >= bound - 1e-9
            assert dvals.min() - bound < 0.05      # sampled minimum approaches the bound


def test_lemma_promethee_i_characterisation():
    """Lemma 2: a R^I b <=> |T(a) - T(b)| > |phi(a) - phi(b)|, T = phi+ + phi-."""
    for _ in range(10):
        X, model = random_instance(m=10)
        pr = promethee(X, model)
        _, _, R = relations(pr, "PI")
        T = pr.phi_plus + pr.phi_minus
        pred = np.abs(T[:, None] - T[None, :]) > np.abs(pr.phi[:, None] - pr.phi[None, :]) + 1e-12
        np.fill_diagonal(pred, False)
        assert np.array_equal(R, pred)


# ----------------------------------------------------------------------------- Section 5


# ----------------------------------------------------------------------------- Section 6
@pytest.mark.parametrize("source", ["PI", "gamma"])
def test_proposition_degenerate_incomparability_minimiser(source):
    """Previous draft (genetic calibration): a single criterion carrying all the weight gives no incomparability."""
    for t in range(10):
        X, model = random_instance(m=12, rng=np.random.default_rng(50 + t))
        for j in range(model.n):
            w = np.full(model.n, 0.0); w[j] = 1.0
            pr = promethee(X, model.with_(w=w + 0.0))
            assert core.incomparability_count(pr, source) == 0


def test_proposition_linear_unsaturated_is_cycle_free():
    """Proposition 2(i): q = 0 and p_j >= range_j give K = 0 for any weights."""
    X = RNG.uniform(0, 10, size=(15, 5))
    model = gradient_model(X)
    for _ in range(5):
        P = promethee(X, model.with_(w=RNG.random(5))).P
        assert cyclicity_index(P) < 1e-20


def test_exact_loo_rate_matches_rerankings():
    """Theorem 4 gives the leave-one-out reversal rate of PROMETHEE II without re-ranking."""
    from gpii import data
    from gpii.reversal import loo_reversal_rate, rr_deletion
    for name in ("ELECTRIC", "BLI"):
        ds = data.load(name)
        P = promethee(ds.X, ds.model).P
        emp = rr_deletion(lambda X, ids: promethee(X, ds.model).phi, ds.X)["pair_rr"]
        assert abs(loo_reversal_rate(P) - emp) < 1e-12


# ----------------------------------------------------------------------------- new results


def test_example_1_triple_reversal():
    """Example 1: adding c reverses a and b through C(a, b, c) = 1 > 3 P(a, b) = 0.6."""
    X = np.array([[1.0, 0.0], [0.0, 2.0], [2.0, 1.0]])
    model = Model([True, True], [0.6, 0.4], [0, 0], [1, 1])
    pr = promethee(X, model)
    P = pr.P
    assert np.isclose(P[0, 1], 0.2) and np.isclose(P[1, 2], -0.2) and np.isclose(P[2, 0], 1.0)
    assert np.allclose(pr.phi, [-0.4, -0.2, 0.6])
    assert promethee(X[:2], model).phi[0] > promethee(X[:2], model).phi[1]


