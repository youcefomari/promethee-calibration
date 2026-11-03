"""Symbolic verification (SymPy) of the algebraic identities of the manuscript.

The identities are checked for arbitrary values: the preferences pi(a, b) of a set of
alternatives are free symbols, so a passing test proves the identity for every PROMETHEE
model on that number of alternatives (and the proofs do not depend on the number).
"""
import itertools

import pytest

sp = pytest.importorskip("sympy")

M = 5            # number of alternatives in the symbolic checks


def free_preferences(m=M, name="p"):
    """pi(a, b) >= 0 free symbols, and Pi = pi - pi^T."""
    pi = [[sp.Integer(0) if a == b else sp.Symbol(f"{name}_{a}{b}", nonnegative=True)
           for b in range(m)] for a in range(m)]
    Pi = [[pi[a][b] - pi[b][a] for b in range(m)] for a in range(m)]
    return pi, Pi


def D(Pi, B, a, b):
    return sum(Pi[a][s] - Pi[b][s] for s in B)


def C(Pi, a, b, x):
    return Pi[a][b] + Pi[b][x] + Pi[x][a]


def zero(expr):
    return sp.simplify(sp.expand(expr)) == 0


def test_lemma1_ii_iii():
    pi, Pi = free_preferences()
    m = M
    for B in [list(range(m)), [0, 1, 3], [1, 2, 3, 4]]:
        k = len(B)
        phi_plus = {a: sum(pi[a][x] for x in B) / (k - 1) for a in B}
        phi_minus = {a: sum(pi[x][a] for x in B) / (k - 1) for a in B}
        for a in B:
            # (ii) phi = sum_x Pi(a, x) / (|B| - 1)
            assert zero(phi_plus[a] - phi_minus[a] - sum(Pi[a][x] for x in B) / (k - 1))
        for a, b in itertools.combinations(B, 2):
            # (iii) D_B(a, b) = (|B| - 1)(phi_B(a) - phi_B(b))
            lhs = D(Pi, B, a, b)
            rhs = (k - 1) * ((phi_plus[a] - phi_minus[a]) - (phi_plus[b] - phi_minus[b]))
            assert zero(lhs - rhs)


def test_theorem1():
    _, Pi = free_preferences()
    A = list(range(M))
    for a, b in itertools.permutations(A, 2):
        for x in A:
            if x in (a, b):
                continue
            delta = Pi[a][x] - Pi[b][x]
            assert zero(D(Pi, [s for s in A if s != x], a, b) - (D(Pi, A, a, b) - delta))
            assert zero(delta - (Pi[a][x] + Pi[x][b]))
            assert zero(delta - (Pi[a][b] - C(Pi, a, b, x)))
    # insertion: compute on a larger set with one extra alternative
    _, Pi6 = free_preferences(M + 1, "q")
    A = list(range(M))
    for a, b in itertools.combinations(A, 2):
        assert zero(D(Pi6, A + [M], a, b) - D(Pi6, A, a, b) - (Pi6[a][M] - Pi6[b][M]))


def test_theorem2():
    _, Pi = free_preferences()
    m = M
    u = sp.symbols(f"u0:{m}")
    J = sum((Pi[a][b] - u[a] + u[b]) ** 2 for a in range(m) for b in range(m))
    v = [sum(Pi[a][x] for x in range(m)) / m for a in range(m)]
    sub = dict(zip(u, v))
    # Step 1-2: the gradient vanishes at u = v
    for c in range(m):
        assert zero(sp.diff(J, u[c]).subs(sub))
    # the gradient has the closed form of Step 1
    for c in range(m):
        g = -4 * sum(Pi[c][x] - u[c] + u[x] for x in range(m))
        assert zero(sp.diff(J, u[c]) - g)
    G = [[v[a] - v[b] for b in range(m)] for a in range(m)]
    R = [[Pi[a][b] - G[a][b] for b in range(m)] for a in range(m)]
    # Step 3: row sums of R vanish
    for a in range(m):
        assert zero(sum(R[a][b] for b in range(m)))
    # (ii) Pythagoras
    nPi = sum(Pi[a][b] ** 2 for a in range(m) for b in range(m))
    nG = sum(G[a][b] ** 2 for a in range(m) for b in range(m))
    nR = sum(R[a][b] ** 2 for a in range(m) for b in range(m))
    assert zero(nPi - nG - nR)
    # (iii) R is the mean circulation
    for a, b in itertools.permutations(range(m), 2):
        assert zero(R[a][b] - sum(C(Pi, a, b, x) for x in range(m)) / m)


def test_theorem3():
    m = M
    u = sp.symbols(f"u0:{m}")
    Pi = [[u[a] - u[b] for b in range(m)] for a in range(m)]
    for a, b, x in itertools.permutations(range(m), 3):
        assert zero(C(Pi, a, b, x))
    for B in [list(range(m)), [0, 2, 4], [1, 3]]:
        for a, b in itertools.combinations(B, 2):
            assert zero(D(Pi, B, a, b) - len(B) * (u[a] - u[b]))
    # (ii) => (iii): u(a) = Pi(a, r) under vanishing circulations
    _, P = free_preferences()
    r = 0
    for a, b in itertools.permutations(range(m), 2):
        # Pi(a,b) - (Pi(a,r) - Pi(b,r)) equals the circulation C(a, b, r) up to sign
        assert zero(P[a][b] - (P[a][r] - P[b][r]) - C(P, a, b, r))


def test_theorem4_ii_linear_unsaturated():
    d, p = sp.symbols("d p", positive=True)
    F = lambda t: sp.Piecewise((0, t <= 0), (t / p, True))        # unsaturated on the data range
    assert zero(F(d) - F(-d) - d / p)                              # d > 0
    assert zero(F(-d) - F(d) - (-d) / p)                           # difference -d < 0


def test_prop2_quadratic_forms():
    """||R||^2 = w'Aw and ||Pi||^2 = w'Bw with Pi = w1 Pi_1 + w2 Pi_2 (n = 2 criteria)."""
    m = 4
    _, P1 = free_preferences(m, "a")
    _, P2 = free_preferences(m, "b")
    w1, w2 = sp.symbols("w1 w2")

    def residual(P):
        v = [sum(P[a][x] for x in range(m)) / m for a in range(m)]
        return [[P[a][b] - (v[a] - v[b]) for b in range(m)] for a in range(m)]

    inner = lambda X, Y: sum(X[a][b] * Y[a][b] for a in range(m) for b in range(m))
    Pi = [[w1 * P1[a][b] + w2 * P2[a][b] for b in range(m)] for a in range(m)]
    R, R1, R2 = residual(Pi), residual(P1), residual(P2)
    A = [[inner(R1, R1), inner(R1, R2)], [inner(R2, R1), inner(R2, R2)]]
    B = [[inner(P1, P1), inner(P1, P2)], [inner(P2, P1), inner(P2, P2)]]
    w = [w1, w2]
    quad = lambda Mx: sum(w[i] * w[j] * Mx[i][j] for i in range(2) for j in range(2))
    assert zero(inner(R, R) - quad(A))
    assert zero(inner(Pi, Pi) - quad(B))


def test_lemma2_shift_invariance():
    """(w - w')'c = (w - w')'(c - gamma 1) when both weight vectors sum to one."""
    n = 4
    w = sp.symbols(f"w0:{n}")
    v = sp.symbols(f"v0:{n}")
    c = sp.symbols(f"c0:{n}")
    g = sp.Symbol("gamma")
    lhs = sum((w[j] - v[j]) * c[j] for j in range(n))
    rhs = sum((w[j] - v[j]) * (c[j] - g) for j in range(n))
    diff = sp.expand(lhs - rhs)                                   # = gamma (sum w - sum v)
    diff = diff.subs({w[-1]: 1 - sum(w[:-1]), v[-1]: 1 - sum(v[:-1])})
    assert zero(diff)


@pytest.mark.parametrize("sizes", [(1, 1, 1), (2, 1, 3), (1, 3), (3, 2, 1, 2)])
def test_prop3_inverse_telescopes(sizes):
    k = len(sizes)
    c = sp.symbols(f"c1:{k + 1}")
    cc = list(c) + [0]
    T = [sum(sizes[:l + 1]) for l in range(k)]
    lam = [T[l] * (cc[l] - cc[l + 1]) for l in range(k)]
    # weight of a criterion in group i: sum_{l >= i} lam_l / |T_l| = c_i
    for i in range(k):
        assert zero(sum(lam[l] / T[l] for l in range(i, k)) - c[i])
    # sum of lam = sum_i |L_i| c_i (= 1 for a weight vector)
    assert zero(sum(lam) - sum(sizes[i] * c[i] for i in range(k)))


def test_example1_exact_arithmetic():
    R = sp.Rational
    w1, w2 = R(3, 5), R(2, 5)
    clip = lambda d: min(max(d, 0), 1)
    a, b, c = (1, 0), (0, 2), (2, 1)

    def Pi(x, y):
        return sum(w * (clip(x[j] - y[j]) - clip(y[j] - x[j])) for j, w in enumerate((w1, w2)))
    assert Pi(a, b) == R(1, 5) and Pi(b, c) == R(-1, 5) and Pi(c, a) == 1
    phi = {k: (Pi(x, y1) + Pi(x, y2)) / 2 for k, (x, y1, y2) in
           {"a": (a, b, c), "b": (b, a, c), "c": (c, a, b)}.items()}
    assert phi == {"a": R(-2, 5), "b": R(-1, 5), "c": R(3, 5)}
    D_ba = 2 * (phi["b"] - phi["a"])
    Cbac = Pi(b, a) + Pi(a, c) + Pi(c, b)
    assert D_ba == R(2, 5) and Cbac == -1 and Pi(b, a) - Cbac == R(4, 5)
    assert D_ba - (Pi(b, a) - Cbac) == R(-2, 5)
    # K = ||R||^2 / ||Pi||^2 = (6 / 9) / 2.16 = 25 / 81
    num = 6 * R(1, 9)
    den = 2 * (Pi(a, b) ** 2 + Pi(b, c) ** 2 + Pi(c, a) ** 2)
    assert num / den == R(25, 81)
