"""Two-cycle / gradient / cyclic decomposition of a PROMETHEE preference matrix.

For pi on a set A of m alternatives (manuscript, Section 3):

* symmetric part     pi(a,b) + pi(b,a) = |P(a,b)| + 2 CI(a,b)  (2-cycles, cancelled mass);
* gradient part      v(a) - v(b) with v = phi_A = (m-1)/m * phi   (least-squares potential);
* cyclic residual    Rc(a,b) = P(a,b) - (v(a) - v(b)) = mean_{x in A} C(a,b,x),
                     where C(a,b,x) = P(a,b) + P(b,x) + P(x,a) is the triangle circulation.

Indices
-------
two_cycle_index   I = sum_{a<b} 2 CI / sum pi   (Coquelet, Dejaegere & De Smet, 2025)
cyclicity_index   K = ||Rc||^2 / ||P||^2 in [0, 1]  (share of the antisymmetric part that
                  no ranking potential explains)
inversion_share   share of pairs whose PROMETHEE II order contradicts the direct pairwise
                  preference sign(P(a,b)) -- possible only through 3-cycles (Theorem 1).
"""
from __future__ import annotations

import numpy as np


def antisymmetric(pi: np.ndarray) -> np.ndarray:
    return pi - pi.T


def potential(P: np.ndarray) -> np.ndarray:
    """Least-squares ranking potential on the complete graph: v(a) = (1/m) sum_x P(a, x)."""
    return P.mean(1)


def cyclic_residual(P: np.ndarray) -> np.ndarray:
    v = potential(P)
    return P - (v[:, None] - v[None, :])


def triangle(P: np.ndarray, a: int, b: int, x: int) -> float:
    """Circulation C(a, b, x) = P(a,b) + P(b,x) + P(x,a)."""
    return P[a, b] + P[b, x] + P[x, a]


def mean_triangle(P: np.ndarray) -> np.ndarray:
    """Matrix of mean_{x in A} C(a, b, x) computed directly from the definition (O(m^2) memory)."""
    m = P.shape[0]
    r = P.sum(1)
    # sum_x [P(a,b) + P(b,x) + P(x,a)] = m P(a,b) + r_b - r_a
    return (m * P + r[None, :] - r[:, None]) / m


def two_cycle_index(pi: np.ndarray) -> float:
    tot = pi.sum()
    return float(2 * np.triu(np.minimum(pi, pi.T), 1).sum() / tot) if tot > 0 else 0.0


def cyclicity_index(P: np.ndarray) -> float:
    den = (P ** 2).sum()
    return float((cyclic_residual(P) ** 2).sum() / den) if den > 0 else 0.0


def inversion_share(P: np.ndarray, phi: np.ndarray) -> float:
    """Share of pairs with P(a,b) != 0 whose net-flow order contradicts sign(P(a,b))."""
    iu = np.triu_indices(P.shape[0], 1)
    sp = np.sign(P[iu])
    sf = np.sign((phi[:, None] - phi[None, :])[iu])
    mask = sp != 0
    return float(np.mean(sp[mask] != sf[mask])) if mask.any() else 0.0


def decomposition_summary(pi: np.ndarray) -> dict:
    P = antisymmetric(pi)
    m = P.shape[0]
    phi = P.sum(1) / (m - 1)
    return {
        "I_two_cycle": two_cycle_index(pi),
        "K_cyclicity": cyclicity_index(P),
        "inversion_share": inversion_share(P, phi),
        "max_identity_error": float(np.abs(cyclic_residual(P) - mean_triangle(P)).max()),
    }
