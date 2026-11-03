"""Unicriterion preference functions of the PROMETHEE family (Brans & Vincke, 1985).

Every function maps a difference ``d = g(a) - g(b)`` (already oriented so that larger is
better) to a preference degree in [0, 1] and is non-decreasing with F(d) = 0 for d <= 0.

The six classical shapes are supported and referred to by a short code:

=====  ==========  ==============================  ==========
code   name        definition (d > 0)              parameters
=====  ==========  ==============================  ==========
usual  Type I      1                               --
ushape Type II     1 if d > q else 0               q
vshape Type III    min(d / p, 1)                   p
level  Type IV     0 (d<=q), 1/2 (q<d<=p), 1 (d>p) q, p
linear Type V      0 (d<=q), (d-q)/(p-q), 1 (d>p)  q, p
gauss  Type VI     1 - exp(-d^2 / (2 s^2))         s
=====  ==========  ==============================  ==========

Bug fixes with respect to the original notebook
-----------------------------------------------
* the notebook compared string codes ('li', 'u', ...) with integers (``f == 0``), so every
  criterion silently fell through to the fallback ``max(d, 0)`` and the thresholds q, p and
  the function type had no effect at all;
* its U-shape branch tested ``diff > x[0]`` (a data value) instead of ``diff > q``.
Both are corrected here; every shape is unit-tested in ``tests/test_preference.py``.
"""
from __future__ import annotations

import numpy as np

CODES = ("usual", "ushape", "vshape", "level", "linear", "gauss")
#: integer encoding used by the genetic algorithm (gene value -> code)
INDEX_TO_CODE = dict(enumerate(CODES))
CODE_TO_INDEX = {c: i for i, c in INDEX_TO_CODE.items()}


def preference_degree(d: np.ndarray, code: str, q: float = 0.0, p: float = 1.0,
                      s: float | None = None) -> np.ndarray:
    """Vectorised preference degree F(d) for one criterion.

    Parameters
    ----------
    d : array of oriented differences g(a) - g(b).
    code : one of :data:`CODES`.
    q, p : indifference and preference thresholds (q <= p is assumed; it is enforced by
        the callers). For ``vshape`` only ``p`` is used, for ``ushape`` only ``q``.
    s : inflexion point of the Gaussian shape; defaults to (q + p) / 2.
    """
    d = np.asarray(d, dtype=float)
    out = np.zeros_like(d)
    pos = d > 0
    if code == "usual":
        out[pos] = 1.0
    elif code == "ushape":
        out[d > q] = 1.0
    elif code == "vshape":
        if p <= 0:
            out[pos] = 1.0
        else:
            out[pos] = np.minimum(d[pos] / p, 1.0)
    elif code == "level":
        out[(d > q) & (d <= p)] = 0.5
        out[d > p] = 1.0
    elif code == "linear":
        if p <= q:  # degenerate linear function = U-shape at q
            out[d > q] = 1.0
        else:
            mid = (d > q) & (d < p)
            out[mid] = (d[mid] - q) / (p - q)
            out[d >= p] = 1.0
    elif code == "gauss":
        s = (q + p) / 2.0 if s is None else s
        if s <= 0:
            out[pos] = 1.0
        else:
            out[pos] = 1.0 - np.exp(-d[pos] ** 2 / (2.0 * s ** 2))
    else:
        raise ValueError(f"unknown preference function {code!r}; expected one of {CODES}")
    return out


def breakpoints(code: str, q: float, p: float) -> list[float]:
    """Points where a piecewise-linear F (or its reflection) changes slope or jumps.

    Used by the exact worst-case insertion bound (``reversal.insertion_worst_case``): the
    minimum of a sum of reflected piecewise-linear functions is attained at a breakpoint.
    """
    if code == "usual":
        return [0.0]
    if code == "ushape":
        return [0.0, q]
    if code == "vshape":
        return [0.0, p]
    if code in ("level", "linear"):
        return [0.0, q, p]
    return []  # Gaussian: not piecewise linear, handled by a dense grid
