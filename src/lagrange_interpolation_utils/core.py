"""Core implementation of Lagrange interpolation.

The Lagrange interpolating polynomial through points (x_i, y_i) is

    L(x) = sum_i y_i * prod_{j != i} (x - x_j) / (x_i - x_j)

This module represents the polynomial in the barycentric form, which is
numerically stable and makes evaluation O(n) per point. The weights are
computed once in the constructor.

Barycentric form:

    L(x) = sum_i (w_i / (x - x_i)) * y_i  /  sum_i (w_i / (x - x_i))

where w_i = 1 / prod_{j != i} (x_i - x_j).

At a node x = x_i the formula is singular; we return y_i directly.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple


class LagrangePolynomial:
    """A Lagrange interpolating polynomial through a fixed set of points.

    The constructor validates that the points have distinct abscissas,
    because interpolation is undefined otherwise. The barycentric weights
    are precomputed so that ``__call__`` is O(n) rather than O(n^2).

    Parameters
    ----------
    points:
        An iterable of ``(x, y)`` pairs. ``x`` values must be distinct.

    Raises
    ------
    ValueError
        If ``points`` is empty, contains a non-finite value, or has
        duplicate ``x`` values.
    """

    __slots__ = ("_xs", "_ys", "_weights")

    def __init__(self, points: Iterable[Tuple[float, float]]) -> None:
        xs: List[float] = []
        ys: List[float] = []
        for index, point in enumerate(points):
            try:
                x, y = point
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"point at index {index} is not a pair: {point!r}"
                ) from exc
            xf = float(x)
            yf = float(y)
            if not (math.isfinite(xf) and math.isfinite(yf)):
                raise ValueError(
                    f"point at index {index} is not finite: ({xf!r}, {yf!r})"
                )
            xs.append(xf)
            ys.append(yf)

        if not xs:
            raise ValueError("at least one point is required")

        seen = set()
        for x in xs:
            if x in seen:
                raise ValueError(f"duplicate abscissa: {x!r}")
            seen.add(x)

        self._xs = tuple(xs)
        self._ys = tuple(ys)
        self._weights = _barycentric_weights(self._xs)

    @property
    def xs(self) -> Tuple[float, ...]:
        """The abscissas of the interpolation nodes, in input order."""
        return self._xs

    @property
    def ys(self) -> Tuple[float, ...]:
        """The ordinates of the interpolation nodes, in input order."""
        return self._ys

    def __len__(self) -> int:
        return len(self._xs)

    def __call__(self, x: float) -> float:
        """Evaluate the interpolating polynomial at ``x``.

        If ``x`` coincides exactly with a node, the corresponding ``y``
        is returned without evaluating the barycentric sum, which would
        otherwise divide by zero.
        """
        xf = float(x)
        if not math.isfinite(xf):
            raise ValueError(f"cannot evaluate at non-finite x: {xf!r}")

        # Exact node hit: return the tabulated value.
        for xi, yi in zip(self._xs, self._ys):
            if xf == xi:
                return yi

        numerator = 0.0
        denominator = 0.0
        for xi, yi, wi in zip(self._xs, self._ys, self._weights):
            term = wi / (xf - xi)
            numerator += term * yi
            denominator += term
        return numerator / denominator


def _barycentric_weights(xs: Sequence[float]) -> Tuple[float, ...]:
    """Compute barycentric weights for the given distinct abscissas.

    For a set of nodes, the weight at node ``i`` is

        w_i = 1 / prod_{j != i} (x_i - x_j)

    The weights are computed in two passes to keep the running products
    balanced, which reduces overflow for widely spaced nodes.
    """
    n = len(xs)
    if n == 1:
        return (1.0,)

    weights = [1.0] * n
    for i in range(n):
        xi = xs[i]
        left = 1.0
        right = 1.0
        for j in range(i):
            left *= xi - xs[j]
        for j in range(i + 1, n):
            right *= xi - xs[j]
        weights[i] = 1.0 / (left * right)
    return tuple(weights)


def lagrange_interpolate(
    points: Iterable[Tuple[float, float]], x: float
) -> float:
    """Convenience wrapper: build the polynomial and evaluate it once.

    This is provided for callers that only need a single evaluation and
    do not want to manage a ``LagrangePolynomial`` object.
    """
    return LagrangePolynomial(points)(x)
