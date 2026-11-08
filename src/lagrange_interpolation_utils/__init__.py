"""Lagrange interpolation utilities.

This package provides a small, dependency-free implementation of the
Lagrange interpolating polynomial. The public API is intentionally
minimal: build a polynomial from points, then evaluate it.
"""

from .core import LagrangePolynomial, lagrange_interpolate

__all__ = ["LagrangePolynomial", "lagrange_interpolate"]
