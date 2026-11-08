# lagrange_interpolation_utils

Build and evaluate the Lagrange interpolating polynomial through a set of
points, using only the Python standard library.

```python
from lagrange_interpolation_utils import LagrangePolynomial, lagrange_interpolate

points = [(0.0, 1.0), (1.0, 3.0), (2.0, 2.0)]

poly = LagrangePolynomial(points)
print(poly(1.5))          # evaluate the interpolant
print(poly.xs)            # (0.0, 1.0, 2.0)
print(len(poly))          # 3

# Or, for a single evaluation without keeping the object:
print(lagrange_interpolate(points, 1.5))
```

## Why this exists

Lagrange interpolation is a small, self-contained numerical primitive that
appears in plotting, curve fitting, and numerical analysis code. Pulling in a
full scientific stack for it is overkill, and a naive O(n^2) evaluation is
both slower and less stable than necessary. This library precomputes the
barycentric weights once in the constructor, so each evaluation is O(n). The
trade-off is that the weights are stored eagerly: construction is O(n^2) and
memory is O(n), which is fine for the modest point counts this is meant for.

## API

The package exports two names:

- `LagrangePolynomial(points)` — construct from an iterable of `(x, y)`
pairs. The `x` values must be distinct. Evaluating the object at a node
returns the tabulated `y` exactly; elsewhere it uses the barycentric form.
The `.xs` and `.ys` properties expose the input points in order, and
`len(poly)` gives the number of points.
- `lagrange_interpolate(points, x)` — convenience wrapper that builds the
polynomial and evaluates it once.

Both raise `ValueError` for empty input, duplicate abscissas, non-finite
values, or malformed points.

## The awkward edge

If you evaluate at a point that is not exactly a node, the barycentric sum
is used and the result is subject to ordinary floating-point rounding. At a
node, the implementation returns the stored ordinate verbatim, so you get
the exact input value even when the interpolant would otherwise round it.
This matters when your ordinates are not exactly representable and you rely
on the interpolant passing through its own data.
