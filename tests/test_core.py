import math
import unittest

from lagrange_interpolation_utils import LagrangePolynomial, lagrange_interpolate


EPS = 1e-12


def assert_close(testcase, got, want, eps=EPS):
    testcase.assertTrue(
        math.isclose(got, want, rel_tol=eps, abs_tol=eps),
        f"got {got!r}, want {want!r}",
    )


class TestConstruction(unittest.TestCase):
    def test_single_point_evaluates_to_itself(self):
        poly = LagrangePolynomial([(2.0, 7.0)])
        assert_close(self, poly(2.0), 7.0)
        assert_close(self, poly(0.0), 7.0)
        assert_close(self, poly(100.0), 7.0)

    def test_two_points_define_a_line(self):
        poly = LagrangePolynomial([(0.0, 1.0), (2.0, 5.0)])
        assert_close(self, poly(0.0), 1.0)
        assert_close(self, poly(2.0), 5.0)
        assert_close(self, poly(1.0), 3.0)
        assert_close(self, poly(3.0), 7.0)

    def test_duplicate_abscissas_rejected(self):
        with self.assertRaises(ValueError):
            LagrangePolynomial([(1.0, 0.0), (1.0, 1.0)])

    def test_empty_points_rejected(self):
        with self.assertRaises(ValueError):
            LagrangePolynomial([])

    def test_non_finite_values_rejected(self):
        with self.assertRaises(ValueError):
            LagrangePolynomial([(0.0, math.inf)])
        with self.assertRaises(ValueError):
            LagrangePolynomial([(math.nan, 0.0)])

    def test_malformed_point_rejected(self):
        with self.assertRaises(ValueError):
            LagrangePolynomial([(1.0, 2.0, 3.0)])

    def test_properties_preserve_input_order(self):
        pts = [(3.0, -1.0), (1.0, 4.0), (2.0, 0.0)]
        poly = LagrangePolynomial(pts)
        self.assertEqual(poly.xs, (3.0, 1.0, 2.0))
        self.assertEqual(poly.ys, (-1.0, 4.0, 0.0))
        self.assertEqual(len(poly), 3)


class TestEvaluation(unittest.TestCase):
    def test_interpolates_a_parabola(self):
        # y = x^2 through three points.
        poly = LagrangePolynomial([(-1.0, 1.0), (0.0, 0.0), (2.0, 4.0)])
        for x in (-1.0, 0.0, 2.0):
            assert_close(self, poly(x), x * x)
        assert_close(self, poly(1.0), 1.0)
        assert_close(self, poly(0.5), 0.25)

    def test_interpolates_a_cubic(self):
        # y = x^3 - 2x + 1 through four points.
        def f(x):
            return x * x * x - 2.0 * x + 1.0

        xs = [-2.0, -1.0, 0.0, 3.0]
        poly = LagrangePolynomial([(x, f(x)) for x in xs])
        for x in xs:
            assert_close(self, poly(x), f(x))
        assert_close(self, poly(1.5), f(1.5))
        assert_close(self, poly(-0.5), f(-0.5))

    def test_extrapolation_matches_polynomial(self):
        def f(x):
            return 2.0 * x + 3.0

        poly = LagrangePolynomial([(0.0, f(0.0)), (1.0, f(1.0))])
        assert_close(self, poly(-10.0), f(-10.0))
        assert_close(self, poly(100.0), f(100.0))

    def test_evaluation_at_non_finite_raises(self):
        poly = LagrangePolynomial([(0.0, 0.0), (1.0, 1.0)])
        with self.assertRaises(ValueError):
            poly(math.inf)
        with self.assertRaises(ValueError):
            poly(math.nan)

    def test_node_hit_returns_exact_ordinate(self):
        # Use values that are not exactly representable so that a
        # barycentric sum would likely be off by a tiny amount.
        pts = [(0.1, 0.3), (0.2, 0.7), (0.4, 0.9)]
        poly = LagrangePolynomial(pts)
        for x, y in pts:
            self.assertEqual(poly(x), y)

    def test_irrational_nodes(self):
        # sqrt(2) and sqrt(3) are not exactly representable; the
        # interpolant should still hit the tabulated values.
        pts = [(math.sqrt(2.0), 1.0), (math.sqrt(3.0), 2.0)]
        poly = LagrangePolynomial(pts)
        for x, y in pts:
            self.assertEqual(poly(x), y)


class TestConvenienceFunction(unittest.TestCase):
    def test_matches_class_evaluation(self):
        pts = [(0.0, 1.0), (1.0, 3.0), (2.0, 2.0)]
        poly = LagrangePolynomial(pts)
        for x in (-1.0, 0.5, 1.5, 3.0):
            assert_close(self, lagrange_interpolate(pts, x), poly(x))

    def test_single_point(self):
        assert_close(self, lagrange_interpolate([(5.0, -2.0)], 100.0), -2.0)

    def test_rejects_duplicate_points(self):
        with self.assertRaises(ValueError):
            lagrange_interpolate([(0.0, 0.0), (0.0, 1.0)], 0.5)


class TestNumericalStability(unittest.TestCase):
    def test_high_degree_runge_function(self):
        # The Runge function is a classic stress test for interpolation.
        # We only check at the nodes themselves and at a point where the
        # interpolant is known to be well-behaved for equidistant nodes.
        def f(x):
            return 1.0 / (1.0 + 25.0 * x * x)

        xs = [i / 5.0 for i in range(-5, 6)]
        poly = LagrangePolynomial([(x, f(x)) for x in xs])
        for x in xs:
            assert_close(self, poly(x), f(x))

    def test_large_ordinates(self):
        pts = [(0.0, 1e12), (1.0, -1e12)]
        poly = LagrangePolynomial(pts)
        assert_close(self, poly(0.5), 0.0, eps=1e-6)


if __name__ == "__main__":
    unittest.main()
