from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from symbolic_discovery.utils.metrics import (
    calculate_mae,
    calculate_mse,
    calculate_r,
    calculate_r2,
    equation_to_metrics,
)


# calculate_r

class TestCalculateR:
    @pytest.mark.parametrize("X,Y,expected", [
        # r = 1.
        (np.array([1, 2, 3, 4, 5]), np.array([2, 4, 6, 8, 10]), 1.0),
        # r = -1.
        (np.array([1, 2, 3, 4, 5]), np.array([10, 8, 6, 4, 2]), -1.0),
        # r = 1.
        (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), 1.0),
    ])
    def test_correlation_extremes(self, X, Y, expected):
        assert calculate_r(X, Y) == pytest.approx(expected)

    def test_zero_variance_x_returns_zero(self):
        assert calculate_r(np.array([1, 1, 1, 1]), np.array([2, 4, 6, 8])) == 0.0

    def test_zero_variance_y_returns_zero(self):
        assert calculate_r(np.array([2, 4, 6, 8]), np.array([1, 1, 1, 1])) == 0.0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            calculate_r(np.array([1, 2, 3]), np.array([1, 2]))

    def test_returns_float_type(self):
        result = calculate_r(np.array([1, 2, 3]), np.array([1, 2, 3]))
        assert isinstance(result, float)


# calculate_r2

class TestCalculateR2:
    def test_perfect_prediction(self):
        y = np.array([1, 2, 3, 4, 5])
        assert calculate_r2(y, y) == pytest.approx(1.0)

    def test_inverted_prediction_is_negative(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([5, 4, 3, 2, 1])
        assert calculate_r2(y_true, y_pred) < 0

    def test_constant_target_perfect_prediction(self):
        y = np.array([5, 5, 5, 5])
        assert calculate_r2(y, y) == 1.0

    def test_constant_target_mismatched_prediction(self):
        y_true = np.array([5.0, 5.0, 5.0, 5.0])
        y_pred = np.array([4.0, 4.0, 4.0, 4.0])
        assert calculate_r2(y_true, y_pred) == 0.0

    def test_mean_prediction_is_zero(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean = float(np.mean(y_true))
        y_pred = np.full_like(y_true, mean)
        assert calculate_r2(y_true, y_pred) == pytest.approx(0.0, abs=1e-9)


# calculate_mse / calculate_mae

class TestCalculateMSE:
    @pytest.mark.parametrize("y_true,y_pred,expected", [
        (np.array([1, 2, 3]), np.array([1, 2, 3]), 0.0),
        (np.array([1, 2, 3]), np.array([2, 3, 4]), 1.0),
        (np.array([0, 0, 0]), np.array([1, -1, 0]), 2.0 / 3),
    ])
    def test_known_values(self, y_true, y_pred, expected):
        assert calculate_mse(y_true, y_pred) == pytest.approx(expected)


class TestCalculateMAE:
    @pytest.mark.parametrize("y_true,y_pred,expected", [
        (np.array([1, 2, 3]), np.array([1, 2, 3]), 0.0),
        (np.array([0, 2, 3]), np.array([2, 4, 5]), 2.0),
        (np.array([0, 0, 0]), np.array([-1, 1, 0]), 2.0 / 3),
    ])
    def test_known_values(self, y_true, y_pred, expected):
        assert calculate_mae(y_true, y_pred) == pytest.approx(expected)


# equation_to_metrics

class TestEquationToMetrics:
    def test_linear_perfect_fit(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
        r2, mse, mae = equation_to_metrics("y = 2*x", df, "y")
        assert r2 == pytest.approx(1.0)
        assert mse == pytest.approx(0.0)
        assert mae == pytest.approx(0.0)

    def test_constant_equation(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [5.0, 5.0, 5.0]})
        r2, mse, mae = equation_to_metrics("y = 5", df, "y")
        assert mse == pytest.approx(0.0)
        assert mae == pytest.approx(0.0)

    def test_no_law_sentinel(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
        r2, mse, mae = equation_to_metrics("No law found", df, "y")
        assert r2 == 0.0
        assert mse == float("inf")
        assert mae == float("inf")

    def test_missing_target_raises(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="Target column"):
            equation_to_metrics("y = x", df, "y")

    def test_handles_equation_without_lhs(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
        r2, mse, mae = equation_to_metrics("2*x", df, "y")
        assert r2 == pytest.approx(1.0)

    def test_multivariate_expression(self):
        df = pd.DataFrame({
            "x1": [1.0, 2.0, 3.0],
            "x2": [10.0, 20.0, 30.0],
            "y":  [11.0, 22.0, 33.0],
        })
        r2, _, _ = equation_to_metrics("y = x1 + x2", df, "y")
        assert r2 == pytest.approx(1.0)

    def test_imperfect_fit_gives_finite_metrics(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [2.1, 4.05, 5.95, 8.1]})
        r2, mse, mae = equation_to_metrics("y = 2*x", df, "y")
        assert np.isfinite(r2)
        assert np.isfinite(mse)
        assert np.isfinite(mae)
        assert r2 > 0.99
