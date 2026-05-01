"""Tests for BACON.7F internals."""

from __future__ import annotations

import numpy as np
import sympy
import pytest

from symbolic_discovery.algorithms import BACON7F
from symbolic_discovery.algorithms.bacon7f import Term


def _term(name: str, values) -> Term:
    return Term(sympy.Symbol(name), np.asarray(values, dtype=float))


# _check

class TestCheck:
    def test_returns_string_not_term(self):
        solver = BACON7F(initial_delta=0.1, initial_epsilon=0.01)
        x = _term("x", [1.0, 2.0, 3.0, 4.0, 5.0])
        y = _term("y", [5.0, 5.0, 5.0, 5.0, 5.0])
        assert solver._check(y, x) == "Constant"
        assert isinstance(solver._check(y, x), str)

    def test_linear(self):
        solver = BACON7F(initial_epsilon=0.01)
        x_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_vals = 2.0 * x_vals + 1.0
        assert solver._check(_term("y", y_vals), _term("x", x_vals)) == "Linear"

    def test_ratio(self):
        solver = BACON7F(initial_epsilon=0.001)
        x_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_vals = x_vals ** 2
        assert solver._check(_term("y", y_vals), _term("x", x_vals)) == "Ratio"


# _election

class TestElection:
    @pytest.fixture
    def solver(self):
        return BACON7F()

    def test_majority_wins(self, solver):
        assert solver._election(["Ratio", "Ratio", "Linear"]) == "Ratio"

    @pytest.mark.parametrize("votes,expected", [
        (["Constant", "Linear"], "Constant"),
        (["Linear", "Ratio"], "Linear"),
        (["Ratio", "Product"], "Ratio"),
    ])
    def test_priority_tiebreak(self, solver, votes, expected):
        assert solver._election(votes) == expected

    def test_unanimous_vote(self, solver):
        assert solver._election(["Linear"] * 5) == "Linear"


# _apply

class TestApply:
    @pytest.fixture
    def solver(self):
        s = BACON7F()
        s.known_expressions = set()
        return s

    def test_ratio_constructs_y_over_x(self, solver):
        x_vals = np.array([1.0, 2.0, 4.0])
        y_vals = np.array([2.0, 4.0, 8.0])
        result = solver._apply("Ratio", _term("y", y_vals), _term("x", x_vals))
        assert result is not None
        assert "y/x" in str(result.symbol)
        np.testing.assert_array_almost_equal(result.values, y_vals / x_vals)

    def test_product_constructs_y_times_x(self, solver):
        x_vals = np.array([1.0, 2.0, 4.0])
        y_vals = np.array([8.0, 4.0, 2.0])
        result = solver._apply("Product", _term("y", y_vals), _term("x", x_vals))
        assert result is not None
        sym_str = str(result.symbol)
        assert "y*x" in sym_str or "x*y" in sym_str
        np.testing.assert_array_almost_equal(result.values, y_vals * x_vals)

    def test_linear_negligible_intercept_yields_ratio_form(self):
        solver = BACON7F(c_val=0.1)
        solver.known_expressions = set()
        x_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_vals = 2.0 * x_vals
        result = solver._apply("Linear",
                                 _term("y", y_vals), _term("x", x_vals))
        assert result is not None
        assert "/" in str(result.symbol)

    def test_linear_significant_intercept_yields_subtraction_form(self):
        solver = BACON7F(c_val=0.01)
        solver.known_expressions = set()
        x_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_vals = 2.0 * x_vals + 100.0
        result = solver._apply("Linear",
                                 _term("y", y_vals), _term("x", x_vals))
        assert result is not None
        sym_str = str(result.symbol)
        assert "-" in sym_str or "+" in sym_str

    def test_constant_returns_dependent_unchanged(self, solver):
        x = _term("x", [1.0, 2.0, 3.0])
        y = _term("y", [5.0, 5.0, 5.0])
        assert solver._apply("Constant", y, x) is y