from __future__ import annotations

import numpy as np
import sympy
import pytest

from symbolic_discovery.algorithms import BACON3F
from symbolic_discovery.algorithms.bacon3f import Term


def _term(name: str, values) -> Term:
    return Term(sympy.Symbol(name), np.asarray(values, dtype=float))


# _check

class TestCheckClassifier:
    @pytest.fixture
    def solver(self) -> BACON3F:
        return BACON3F(constancy_threshold=0.1)

    def test_constant_dependent(self, solver):
        x = _term("x", [1, 2, 3, 4, 5])
        y = _term("y", [5.0] * 5)
        result, rel_type = solver._check(y, x)
        assert rel_type == "Constant"
        assert result is not None

    def test_linear_no_intercept_returns_ratio_form(self, solver):
        x_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_vals = 2.0 * x_vals
        result, rel_type = solver._check(_term("y", y_vals), _term("x", x_vals))
        assert rel_type == "Linear"
        assert result is not None
        assert "y/x" in str(result.symbol)

    def test_linear_with_significant_intercept(self, solver):
        x_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_vals = 2.0 * x_vals + 10.0
        result, rel_type = solver._check(_term("y", y_vals), _term("x", x_vals))
        assert rel_type == "Linear"
        assert result is not None
        sym_str = str(result.symbol)
        assert "+" in sym_str or "-" in sym_str

    def test_positive_correlation_below_linear_is_ratio(self):
        solver = BACON3F(constancy_threshold=0.01)
        x_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_vals = x_vals ** 2
        _, rel_type = solver._check(_term("y", y_vals), _term("x", x_vals))
        assert rel_type == "Ratio"

    def test_negative_correlation_is_product(self):
        solver = BACON3F(constancy_threshold=0.01)
        x_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_vals = 10.0 / x_vals
        _, rel_type = solver._check(_term("y", y_vals), _term("x", x_vals))
        assert rel_type == "Product"

    def test_uncorrelated_returns_null(self, solver):
        x_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_vals = np.array([2.0, 5.0, 1.0, 4.0, 3.0])
        result, rel_type = solver._check(_term("y", y_vals), _term("x", x_vals))
        assert rel_type == "Null"
        assert result is None


# _contains_target

class TestContainsTarget:
    @pytest.fixture
    def solver(self):
        s = BACON3F()
        s.target_var = sympy.Symbol("y")
        return s

    def test_finds_target_in_product(self, solver):
        assert solver._contains_target(sympy.sympify("y*x")) is True

    def test_finds_target_at_top_level(self, solver):
        assert solver._contains_target(sympy.Symbol("y")) is True

    def test_misses_when_target_absent(self, solver):
        assert solver._contains_target(sympy.sympify("x*z")) is False

    def test_finds_target_in_complex_expr(self, solver):
        assert solver._contains_target(sympy.sympify("a*(y + b**2)")) is True


# _rearrange

class TestRearrange:
    @pytest.fixture
    def solver(self):
        s = BACON3F()
        s.target_var = sympy.Symbol("y")
        return s

    def test_solves_ratio_for_target(self, solver):
        x_sym, y_sym = sympy.Symbol("x"), sympy.Symbol("y")
        term = Term(y_sym / x_sym, np.array([2.0, 2.0, 2.0])) # type: ignore
        result = solver._rearrange(term)
        assert result is not None
        assert x_sym in result.free_symbols
