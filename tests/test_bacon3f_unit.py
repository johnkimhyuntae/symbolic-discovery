"""Unit tests for BACON.3F internal functions."""
import pytest
import numpy as np
import sympy
from symbolic_discovery.algorithms import BACON3F
from symbolic_discovery.algorithms.bacon3f import Term


# _check: relation detection

def test_check_detects_constant():
    """_check should detect when dependent is constant."""
    solver = BACON3F(constancy_threshold=0.1)
    x = Term(sympy.Symbol("x"), np.array([1, 2, 3, 4, 5]))
    y = Term(sympy.Symbol("y"), np.array([5.0, 5.0, 5.0, 5.0, 5.0]))
    
    result, rel_type = solver._check(y, x)
    assert rel_type == "Constant"
    assert result is not None


def test_check_detects_linear_no_intercept():
    """_check should detect y = 2x as linear with negligible intercept."""
    solver = BACON3F(constancy_threshold=0.1)
    x_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_vals = 2.0 * x_vals
    
    x = Term(sympy.Symbol("x"), x_vals)
    y = Term(sympy.Symbol("y"), y_vals)
    
    result, rel_type = solver._check(y, x)
    assert rel_type == "Linear"
    assert result is not None
    # Should produce y/x since intercept is negligible
    assert "y/x" in str(result.symbol)


def test_check_detects_linear_with_intercept():
    """_check should detect y = 2x + 10 as linear with significant intercept."""
    solver = BACON3F(constancy_threshold=0.1)
    x_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_vals = 2.0 * x_vals + 10.0
    
    x = Term(sympy.Symbol("x"), x_vals)
    y = Term(sympy.Symbol("y"), y_vals)
    
    result, rel_type = solver._check(y, x)
    assert rel_type == "Linear"
    assert result is not None
    # Should produce y - mx since intercept is significant
    assert "-" in str(result.symbol) or "+" in str(result.symbol)


def test_check_detects_ratio():
    """_check should detect positive correlation -> ratio."""
    solver = BACON3F(constancy_threshold=0.01)  # Tight threshold to avoid linearity
    x_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_vals = x_vals ** 2  # Positively correlated but not linear
    
    x = Term(sympy.Symbol("x"), x_vals)
    y = Term(sympy.Symbol("y"), y_vals)
    
    result, rel_type = solver._check(y, x)
    assert rel_type == "Ratio"


def test_check_detects_product():
    """_check should detect negative correlation -> product."""
    solver = BACON3F(constancy_threshold=0.01)
    x_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_vals = 10.0 / x_vals  # Inversely correlated
    
    x = Term(sympy.Symbol("x"), x_vals)
    y = Term(sympy.Symbol("y"), y_vals)
    
    result, rel_type = solver._check(y, x)
    assert rel_type == "Product"


def test_check_returns_null_uncorrelated():
    """_check should return Null for uncorrelated data."""
    solver = BACON3F(constancy_threshold=0.1)
    x_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_vals = np.array([2.0, 5.0, 1.0, 4.0, 3.0])  # Near-zero correlation
    
    x = Term(sympy.Symbol("x"), x_vals)
    y = Term(sympy.Symbol("y"), y_vals)
    
    result, rel_type = solver._check(y, x)
    assert rel_type == "Null"
    assert result is None


# _contains_target

def test_contains_target():
    """_contains_target should detect target in expression."""
    solver = BACON3F()
    solver.target_var = sympy.Symbol("y")
    
    expr1 = sympy.sympify("y*x")
    expr2 = sympy.sympify("x*z")
    
    assert solver._contains_target(expr1) is True
    assert solver._contains_target(expr2) is False


# _rearrange

def test_rearrange_simple():
    """_rearrange should solve y/x = k for y."""
    solver = BACON3F()
    solver.target_var = sympy.Symbol("y")
    
    x_sym = sympy.Symbol("x")
    y_sym = sympy.Symbol("y")
    term = Term(y_sym / x_sym, np.array([2.0, 2.0, 2.0]))  # type: ignore
    
    result = solver._rearrange(term)
    assert result is not None
    # y/x = 2 -> y = 2*x
    assert x_sym in result.free_symbols
