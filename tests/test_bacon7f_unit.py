"""Unit tests for BACON.7F internal functions."""
import pytest
import numpy as np
import sympy
from symbolic_discovery.algorithms import BACON7F
from symbolic_discovery.algorithms.bacon7f import Term


# _check: relation detection (returns string)

def test_check_returns_string():
    """_check should return relation type string, not a Term."""
    solver = BACON7F(initial_delta=0.1, initial_epsilon=0.01)
    x = Term(sympy.Symbol("x"), np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    y = Term(sympy.Symbol("y"), np.array([5.0, 5.0, 5.0, 5.0, 5.0]))
    
    result = solver._check(y, x)
    assert isinstance(result, str)
    assert result == "Constant"


def test_check_detects_linear():
    """_check should detect high |r| as linear."""
    solver = BACON7F(initial_epsilon=0.01)
    x_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_vals = 2.0 * x_vals + 1.0  # r = 1.0
    
    x = Term(sympy.Symbol("x"), x_vals)
    y = Term(sympy.Symbol("y"), y_vals)
    
    result = solver._check(y, x)
    assert result == "Linear"


def test_check_detects_ratio():
    """_check should detect positive correlation below linear threshold as ratio."""
    solver = BACON7F(initial_epsilon=0.001)  # Very tight epsilon
    x_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_vals = x_vals ** 2  # Positively correlated but not perfectly linear
    
    x = Term(sympy.Symbol("x"), x_vals)
    y = Term(sympy.Symbol("y"), y_vals)
    
    result = solver._check(y, x)
    assert result == "Ratio"


# _election: voting logic

def test_election_majority_wins():
    """_election should return majority vote."""
    solver = BACON7F()
    votes = ["Ratio", "Ratio", "Linear"]
    assert solver._election(votes) == "Ratio"


def test_election_tiebreak_priority():
    """_election should use priority tiebreak: Constant > Linear > Ratio > Product."""
    solver = BACON7F()
    
    # Tie between Constant and Linear -> Constant wins
    votes1 = ["Constant", "Linear"]
    assert solver._election(votes1) == "Constant"
    
    # Tie between Linear and Ratio -> Linear wins
    votes2 = ["Linear", "Ratio"]
    assert solver._election(votes2) == "Linear"
    
    # Tie between Ratio and Product -> Ratio wins
    votes3 = ["Ratio", "Product"]
    assert solver._election(votes3) == "Ratio"


# _average: Term construction

def test_average_ratio():
    """_average should construct y/x Term for Ratio."""
    solver = BACON7F()
    solver.known_expressions = set()
    
    x_vals = np.array([1.0, 2.0, 4.0])
    y_vals = np.array([2.0, 4.0, 8.0])
    
    x = Term(sympy.Symbol("x"), x_vals)
    y = Term(sympy.Symbol("y"), y_vals)
    
    result = solver._average("Ratio", y, x)
    assert result is not None
    assert "y/x" in str(result.symbol)
    np.testing.assert_array_almost_equal(result.values, y_vals / x_vals)


def test_average_product():
    """_average should construct y*x Term for Product."""
    solver = BACON7F()
    solver.known_expressions = set()
    
    x_vals = np.array([1.0, 2.0, 4.0])
    y_vals = np.array([8.0, 4.0, 2.0])
    
    x = Term(sympy.Symbol("x"), x_vals)
    y = Term(sympy.Symbol("y"), y_vals)
    
    result = solver._average("Product", y, x)
    assert result is not None
    assert "y*x" in str(result.symbol) or "x*y" in str(result.symbol)
    np.testing.assert_array_almost_equal(result.values, y_vals * x_vals)


def test_average_linear_negligible_intercept():
    """_average should produce ratio for linear with negligible intercept."""
    solver = BACON7F(c_val=0.1)
    solver.known_expressions = set()
    
    x_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_vals = 2.0 * x_vals  # y = 2x, no intercept
    
    x = Term(sympy.Symbol("x"), x_vals)
    y = Term(sympy.Symbol("y"), y_vals)
    
    result = solver._average("Linear", y, x)
    assert result is not None
    assert "/" in str(result.symbol)  # Should be y/x


def test_average_linear_significant_intercept():
    """_average should produce y - mx for linear with significant intercept."""
    solver = BACON7F(c_val=0.01)  # Tight threshold
    solver.known_expressions = set()
    
    x_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_vals = 2.0 * x_vals + 100.0  # Large intercept
    
    x = Term(sympy.Symbol("x"), x_vals)
    y = Term(sympy.Symbol("y"), y_vals)
    
    result = solver._average("Linear", y, x)
    assert result is not None
    assert "-" in str(result.symbol) or "+" in str(result.symbol)


def test_average_constant_returns_dependent():
    """_average should return dependent unchanged for Constant."""
    solver = BACON7F()
    
    x = Term(sympy.Symbol("x"), np.array([1.0, 2.0, 3.0]))
    y = Term(sympy.Symbol("y"), np.array([5.0, 5.0, 5.0]))
    
    result = solver._average("Constant", y, x)
    assert result is y
