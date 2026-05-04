"""Tests for the solver wrappers."""

from __future__ import annotations

import pandas as pd
import pytest

from symbolic_discovery.solvers import SolverResult
from symbolic_discovery.solvers.bacon3f import BACON3FSolver
from symbolic_discovery.solvers.bacon7f import BACON7FSolver


WRAPPERS = pytest.mark.parametrize(
    "WrapperClass", [BACON3FSolver, BACON7FSolver],
    ids=["bacon3f", "bacon7f"],
)


@WRAPPERS
def test_returns_solver_result(WrapperClass, simple_train_test):
    train, test = simple_train_test
    solver = WrapperClass()
    result = solver.solve(train, test, target_col="y", seed=73)
    assert isinstance(result, SolverResult)


@WRAPPERS
def test_finds_law_on_simple_data(WrapperClass, simple_train_test):
    train, test = simple_train_test
    solver = WrapperClass()
    result = solver.solve(train, test, target_col="y", seed=73)
    assert result.status == "Found"
    assert result.r2 > 0.99


@WRAPPERS
def test_missing_target_returns_error(WrapperClass):
    train = pd.DataFrame({"x": [1, 2, 3]})
    test = pd.DataFrame({"x": [4, 5]})
    solver = WrapperClass()
    result = solver.solve(train, test, target_col="y", seed=73)
    assert result.status == "Error"
    assert "y" in result.raw_equation


@WRAPPERS
def test_kwargs_forwarded_to_underlying_model(WrapperClass, simple_train_test):
    train, test = simple_train_test
    solver = WrapperClass(r2_threshold=0.999999)
    result = solver.solve(train, test, target_col="y", seed=73)
    assert result.status == "Found"


@WRAPPERS
def test_no_law_found_is_failure_not_error(WrapperClass):
    train = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0]})
    test = pd.DataFrame({"y": [6.0, 7.0]})
    solver = WrapperClass()
    result = solver.solve(train, test, target_col="y", seed=73)
    assert result.status == "Failure"
    assert result.equation == "No law found"


@WRAPPERS
def test_result_fields_populated(WrapperClass, simple_train_test):
    train, test = simple_train_test
    solver = WrapperClass()
    r = solver.solve(train, test, target_col="y", seed=73)
    assert isinstance(r.equation, str)
    assert isinstance(r.raw_equation, str)
    assert isinstance(r.r2, float)
    assert isinstance(r.mse, float)
    assert isinstance(r.mae, float)
    assert isinstance(r.time_sec, float)
    assert r.time_sec >= 0.0
    assert r.status in {"Found", "Failure", "Error"}


class TestSolverResult:
    def test_construction_with_all_fields(self):
        r = SolverResult(
            equation="y = x", raw_equation="y = x",
            r2=1.0, mse=0.0, mae=0.0,
            time_sec=0.01, status="Found",
        )
        assert r.equation == "y = x"
        assert r.r2 == 1.0
        assert r.status == "Found"