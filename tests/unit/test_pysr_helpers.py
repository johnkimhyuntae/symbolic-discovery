"""Tests for private helpers in symbolic_discovery.solvers.pysr."""

from __future__ import annotations

import pandas as pd
import pytest

from symbolic_discovery.solvers.pysr import (
    _sanitise_columns,
    _unsanitise_equation,
)


# _sanitise_columns

class TestSanitiseColumns:
    def test_safe_columns_unchanged(self):
        X = pd.DataFrame({"x": [1], "y_2": [2], "_z": [3]})
        out, rmap = _sanitise_columns(X)
        assert list(out.columns) == ["x", "y_2", "_z"]
        assert rmap == {}

    @pytest.mark.parametrize("reserved", ["I", "E", "pi"])
    def test_sympy_reserved_renamed(self, reserved):
        X = pd.DataFrame({reserved: [1, 2], "x": [3, 4]})
        out, rmap = _sanitise_columns(X)
        assert reserved not in out.columns
        assert reserved in rmap

    def test_columns_with_spaces_renamed(self):
        X = pd.DataFrame({"current (A)": [1, 2], "x": [3, 4]})
        out, rmap = _sanitise_columns(X)
        assert "current (A)" not in out.columns
        assert "current (A)" in rmap

    def test_columns_starting_with_digit_renamed(self):
        X = pd.DataFrame({"1st_col": [1, 2], "x": [3, 4]})
        out, rmap = _sanitise_columns(X)
        assert "1st_col" not in out.columns

    def test_rename_does_not_collide_with_existing_safe_column(self):
        X = pd.DataFrame({"x1": [1], "I": [2], "x2": [3]})
        out, rmap = _sanitise_columns(X)
        assert len(set(out.columns)) == len(out.columns)

    def test_does_not_mutate_input(self):
        X = pd.DataFrame({"I": [1, 2]})
        before = list(X.columns)
        _sanitise_columns(X)
        assert list(X.columns) == before

    def test_empty_dataframe(self):
        X = pd.DataFrame()
        out, rmap = _sanitise_columns(X)
        assert out.empty
        assert rmap == {}


# _unsanitise_equation

class TestUnsanitiseEquation:
    def test_substitutes_renamed_back(self):
        eq = "y = x1 * x2"
        rmap = {"current (A)": "x1", "voltage (V)": "x2"}
        out = _unsanitise_equation(eq, rmap)
        assert out == "y = current (A) * voltage (V)"

    def test_empty_rename_map_returns_input(self):
        eq = "y = x1 + x2"
        assert _unsanitise_equation(eq, {}) == eq

    def test_word_boundary_respected(self):
        rmap = {"original_name": "x1"}
        eq = "y = x1 + x10"
        out = _unsanitise_equation(eq, rmap)
        assert "original_name" in out
        assert "x10" in out