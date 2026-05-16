from __future__ import annotations

import pytest

from symbolic_discovery.data.api import pretty_equation, _build_pretty_map


# pretty_equation

class TestPrettyEquationHappy:
    def test_simple_substitution(self):
        out = pretty_equation("y = x1 + x2", {"x1": "θ", "x2": "φ"})
        assert out == "y = θ + φ"

    def test_unmatched_columns_left_alone(self):
        out = pretty_equation("y = x1 + z", {"x1": "θ"})
        assert out == "y = θ + z"

    def test_repeated_occurrences_all_replaced(self):
        out = pretty_equation("x1 + x1*x1", {"x1": "θ"})
        assert out == "θ + θ*θ"


class TestPrettyEquationEdgeCases:
    def test_empty_equation_returns_empty(self):
        assert pretty_equation("", {"x1": "θ"}) == ""

    def test_none_pretty_map_returns_input_unchanged(self):
        eq = "y = x1 + x2"
        assert pretty_equation(eq, None) == eq

    def test_empty_pretty_map_returns_input_unchanged(self):
        eq = "y = x1 + x2"
        assert pretty_equation(eq, {}) == eq

    def test_identity_mapping_skipped(self):
        eq = "y = x1"
        assert pretty_equation(eq, {"x1": "x1"}) == eq

    def test_empty_value_mapping_skipped(self):
        eq = "y = x1 + x2"
        assert pretty_equation(eq, {"x1": "", "x2": "φ"}) == "y = x1 + φ"


class TestPrettyEquationOrderSensitivity:
    def test_longer_keys_replaced_first(self):
        out = pretty_equation("x1² + x1", {"x1": "θ", "x1²": "(θ)²"})
        assert out == "(θ)² + θ"

    def test_word_boundary_respected(self):
        out = pretty_equation("x10 + x1", {"x1": "θ"})
        assert out == "x10 + θ"


# _build_pretty_map

class TestBuildPrettyMap:
    def test_one_to_one_mapping(self):
        pm = _build_pretty_map(["θ", "φ", "y"], ["x1", "x2", "y"])
        assert pm["x1"] == "θ"
        assert pm["x2"] == "φ"
        assert pm["y"] == "y"

    def test_squared_and_cubed_forms_added(self):
        pm = _build_pretty_map(["θ"], ["x1"])
        assert pm["x1²"] == "(θ)²"
        assert pm["x1³"] == "(θ)³"

    def test_extra_columns_map_to_themselves(self):
        pm = _build_pretty_map(["θ"], ["x1", "x2"])
        assert pm["x1"] == "θ"
        assert pm["x2"] == "x2"

    def test_empty_inputs(self):
        assert _build_pretty_map([], []) == {}
        