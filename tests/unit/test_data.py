from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from symbolic_discovery.data import CATALOGUE, DatasetConfig, expand_datasets, resolve
from symbolic_discovery.data.api import inject_noise
from symbolic_discovery.data.synthetic import generate


# CATALOGUE

class TestCatalogue:
    @pytest.mark.parametrize("key", ["S1", "S2", "S3", "T1", "T2", "T3", "T4", "T5"])
    def test_built_in_keys_present(self, key):
        assert key in CATALOGUE
        assert CATALOGUE[key].key == key

    def test_synthetic_have_two_var_domains(self):
        cfg = CATALOGUE["S1"]
        assert cfg.family == "S"
        assert set(cfg.variables) == {"x1", "x2"}
        assert cfg.target == "y"
        assert "x1" in cfg.domain and "x2" in cfg.domain

    def test_textbook_uses_physics_symbols(self):
        cfg = CATALOGUE["T1"]
        assert cfg.target == "V"
        assert "I" in cfg.variables and "R" in cfg.variables

    def test_domain_low_less_than_high(self):
        for key, cfg in CATALOGUE.items():
            for var, (lo, hi) in cfg.domain.items():
                assert lo < hi, f"{key}.{var}: low {lo} not < high {hi}"

    def test_dataset_config_dataclass_fields(self):
        cfg = DatasetConfig(
            key="X1", eq_id="X1", family="S",
            variables=["a"], target="b", formula="a",
        )
        assert cfg.domain == {}


# expand_datasets

class TestExpandDatasets:
    def test_passthrough_for_specific_keys(self):
        out = expand_datasets(["S1", "T1"])
        assert "S1" in out and "T1" in out

    def test_synthetic_family_expands_to_all_S(self):
        out = expand_datasets(["S"])
        for key in ["S1", "S2", "S3"]:
            assert key in out

    def test_textbook_family_expands_to_all_T(self):
        out = expand_datasets(["T"])
        for key in ["T1", "T2", "T3", "T4", "T5"]:
            assert key in out

    def test_mixed_family_and_specific(self):
        out = expand_datasets(["S1", "T"])
        assert "S1" in out
        for key in ["T1", "T2", "T3", "T4", "T5"]:
            assert key in out

    def test_strips_whitespace_in_tokens(self):
        out = expand_datasets(["                 S1                                                  "])
        assert "S1" in out


# resolve

class TestResolve:
    def test_resolves_synthetic_key(self):
        cfg = resolve("S1")
        assert cfg.key == "S1"
        assert cfg.family == "S"

    def test_resolves_textbook_key(self):
        cfg = resolve("T1")
        assert cfg.key == "T1"
        assert cfg.family == "T"

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            resolve("Z99")

    def test_custom_csv_requires_target(self):
        with pytest.raises(ValueError, match="target"):
            resolve("/tmp/nonexistent.csv")

    def test_custom_csv_with_target_returns_C_family(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("a,b\n1,2\n")
        cfg = resolve(str(path), target="b")
        assert cfg.family == "C"
        assert cfg.target == "b"


# generate

class TestGenerate:
    def test_shape(self):
        cfg = CATALOGUE["S2"]
        df = generate(cfg, n_samples=100, seed=73)
        assert len(df) == 100
        for var in cfg.variables:
            assert var in df.columns
        assert cfg.target in df.columns

    def test_deterministic_for_same_seed(self):
        cfg = CATALOGUE["S2"]
        df1 = generate(cfg, n_samples=50, seed=73)
        df2 = generate(cfg, n_samples=50, seed=73)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_data(self):
        cfg = CATALOGUE["S2"]
        df1 = generate(cfg, n_samples=50, seed=1)
        df2 = generate(cfg, n_samples=50, seed=2)
        assert not df1.equals(df2)

    def test_target_satisfies_formula(self):
        cfg = CATALOGUE["S1"]
        df = generate(cfg, n_samples=50, seed=73)
        np.testing.assert_allclose(df["y"], df["x1"] + df["x2"])

    def test_variable_values_within_domain(self):
        cfg = CATALOGUE["S1"]
        df = generate(cfg, n_samples=200, seed=7)
        for var, (lo, hi) in cfg.domain.items():
            assert df[var].min() >= lo
            assert df[var].max() <= hi


# inject_noise

class TestInjectNoise:
    def test_zero_noise_returns_unchanged(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
        result = inject_noise(df, "y", 0.0, seed=73)
        pd.testing.assert_frame_equal(df, result)

    def test_negative_noise_returns_unchanged(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
        result = inject_noise(df, "y", -0.1, seed=73)
        pd.testing.assert_frame_equal(df, result)

    def test_missing_target_returns_unchanged(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = inject_noise(df, "y_not_here", 0.1, seed=73)
        pd.testing.assert_frame_equal(df, result)

    def test_nonzero_noise_changes_target(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [10.0, 20.0, 30.0]})
        result = inject_noise(df, "y", 0.1, seed=73)
        assert not df["y"].equals(result["y"])

    def test_only_target_column_changes(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
        result = inject_noise(df, "y", 0.1, seed=73)
        pd.testing.assert_series_equal(result["x"], df["x"])

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
        original_y = df["y"].copy()
        _ = inject_noise(df, "y", 0.5, seed=73)
        pd.testing.assert_series_equal(df["y"], original_y)

    def test_deterministic_for_same_seed(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
        a = inject_noise(df, "y", 0.1, seed=73)
        b = inject_noise(df, "y", 0.1, seed=73)
        pd.testing.assert_frame_equal(a, b)

    def test_multiplicative_scales_with_value(self):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5],
            "y": [1.0, 1.0, 100.0, 100.0, 100.0],
        })
        result = inject_noise(df, "y", 0.1, noise_type="multiplicative", seed=73)
        small_dev = float(np.mean(np.abs(result["y"][:2] - df["y"][:2])))
        large_dev = float(np.mean(np.abs(result["y"][2:] - df["y"][2:])))
        assert large_dev > small_dev * 5

    def test_additive_scales_with_range(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [0.0, 10.0, 20.0, 30.0]})
        a = inject_noise(df, "y", 0.1, noise_type="additive", seed=73)
        b = inject_noise(df, "y", 0.5, noise_type="additive", seed=73)
        dev_a = float(np.mean(np.abs(a["y"] - df["y"])))
        dev_b = float(np.mean(np.abs(b["y"] - df["y"])))
        assert dev_b > dev_a * 2  # generous bound
