"""Integration tests for the data layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from symbolic_discovery.data import (
    expand_datasets,
    load,
    resolve,
)
from symbolic_discovery.data.synthetic import CATALOGUE


FEYNMAN_ROOT = "feynman"
_FEYNMAN_PRESENT = Path(FEYNMAN_ROOT).exists()
_no_feynman = pytest.mark.skipif(
    not _FEYNMAN_PRESENT,
    reason="Feynman data directory not present",
)


# load() for S / T

class TestLoadSyntheticAndTextbook:
    @pytest.mark.parametrize("key", ["S1", "S2", "S3", "T1", "T2", "T3", "T4", "T5"])
    def test_load_returns_train_test(self, key):
        cfg = CATALOGUE[key]
        train_df, test_df, pretty_map = load(cfg, n_samples=100, seed=73)

        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)
        assert pretty_map is not None

    def test_target_column_present_in_both_splits(self):
        cfg = CATALOGUE["S1"]
        train_df, test_df, _ = load(cfg, n_samples=100, seed=73)
        assert cfg.target in train_df.columns
        assert cfg.target in test_df.columns

    def test_all_variables_present_in_both_splits(self):
        cfg = CATALOGUE["S2"]
        train_df, test_df, _ = load(cfg, n_samples=100, seed=73)
        for var in cfg.variables:
            assert var in train_df.columns, f"missing {var} in train"
            assert var in test_df.columns, f"missing {var} in test"

    @pytest.mark.parametrize("n_samples", [50, 100, 250, 500])
    def test_eighty_twenty_split(self, n_samples):
        cfg = CATALOGUE["S1"]
        train_df, test_df, _ = load(cfg, n_samples=n_samples, seed=73)
        assert len(train_df) == int(n_samples * 0.8)
        assert len(test_df) == n_samples - int(n_samples * 0.8)
        assert len(train_df) + len(test_df) == n_samples

    def test_split_is_deterministic(self):
        cfg = CATALOGUE["S1"]
        a_train, a_test, _ = load(cfg, n_samples=100, seed=73)
        b_train, b_test, _ = load(cfg, n_samples=100, seed=73)
        pd.testing.assert_frame_equal(a_train, b_train)
        pd.testing.assert_frame_equal(a_test, b_test)

    def test_noise_perturbs_target_only(self):
        cfg = CATALOGUE["S1"]
        clean_train, _, _ = load(cfg, noise=0.0, n_samples=100, seed=73)
        noisy_train, _, _ = load(cfg, noise=0.1, n_samples=100, seed=73)

        # Target column should differ; feature columns should not.
        assert not clean_train[cfg.target].equals(noisy_train[cfg.target])
        for var in cfg.variables:
            pd.testing.assert_series_equal(
                clean_train[var], noisy_train[var], check_names=False,
            )

    def test_zero_noise_is_identical_regardless_of_type(self):
        cfg = CATALOGUE["T1"]
        clean_train, clean_test, _ = load(cfg, noise=0.0, noise_type="multiplicative", n_samples=100, seed=73)
        also_clean_train, also_clean_test, _ = load(
            cfg, noise=0.0, noise_type="additive", n_samples=100, seed=73,
        )
        pd.testing.assert_frame_equal(clean_train, also_clean_train)
        pd.testing.assert_frame_equal(clean_test, also_clean_test)

    @pytest.mark.parametrize("noise_type", ["multiplicative", "additive"])
    def test_both_noise_types_produce_finite_values(self, noise_type):
        cfg = CATALOGUE["S2"]
        train_df, test_df, _ = load(
            cfg, noise=0.05, noise_type=noise_type, n_samples=100, seed=73,
        )
        assert np.isfinite(train_df[cfg.target]).all()
        assert np.isfinite(test_df[cfg.target]).all()

    def test_different_seeds_produce_different_data(self):
        cfg = CATALOGUE["S2"]
        a_train, _, _ = load(cfg, n_samples=100, seed=73)
        b_train, _, _ = load(cfg, n_samples=100, seed=74)
        # Synthetic data is sampled per-seed so the two should differ.
        assert not a_train.equals(b_train)


# expand_datasets with load

class TestExpandThenLoad:
    def test_S_family_wildcard_expands_and_each_loads(self):
        keys = expand_datasets(["S"])
        assert keys == ["S1", "S2", "S3"]
        for k in keys:
            cfg = resolve(k)
            train_df, test_df, _ = load(cfg, n_samples=50, seed=73)
            assert len(train_df) > 0 and len(test_df) > 0

    def test_T_family_wildcard_expands_and_each_loads(self):
        keys = expand_datasets(["T"])
        assert keys == ["T1", "T2", "T3", "T4", "T5"]
        for k in keys:
            cfg = resolve(k)
            train_df, test_df, _ = load(cfg, n_samples=50, seed=73)
            assert cfg.target in train_df.columns

    def test_mixed_selectors_preserve_order(self):
        keys = expand_datasets(["S1", "T", "S3"])
        assert keys[0] == "S1"
        assert keys[-1] == "S3"
        assert "T1" in keys and "T5" in keys

    def test_unknown_key_raises_at_resolve(self):
        with pytest.raises(ValueError, match="Unknown dataset key"):
            resolve("Z99")


# Custom CSV

class TestCustomCsv:
    def test_custom_csv_requires_target(self):
        with pytest.raises(ValueError, match="--target is required"):
            resolve("anything.csv")

    def test_custom_csv_resolves_with_target(self, tmp_path):
        # resolve doesn't read the file; only load does. So this should
        # succeed even though the file doesn't exist yet.
        cfg = resolve("synthetic.csv", target="y")
        assert cfg.family == "C"
        assert cfg.target == "y"
        assert cfg.variables == []
        assert cfg.formula == "Unknown"

    def test_custom_csv_load_round_trip(self, tmp_path):
        # Create a minimal CSV on disk and round-trip it through load.
        csv_path = tmp_path / "toy.csv"
        df = pd.DataFrame({
            "x1": np.linspace(0.0, 1.0, 20),
            "x2": np.linspace(1.0, 2.0, 20),
            "y": np.linspace(2.0, 3.0, 20),
        })
        df.to_csv(csv_path, index=False)

        cfg = resolve(str(csv_path), target="y")
        train_df, test_df, pretty_map = load(cfg, n_samples=20, seed=73)

        assert pretty_map is not None
        assert "y" in train_df.columns
        assert len(train_df) + len(test_df) == 20

    def test_custom_csv_noise_applies_to_target(self, tmp_path):
        csv_path = tmp_path / "toy.csv"
        pd.DataFrame({
            "x": np.arange(50, dtype=float),
            "y": np.arange(50, dtype=float) * 2.0,
        }).to_csv(csv_path, index=False)

        cfg = resolve(str(csv_path), target="y")
        clean, _, _ = load(cfg, noise=0.0, n_samples=50, seed=73)
        noisy, _, _ = load(cfg, noise=0.1, n_samples=50, seed=73)

        assert not clean["y"].equals(noisy["y"])
        pd.testing.assert_series_equal(
            clean["x"], noisy["x"], check_names=False,
        )


# Feynman / Bonus - only ran if root exists

@_no_feynman
class TestFeynmanLoadPipeline:
    def test_F1_resolves_and_loads(self):
        cfg = resolve("F1", feynman_root=FEYNMAN_ROOT)
        assert cfg.key == "F1"
        assert cfg.family == "F"
        assert cfg.eq_id
        assert len(cfg.variables) > 0

        train_df, test_df, pretty_map = load(
            cfg, n_samples=50, seed=73, feynman_root=FEYNMAN_ROOT,
        )
        assert cfg.target in train_df.columns
        assert cfg.target in test_df.columns
        assert pretty_map is not None
        for col in train_df.columns:
            assert col in pretty_map

    def test_F_family_wildcard_returns_only_files_present(self):
        keys = expand_datasets(["F"], feynman_root=FEYNMAN_ROOT)
        assert len(keys) > 0
        for k in keys[:3]:
            cfg = resolve(k, feynman_root=FEYNMAN_ROOT)
            assert cfg.family == "F"

    def test_B_family_wildcard_returns_only_files_present(self):
        keys = expand_datasets(["B"], feynman_root=FEYNMAN_ROOT)
        for k in keys[:3]:
            cfg = resolve(k, feynman_root=FEYNMAN_ROOT)
            assert cfg.family == "B"

    def test_pretty_map_includes_squared_and_cubed_variants(self):
        cfg = resolve("F1", feynman_root=FEYNMAN_ROOT)
        _, _, pretty_map = load(
            cfg, n_samples=50, seed=73, feynman_root=FEYNMAN_ROOT,
        )
        assert pretty_map is not None
        # Every safe column should also have x_n² and x_n³ keys.
        for safe, pretty in list(pretty_map.items()):
            if safe.endswith("²") or safe.endswith("³"):
                continue
            assert f"{safe}²" in pretty_map
            assert f"{safe}³" in pretty_map

    def test_feynman_load_is_deterministic(self):
        cfg = resolve("F1", feynman_root=FEYNMAN_ROOT)
        a_train, a_test, _ = load(
            cfg, n_samples=50, seed=73, feynman_root=FEYNMAN_ROOT,
        )
        b_train, b_test, _ = load(
            cfg, n_samples=50, seed=73, feynman_root=FEYNMAN_ROOT,
        )
        pd.testing.assert_frame_equal(a_train, b_train)
        pd.testing.assert_frame_equal(a_test, b_test)


@_no_feynman
class TestFeynmanDirectoryStructure:
    # If root exists, sanity check the expected files and dirs.
    def test_feynman_metadata_csv_present(self):
        assert (Path(FEYNMAN_ROOT) / "FeynmanEquations.csv").exists()

    def test_feynman_data_dir_present(self):
        assert (Path(FEYNMAN_ROOT) / "Feynman_with_units").exists()

    def test_bonus_metadata_csv_present(self):
        assert (Path(FEYNMAN_ROOT) / "BonusEquations.csv").exists()

    def test_bonus_data_dir_present(self):
        assert (Path(FEYNMAN_ROOT) / "bonus_with_units").exists()
