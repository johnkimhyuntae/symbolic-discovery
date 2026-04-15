import pytest
import numpy as np
import pandas as pd
from symbolic_discovery.data import CATALOGUE, load, expand_datasets, resolve
from symbolic_discovery.data.synthetic import generate, inject_noise


# CATALOGUE

def test_catalogue_has_expected_keys():
    assert "S1" in CATALOGUE
    assert "S2" in CATALOGUE
    assert "T1" in CATALOGUE


def test_catalogue_config_structure():
    cfg = CATALOGUE["S1"]
    assert cfg.key == "S1"
    assert cfg.family == "S"
    assert cfg.target == "y"
    assert len(cfg.variables) > 0


# expand_datasets

def test_expand_datasets_passthrough():
    result = expand_datasets(["S1", "T1"])
    assert "S1" in result
    assert "T1" in result


def test_expand_datasets_family():
    result = expand_datasets(["S"])
    assert "S1" in result
    assert "S2" in result
    assert "S3" in result
    assert "S4" in result


# resolve

def test_resolve_synthetic():
    cfg = resolve("S1")
    assert cfg.key == "S1"
    assert cfg.family == "S"


def test_resolve_textbook():
    cfg = resolve("T1")
    assert cfg.key == "T1"
    assert cfg.family == "T"


# load

def test_load_returns_dataframes():
    cfg = CATALOGUE["S1"]
    train_df, test_df, _ = load(cfg, noise=0.0)
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    assert cfg.target in train_df.columns
    assert cfg.target in test_df.columns


def test_load_with_noise():
    cfg = CATALOGUE["S1"]
    train_clean, _, _ = load(cfg, noise=0.0)
    train_noisy, _, _ = load(cfg, noise=0.1)
    assert not train_clean[cfg.target].equals(train_noisy[cfg.target])


# generate

def test_generate_shape():
    cfg = CATALOGUE["S2"]
    df = generate(cfg, n_samples=100, seed=42)
    assert len(df) == 100
    assert cfg.target in df.columns
    for var in cfg.variables:
        assert var in df.columns


def test_generate_deterministic():
    cfg = CATALOGUE["S2"]
    df1 = generate(cfg, n_samples=50, seed=42)
    df2 = generate(cfg, n_samples=50, seed=42)
    pd.testing.assert_frame_equal(df1, df2)


# inject_noise

def test_inject_noise_zero():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
    result = inject_noise(df, "y", 0.0, seed=42)
    pd.testing.assert_frame_equal(df, result)


def test_inject_noise_nonzero():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [10.0, 20.0, 30.0]})
    result = inject_noise(df, "y", 0.1, seed=42)
    assert not df["y"].equals(result["y"])
