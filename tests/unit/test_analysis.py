"""Tests for symbolic_discovery.analysis."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from symbolic_discovery.analysis import (
    aggregate_seeds,
    expand_params,
    noise_curve,
    pivot_compare,
    success_rate,
    successful,
    summarise_by_variant,
)


def _runner_row(**overrides) -> dict:
    row = {
        "variant": "a",
        "method": "bacon3f",
        "dataset": "S1",
        "noise": 0.0,
        "noise_type": "multiplicative",
        "n_samples": 100,
        "seed": 1,
        "r2": 0.99,
        "mse": 0.01,
        "mae": 0.005,
        "time_s": 0.1,
        "status": "Found",
        "params_json": '{"k": 1}',
    }
    row.update(overrides)
    return row


@pytest.fixture
def runner_df() -> pd.DataFrame:
    rows = [
        _runner_row(),
        _runner_row(seed=2, r2=0.97, mse=0.02, mae=0.010, time_s=0.2),
        _runner_row(dataset="S2", r2=0.95, mse=0.05, mae=0.02, time_s=0.3),
        _runner_row(
            dataset="S2",
            seed=2,
            r2=0.0,
            mse=float("inf"),
            mae=float("inf"),
            time_s=0.0,
            status="Failure",
        ),
        _runner_row(
            variant="b",
            method="bacon7f",
            seed=1,
            r2=0.80,
            mse=0.10,
            mae=0.05,
            time_s=1.0,
            params_json='{"k": 5}',
        ),
        _runner_row(
            variant="b",
            method="bacon7f",
            seed=2,
            r2=0.85,
            mse=0.08,
            mae=0.04,
            time_s=1.2,
            params_json='{"k": 5}',
        ),
    ]
    return pd.DataFrame(rows)


# successful

class TestSuccessful:
    def test_keeps_only_found(self, runner_df):
        out = successful(runner_df)
        assert (out["status"] == "Found").all()
        assert len(out) == 5  # one Failure dropped

    def test_returns_a_copy(self, runner_df):
        # Modifying the output should not bleed into the caller's frame.
        out = successful(runner_df)
        out.loc[out.index[0], "r2"] = -999
        assert runner_df.loc[runner_df.index[0], "r2"] != -999


# expand_params

class TestExpandParams:
    def test_promotes_param_to_column(self, runner_df):
        out = expand_params(runner_df, ["k"])
        assert "k" in out.columns
        assert out["k"].iloc[0] == 1
        assert out["k"].iloc[-1] == 5

    def test_unknown_param_yields_none(self, runner_df):
        # Asking for a param that no row has should produce a column of None,
        # not raise. This protects analysis code that asks for the union of
        # params across multiple sweeps.
        out = expand_params(runner_df, ["nonexistent"])
        assert "nonexistent" in out.columns
        assert out["nonexistent"].isna().all()

    def test_handles_empty_params_json(self):
        df = pd.DataFrame({"params_json": ["", "{}", '{"k": 1}']})
        out = expand_params(df, ["k"])
        assert math.isnan(out["k"].iloc[0]) or out["k"].iloc[0] is None
        assert out["k"].iloc[2] == 1


# aggregate_seeds

class TestAggregateSeeds:
    def test_collapses_seed_dimension(self, runner_df):
        out = aggregate_seeds(runner_df, group_by=["variant", "dataset"])
        # 3 distinct (variant, dataset) cells: (a,S1), (a,S2), (b,S1).
        assert len(out) == 3

    def test_outputs_mean_and_std_columns(self, runner_df):
        out = aggregate_seeds(runner_df, group_by=["variant", "dataset"])
        assert "r2_mean" in out.columns
        assert "r2_std" in out.columns
        assert "n_runs" in out.columns

    def test_n_runs_counts_per_cell(self, runner_df):
        out = aggregate_seeds(runner_df, group_by=["variant", "dataset"])
        # Each (variant, dataset) cell has 2 seeds in our fixture.
        assert (out["n_runs"] == 2).all()

    def test_ignores_missing_group_keys(self, runner_df):
        # Asking to group by a column that isn't there should silently
        # drop it rather than KeyError.
        out = aggregate_seeds(runner_df, group_by=["variant", "no_such_col"])
        assert "no_such_col" not in out.columns


# success_rate

class TestSuccessRate:
    def test_per_dataset(self, runner_df):
        out = success_rate(runner_df, by=["variant", "dataset"])
        # (a, S2) had 1 of 2 Found.
        a_s2 = out[(out["variant"] == "a") & (out["dataset"] == "S2")].iloc[0]
        assert a_s2["success_rate"] == pytest.approx(0.5)
        assert a_s2["n_runs"] == 2

    def test_all_found_gives_one(self, runner_df):
        out = success_rate(runner_df, by=["variant", "dataset"])
        b_s1 = out[(out["variant"] == "b") & (out["dataset"] == "S1")].iloc[0]
        assert b_s1["success_rate"] == 1.0


# summarise_by_variant

class TestSummariseByVariant:
    def test_one_row_per_variant(self, runner_df):
        out = summarise_by_variant(runner_df)
        assert set(out["variant"]) == {"a", "b"}
        assert len(out) == 2

    def test_sorted_by_r2_desc(self, runner_df):
        out = summarise_by_variant(runner_df)
        # The function uses every row for the mean (failures contribute
        # r²=0), so we don't hand-compute the order — we just assert it
        # really is descending.
        means = out["r2_mean"].tolist()
        assert means == sorted(means, reverse=True)
        assert set(out["variant"]) == {"a", "b"}

    def test_columns_present(self, runner_df):
        out = summarise_by_variant(runner_df)
        for col in ("n_runs", "success_rate", "r2_mean", "r2_std",
                    "time_s_mean", "time_s_total"):
            assert col in out.columns


# pivot_compare

class TestPivotCompare:
    def test_default_pivot(self, runner_df):
        out = pivot_compare(runner_df, metric="r2")
        # rows = dataset, cols = variant.
        assert "S1" in out.index
        assert "a" in out.columns

    def test_handles_missing_cell(self, runner_df):
        # Only variant 'a' was run on S2; the (b, S2) cell should be NaN.
        out = pivot_compare(runner_df, metric="r2")
        assert pd.isna(out.loc["S2", "b"])


# noise_curve

class TestNoiseCurve:
    def test_returns_long_format_frame(self, runner_df):
        out = noise_curve(runner_df, metric="r2", group_by=["variant", "noise"])
        assert "r2_mean" in out.columns
        assert "r2_std" in out.columns
        assert "n" in out.columns

    def test_one_row_per_group(self, runner_df):
        # Only one noise level (0.0) and two variants -> two rows.
        out = noise_curve(runner_df)
        assert len(out) == 2