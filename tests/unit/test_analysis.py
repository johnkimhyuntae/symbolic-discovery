from __future__ import annotations

import math

import pandas as pd
import pytest

from symbolic_discovery.utils.analysis import (
    aggregate,
    equivalence_ratio,
    load_csvs,
    parse_variants,
)


def _runner_row(**overrides) -> dict:
    row = {
        "run_id":       "r1",
        "variant":      "a",
        "method":       "bacon3f",
        "dataset":      "T1",
        "noise":        0.0,
        "noise_type":   "multiplicative",
        "n_samples":    100,
        "seed":         1,
        "equation":     "V = I*R",
        "raw_equation": "V = x1*x2",
        "r2":           0.99,
        "mse":          0.01,
        "mae":          0.005,
        "time_s":       0.1,
        "status":       "Found",
        "params_json":  '{"k": 1}',
    }
    row.update(overrides)
    return row


@pytest.fixture
def runner_df() -> pd.DataFrame:
    rows = [
        _runner_row(),
        _runner_row(seed=2, r2=0.97, time_s=0.2),
        _runner_row(dataset="T2", r2=0.95, time_s=0.3),
        _runner_row(
            dataset="T2", seed=2,
            r2=0.0, mse=float("inf"), mae=float("inf"),
            time_s=0.0, status="Failure",
        ),
        _runner_row(variant="b", method="bacon7f", r2=0.80, time_s=1.0,
                    params_json='{"k": 5}'),
        _runner_row(variant="b", method="bacon7f", seed=2, r2=0.85,
                    time_s=1.2, params_json='{"k": 5}'),
    ]
    df = pd.DataFrame(rows)
    df["found"] = df["status"] == "Found"
    return df


# load_csvs

class TestLoadCsvs:
    def test_returns_none_when_no_paths_exist(self, tmp_path):
        assert load_csvs([tmp_path / "missing.csv"]) is None

    def test_loads_single_csv(self, tmp_path, runner_df):
        path = tmp_path / "results.csv"
        runner_df.drop(columns="found").to_csv(path, index=False)
        out = load_csvs([path])
        assert out is not None
        assert len(out) == len(runner_df)

    def test_concatenates_multiple_csvs(self, tmp_path, runner_df):
        clean = runner_df.drop(columns="found")
        p1 = tmp_path / "a.csv"
        p2 = tmp_path / "b.csv"
        clean.iloc[:3].to_csv(p1, index=False)
        clean.iloc[3:].to_csv(p2, index=False)
        out = load_csvs([p1, p2])
        assert len(out) == len(runner_df) # type: ignore

    def test_dedupes_on_experiment_key(self, tmp_path, runner_df):
        clean = runner_df.drop(columns="found")
        p1 = tmp_path / "a.csv"
        p2 = tmp_path / "b.csv"
        clean.to_csv(p1, index=False)
        clean.to_csv(p2, index=False)
        out = load_csvs([p1, p2])
        assert len(out) == len(runner_df) # type: ignore

    def test_found_column_added(self, tmp_path, runner_df):
        path = tmp_path / "results.csv"
        runner_df.drop(columns="found").to_csv(path, index=False)
        out = load_csvs([path])
        assert out["found"].dtype == bool # type: ignore
        assert (~out["found"]).sum() == 1 # type: ignore


# aggregate

class TestAggregate:
    def test_collapses_seed_dimension(self, runner_df):
        out = aggregate(runner_df, ["variant", "dataset"])
        assert len(out) == 3

    def test_n_runs_counts_all_runs_including_failures(self, runner_df):
        out = aggregate(runner_df, ["variant", "dataset"])
        a_t2 = out[(out["variant"] == "a") & (out["dataset"] == "T2")].iloc[0]
        assert a_t2["n_runs"] == 2

    def test_success_rate_correct(self, runner_df):
        out = aggregate(runner_df, ["variant", "dataset"])
        a_t2 = out[(out["variant"] == "a") & (out["dataset"] == "T2")].iloc[0]
        assert a_t2["success_rate"] == pytest.approx(0.5)

    def test_r2_mean_uses_successful_runs_only(self, runner_df):
        out = aggregate(runner_df, ["variant", "dataset"])
        a_t2 = out[(out["variant"] == "a") & (out["dataset"] == "T2")].iloc[0]
        assert a_t2["r2_mean"] == pytest.approx(0.95)

    def test_time_mean_includes_all_runs(self, runner_df):
        out = aggregate(runner_df, ["variant", "dataset"])
        a_t2 = out[(out["variant"] == "a") & (out["dataset"] == "T2")].iloc[0]
        assert a_t2["time_mean"] == pytest.approx(0.15) # (0.3 + 0.0) / 2

    def test_time_mean_found_excludes_failures(self, runner_df):
        out = aggregate(runner_df, ["variant", "dataset"])
        a_t2 = out[(out["variant"] == "a") & (out["dataset"] == "T2")].iloc[0]
        assert a_t2["time_mean_found"] == pytest.approx(0.3)

    def test_all_documented_columns_present(self, runner_df):
        out = aggregate(runner_df, ["variant", "dataset"])
        expected = {
            "n_runs", "success_rate",
            "r2_mean", "r2_std", "r2_sem",
            "time_mean", "time_std", "time_sem",
            "time_mean_found", "time_std_found", "time_sem_found",
        }
        assert expected.issubset(out.columns)

    def test_single_seed_cell_yields_nan_std_and_sem(self, runner_df):
        out = aggregate(runner_df.head(1), ["variant", "dataset"])
        assert math.isnan(out["r2_std"].iloc[0])
        assert math.isnan(out["r2_sem"].iloc[0])


# parse_variants

class TestParseVariants:
    def test_splits_variant_string_into_param_columns(self):
        df = pd.DataFrame({"variant": ["b3f_md_4", "b3f_md_6", "b3f_baseline"]})
        out = parse_variants(
            df,
            regex=r"^b3f_(md)_([\d.]+)$",
            param_map={"md": ("max_depth", int)},
            baseline_variant="b3f_baseline",
            baseline_params={"max_depth": 6},
        )
        # 2 swept rows + 1 baseline echo for max_depth.
        md_rows = out[out["param"] == "max_depth"]
        assert len(md_rows) == 3
        assert sorted(md_rows["param_value"]) == [4, 6, 6]

    def test_echoes_baseline_once_per_param(self):
        df = pd.DataFrame({"variant": ["b3f_md_4", "b3f_ct_0.2", "b3f_baseline"]})
        out = parse_variants(
            df,
            regex=r"^b3f_(md|ct)_([\d.]+)$",
            param_map={
                "md": ("max_depth", int),
                "ct": ("constancy_threshold", float),
            },
            baseline_variant="b3f_baseline",
            baseline_params={"max_depth": 6, "constancy_threshold": 0.1},
        )
        assert (out["param"] == "max_depth").sum() == 2
        assert (out["param"] == "constancy_threshold").sum() == 2

    def test_type_casting_applied_to_param_value(self):
        df = pd.DataFrame({"variant": ["b3f_md_4"]})
        out = parse_variants(
            df,
            regex=r"^b3f_(md)_([\d.]+)$",
            param_map={"md": ("max_depth", int)},
            baseline_variant="b3f_baseline",
            baseline_params={"max_depth": 6},
        )
        v = out[out["param"] == "max_depth"].iloc[0]["param_value"]
        assert v == 4
        assert pd.api.types.is_integer(v)


# equivalence_ratio

class TestEquivalenceRatio:

    def test_exact_match_returns_one(self):
        assert equivalence_ratio("V = I*R", "T1") == pytest.approx(1.0)

    def test_scaled_match_returns_constant(self):
        assert equivalence_ratio("V = 2*I*R", "T1") == pytest.approx(2.0)

    def test_wrong_form_returns_zero(self):
        assert equivalence_ratio("V = I + R", "T1") == 0.0

    def test_lhs_stripped_when_present(self):
        with_lhs    = equivalence_ratio("V = I*R", "T1")
        without_lhs = equivalence_ratio("I*R",     "T1")
        assert with_lhs == without_lhs == pytest.approx(1.0)

    def test_constant_folding(self):
        assert equivalence_ratio("s = 4.905*t**2", "T3") == pytest.approx(1.0)

    def test_wrong_power_returns_zero(self):
        assert equivalence_ratio("s = 4.905*t**3", "T3") == 0.0

    def test_extra_additive_term_returns_zero(self):
        assert equivalence_ratio("V = I*R + 0.001", "T1") == 0.0

    def test_multivariable_with_floating_point_constant(self):
        result = equivalence_ratio("T = 0.12028*P*V/n", "T4")
        assert result == pytest.approx(1.0, rel=1e-3)
