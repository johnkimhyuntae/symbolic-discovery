"""
Unit tests for symbolic_discovery.experiments.runner.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from types import SimpleNamespace

import pytest
import yaml

from symbolic_discovery.solvers import SOLVER_REGISTRY, SolverResult
from symbolic_discovery.experiments.runner import (
    _build_runs,
    _print_progress,
    _validate_args,
    _write_row,
    build_arg_parser,
)
from symbolic_discovery.experiments.plan import Run, Variant


KNOWN_MODEL = next(iter(SOLVER_REGISTRY))


# _validate_args

class TestValidateArgs:
    """
    Cross-flag validation: at least one of --models / --variant / --sweep /
    --study must be supplied, and --datasets is required unless --study
    provides them.
    """

    def _parser(self):
        # Each test gets a fresh parser; argparse calls sys.exit on
        # parser.error, so we can catch SystemExit cleanly.
        return argparse.ArgumentParser()

    def test_no_solver_source_raises(self, default_runner_args):
        args = default_runner_args(datasets=["S1"])
        with pytest.raises(SystemExit):
            _validate_args(self._parser(), args)

    def test_no_datasets_without_study_raises(self, default_runner_args):
        args = default_runner_args(models=[KNOWN_MODEL])
        with pytest.raises(SystemExit):
            _validate_args(self._parser(), args)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"models": [KNOWN_MODEL], "datasets": ["S1"]},
            {"study": "some.yaml"},
            {"variant": [f"x={KNOWN_MODEL}"], "datasets": ["S1"]},
            {"sweep": [f"{KNOWN_MODEL}.k=1,2"], "datasets": ["S1"]},
        ],
        ids=["models", "study", "variant", "sweep"],
    )
    def test_accepts_valid_solver_sources(self, default_runner_args, kwargs):
        args = default_runner_args(**kwargs)
        _validate_args(self._parser(), args)


# build_arg_parser

class TestBuildArgParser:
    def test_models_and_datasets(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--models", KNOWN_MODEL, "--datasets", "S1"])
        assert args.models == [KNOWN_MODEL]
        assert args.datasets == ["S1"]

    def test_variant_is_appendable(self):
        # --variant is action="append": each occurrence accumulates.
        parser = build_arg_parser()
        args = parser.parse_args([
            "--variant", f"a={KNOWN_MODEL}",
            "--variant", f"b={KNOWN_MODEL}",
            "--datasets", "S1",
        ])
        assert args.variant == [f"a={KNOWN_MODEL}", f"b={KNOWN_MODEL}"]

    def test_sweep_is_appendable(self):
        parser = build_arg_parser()
        args = parser.parse_args([
            "--sweep", f"{KNOWN_MODEL}.k=1,2",
            "--sweep", f"{KNOWN_MODEL}.j=3,4",
            "--datasets", "S1",
        ])
        assert args.sweep == [f"{KNOWN_MODEL}.k=1,2", f"{KNOWN_MODEL}.j=3,4"]

    def test_noise_types_dashes_become_underscores(self):
        parser = build_arg_parser()
        args = parser.parse_args([
            "--models", KNOWN_MODEL, "--datasets", "S1",
            "--noise-types", "additive", "multiplicative",
        ])
        assert args.noise_types == ["additive", "multiplicative"]

    def test_n_samples_parses_multiple_ints(self):
        parser = build_arg_parser()
        args = parser.parse_args([
            "--models", KNOWN_MODEL, "--datasets", "S1",
            "--n-samples", "50", "100", "1000",
        ])
        assert args.n_samples == [50, 100, 1000]

    def test_unknown_model_rejected(self):
        # argparse `choices` enforces the set of known solvers.
        parser = build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--models", "not_a_real_solver",
                "--datasets", "S1",
            ])

    def test_unknown_noise_type_rejected(self):
        parser = build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--models", KNOWN_MODEL, "--datasets", "S1",
                "--noise-types", "bogus_noise",
            ])


# _build_runs

class TestBuildRuns:
    def test_models_only(self, default_runner_args):
        args = default_runner_args(models=[KNOWN_MODEL], datasets=["S1"])
        runs = _build_runs(args)
        assert len(runs) == 1
        assert runs[0].variant.name == KNOWN_MODEL
        assert runs[0].variant.model == KNOWN_MODEL
        assert runs[0].variant.params == {}

    def test_models_use_default_axis_values(self, default_runner_args):
        # Defaults defined in _build_runs when neither CLI nor study
        # supply them. These pin the framework's documented behaviour.
        args = default_runner_args(models=[KNOWN_MODEL], datasets=["S1"])
        runs = _build_runs(args)
        r = runs[0]
        assert r.noise == 0.0
        assert r.noise_type == "multiplicative"
        assert r.n_samples == 1000
        assert r.seed == 73

    def test_variant_spec(self, fake_solver, default_runner_args):
        args = default_runner_args(
            variant=["custom=fake:k=7"],
            datasets=["S1"],
        )
        runs = _build_runs(args)
        assert len(runs) == 1
        assert runs[0].variant.name == "custom"
        assert runs[0].variant.model == "fake"
        assert runs[0].variant.params == {"k": 7}

    def test_sweep_spec(self, fake_solver, default_runner_args):
        args = default_runner_args(
            sweep=["fake.k=1,2,3"],
            datasets=["S1"],
        )
        runs = _build_runs(args)
        assert len(runs) == 3
        assert [r.variant.params["k"] for r in runs] == [1, 2, 3]

    def test_models_variants_sweeps_concatenate(self, fake_solver, default_runner_args):
        # 1 model + 1 variant + 2 swept = 4 distinct variants in the run list.
        args = default_runner_args(
            models=[KNOWN_MODEL],
            variant=["custom=fake:k=1"],
            sweep=["fake.j=2,3"],
            datasets=["S1"],
        )
        runs = _build_runs(args)
        assert len({r.variant.name for r in runs}) == 4

    def test_full_grid_cardinality(self, default_runner_args):
        args = default_runner_args(
            models=[KNOWN_MODEL],
            datasets=["S1", "S2"],
            noise=[0.0, 0.1],
            noise_types=["additive", "multiplicative"],
            n_samples=[100, 200],
            seeds=[1, 2, 3],
        )
        runs = _build_runs(args)
        # 1 variant × 2 datasets × 2 noise × 2 noise_types × 2 n_samples × 3 seeds
        assert len(runs) == 1 * 2 * 2 * 2 * 2 * 3

    def test_no_variants_raises(self, default_runner_args):
        args = default_runner_args(datasets=["S1"])
        with pytest.raises(ValueError, match="No variants"):
            _build_runs(args)

    def test_no_datasets_raises(self, default_runner_args):
        args = default_runner_args(models=[KNOWN_MODEL])
        with pytest.raises(ValueError, match="No datasets"):
            _build_runs(args)

    def test_study_provides_axes_when_cli_silent(self, default_runner_args, tmp_path):
        study_path = tmp_path / "s.yaml"
        study_path.write_text(yaml.safe_dump({
            "variants": [{"name": "v1", "model": KNOWN_MODEL}],
            "datasets": ["S1", "S2"],
            "noise": [0.0, 0.05],
            "seeds": [1, 2],
        }))
        args = default_runner_args(study=str(study_path))
        runs = _build_runs(args)
        # 1 variant × 2 datasets × 2 noise × 1 noise_type × 1 n_samples × 2 seeds
        assert len(runs) == 8

    def test_cli_overrides_study(self, default_runner_args, tmp_path):
        # Documented precedence: --datasets / --models / --seeds on the
        # CLI replace, not merge, the study's values.
        study_path = tmp_path / "s.yaml"
        study_path.write_text(yaml.safe_dump({
            "variants": [{"name": "v_study", "model": KNOWN_MODEL}],
            "datasets": ["S1", "S2"],
            "seeds": [1, 2, 3],
        }))
        args = default_runner_args(
            study=str(study_path),
            models=[KNOWN_MODEL],   # overrides study variants
            datasets=["S3"],         # overrides study datasets
            seeds=[99],              # overrides study seeds
        )
        runs = _build_runs(args)
        assert all(r.dataset == "S3" for r in runs)
        assert all(r.seed == 99 for r in runs)
        assert all(r.variant.name == KNOWN_MODEL for r in runs)


# _write_row

class TestWriteRow:
    def _setup(self, csv_fieldnames):
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=csv_fieldnames)
        writer.writeheader()
        return buf, writer

    def _make(self):
        v = Variant(name="v1", model=KNOWN_MODEL, params={"a": 1, "b": 2.5})
        run = Run(
            variant=v, dataset="S1", noise=0.05,
            noise_type="additive", n_samples=500, seed=73,
        )
        config = SimpleNamespace(key="S1", target="y")
        result = SolverResult(
            equation="y = x", raw_equation="y = x",
            r2=0.99, mse=0.01, mae=0.01,
            time_sec=0.5, status="Found",
        )
        return v, run, config, result

    def test_writes_all_fields(self, csv_fieldnames):
        buf, writer = self._setup(csv_fieldnames)
        v, run, config, result = self._make()
        _write_row(writer, "rid_1", run, config, v, result, None)
        rows = list(csv.DictReader(io.StringIO(buf.getvalue())))
        assert len(rows) == 1
        row = rows[0]
        assert row["run_id"] == "rid_1"
        assert row["dataset"] == "S1"
        assert row["method"] == KNOWN_MODEL
        assert row["variant"] == "v1"
        assert row["noise"] == "0.05"
        assert row["noise_type"] == "additive"
        assert row["n_samples"] == "500"
        assert row["seed"] == "73"
        assert row["status"] == "Found"

    def test_params_json_is_valid_and_sorted(self, csv_fieldnames):
        # `sort_keys=True` is non-decorative: it means two rows with the
        # same params produce identical JSON strings, so analysis can do
        # df.groupby("params_json") to identify sweep variants.
        buf, writer = self._setup(csv_fieldnames)
        v, run, config, result = self._make()
        _write_row(writer, "rid_1", run, config, v, result, None)
        row = next(csv.DictReader(io.StringIO(buf.getvalue())))
        parsed = json.loads(row["params_json"])
        assert parsed == {"a": 1, "b": 2.5}
        assert row["params_json"] == '{"a": 1, "b": 2.5}'

    def test_empty_params_serializes_to_empty_object(self, csv_fieldnames):
        buf, writer = self._setup(csv_fieldnames)
        v = Variant(name="v1", model=KNOWN_MODEL)
        run = Run(
            variant=v, dataset="S1", noise=0.0,
            noise_type="multiplicative", n_samples=100, seed=1,
        )
        config = SimpleNamespace(key="S1", target="y")
        result = SolverResult(
            equation="", raw_equation="",
            r2=0.0, mse=0.0, mae=0.0,
            time_sec=0.0, status="Failure",
        )
        _write_row(writer, "rid_1", run, config, v, result, None)
        row = next(csv.DictReader(io.StringIO(buf.getvalue())))
        assert json.loads(row["params_json"]) == {}


# _print_progress

class TestPrintProgress:
    def _make(self, status="Found", equation="y = x", raw="y = x"):
        v = Variant(name="v1", model=KNOWN_MODEL)
        run = Run(
            variant=v, dataset="S1", noise=0.0,
            noise_type="multiplicative", n_samples=100, seed=73,
        )
        config = SimpleNamespace(key="S1", target="y")
        result = SolverResult(
            equation=equation, raw_equation=raw,
            r2=0.99, mse=0.01, mae=0.01,
            time_sec=0.5, status=status,
        )
        return run, v, config, result

    def test_prints_summary_line(self, capsys):
        run, v, config, result = self._make()
        _print_progress("rid_1", run, config, v, result, None)
        out = capsys.readouterr().out
        # Identifying info, not exact format — the format may evolve.
        assert "v1" in out
        assert "S1" in out
        assert "Found" in out
        assert "0.9900" in out

    def test_prints_equation_when_found(self, capsys):
        run, v, config, result = self._make(equation="y = 2*x")
        _print_progress("rid_1", run, config, v, result, None)
        out = capsys.readouterr().out
        assert "y = 2*x" in out

    def test_prints_details_for_failure(self, capsys):
        # On failure the equation is "No law found"; we want the more
        # informative raw_equation surfaced instead.
        run, v, config, result = self._make(
            status="Failure", equation="No law found",
            raw="Excluded: too complex",
        )
        _print_progress("rid_1", run, config, v, result, None)
        out = capsys.readouterr().out
        assert "Excluded: too complex" in out

    def test_truncates_long_equations(self, capsys):
        # 100-term polynomials look ugly in the terminal; truncate.
        long_eq = "y = " + " + ".join([f"a{i}*x{i}" for i in range(100)])
        run, v, config, result = self._make(equation=long_eq, raw=long_eq)
        _print_progress("rid_1", run, config, v, result, None)
        out = capsys.readouterr().out
        assert "..." in out