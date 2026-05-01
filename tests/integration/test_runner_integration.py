"""
Integration tests for symbolic_discovery.experiments.runner.
"""

from __future__ import annotations

import csv
import json

import pytest
import yaml

from symbolic_discovery.experiments.runner import (
    main as runner_main,
    run_experiment,
    _execute_one,
)
from symbolic_discovery.solvers import SOLVER_REGISTRY


pytestmark = pytest.mark.integration


# Shared helpers

def _read_rows(csv_path):
    with open(csv_path) as f:
        return list(csv.DictReader(f))


# Single-run sanity

class TestSingleRun:

    def test_writes_one_row(self, fake_solver, default_runner_args, tmp_path):
        out = tmp_path / "results.csv"
        args = default_runner_args(
            variant=["custom=fake"],
            datasets=["S1"],
            n_samples=[50],
            output=str(out),
        )
        run_experiment(args)
        rows = _read_rows(out)
        assert len(rows) == 1
        assert rows[0]["variant"] == "custom"
        assert rows[0]["method"] == "fake"
        assert rows[0]["dataset"] == "S1"
        assert rows[0]["status"] == "Found"

    def test_run_id_includes_noise_type_initial(
        self, fake_solver, default_runner_args, tmp_path,
    ):
        # The deterministic run_id encodes axes for grep-friendliness;
        # noise_type is encoded as its first letter.
        out = tmp_path / "results.csv"
        args = default_runner_args(
            variant=["v=fake"],
            datasets=["S1"],
            noise=[0.1],
            noise_types=["gaussian"],
            n_samples=[50],
            output=str(out),
        )
        run_experiment(args)
        rows = _read_rows(out)
        assert "0.1g" in rows[0]["run_id"]


# CSV append behaviour

class TestCsvAppend:

    def test_header_written_only_once(
        self, fake_solver, default_runner_args, tmp_path,
    ):
        out = tmp_path / "results.csv"

        # First invocation: header + 1 row.
        args1 = default_runner_args(
            variant=["a=fake"], datasets=["S1"],
            n_samples=[50], output=str(out),
        )
        run_experiment(args1)

        # Second invocation: append, no second header.
        args2 = default_runner_args(
            variant=["b=fake"], datasets=["S2"],
            n_samples=[50], output=str(out),
        )
        run_experiment(args2)

        with open(out) as f:
            lines = f.readlines()
        assert len(lines) == 3            # header + 2 data rows
        assert lines[0].startswith("run_id,")


# Sweeps

class TestSweep:

    def test_one_row_per_swept_value(
        self, fake_solver, default_runner_args, tmp_path,
    ):
        out = tmp_path / "results.csv"
        args = default_runner_args(
            sweep=["fake.k=1,2,3,4"],
            datasets=["S1"],
            n_samples=[50],
            output=str(out),
        )
        run_experiment(args)
        rows = _read_rows(out)
        assert len(rows) == 4
        # Each row carries its own k via params_json.
        ks = [json.loads(r["params_json"])["k"] for r in rows]
        assert sorted(ks) == [1, 2, 3, 4]


# Full grid

class TestFullGrid:

    def test_cardinality(self, fake_solver, default_runner_args, tmp_path):
        out = tmp_path / "results.csv"
        args = default_runner_args(
            variant=["v1=fake", "v2=fake"],
            datasets=["S1", "S2"],
            noise=[0.0, 0.05],
            noise_types=["multiplicative"],
            n_samples=[50],
            seeds=[1, 2],
            output=str(out),
        )
        run_experiment(args)
        rows = _read_rows(out)
        # 2 variants × 2 datasets × 2 noise × 1 noise_type × 1 n_samples × 2 seeds = 16
        assert len(rows) == 16


# Variant params forwarding

class TestVariantParamsForwarding:

    def test_variant_params_reach_solver(
        self, fake_solver, default_runner_args, tmp_path,
    ):
        # The runner instantiates SolverClass(verbose=..., **params).
        # FakeSolver records its kwargs so we can inspect them.
        out = tmp_path / "results.csv"
        args = default_runner_args(
            variant=["custom=fake:my_param=99,verbose_extra=true"],
            datasets=["S1"],
            n_samples=[50],
            output=str(out),
        )
        run_experiment(args)

        # The fake_solver fixture yields the FakeSolver class itself.
        assert len(fake_solver.instances) == 1
        kwargs = fake_solver.instances[0].kwargs
        assert kwargs["my_param"] == 99
        assert kwargs["verbose_extra"] is True
        assert "verbose" in kwargs   # injected by the runner itself


# Study file

class TestStudyFile:

    def test_full_run_from_study_file(
        self, fake_solver, default_runner_args, tmp_path,
    ):
        study_path = tmp_path / "study.yaml"
        study_path.write_text(yaml.safe_dump({
            "variants": [
                {"name": "a", "model": "fake", "params": {"k": 1}},
                {"name": "b", "model": "fake", "params": {"k": 2}},
            ],
            "datasets": ["S1"],
            "noise": [0.0],
            "noise_types": ["multiplicative"],
            "n_samples": [50],
            "seeds": [42, 43],
        }))
        out = tmp_path / "results.csv"
        args = default_runner_args(study=str(study_path), output=str(out))
        run_experiment(args)

        rows = _read_rows(out)
        assert len(rows) == 4
        assert {r["variant"] for r in rows} == {"a", "b"}
        assert {r["seed"] for r in rows} == {"42", "43"}


# main() entry point

class TestMainEntryPoint:

    def test_main_argv_to_csv(self, fake_solver, tmp_path):
        # Exercises the full main() entry point: argparse construction,
        # validation, build_runs, execute, write — all from a list of
        # CLI tokens.
        out = tmp_path / "results.csv"
        runner_main([
            "--variant", "v=fake",
            "--datasets", "S1",
            "--n-samples", "50",
            "--output", str(out),
        ])
        rows = _read_rows(out)
        assert len(rows) == 1