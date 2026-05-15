from __future__ import annotations

import csv
import json

import pytest
import yaml

from symbolic_discovery.experiments.runner import (
    main as runner_main,
    run_experiment,
)

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
        out = tmp_path / "results.csv"
        args = default_runner_args(
            variant=["v=fake"],
            datasets=["S1"],
            noise=[0.1],
            noise_types=["additive"],
            n_samples=[50],
            output=str(out),
        )
        run_experiment(args)
        rows = _read_rows(out)
        assert "0.1a" in rows[0]["run_id"]


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
        assert len(lines) == 3 # header + 2 data rows
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
        assert len(rows) == 16


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
            "seeds": [73, 74],
        }))
        out = tmp_path / "results.csv"
        args = default_runner_args(study=str(study_path), output=str(out))
        run_experiment(args)

        rows = _read_rows(out)
        assert len(rows) == 4
        assert {r["variant"] for r in rows} == {"a", "b"}
        assert {r["seed"] for r in rows} == {"73", "74"}


# main() entry point

class TestMainEntryPoint:
    def test_main_argv_to_csv(self, fake_solver, tmp_path):
        # full main() entry point
        out = tmp_path / "results.csv"
        runner_main([
            "--variant", "v=fake",
            "--datasets", "S1",
            "--n-samples", "50",
            "--output", str(out),
        ])
        rows = _read_rows(out)
        assert len(rows) == 1
