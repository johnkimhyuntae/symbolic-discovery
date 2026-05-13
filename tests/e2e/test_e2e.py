"""End-to-end checks for the top-level CLI."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from symbolic_discovery.cli.main import main as cli_main


def _read_csv_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# The round-trip

class TestCliRoundTrip:
    """Round-trip checks through the top-level CLI."""

    def test_run_then_view(self, fake_solver, tmp_path, capsys):
        out = tmp_path / "results.csv"

        cli_main([
            "run",
            "--variant", "v=fake",
            "--datasets", "S1", "S2",
            "--n-samples", "50",
            "--seeds", "73", "74",
            "--output", str(out),
        ])
        assert out.exists(), "run subcommand did not produce the expected CSV"

        rows = _read_csv_rows(out)
        assert len(rows) == 4
        assert {r["method"] for r in rows} == {"fake"}
        assert {r["variant"] for r in rows} == {"v"}
        assert {r["dataset"] for r in rows} == {"S1", "S2"}
        assert {r["status"] for r in rows} == {"Found"}

        with open(out) as f:
            header = f.readline().strip().split(",")
        assert header == [
            "run_id", "dataset", "method", "variant", "params_json",
            "noise", "noise_type", "n_samples", "seed",
            "equation", "raw_equation", "r2", "mse", "mae", "time_s", "status",
        ]

        capsys.readouterr()

        cli_main(["view", str(out)])
        captured = capsys.readouterr()
        assert "results.csv" in captured.out

    def test_run_with_sweep_and_view_stats(self, fake_solver, tmp_path, capsys):
        """Sweeps expand into variants and stats mode writes a sibling CSV."""
        out = tmp_path / "sweep.csv"

        cli_main([
            "run",
            "--sweep", "fake.k=1,2,3",
            "--datasets", "S1",
            "--n-samples", "50",
            "--seeds", "73", "74",
            "--output", str(out),
        ])
        rows = _read_csv_rows(out)
        assert len(rows) == 6

        ks = {r["variant"] for r in rows}
        assert ks == {"fake_k_1", "fake_k_2", "fake_k_3"}

        capsys.readouterr()

        cli_main(["view", str(out), "--stats"])
        stats_path = out.with_name("sweep_stats.csv")
        assert stats_path.exists(), "stats mode did not produce the expected CSV"


# Failure modes

class TestCliFailureModes:
    """CLI misuse should fail explicitly but predictably."""

    def test_no_args_prints_help_and_returns(self, capsys):
        cli_main([])
        captured = capsys.readouterr()
        assert "symbolic-discovery" in captured.out

    def test_run_without_models_or_variant_errors(self, tmp_path):
        out = tmp_path / "results.csv"
        with pytest.raises(SystemExit):
            cli_main([
                "run",
                "--datasets", "S1",
                "--output", str(out),
            ])

    def test_run_without_datasets_errors(self, fake_solver, tmp_path):
        out = tmp_path / "results.csv"
        with pytest.raises(SystemExit):
            cli_main([
                "run",
                "--variant", "v=fake",
                "--output", str(out),
            ])

    def test_view_missing_csv_does_not_crash(self, tmp_path, capsys):
        cli_main(["view", str(tmp_path / "nonexistent.csv")])
        captured = capsys.readouterr()
        assert "not found" in captured.out.lower()