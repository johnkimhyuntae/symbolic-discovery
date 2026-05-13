"""
Tests for symbolic_discovery.cli.main.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from symbolic_discovery.cli.main import build_arg_parser, main
from symbolic_discovery.experiments import runner, viewer


# build_arg_parser

class TestBuildArgParser:
    def test_run_subcommand_recognised(self):
        parser = build_arg_parser()
        ns = parser.parse_args(["run"])
        assert ns.command == "run"

    def test_view_subcommand_recognised(self):
        parser = build_arg_parser()
        ns = parser.parse_args(["view"])
        assert ns.command == "view"

    def test_unknown_subcommand_rejected(self):
        parser = build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["churchill"])

    def test_subcommand_is_required(self):
        parser = build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])


# Empty argv -> print help, no dispatch

class TestMainEmptyArgv:
    def test_prints_help(self, capsys):
        main([])
        out = capsys.readouterr().out
        assert "usage" in out.lower()
        assert "symbolic-discovery" in out

    def test_does_not_dispatch(self, monkeypatch):
        runner_main = Mock()
        viewer_main = Mock()
        monkeypatch.setattr(runner, "main", runner_main)
        monkeypatch.setattr(viewer, "main", viewer_main)
        main([])
        runner_main.assert_not_called()
        viewer_main.assert_not_called()

    def test_returns_cleanly(self):
        main([])  # would raise if it did


# årun dispatches to runner.main

class TestMainRunDispatch:
    def test_forwards_argv_with_subcommand_stripped(self, monkeypatch):
        runner_main = Mock()
        monkeypatch.setattr(runner, "main", runner_main)
        main(["run", "--models", "bacon3f", "--datasets", "S1"])
        runner_main.assert_called_once_with(
            ["--models", "bacon3f", "--datasets", "S1"]
        )

    def test_strips_subcommand_even_with_no_further_args(self, monkeypatch):
        runner_main = Mock()
        monkeypatch.setattr(runner, "main", runner_main)
        main(["run"])
        runner_main.assert_called_once_with([])

    def test_does_not_call_viewer(self, monkeypatch):
        monkeypatch.setattr(runner, "main", Mock())
        viewer_main = Mock()
        monkeypatch.setattr(viewer, "main", viewer_main)
        main(["run", "--datasets", "S1"])
        viewer_main.assert_not_called()

    def test_returns_cleanly_on_run(self, monkeypatch):
        monkeypatch.setattr(runner, "main", lambda *_a, **_k: None)
        main(["run", "--datasets", "S1"])


# view dispatches to viewer.main

class TestMainViewDispatch:
    def test_forwards_argv_with_subcommand_stripped(self, monkeypatch):
        viewer_main = Mock()
        monkeypatch.setattr(viewer, "main", viewer_main)
        main(["view", "results.csv", "--stats"])
        viewer_main.assert_called_once_with(["results.csv", "--stats"])

    def test_strips_subcommand_even_with_no_further_args(self, monkeypatch):
        viewer_main = Mock()
        monkeypatch.setattr(viewer, "main", viewer_main)
        main(["view"])
        viewer_main.assert_called_once_with([])

    def test_does_not_call_runner(self, monkeypatch):
        monkeypatch.setattr(viewer, "main", Mock())
        runner_main = Mock()
        monkeypatch.setattr(runner, "main", runner_main)
        main(["view", "results.csv"])
        runner_main.assert_not_called()


# Unknown command -> SystemExit, no dispatch

class TestMainUnknownCommand:
    def test_unknown_command_raises_systemexit(self, monkeypatch):
        monkeypatch.setattr(runner, "main", lambda *_a, **_k: None)
        monkeypatch.setattr(viewer, "main", lambda *_a, **_k: None)
        with pytest.raises(SystemExit):
            main(["bogus"])

    def test_unknown_command_does_not_dispatch(self, monkeypatch):
        runner_main = Mock()
        viewer_main = Mock()
        monkeypatch.setattr(runner, "main", runner_main)
        monkeypatch.setattr(viewer, "main", viewer_main)
        with pytest.raises(SystemExit):
            main(["bogus"])
        runner_main.assert_not_called()
        viewer_main.assert_not_called()

    @pytest.mark.parametrize("command", ["cambridge", "hi", "View", "RUN"])
    def test_typos_and_case_variants_all_rejected(self, command, monkeypatch):
        monkeypatch.setattr(runner, "main", lambda *_a, **_k: None)
        monkeypatch.setattr(viewer, "main", lambda *_a, **_k: None)
        with pytest.raises(SystemExit):
            main([command])