import pytest
import os
import tempfile
from argparse import Namespace
from symbolic_discovery.experiments.runner import run_experiment


def test_runner_bacon3f_s1():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        output_path = f.name
    os.remove(output_path)

    try:
        args = Namespace(
            models=["bacon3f"],
            datasets=["S1"],
            target=None,
            noise=[0.0],
            seeds=[42],
            output=output_path,
            verbose=False,
            n_samples=100,
            feynman_root="feynman",
        )
        run_experiment(args)

        assert os.path.exists(output_path)
        with open(output_path, "r") as f:
            lines = f.readlines()
        assert len(lines) == 2  # header + 1 result
        assert "bacon3f" in lines[1]
        assert "S1" in lines[1]
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)


def test_runner_bacon7f_s2():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        output_path = f.name
    os.remove(output_path)

    try:
        args = Namespace(
            models=["bacon7f"],
            datasets=["S2"],
            target=None,
            noise=[0.0],
            seeds=[42],
            output=output_path,
            verbose=False,
            n_samples=100,
            feynman_root="feynman",
        )
        run_experiment(args)

        assert os.path.exists(output_path)
        with open(output_path, "r") as f:
            lines = f.readlines()
        assert len(lines) == 2
        assert "bacon7f" in lines[1]
        assert "S2" in lines[1]
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)
