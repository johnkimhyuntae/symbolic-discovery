"""
Synthetic data generation and the built-in synthetic (S) and textbook (T) datasets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DatasetConfig

# Built-in datasets
CATALOGUE: dict[str, DatasetConfig] = {
    # Synthetic
    "S1": DatasetConfig("S1", "S1", "S", ["x1", "x2"], "y", "x1 + x2",
                        {"x1": (-5, 5), "x2": (-5, 5)}),
    "S2": DatasetConfig("S2", "S2", "S", ["x1", "x2"], "y", "x1 * x2",
                        {"x1": (1, 5), "x2": (1, 5)}),
    "S3": DatasetConfig("S3", "S3", "S", ["x1", "x2"], "y", "x1 / (x2 + 1)",
                        {"x1": (1, 10), "x2": (1, 10)}),

    # Textbook laws
    "T1": DatasetConfig("T1", "T1", "T", ["I", "R"], "V", "I * R",
                        {"I": (0, 2), "R": (1, 10)}),
    "T2": DatasetConfig("T2", "T2", "T", ["k", "x"], "F", "k * x",
                        {"k": (1, 10), "x": (-1, 1)}),
    "T3": DatasetConfig("T3", "T3", "T", ["t"], "s", "0.5 * 9.81 * t**2",
                        {"t": (0, 2)}),
    "T4": DatasetConfig("T4", "T4", "T", ["P", "V", "n"], "T",
                        "(P * V) / (n * 8.314)",
                        {"P": (1, 5), "V": (10, 30), "n": (1, 2)}),
    "T5": DatasetConfig("T5", "T5", "T", ["T"], "P", "5.67e-8 * T**4",
                        {"T": (100, 500)}),
}


def generate(
    config: DatasetConfig,
    *,
    n_samples: int = 1000,
    seed: int = 73,
) -> pd.DataFrame:
    """
    Generate a synthetic DataFrame from a catalogue config.

    The DataFrame is returned with sympy-safe input column names
    ``x1, x2, ...``. The original symbols live in ``config.variables`` 
    and surface via the pretty_map built by :func:`api.load`.
    """
    rng = np.random.default_rng(seed)

    data: dict[str, np.ndarray] = {}
    for var in config.variables:
        low, high = config.domain[var]
        data[var] = rng.uniform(low, high, n_samples)

    df = pd.DataFrame(data)
    df[config.target] = df.eval(config.formula)

    rename = {v: f"x{i + 1}" for i, v in enumerate(config.variables)}
    return df.rename(columns=rename)
