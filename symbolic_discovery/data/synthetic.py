"""
Synthetic data generation and noise injection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .catalogue import DatasetConfig


def generate(
    config: DatasetConfig,
    *,
    n_samples: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic DataFrame from a catalogue config."""
    rng = np.random.default_rng(seed)

    data: dict[str, np.ndarray] = {}
    for var in config.variables:
        low, high = config.domain[var]
        data[var] = rng.uniform(low, high, n_samples)

    df = pd.DataFrame(data)
    df[config.target] = df.eval(config.formula)

    return df