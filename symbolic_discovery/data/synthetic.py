from __future__ import annotations

import numpy as np
import pandas as pd

from .catalogue import CATALOGUE, DatasetConfig


class DatasetGenerator:
    """Generate synthetic catalogue datasets.

    Handles data generation, noise injection, and splitting.
    Ensures reproducibility via fixed seeds.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    def generate(
        self, config_id: str, noise_level: float = 0.0
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if config_id not in CATALOGUE:
            raise ValueError(f"Unknown dataset ID: {config_id}")

        config = CATALOGUE[config_id]

        data: dict[str, np.ndarray] = {}
        for var in config.variables:
            low, high = config.domain[var]
            data[var] = self.rng.uniform(low, high, config.n_samples)

        df = pd.DataFrame(data)
        df[config.target] = df.eval(config.formula)

        if noise_level > 0.0:
            target_range = df[config.target].max() - df[config.target].min()
            scale = float(target_range) if float(target_range) > 1e-9 else 1.0
            noise = self.rng.normal(loc=0.0, scale=noise_level * scale, size=len(df))
            df[config.target] += noise

        train_df = df.sample(frac=0.75, random_state=self.seed)
        test_df = df.drop(train_df.index)
        extra_df = self._generate_extrapolation_slab(config)

        return train_df, test_df, extra_df

    def _generate_extrapolation_slab(self, config: DatasetConfig) -> pd.DataFrame:
        data: dict[str, np.ndarray] = {}
        n_extra = config.n_samples // 4

        for var in config.variables:
            low, high = config.domain[var]
            span = high - low
            new_low = high
            new_high = high + (0.25 * span)
            data[var] = self.rng.uniform(new_low, new_high, n_extra)

        df = pd.DataFrame(data)
        df[config.target] = df.eval(config.formula)
        return df


def split_train_test(
    df: pd.DataFrame,
    *,
    train_frac: float = 0.75,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not (0.0 < train_frac < 1.0):
        raise ValueError("train_frac must be between 0 and 1")
    train_df = df.sample(frac=train_frac, random_state=seed)
    test_df = df.drop(train_df.index)
    return train_df, test_df
