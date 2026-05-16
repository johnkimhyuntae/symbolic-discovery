"""
Unified dataset interface.

This is the module that :mod:`symbolic_discovery.experiments.runner`
(or any other consumer) should import from ``symbolic_discovery.data``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from .config import DatasetConfig
from . import feynman as _feyn
from . import custom as _custom
from . import synthetic as _synthetic


_KEY_RE = re.compile(r"^([STFB])(\d+)$")
_FAMILY_RE = re.compile(r"^[STFB]$")


# Noise injection

def inject_noise(
    df: pd.DataFrame,
    target: str,
    noise: float = 0.0,
    noise_type: str = "multiplicative",
    seed: int = 73,
) -> pd.DataFrame:
    """
    Add Gaussian noise scaled to the target column's range.

    Returns *df* unchanged when *noise* <= 0.
    """
    if noise <= 0.0 or target not in df.columns:
        return df
    y = df[target].to_numpy()
    y_range = float(np.ptp(y))
    scale = y_range if y_range > 1e-9 else 1.0
    rng = np.random.default_rng(seed)
    out = df.copy()
    if noise_type == "multiplicative":
        out[target] = y * (1 + rng.normal(0.0, noise, len(y)))
    else: # additive
        out[target] = y + rng.normal(0.0, noise * scale, len(y))
    return out


# Expand datasets

def expand_datasets(
    raw: list[str],
    feynman_root: str = "feynman",
) -> list[str]:
    """
    Expand bare-family wildcards; pass numbered keys through.

    ``["S", "F8", "T"]``  to  ``["S1", "S2", "S3", "F8", "T1", ..., "T5"]``
    """
    out: list[str] = []
    for token in raw:
        token = token.strip()
        if _FAMILY_RE.match(token):
            out.extend(_expand_family(token, feynman_root))
        else:
            out.append(token)
    return out


def _expand_family(family: str, feynman_root: str) -> list[str]:
    if family in ("S", "T"):
        return sorted(
            (k for k in _synthetic.CATALOGUE if k.startswith(family)),
            key=lambda k: int(k[1:]),
        )
    # F or B — derive keys from metadata + available data files
    nums = _feyn.equation_numbers(feynman_root, family)
    return [f"{family}{n}" for n in nums]


# Resolve keys to configs

def resolve(
    key: str,
    feynman_root: str = "feynman",
    target: str | None = None,
) -> DatasetConfig:
    """
    Resolve any dataset key to a :class:`DatasetConfig`.

    Supports ``S1``-``S3``, ``T1``-``T5``, ``F1``-``F100``, ``B1``-``B20``,
    and paths to custom ``.csv`` files (requires *target*).
    """
    key = key.strip()

    # Custom CSV
    if key.endswith(".csv"):
        if not target:
            raise ValueError(
                f"--target is required for custom CSV files: {key}"
            )
        name = Path(key).stem
        return DatasetConfig(
            key=name,
            eq_id=key,       # store the file path
            family="C",
            variables=[],    # unknown until loaded
            target=target,
            formula="Unknown",
        )

    # S / T — direct catalogue lookup
    if key in _synthetic.CATALOGUE:
        return _synthetic.CATALOGUE[key]

    # F / B — resolve via Feynman metadata
    m = _KEY_RE.match(key)
    if m and m.group(1) in ("F", "B"):
        family = m.group(1)
        number = int(m.group(2))
        row = _feyn.get_metadata_row(feynman_root, family, number=number)

        eq_id = str(row["Filename"]).strip()
        formula = (row.get("Formula", "") or "").strip() or "Unknown"
        target_var = (row.get("Output", "") or "").strip() or "y"
        variables = _feyn.extract_variables(row)

        return DatasetConfig(
            key=key,
            eq_id=eq_id,
            family=family,
            variables=variables,
            target=target_var,
            formula=formula,
        )

    raise ValueError(f"Unknown dataset key: {key!r}")


# Load data

def load(
    config: DatasetConfig,
    *,
    noise: float = 0.0,
    noise_type: str = "multiplicative",
    seed: int = 73,
    n_samples: int = 1000,
    feynman_root: str = "feynman",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """
    Load data for any dataset.

    Returns ``(train_df, test_df, pretty_map)`` where 
    data split is 0.8, and pretty_map is a dictionary mapping safe 
    column names to raw dataset symbols.
    """
    # S / T — synthetic generation
    if config.family in ("S", "T"):
        df = _synthetic.generate(config, n_samples=n_samples, seed=seed)
        if noise > 0:
            df = inject_noise(df, config.target, noise, noise_type, seed)

    # C — custom CSV
    elif config.family == "C":
        df = _custom.load_csv(config.eq_id, n_samples=n_samples, seed=seed)
        if noise > 0:
            df = inject_noise(df, config.target, noise, noise_type, seed)

    # F / B — feynman databases
    else:
        df = _feyn.load_data(
            config.eq_id, feynman_root, config.family, n_samples, seed,
            target=config.target,
        )
        if noise > 0:
            df = inject_noise(df, config.target, noise, noise_type, seed)

    feature_cols = [c for c in df.columns]
    pretty_map = _build_pretty_map(config.variables + [config.target], feature_cols)
    return df[:int(len(df) * 0.8)], df[int(len(df) * 0.8):], pretty_map


def _build_pretty_map(
    variables: list[str],
    feature_cols: list[str],
) -> dict[str, str]:
    """Map safe column names (x1, x2, ...) to raw symbols (θ, ...)."""
    pm: dict[str, str] = {}
    for idx, safe in enumerate(feature_cols):
        pretty = variables[idx] if idx < len(variables) else safe
        pm[safe] = pretty
    # Extend with squared / cubed forms for pretty-printing.
    for safe, pretty in list(pm.items()):
        pm[f"{safe}²"] = f"({pretty})²"
        pm[f"{safe}³"] = f"({pretty})³"
    return pm


# Feynman exclusions

def get_exclusion_reason(config: DatasetConfig) -> str | None:
    """
    Return the exclusion reason for *config*, or ``None``.
    """
    return _feyn.get_exclusion_reason(config.key)


# Pretty-printing

def pretty_equation(
    eq: str,
    pretty_map: Optional[dict[str, str]],
) -> str:
    """Substitute safe column names (x1, x2, ...) for raw symbols."""
    if not eq or not pretty_map:
        return eq
    # Replace longer tokens first to avoid partial overlaps (x1³ before x1).
    keys = sorted(pretty_map.keys(), key=len, reverse=True)
    out = eq
    for k in keys:
        v = pretty_map[k]
        if not v or v == k:
            continue
        out = re.sub(rf"\b{re.escape(k)}\b", v, out)
    return out
