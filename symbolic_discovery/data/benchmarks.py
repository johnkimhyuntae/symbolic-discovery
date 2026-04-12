from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .catalogue import DatasetConfig


def _read_whitespace_matrix(file_path: Path, n_samples: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if len(rows) >= n_samples:
                break
            arr = np.fromstring(line.strip(), sep=" ")
            if arr.size == 0:
                continue
            rows.append(arr)

    if len(rows) < n_samples:
        raise ValueError(
            f"Equation file {file_path.name} had only {len(rows)} usable rows; requested {n_samples}"
        )
    return np.vstack(rows)


_BENCHMARK_META_FILES: dict[tuple[str, bool], str] = {
    ("feynman", True): "FeynmanEquationsDimensionless.csv",
    ("feynman", False): "FeynmanEquations.csv",
    ("bonus", True): "BonusEquationsDimensionless.csv",
    ("bonus", False): "BonusEquations.csv",
}

_BENCHMARK_DATA_DIRS: dict[tuple[str, bool], str] = {
    ("feynman", True): "Feynman_without_units",
    ("feynman", False): "Feynman_with_units",
    ("bonus", True): "bonus_without_units",
    ("bonus", False): "bonus_with_units",
}


@lru_cache(maxsize=16)
def _load_benchmark_metadata(root_dir: str, family: str, dimensionless: bool) -> pd.DataFrame:
    family_key = str(family).strip().lower()
    key = (family_key, bool(dimensionless))
    if key not in _BENCHMARK_META_FILES:
        raise ValueError(f"Unknown benchmark family/variant: {key}")

    root = Path(root_dir)
    meta_path = root / _BENCHMARK_META_FILES[key]
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing benchmark metadata CSV: {meta_path}")

    return pd.read_csv(meta_path, dtype=str).fillna("")


def list_benchmark_equations(
    *,
    root_dir: str = "feynman",
    family: str = "feynman",
    dimensionless: bool = True,
    require_data_file: bool = True,
) -> list[str]:
    meta = _load_benchmark_metadata(root_dir, family, dimensionless)
    eqs = [
        str(x).strip()
        for x in meta.get("Filename", pd.Series([], dtype=str)).tolist()
        if str(x).strip()
    ]
    if not require_data_file:
        return eqs

    family_key = str(family).strip().lower()
    data_dir_name = _BENCHMARK_DATA_DIRS[(family_key, bool(dimensionless))]
    data_dir = Path(root_dir) / data_dir_name
    out: list[str] = []
    for eq in eqs:
        if (data_dir / eq).exists():
            out.append(eq)
    return out


def get_benchmark_config(
    equation_id: str,
    *,
    root_dir: str = "feynman",
    family: str = "feynman",
    dimensionless: bool = True,
    target: str = "y",
) -> DatasetConfig:
    meta = _load_benchmark_metadata(root_dir, family, dimensionless)
    row = meta.loc[meta["Filename"] == equation_id]
    if row.empty:
        raise ValueError(f"Unknown {family} equation_id: {equation_id}")

    row0 = row.iloc[0]
    formula = (row0.get("Formula", "") or "").strip() or "Unknown"

    variables: list[str] = []
    if dimensionless:
        for key in ["var1", "var2", "var3", "var4", "var5", "var6"]:
            v = row0.get(key, "")
            if isinstance(v, str) and v.strip():
                variables.append(v.strip())
    else:
        n_vars = None
        try:
            n_vars = int(str(row0.get("# variables", "")).strip() or "")
        except Exception:
            n_vars = None

        name_keys = [f"v{i}_name" for i in range(1, 11)]
        if n_vars is not None:
            name_keys = name_keys[: max(0, min(10, n_vars))]

        for key in name_keys:
            v = row0.get(key, "")
            if isinstance(v, str) and v.strip():
                variables.append(v.strip())

    return DatasetConfig(
        id=equation_id,
        variables=variables,
        target=target,
        formula=formula,
        domain={},
        n_samples=0,
    )


def load_benchmark_df(
    equation_id: str,
    *,
    root_dir: str = "feynman",
    family: str = "feynman",
    dimensionless: bool = True,
    n_samples: int = 400,
    seed: int = 42,
    target: str = "y",
) -> pd.DataFrame:
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")

    _ = get_benchmark_config(
        equation_id,
        root_dir=root_dir,
        family=family,
        dimensionless=dimensionless,
        target=target,
    )

    family_key = str(family).strip().lower()
    data_dir_name = _BENCHMARK_DATA_DIRS[(family_key, bool(dimensionless))]
    file_path = Path(root_dir) / data_dir_name / equation_id
    if not file_path.exists():
        raise FileNotFoundError(f"Missing {family} equation file: {file_path}")

    mat = _read_whitespace_matrix(file_path, n_samples)
    n_cols = mat.shape[1]
    if n_cols == 1:
        x_names: list[str] = []
    else:
        x_names = [f"x{i+1}" for i in range(n_cols - 1)]

    data: dict[str, np.ndarray] = {}
    if x_names:
        for i, name in enumerate(x_names):
            data[name] = mat[:, i]
        data[target] = mat[:, -1]
    else:
        data[target] = mat[:, 0]

    df = pd.DataFrame(data)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def load_benchmark_df_and_pretty_map(
    equation_id: str,
    *,
    root_dir: str = "feynman",
    family: str = "feynman",
    dimensionless: bool = True,
    n_samples: int = 400,
    seed: int = 42,
    target: str = "y",
) -> tuple[pd.DataFrame, dict[str, str]]:
    meta_cfg = get_benchmark_config(
        equation_id,
        root_dir=root_dir,
        family=family,
        dimensionless=dimensionless,
        target=target,
    )
    df = load_benchmark_df(
        equation_id,
        root_dir=root_dir,
        family=family,
        dimensionless=dimensionless,
        n_samples=n_samples,
        seed=seed,
        target=target,
    )

    feature_cols = [c for c in df.columns if c != target]
    pretty_map: dict[str, str] = {}
    for idx, safe_name in enumerate(feature_cols, start=1):
        pretty = None
        if idx - 1 < len(meta_cfg.variables):
            candidate = (meta_cfg.variables[idx - 1] or "").strip()
            if candidate:
                pretty = candidate
        pretty_map[safe_name] = pretty or safe_name

    for safe_name, pretty in list(pretty_map.items()):
        pretty_map[f"{safe_name}²"] = f"({pretty})²"
        pretty_map[f"{safe_name}³"] = f"({pretty})³"

    return df, pretty_map


# Convenience wrappers for the original dimensionless Feynman helpers

def list_feynman_dimensionless_equations(
    *,
    root_dir: str = "feynman",
    require_data_file: bool = True,
) -> list[str]:
    return list_benchmark_equations(
        root_dir=root_dir,
        family="feynman",
        dimensionless=True,
        require_data_file=require_data_file,
    )


def get_feynman_dimensionless_config(
    equation_id: str,
    *,
    root_dir: str = "feynman",
    target: str = "y",
) -> DatasetConfig:
    return get_benchmark_config(
        equation_id,
        root_dir=root_dir,
        family="feynman",
        dimensionless=True,
        target=target,
    )


def load_feynman_dimensionless_df(
    equation_id: str,
    *,
    root_dir: str = "feynman",
    n_samples: int = 400,
    seed: int = 42,
    target: str = "y",
) -> pd.DataFrame:
    return load_benchmark_df(
        equation_id,
        root_dir=root_dir,
        family="feynman",
        dimensionless=True,
        n_samples=n_samples,
        seed=seed,
        target=target,
    )


def load_feynman_dimensionless_df_and_pretty_map(
    equation_id: str,
    *,
    root_dir: str = "feynman",
    n_samples: int = 400,
    seed: int = 42,
    target: str = "y",
) -> tuple[pd.DataFrame, dict[str, str]]:
    return load_benchmark_df_and_pretty_map(
        equation_id,
        root_dir=root_dir,
        family="feynman",
        dimensionless=True,
        n_samples=n_samples,
        seed=seed,
        target=target,
    )
