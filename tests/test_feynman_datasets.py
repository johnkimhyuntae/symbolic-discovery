from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
import pytest

from symbolic_discovery.algorithms.bacon3 import BACON3
from symbolic_discovery.algorithms.bacon7 import BACON7


ROOT = Path(__file__).resolve().parents[1] / "feynman"
META = ROOT / "FeynmanEquations.csv"
DATA_DIR = ROOT / "Feynman_without_units"

HAS_FEYNMAN = META.exists() and DATA_DIR.exists()
EQUATION_IDS = sorted([p.name for p in DATA_DIR.iterdir() if p.is_file()]) if HAS_FEYNMAN else []

# A small curated set that both solvers should reliably solve (in this checkout).
MUST_SOLVE_EQUATIONS = [
    "I.12.1",
    "I.12.2",
    "I.39.22",
    "II.11.20",
    "II.34.11",
    "II.34.29b",
    "II.38.3",
    "III.12.43",
]


def _load_formula_map() -> dict[str, str]:
    if not META.exists():
        return {}
    df = pd.read_csv(META)
    if "Filename" not in df.columns or "Formula" not in df.columns:
        return {}
    out: dict[str, str] = {}
    for _, row in df[["Filename", "Formula"]].iterrows():
        key = str(row["Filename"]).strip()
        val = str(row["Formula"]).strip()
        if key and key != "nan":
            out[key] = val
    return out


FORMULA_BY_ID = _load_formula_map() if HAS_FEYNMAN else {}


def _load_equation_df(eq_id: str, *, n_samples: int, seed: int) -> pd.DataFrame:
    # Note: these files are whitespace-delimited and do not include a header row.
    # Convention: last column is the target.
    path = DATA_DIR / eq_id
    df = pd.read_csv(path, sep=r"\s+", header=None, nrows=n_samples)
    if df.shape[1] < 1:
        raise ValueError(f"Expected at least 1 column in {path}, got {df.shape[1]}")

    # Make internal column names sympy-safe for BACON.
    if df.shape[1] == 1:
        # Constant-only dataset (no independent variables). Treat as a valid load.
        df.columns = ["y"]
    else:
        feature_cols = [f"x{i}" for i in range(1, df.shape[1])]
        df.columns = feature_cols + ["y"]

    # Shuffle deterministically so we don't always take a potentially-structured prefix.
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def _expected_failure_reason(formula: str) -> str | None:
    if not formula or formula.lower() == "nan":
        return "No metadata formula available"

    f = formula.strip()

    # Anything outside a simple algebraic/power-law family is an expected limitation
    # for our current BACON search spaces.
    if re.search(r"\b[A-Za-z_][A-Za-z0-9_]*\s*\(", f):
        return "Contains function call (e.g., sin/cos/exp/log/tanh/...)"

    # Additive/subtractive structure is generally outside the multiplicative invariant search.
    if "+" in f:
        return "Contains additive structure (+)"
    if re.search(r"[A-Za-z0-9\)]\s*-\s*[A-Za-z0-9\(]", f):
        return "Contains subtractive structure (-)"

    # Non-integer powers are not in our current expected-robust set.
    # Accept exponents that are "effectively" small integers (including negatives), e.g. **(-2.0).
    for m in re.finditer(r"\*\*\s*\(?\s*([-+]?[0-9]*\.?[0-9]+)", f):
        try:
            exp_val = float(m.group(1))
        except ValueError:
            continue
        if abs(exp_val - round(exp_val)) > 1e-9:
            return "Contains non-integer power"

    return None


@pytest.mark.parametrize("eq_id", EQUATION_IDS)
def test_feynman_all_equations_loadable(eq_id: str):
    if not HAS_FEYNMAN:
        pytest.skip("Feynman datasets not present in this checkout")

    df = _load_equation_df(eq_id, n_samples=50, seed=123)
    assert len(df) == 50
    assert "y" in df.columns
    assert df.isna().sum().sum() == 0


@pytest.mark.parametrize("eq_id", MUST_SOLVE_EQUATIONS)
def test_feynman_must_solve_equations_bacon3_and_bacon7(eq_id: str):
    if not HAS_FEYNMAN:
        pytest.skip("Feynman datasets not present in this checkout")
    if eq_id not in EQUATION_IDS:
        pytest.skip(f"Equation {eq_id} not present in local data files")

    train_df = _load_equation_df(eq_id, n_samples=400, seed=42)
    if len(train_df.columns) < 2:
        pytest.skip(f"Equation {eq_id} is constant-only in this checkout")

    # BACON.3
    eq3, d3 = BACON3(max_depth=3, r2_threshold=0.98, verbose=False).discover(train_df, target_col="y")
    assert eq3 is not None
    assert "Failed" not in eq3
    assert float((d3 or {}).get("R-squared", 0.0)) >= 0.98

    # BACON.7
    eq7, d7 = BACON7(max_depth=4, r2_threshold=0.98, verbose=False).discover(train_df, target_col="y")
    assert eq7 is not None
    assert "No law found" not in eq7
    assert float((d7 or {}).get("R-squared", 0.0)) >= 0.98


@pytest.mark.parametrize("eq_id", EQUATION_IDS)
def test_feynman_bacon7_solves_algebraic_equations(eq_id: str):
    if not HAS_FEYNMAN:
        pytest.skip("Feynman datasets not present in this checkout")

    formula = FORMULA_BY_ID.get(eq_id, "")
    reason = _expected_failure_reason(formula)
    if reason is not None:
        pytest.xfail(f"{eq_id}: {reason}")

    train_df = _load_equation_df(eq_id, n_samples=200, seed=42)
    if len(train_df.columns) < 2:
        pytest.xfail(f"{eq_id}: constant-only dataset (no independent variables)")

    solver = BACON7(max_depth=4, r2_threshold=0.98, verbose=False)
    equation, diagnostics = solver.discover(train_df, target_col="y")

    assert equation is not None
    assert "No law found" not in equation
    r2 = float(diagnostics.get("R-squared", 0.0))
    assert r2 >= 0.98


@pytest.mark.parametrize("eq_id", EQUATION_IDS)
def test_feynman_bacon3_runs_on_algebraic_equations(eq_id: str):
    if not HAS_FEYNMAN:
        pytest.skip("Feynman datasets not present in this checkout")

    formula = FORMULA_BY_ID.get(eq_id, "")
    reason = _expected_failure_reason(formula)
    if reason is not None:
        pytest.xfail(f"{eq_id}: {reason}")

    train_df = _load_equation_df(eq_id, n_samples=200, seed=42)
    if len(train_df.columns) < 2:
        pytest.xfail(f"{eq_id}: constant-only dataset (no independent variables)")

    # BACON.3 is weaker than BACON.7 on many equations
    # we assert only that it runs and produces a well-formed (non-crashing) result.
    equation, diagnostics = BACON3(max_depth=3, r2_threshold=0.95, verbose=False).discover(train_df, target_col="y")
    assert equation is not None
    assert isinstance(diagnostics, dict)
