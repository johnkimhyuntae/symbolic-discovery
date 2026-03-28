import pytest
from symbolic_discovery._core.bacon3 import BACON3
from symbolic_discovery.data.catalogue import CATALOGUE
from symbolic_discovery.data.synthetic import DatasetGenerator


def is_failure_equation(equation: str) -> bool:
    if not equation:
        return True
    # Keep compatible with both older and newer failure strings.
    return ("No law found" in equation) or ("Failed" in equation) or (equation.strip() == "Error")

def is_term_present(equation: str, expected_term: str) -> bool:
    """
    Checks if the core algebraic term exists in the discovered equation.
    Ignores whitespace and multiplication symbols for robust matching.
    """
    if is_failure_equation(equation):
        return False
        
    eq_clean = equation.replace(" ", "")
    term_clean = expected_term.replace(" ", "")
    
    # Check if the term exists (e.g. "I*R" inside "V = 1.0*(I*R)")
    # We also check the reverse for commutative operations (A*B vs B*A)
    if "*" in term_clean:
        parts = term_clean.split("*")
        if len(parts) == 2:
            rev_term = f"{parts[1]}*{parts[0]}"
            return term_clean in eq_clean or rev_term in eq_clean
            
    return term_clean in eq_clean

# --- Fixtures ---
@pytest.fixture
def generator():
    """Provides a fresh dataset generator for each test."""
    return DatasetGenerator(seed=42)

# --- 1. The Full Baseline Sweep (Clean Data) ---
# This tests Success Criterion 1: Exact recovery on clean benchmarks.
@pytest.mark.parametrize("dataset_id, expected_term", [
    # Synthetic Functions
    ("S-1", "x1+x2"),
    ("S-2", "x1*x2"),       # y = x1 * x2
    pytest.param(
        "S-3",
        "x1/(x2+1)",
        marks=pytest.mark.xfail(reason="Non-homogeneous offset (+1) is typically not recovered by BACON.3's search space."),
    ),
    # S-4 is handled in an explicit expected-failure test (see below)
    
    # Textbook Laws
    ("T-1", "I*R"),         # Ohm's Law (V = IR)
    ("T-2", "k*x"),         # Hooke's Law (F = kx)
    ("T-3", "t**2"),        # Free Fall (s = 0.5gt^2) - Tests Power generation
    # T-4 is handled in an explicit expected-failure test (see below)
    ("T-5", "T**4")         # Stefan-Boltzmann (P = cT^4) - Tests High Power
])
def test_baseline_exactness(generator, dataset_id, expected_term):
    """
    Runs BACON.3 on all catalogue datasets with 0% noise.
    Expectation: High R^2 (>0.999) and correct algebraic form on clean benchmarks.
    Known structural limitations are marked as xfail (kept minimal by design).
    """
    # 1. Generate Clean Data
    train_df, _, _ = generator.generate(dataset_id, noise_level=0.0)
    
    # 2. Run BACON.3
    # We use depth=3 to ensure we can reach terms like (x1^2 + x2^2)
    solver = BACON3(max_depth=3, r2_threshold=0.999, verbose=True)
    equation, diagnostics = solver.discover(train_df, target_col=CATALOGUE[dataset_id].target)
    
    print(f"\n{'='*60}")
    print(f"[Clean] {dataset_id}")
    print(f"Discovered: {equation}")
    print(f"R²: {diagnostics.get('R-squared', 0.0):.6f}")
    print(f"{'='*60}")
    
    # 3. Validation
    # Basic success checks
    assert equation is not None, f"BACON failed to return an equation for {dataset_id}"
    assert not is_failure_equation(equation), f"BACON returned failure status for {dataset_id}: {equation}"
    
    # R-squared check (Strict for clean data)
    r2 = diagnostics.get("R-squared", 0.0)
    assert r2 > 0.999, f"R2 {r2:.6f} below threshold for clean {dataset_id}"
    
    # Algebraic Form Check
    assert is_term_present(equation, expected_term), \
        f"Expected term '{expected_term}' not found in '{equation}'"


@pytest.mark.parametrize("dataset_id", ["S-4", "T-4"])
def test_bacon3_expected_failures_clean(generator, dataset_id):
    """Known BACON.3 limitations on clean data should return a clear failure message."""
    train_df, _, _ = generator.generate(dataset_id, noise_level=0.0)
    solver = BACON3(max_depth=3, r2_threshold=0.999, verbose=True)
    equation, _ = solver.discover(train_df, target_col=CATALOGUE[dataset_id].target)

    assert equation is not None
    assert is_failure_equation(equation)

# --- 2. The Stress Test (Noisy Data) ---
@pytest.mark.parametrize("dataset_id", ["T-1", "T-3", "S-2"])
def test_noise_sensitivity(generator, dataset_id):
    """
    Runs BACON.3 with 5% noise.
    Expectation: Likely failure or degraded R^2, justifying BACON.7.
    """
    # 1. Generate Noisy Data (0.05 is significant for BACON)
    train_df, _, _ = generator.generate(dataset_id, noise_level=0.05)
    
    # 2. Run BACON.3 (Slightly relaxed threshold, but still expecting trouble)
    solver = BACON3(max_depth=3, r2_threshold=0.95, verbose=True)
    equation, diagnostics = solver.discover(train_df, target_col=CATALOGUE[dataset_id].target)
    
    print(f"\n{'='*60}")
    print(f"[Noisy 5%] {dataset_id}")
    print(f"Discovered: {equation}")
    if equation and (not is_failure_equation(equation)):
        r2 = diagnostics.get("R-squared", 0.0)
        print(f"Survived with R²: {r2:.4f}")
    else:
        print("System FAILED (Expected for baseline BACON)")
    print(f"{'='*60}")

    # Normal assertion: should not crash; may succeed or fail depending on dataset/noise.
    assert equation is not None
    