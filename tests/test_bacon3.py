import pytest
from symbolic_discovery.bacon3 import BACON3
from symbolic_discovery.datasets import DatasetGenerator, CATALOGUE

def is_term_present(equation: str, expected_term: str) -> bool:
    """
    Checks if the core algebraic term exists in the discovered equation.
    Ignores whitespace and multiplication symbols for robust matching.
    """
    if not equation or "Failed" in equation:
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
    ("S-1", "x1+x2"),       # y = x1 + x2
    ("S-2", "x1*x2"),       # y = x1 * x2
    ("S-3", "x1/(x2+1)"),   # y = x1 / (x2 + 1) (Ratio check)
    ("S-4", "x1**2+x2**2"), # y = x1^2 + x2^2 (Sum of squares)
    
    # Textbook Laws
    ("T-1", "I*R"),         # Ohm's Law (V = IR)
    ("T-2", "k*x"),         # Hooke's Law (F = kx)
    ("T-3", "t**2"),        # Free Fall (s = 0.5gt^2) - Tests Power generation
    ("T-4", "P*V"),         # Ideal Gas (T = PV/nR) - Checks for primary interaction
    ("T-5", "T**4")         # Stefan-Boltzmann (P = cT^4) - Tests High Power
])
def test_baseline_exactness(generator, dataset_id, expected_term):
    """
    Runs BACON.3 on all catalogue datasets with 0% noise.
    Expectation: High R^2 (>0.999) and correct algebraic form.
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
    assert "Failed" not in equation, f"BACON returned failure status for {dataset_id}"
    
    # R-squared check (Strict for clean data)
    r2 = diagnostics.get("R-squared", 0.0)
    assert r2 > 0.999, f"R2 {r2:.6f} below threshold for clean {dataset_id}"
    
    # Algebraic Form Check
    # For T-4, finding "P*V" or "P/n" proves it found the first layer of the nested law
    if dataset_id == "T-4":
        # Ideal gas is hard; check if it found at least one correct pair interaction
        assert ("P*V" in equation) or ("V*P" in equation) or \
               ("P/n" in equation) or ("V/n" in equation)
    else:
        assert is_term_present(equation, expected_term), \
            f"Expected term '{expected_term}' not found in '{equation}'"

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
    if equation and "Failed" not in equation:
        r2 = diagnostics.get("R-squared", 0.0)
        print(f"Survived with R²: {r2:.4f}")
    else:
        print("System FAILED (Expected for baseline BACON)")
    print(f"{'='*60}")
    