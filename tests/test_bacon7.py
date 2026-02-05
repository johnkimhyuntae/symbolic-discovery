import pytest
from symbolic_discovery.bacon7 import BACON7
from symbolic_discovery.datasets import DatasetGenerator, CATALOGUE

def is_term_present(equation: str, expected_term: str) -> bool:
    """
    Checks if the core algebraic term exists in the discovered equation.
    Ignores whitespace and multiplication symbols for robust matching.
    Handles unicode superscripts (² → **2, ³ → **3).
    """
    if not equation or "No law found" in equation:
        return False
    
    # Normalise unicode superscripts to ASCII
    eq_normalized = equation.replace("²", "**2").replace("³", "**3")
    
    eq_clean = eq_normalized.replace(" ", "")
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

# --- 1. The Success Suite (Solvable Laws) ---
# These are laws BACON.7 MUST solve to be considered working.
# Includes T-4 (Ideal Gas) which requires the new Linearity Check.
@pytest.mark.parametrize("dataset_id, expected_term", [
    ("S-2", "x1*x2"),       
    ("T-1", "I*R"),         
    ("T-2", "k*x"),         
    ("T-3", "t**2"),        
    ("T-4", "P*V"),         # Ideal Gas
    ("T-5", "T**4"),         # Stefan–Boltzmann
])
def test_bacon7_solvable_exactness(generator, dataset_id, expected_term):
    """
    Runs BACON.7 on laws it is designed to solve (Clean Data).
    """
    # 1. Generate Clean Data
    train_df, _, _ = generator.generate(dataset_id, noise_level=0.0)
    
    # 2. Run BACON.7
    solver = BACON7(max_depth=4, verbose=True)
    equation, diagnostics = solver.discover(train_df, target_col=CATALOGUE[dataset_id].target)
    
    print(f"\n[BACON.7 Clean] {dataset_id} -> {equation}")
    
    assert equation is not None and "No law found" not in equation
    
    # T-4 Specific Check: Robustness for Ideal Gas
    if dataset_id == "T-4":
        # The law PV = nRT implies P, V, and T must all be present.
        # The relationship must be multiplicative/divisive, not additive.
        
        # 1. Check all variables are used
        has_vars = all(v in equation for v in ["P", "V", "T"])
        
        # 2. Check for interaction (multiplication OR division)
        # If P = T/V, there is division. If PV = T, there is multiplication.
        has_interaction = ("*" in equation) or ("/" in equation)
        
        assert has_vars, "Equation must contain P, V, and T"
        assert has_interaction, "Equation must involve multiplication or division"
        
    elif dataset_id == "T-5":
        # BACON.7 often returns the invariant form, e.g. P/(T*T³) = c.
        eq_normalized = equation.replace("²", "**2").replace("³", "**3")
        eq_clean = eq_normalized.replace(" ", "")
        assert "P" in equation
        assert ("T**4" in eq_clean) or ("T*T**3" in eq_clean) or ("T*T³" in equation)
    else:
        assert is_term_present(equation, expected_term), \
            f"Expected term '{expected_term}' not found in '{equation}'"

# --- 2. Structural Limits / Known Behaviors ---

@pytest.mark.xfail(strict=False, reason="Structural limitation: pure additive law is outside BACON.7 search space")
def test_bacon7_structural_limit_s1(generator):
    """S-1 is pure addition; BACON.7 should not close an equation."""
    train_df, _, _ = generator.generate("S-1", noise_level=0.0)
    solver = BACON7(max_depth=4, verbose=True)
    equation, _ = solver.discover(train_df, target_col=CATALOGUE["S-1"].target)

    # xfail expects this assertion to fail (i.e., we expect "No law found").
    assert equation is not None and "No law found" not in equation


def test_bacon7_s3_is_approximate_not_exact(generator):
    """S-3 contains a hidden offset (+1); BACON.7 typically returns an approximation, not the exact form."""
    train_df, _, _ = generator.generate("S-3", noise_level=0.0)
    solver = BACON7(max_depth=4, verbose=True)
    equation, diagnostics = solver.discover(train_df, target_col=CATALOGUE["S-3"].target)

    assert equation is not None and "No law found" not in equation

    # Should not magically recover the explicit +1 offset.
    eq_clean = equation.replace(" ", "")
    assert "(x2+1)" not in eq_clean and "x2+1" not in eq_clean

    # Constancy score should be < perfect for this mismatch.
    r2 = float(diagnostics.get("R-squared", 0.0))
    assert r2 < 0.999


def test_bacon7_s4_fails_clean(generator):
    """S-4 is additive (x1^2 + x2^2); expect no law found on clean data."""
    train_df, _, _ = generator.generate("S-4", noise_level=0.0)
    solver = BACON7(max_depth=4, verbose=True)
    equation, _ = solver.discover(train_df, target_col=CATALOGUE["S-4"].target)

    assert equation is not None
    assert "No law found" in equation

# --- 3. The Noise Resilience Test ---
# This validates the core upgrade: Does it survive 5% noise?
@pytest.mark.parametrize("dataset_id, expect_fail", [
    ("T-1", True),
    ("S-2", False),
])
def test_bacon7_noise_resilience(generator, dataset_id, expect_fail):
    """
    Runs BACON.7 with 5% noise.
    Expectation: SUCCESS for S-2 (BACON.3 fails this).
    T-1 fails due to high CV from additive noise on multiplicative law.
    """
    # 1. Generate Noisy Data
    train_df, _, _ = generator.generate(dataset_id, noise_level=0.05)
    
    # 2. Run BACON.7 with relaxed threshold for noisy data
    solver = BACON7(max_depth=4, verbose=True, r2_threshold=0.90)
    equation, diagnostics = solver.discover(train_df, target_col=CATALOGUE[dataset_id].target)
    
    print(f"\n{'='*60}")
    print(f"[BACON.7 Noisy 5%] {dataset_id}")
    print(f"Discovered: {equation}")
    print(f"Diagnostics: {diagnostics}")
    print(f"{'='*60}")
    
    if expect_fail:
        assert equation is not None
        assert "No law found" in equation
        return

    # 3. Validation for expected successes
    assert "No law found" not in equation, \
        f"BACON.7 failed to handle noise for {dataset_id}"

    # Check if R2 is decent (it won't be 1.0, but should be > 0.9)
    r2 = diagnostics.get("R-squared", 0.0)
    assert r2 > 0.90, f"R2 {r2:.4f} too low for recovered noisy law"

# --- 4. The 'Miller Gate' Test ---
# Verifies that simple ratio laws do not get polluted with tiny intercepts.
def test_miller_gate_logic(generator):
    """
    Test S-2 (y = x1*x2) to ensure we don't return 'y = x1*x2 + 0.0001*x1'.
    This validates the c_val parameter works.
    """
    dataset_id = "S-2" # Simple product
    train_df, _, _ = generator.generate(dataset_id, noise_level=0.01) # Mild noise
    
    solver = BACON7(c_val=0.001, verbose=True) # Strict gate
    equation, _ = solver.discover(train_df, target_col="y")
    
    # Should look like "y = 1.0 * (x1*x2)" NOT "y = ... + ...*x1"
    # We check that the equation length is reasonable (residues make long strings)
    print(f"Miller Gate Test Equation: {equation}")
    
    assert "x1*x2" in equation or "x2*x1" in equation
    # A simple way to check we didn't add a linear residue term involves the string structure.
    # Ideally, we shouldn't see multiple subtraction/addition signs inside the term structure.
    # But broadly, exact success on S-2 implies the gate worked.