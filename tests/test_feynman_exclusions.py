"""
Tests for Feynman equation exclusions and filtered benchmarking.
"""

import pytest
from symbolic_discovery.data.feynman_exclusions import (
    get_all_excluded_equations,
    get_strict_exclusions,
    get_all_exclusions,
    get_excluded_ids_by_reason,
    ExclusionReason,
    DEGENERATE_EQUATIONS,
    TRANSCENDENTAL_EQUATIONS,
    SQRT_ONLY_EQUATIONS,
)


class TestExclusionCounts:
    """Test that exclusion counts are accurate."""
    
    def test_degenerate_count(self):
        """Verify number of degenerate equations."""
        assert len(DEGENERATE_EQUATIONS) == 11
    
    def test_transcendental_count(self):
        """Verify number of transcendental equations."""
        assert len(TRANSCENDENTAL_EQUATIONS) == 29
    
    def test_sqrt_only_count(self):
        """Verify number of sqrt-only equations."""
        assert len(SQRT_ONLY_EQUATIONS) == 12
    
    def test_strict_exclusions_count(self):
        """Strict = degenerate + transcendental."""
        strict = get_strict_exclusions()
        assert len(strict) == 40  # 11 + 29
    
    def test_all_exclusions_count(self):
        """All = degenerate + transcendental + sqrt_only."""
        all_exc = get_all_exclusions()
        assert len(all_exc) == 52  # 11 + 29 + 12


class TestExclusionCategories:
    """Test that exclusion categories are properly defined."""
    
    def test_no_overlap_degenerate_transcendental(self):
        """Degenerate and transcendental should not overlap."""
        deg = set(DEGENERATE_EQUATIONS.keys())
        trans = set(TRANSCENDENTAL_EQUATIONS.keys())
        overlap = deg & trans
        assert len(overlap) == 0, f"Overlap found: {overlap}"
    
    def test_no_overlap_degenerate_sqrt(self):
        """Degenerate and sqrt_only should not overlap."""
        deg = set(DEGENERATE_EQUATIONS.keys())
        sqrt = set(SQRT_ONLY_EQUATIONS.keys())
        overlap = deg & sqrt
        assert len(overlap) == 0, f"Overlap found: {overlap}"
    
    def test_no_overlap_transcendental_sqrt(self):
        """Transcendental and sqrt_only should not overlap."""
        trans = set(TRANSCENDENTAL_EQUATIONS.keys())
        sqrt = set(SQRT_ONLY_EQUATIONS.keys())
        overlap = trans & sqrt
        assert len(overlap) == 0, f"Overlap found: {overlap}"
    
    def test_get_by_reason_degenerate(self):
        """Test getting IDs by reason."""
        ids = get_excluded_ids_by_reason(ExclusionReason.DEGENERATE)
        assert "II.13.17" in ids
        assert "I.38.12" in ids
    
    def test_get_by_reason_transcendental(self):
        """Test getting transcendental IDs."""
        ids = get_excluded_ids_by_reason(ExclusionReason.TRANSCENDENTAL)
        assert "I.6.2a" in ids  # Gaussian/exp
        assert "I.30.3" in ids  # sin


class TestSpecificEquations:
    """Test that specific problematic equations are properly categorized."""
    
    @pytest.mark.parametrize("eq_id", [
        "II.13.17",  # 1/(4pi*epsilon0c^2)
        "II.4.23",   # Coulomb constant
        "I.38.12",   # Bohr radius
        "III.7.38",  # Larmor frequency
        "II.34.2a",  # Current
    ])
    def test_known_degenerate(self, eq_id):
        """Known degenerate equations should be excluded."""
        assert eq_id in DEGENERATE_EQUATIONS
        assert eq_id in get_strict_exclusions()
    
    @pytest.mark.parametrize("eq_id,function", [
        ("I.6.2a", "exp"),
        ("I.40.1", "exp"),
        ("I.30.3", "sin"),
        ("I.37.4", "cos"),
        ("I.44.4", "ln"),
        ("I.26.2", "arcsin"),
        ("II.35.21", "tanh"),
    ])
    def test_known_transcendental(self, eq_id, function):
        """Known transcendental equations should be excluded."""
        assert eq_id in TRANSCENDENTAL_EQUATIONS, \
            f"{eq_id} with {function}() should be in TRANSCENDENTAL"
        assert eq_id in get_strict_exclusions()
    
    @pytest.mark.parametrize("eq_id", [
        "I.10.7",    # Lorentz factor
        "I.15.1",    # Relativistic momentum  
        "I.48.2",    # Relativistic energy
        "III.10.19", # B magnitude
    ])
    def test_known_sqrt_only(self, eq_id):
        """Known sqrt-only equations should be in SQRT_ONLY."""
        assert eq_id in SQRT_ONLY_EQUATIONS
        assert eq_id in get_all_exclusions()
        # sqrt_only should NOT be in strict exclusions
        assert eq_id not in get_strict_exclusions()


class TestValidEquations:
    """Test that valid equations are NOT excluded."""
    
    @pytest.mark.parametrize("eq_id", [
        "I.12.1",   # F = mu*Nn (simple product)
        "I.14.4",   # U = k*x**2 (quadratic)
        "I.24.6",   # Energy with omega squared
        "I.39.22",  # Ideal gas
        "II.11.3",  # Driven oscillator
        "II.38.3",  # Hooke's law
    ])
    def test_valid_not_excluded(self, eq_id):
        """Valid polynomial equations should NOT be excluded."""
        assert eq_id not in get_strict_exclusions()
        assert eq_id not in get_all_exclusions()


def filter_benchmark_results(results_df, strict: bool = True):
    """
    Filter benchmark results to exclude expected failures.
    
    Args:
        results_df: DataFrame with 'dataset' column like 'feynman:dimless:I.24.6'
        strict: If True, only exclude degenerate + transcendental.
                If False, also exclude sqrt_only.
    
    Returns:
        Filtered DataFrame
    """
    import pandas as pd
    
    exclusions = get_strict_exclusions() if strict else get_all_exclusions()
    
    def is_excluded(dataset_str):
        # Extract equation ID from dataset string
        # Format: 'feynman:dimless:I.24.6' or 'feynman:units:I.24.6'
        parts = str(dataset_str).split(':')
        if len(parts) >= 3:
            eq_id = parts[-1]
            return eq_id in exclusions
        return False
    
    mask = ~results_df['dataset'].apply(is_excluded)
    return results_df[mask]


def compute_filtered_success_rate(results_df, r2_threshold: float = 0.8, strict: bool = True):
    """
    Compute success rate after filtering out expected failures.
    
    Args:
        results_df: DataFrame with 'dataset', 'method', 'r2' columns
        r2_threshold: R² threshold for success
        strict: Whether to use strict exclusions
    
    Returns:
        Dict[str, float] mapping method to success rate
    """
    filtered = filter_benchmark_results(results_df, strict=strict)
    
    rates = {}
    for method in filtered['method'].unique():
        method_df = filtered[filtered['method'] == method]
        success = (method_df['r2'] > r2_threshold).sum()
        total = len(method_df)
        rates[method] = success / total if total > 0 else 0.0
    
    return rates


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
