import pytest
from symbolic_discovery.bacon3 import BACON3
from symbolic_discovery import datasets

def test_discover_synthetic_product():
    """
    Tests discovery of y = x1 * x2.
    This checks Success Criterion 1 of Part II Project Proposal.
    """
    # 1. Get clean, no-noise data
    data = datasets.get_synthetic_product_data(noise=0.0)
    
    # 2. Run the discovery
    bacon = BACON3(max_depth=2)
    equation, diagnostics = bacon.discover(data, target_col='y')
    
    # 3. Check the results
    assert equation is not None
    assert "Failed" not in equation
    
    # Check that the correct symbolic relation was found
    # (SymPy might order it as x1*x2 or x2*x1)
    assert "(x1*x2)" in equation or "(x2*x1)" in equation
    
    # Check the "residual diagnostics" for a perfect fit
    assert diagnostics["R-squared"] > 0.999

def test_discover_ohms_law():
    """
    Tests discovery of V = I * R.
    This checks Success Criterion 1 of Part II Project Proposal.
    """
    # 1. Get clean, no-noise data
    data = datasets.get_ohms_law_data(noise=0.0)
    
    # 2. Run the discovery
    bacon = BACON3(max_depth=2)
    equation, diagnostics = bacon.discover(data, target_col='V')
    
    # 3. Check the results
    assert equation is not None
    assert "Failed" not in equation
    
    # Check for the correct symbolic relation
    assert "(I*R)" in equation or "(R*I)" in equation
    
    # Check diagnostics
    assert diagnostics["R-squared"] > 0.999
    assert diagnostics["MSE"] < 1e-10

def test_discover_synthetic_ratio():
    """
    Tests discovery of y = x1 / x2.
    This checks Success Criterion 1 of Part II Project Proposal.
    """
    # 1. Get clean, no-noise data
    data = datasets.get_synthetic_ratio_data(noise=0.0)
    
    # 2. Run the discovery
    bacon = BACON3(max_depth=2)
    equation, diagnostics = bacon.discover(data, target_col='y')
    
    # 3. Check the results
    assert equation is not None
    assert "Failed" not in equation
    
    # Check for the correct symbolic relation
    assert "(x1/x2)" in equation
    
    # Check diagnostics
    assert diagnostics["R-squared"] > 0.999