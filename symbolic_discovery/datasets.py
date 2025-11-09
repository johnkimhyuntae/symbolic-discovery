import numpy as np
import pandas as pd

def get_synthetic_product_data(n_points_per_dim: int = 20, noise: float = 0.0) -> pd.DataFrame:
    """
    Generates data for the synthetic law y = x1 * x2. (S-2)
    Uses a meshgrid to ensure x1 and x2 are not correlated.
    """
    x1_vals = np.linspace(1, 10, n_points_per_dim)
    x2_vals = np.linspace(1, 5, n_points_per_dim)
    
    x1, x2 = np.meshgrid(x1_vals, x2_vals)
    
    x1 = x1.flatten()
    x2 = x2.flatten()

    y = (x1 * x2) + np.random.normal(0, noise, x1.shape)
    
    return pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})

def get_ohms_law_data(n_points_per_dim: int = 20, noise: float = 0.0) -> pd.DataFrame:
    """
    Generates data for Ohm's Law (V = I * R). (T-1)
    Uses a meshgrid to ensure I and R are not correlated.
    """
    i_vals = np.linspace(0.5, 5.0, n_points_per_dim)
    r_vals = np.linspace(1.0, 10.0, n_points_per_dim)
    
    I, R = np.meshgrid(i_vals, r_vals)
    
    I = I.flatten()
    R = R.flatten()

    V = (I * R) + np.random.normal(0, noise, I.shape)
    
    return pd.DataFrame({'I': I, 'R': R, 'V': V})

def get_synthetic_ratio_data(n_points_per_dim: int = 20, noise: float = 0.0) -> pd.DataFrame:
    """
    Generates data for the synthetic law y = x1 / x2. (S-3)
    Uses a meshgrid to ensure x1 and x2 are not correlated.
    """
    # 20x20 grid = 400 points, matching N=400 in dossier
    x1_vals = np.linspace(10, 100, n_points_per_dim)
    x2_vals = np.linspace(1, 10, n_points_per_dim)
    
    x1, x2 = np.meshgrid(x1_vals, x2_vals)
    
    x1 = x1.flatten()
    x2 = x2.flatten()
    
    y = (x1 / x2) + np.random.normal(0, noise, x1.shape)
    
    return pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})