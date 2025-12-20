import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class DatasetConfig:
    id: str
    variables: list[str]
    target: str
    formula: str
    domain: dict[str, tuple[float, float]]
    n_samples: int = 400

CATALOGUE = {
    # Synthetic Functions
    "S-1": DatasetConfig("S-1", ["x1", "x2"], "y", "x1 + x2", {"x1": (-5, 5), "x2": (-5, 5)}),
    "S-2": DatasetConfig("S-2", ["x1", "x2"], "y", "x1 * x2", {"x1": (1, 5), "x2": (1, 5)}),
    "S-3": DatasetConfig("S-3", ["x1"], "y", "2 * x1 + 1", {"x1": (-5, 5)}),
    "S-4": DatasetConfig("S-4", ["x1", "x2"], "y", "x1**2 + x2**2", {"x1": (-3, 3), "x2": (-3, 3)}),
    
    # Textbook Laws
    "T-1": DatasetConfig("T-1", ["I", "R"], "V", "I * R", {"I": (0, 2), "R": (1, 10)}),
    "T-2": DatasetConfig("T-2", ["k", "x"], "F", "k * x", {"k": (1, 10), "x": (-1, 1)}),
    "T-3": DatasetConfig("T-3", ["t"], "s", "0.5 * 9.81 * t**2", {"t": (0, 2)}),
    "T-4": DatasetConfig("T-4", ["P", "V", "n"], "T", "(P * V) / (n * 8.314)", {"P": (1, 5), "V": (10, 30), "n": (1, 2)}),
    "T-5": DatasetConfig("T-5", ["T"], "P", "5.67e-8 * T**4", {"T": (100, 500)}),
}

class DatasetGenerator:
    """
    Handles data generation, noise injection, and splitting.
    Ensures reproducibility via fixed seeds as per Methods Dossier Section 0.4.2.
    """
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    def generate(self, config_id: str, noise_level: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Creates the data required for a single experiment run.
        
        Args:
            config_id: Key from the CATALOGUE (e.g. "T-1").
            noise_level: Sigma as a fraction of dynamic range (e.g. 0.01).
            
        Returns:
            (train_df, test_df, extra_df)
        """
        if config_id not in CATALOGUE:
            raise ValueError(f"Unknown dataset ID: {config_id}")
            
        config = CATALOGUE[config_id]
        
        # 1. Sample Inputs for Training/Testing (Interpolation)
        data = {}
        for var in config.variables:
            low, high = config.domain[var]
            data[var] = self.rng.uniform(low, high, config.n_samples)
        
        df = pd.DataFrame(data)
        
        # 2. Compute Ground Truth Target
        df[config.target] = df.eval(config.formula)
        
        # 3. Inject Noise
        if noise_level > 0.0:
            target_range = df[config.target].max() - df[config.target].min()
            # If constant (range=0), assume scale of 1.0 to avoid null noise
            scale = target_range if target_range > 1e-9 else 1.0
            
            noise = self.rng.normal(loc=0.0, scale=noise_level * scale, size=len(df))
            df[config.target] += noise

        # 4. Train/Test Split (75/25) 
        # We use the fixed integer seed here for the pandas sampler to ensure
        # the split is identical even if the generator state shifts slightly.
        train_df = df.sample(frac=0.75, random_state=self.seed)
        test_df = df.drop(train_df.index)
        
        # 5. Generate Extrapolation Slab 
        extra_df = self._generate_extrapolation_slab(config)
        
        return train_df, test_df, extra_df

    def _generate_extrapolation_slab(self, config: DatasetConfig) -> pd.DataFrame:
        """
        Creates data in an 'out-of-range slab' for extrapolation checks.
        Per dossier: e.g. train x in [-3,3], test-extra x in [3,4].
        We shift the domain of all variables by +25% of their span.
        """
        data = {}
        n_extra = config.n_samples // 4
        
        for var in config.variables:
            low, high = config.domain[var]
            span = high - low
            # Shift: Start at 'high', end at 'high + 0.25 * span'
            new_low = high
            new_high = high + (0.25 * span)
            
            data[var] = self.rng.uniform(new_low, new_high, n_extra)
        
        df = pd.DataFrame(data)
        
        # Calculate clean ground truth for the extrapolation set
        df[config.target] = df.eval(config.formula)
        
        return df