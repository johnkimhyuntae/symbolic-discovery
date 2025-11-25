import numpy as np
import pandas as pd
from dataclasses import dataclass
from itertools import combinations

from sympy import Symbol, Expr, symbols
from scipy.stats import spearmanr, iqr

@dataclass
class Variable:
    """
    Represents a variable using a SymPy expression.
    'symbol' holds the symbolic form (e.g., x1, x1*x2).
    'values' holds the numerical data array.
    """
    symbol: Expr    
    # values may come from pandas or numpy arrays; accept either
    values: np.ndarray

class BACON3:
    def __init__(self, 
                 constancy_threshold: float = 0.05,
                 monotonicity_threshold: float = 0.98,
                 max_depth: int = 3):
        """
        Initialises the BACON.3 solver.
        """
        self.constancy_threshold = constancy_threshold
        self.monotonicity_threshold = monotonicity_threshold
        self.max_depth = max_depth
        
        # Internal states
        self.logs: list[str] = []
        self.variable_pool: list[Variable] = []
        self.target_var: Variable | None = None
        self.final_equation: str | None = None
        
        # SymPy states
        self.symbol_map: dict[str, Symbol] = {} 
        self.known_symbols: set[str] = set()

    def _log(self, message: str):
        """
        Logs a decision or step, creating a human-readable trace.
        """
        self.logs.append(message)
        print(message) 

    def _check_constancy(self, v: Variable) -> bool:
        """
        Probes the "Near constancy" cue.
        Per the dossier, this uses median-based robustness.
        """
        median = np.median(v.values)
        iqr_value = iqr(v.values)
        cov = iqr_value / (np.abs(median) + 1e-8)  # Avoid division by zero
        return cov < self.constancy_threshold

    def _check_monotonicity(self, v: Variable) -> bool:
        """
        Probes the "Monotonicity" cue against the target variable.
        Per the dossier, this should be robust (e.g., Spearman's rank).
        """
        if self.target_var is None:
            return False
        corr, _ = spearmanr(v.values, self.target_var.values)
        # spearmanr may return nan for constant arrays
        try:
            corr_arr = np.asarray(corr)
            if corr_arr.size == 1:
                corr_val = float(corr_arr.item())
            else:
                corr_val = float(np.ravel(corr_arr)[0])
        except Exception:
            return False

        if np.isnan(corr_val):
            return False

        return np.abs(corr_val) > self.monotonicity_threshold

    def _generate_composites(self, variables: list[Variable]) -> list[Variable]:
        """
        Generates invariants from ratios and products
        using SymPy to build expressions.
        """
        new_composites: list[Variable] = []

        for v1, v2 in combinations(variables, 2):
            # Product
            prod_symbol = v1.symbol * v2.symbol # type: ignore
            if str(prod_symbol) not in self.known_symbols:
                prod_values = v1.values * v2.values
                new_composites.append(Variable(symbol=prod_symbol, values=prod_values))
                self.known_symbols.add(str(prod_symbol))
            
            # Ratio v1 / v2
            div1_symbol = v1.symbol / v2.symbol # type: ignore
            if str(div1_symbol) not in self.known_symbols:
                div1_values = v1.values / (v2.values + 1e-8)  # Avoid division by zero
                new_composites.append(Variable(symbol=div1_symbol, values=div1_values))
                self.known_symbols.add(str(div1_symbol))

            # Ratio v2 / v1
            div2_symbol = v2.symbol / v1.symbol # type: ignore
            if str(div2_symbol) not in self.known_symbols:
                div2_values = v2.values / (v1.values + 1e-8)  # Avoid division by zero
                new_composites.append(Variable(symbol=div2_symbol, values=div2_values))
                self.known_symbols.add(str(div2_symbol))
        
        return new_composites

    def _initialise_variables(self, data: pd.DataFrame, target_col: str) -> None:
        """
        Populates the initial variable_pool from the input DataFrame
        using SymPy Symbol objects.
        """
        # clear previous state
        self.variable_pool = []
        self.symbol_map = {}
        self.known_symbols = set()
        self.target_var = None

        for col in data.columns:
            sym = symbols(col)
            self.symbol_map[col] = sym
            col_values = np.asarray(data[col].values)
            var = Variable(sym, col_values)
            self.variable_pool.append(var)
            self.known_symbols.add(str(sym))
            if col == target_col:
                self.target_var = var

    def _find_closing_relation(self) -> tuple[str, dict[str, float]] | None:
        """
        Tries to find a simple linear fit (y = a*x + b) between
        the target and all other promoted variables.
        """
        if self.target_var is None:
            return None
        
        best_fit_symbol = None
        best_r2 = -np.inf
        best_coeffs = (0.0, 0.0)
        best_diagnostics = {}

        # make sure numeric arrays are numpy arrays for consistent numeric ops
        y_true = np.asarray(self.target_var.values)
        y_mean = float(np.mean(y_true))

        for v in self.variable_pool:
            if v.symbol == self.target_var.symbol:
                continue
            
            x_values = np.asarray(v.values)
            a = 0.0
            b = 0.0
            try:
                # 1. Fit a Line (y = ax + b)
                coeffs = np.polyfit(x_values, y_true, 1)
                a, b = coeffs
                
                # 2. Get Predictions
                y_pred = (a * x_values) + b
                
                # 3. Calculate Residual Diagnostics
                ss_total = np.sum((y_true - y_mean) ** 2)
                ss_residual = np.sum((y_true - y_pred) ** 2)
                
                if ss_total == 0:
                    r2 = 1.0 if ss_residual == 0 else 0.0
                else:
                    r2 = 1 - (ss_residual / ss_total)
                
                mse = np.mean((y_true - y_pred)**2)
                mae = np.mean(np.abs(y_true - y_pred))
                    
            except (np.linalg.LinAlgError, ValueError):
                # Fit failed
                r2, mse, mae = -np.inf, np.inf, np.inf
            
            self._log(f"  Testing closing relation: {str(self.target_var.symbol)} ~ {str(v.symbol)}. R-squared: {r2:.6f}")

            # 4. Check if this is the best, near-perfect fit
            if r2 > 0.999 and r2 > best_r2: 
                best_r2 = r2
                best_fit_symbol = v.symbol
                best_coeffs = (a, b)
                best_diagnostics = {
                    "R-squared": r2,
                    "MSE": mse,
                    "MAE": mae
                }
        
        # 5. Return the equation string and the diagnostics dict
        if best_fit_symbol:
            a, b = best_coeffs
            equation_string = f"{str(self.target_var.symbol)} = {a:.4f} * ({str(best_fit_symbol)}) + {b:.4f}"
            return equation_string, best_diagnostics
        
        # No fit found
        return None

    def discover(self, data: pd.DataFrame, target_col: str, seed: int = 42) -> tuple[str | None, dict[str, float]]:
        """
        Runs the main BACON.3 "deterministic passes".
        
        This is the main entry point that executes the layered
        discovery loop and attempts to find a closing relation.
        
        Args:
            data: A pandas DataFrame with input variables.
            target_col: The name of the column to use as the target variable.
            seed: A fixed seed for reproducibility and tie-breaking.
            
        Returns:
            A tuple containing:
                - The discovered symbolic law as a string (or a failure message).
                - A dictionary of residual diagnostics (e.g., R-squared, MSE, MAE).
        """
        
        np.random.seed(seed)
        self.logs = []
        
        self._initialise_variables(data, target_col)
        
        if self.target_var is None:
            self._log("Error: Target variable not found.")
            return None, {}

        self._log(f"\nStarting discovery. Target: '{str(self.target_var.symbol)}'. Seed: {seed}")

        # main loop
        for i in range(self.max_depth):
            self._log(f"\n--- Layer {i+1} ---")
            
            vars_to_combine = self.variable_pool 
            
            new_composites = self._generate_composites(vars_to_combine)
            if not new_composites:
                self._log("No new composites generated. Stopping.")
                break
            
            promoted_this_layer: list[Variable] = []
            
            for composite in new_composites:
                
                if self._check_constancy(composite):
                    self._log(f"  -> Cue Hit (Constancy): {str(composite.symbol)} is near-constant. Promoting.")
                    promoted_this_layer.append(composite)
                    continue
                
                if self._check_monotonicity(composite):
                    self._log(f"  -> Cue Hit (Monotonicity): {str(composite.symbol)} is monotonic with {self.target_var.symbol}. Promoting.")
                    promoted_this_layer.append(composite)

            if not promoted_this_layer:
                self._log("No candidates met cues. Stopping.")
                break 
            
            self.variable_pool.extend(promoted_this_layer)
            self._log(f"Layer {i+1} complete. Promoted: {[str(v.symbol) for v in promoted_this_layer]}")
        
        self._log("\n--- Finding Closing Relation ---")
        
        # This will hold the {R2, MSE, MAE} dict
        self.final_diagnostics: dict[str, float] = {} 
        
        # Call the function
        fit_results = self._find_closing_relation()
        
        if fit_results:
            # Unpack the tuple here
            self.final_equation, self.final_diagnostics = fit_results
            self._log(f"Success! Found relation: {self.final_equation}")
            self._log(f"Diagnostics: {self.final_diagnostics}")
        else:
            self.final_equation = "Failed to find a simple closing relation."
            self._log(self.final_equation)
        
        # Return both the equation and its diagnostics
        return self.final_equation, self.final_diagnostics