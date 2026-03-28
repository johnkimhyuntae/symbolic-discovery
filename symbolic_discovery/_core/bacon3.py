import numpy as np
import pandas as pd
from dataclasses import dataclass
from itertools import combinations

from sympy import Symbol, Expr, symbols
from scipy.stats import spearmanr, iqr

from ..utils import fit_linear_model

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

    def __post_init__(self):
        self.values = np.asarray(self.values)

class BACON3:
    def __init__(self, 
                 constancy_threshold: float = 0.05,
                 monotonicity_threshold: float = 0.98,
                 r2_threshold: float = 0.99,
                 max_depth: int = 3,
                 verbose: bool = False):
        """
        Initialises the BACON.3 solver.
        """
        self.constancy_threshold = constancy_threshold
        self.monotonicity_threshold = monotonicity_threshold
        self.r2_threshold = r2_threshold
        self.max_depth = max_depth
        self.verbose = verbose
        
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
        if self.verbose:
            if message:
                print(f"[BACON.3] {message}")
            else:
                print("")

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
        Generates invariants from ratios, products, and small integer powers (squares).
        Uses SymPy to manage symbolic expressions.
        """
        new_composites: list[Variable] = []

        # Powers
        for v in variables:
            # Square
            square_symbol = v.symbol ** 2
            if str(square_symbol) not in self.known_symbols:
                square_values = v.values ** 2
                new_composites.append(Variable(symbol=square_symbol, values=square_values))
                self.known_symbols.add(str(square_symbol))

        for v1, v2 in combinations(variables, 2):
            # Sum
            sum_symbol = v1.symbol + v2.symbol  # type: ignore[operator]
            if str(sum_symbol) not in self.known_symbols:
                sum_values = v1.values + v2.values
                new_composites.append(Variable(symbol=sum_symbol, values=sum_values))
                self.known_symbols.add(str(sum_symbol))

            # Differences
            diff1_symbol = v1.symbol - v2.symbol  # type: ignore[operator]
            if str(diff1_symbol) not in self.known_symbols:
                diff1_values = v1.values - v2.values
                new_composites.append(Variable(symbol=diff1_symbol, values=diff1_values))
                self.known_symbols.add(str(diff1_symbol))

            diff2_symbol = v2.symbol - v1.symbol  # type: ignore[operator]
            if str(diff2_symbol) not in self.known_symbols:
                diff2_values = v2.values - v1.values
                new_composites.append(Variable(symbol=diff2_symbol, values=diff2_values))
                self.known_symbols.add(str(diff2_symbol))

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
            if col == target_col:
                # Target is tracked separately and must not participate in composite generation;
                # otherwise BACON.3 can generate self-referential closures involving y.
                self.target_var = var
                continue

            self.variable_pool.append(var)
            self.known_symbols.add(str(sym))

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
            a, b, diagnostics = fit_linear_model(x_values, y_true)
            r2 = diagnostics["R-squared"]
            mse = diagnostics["MSE"]
            mae = diagnostics["MAE"]
            
            self._log(f"  Testing closing relation: {str(self.target_var.symbol)} ~ {str(v.symbol)}. R-squared: {r2:.6f}")

            # Check if this is the best, near-perfect fit
            if r2 > self.r2_threshold and r2 > best_r2: 
                best_r2 = r2
                best_fit_symbol = v.symbol
                best_coeffs = (a, b)
                best_diagnostics = diagnostics
        
        if best_fit_symbol:
            a, b = best_coeffs
            if abs(a) < 1e-12:
                a = 0.0
            if abs(b) < 1e-12:
                b = 0.0

            a_str = f"{a:.4g}"
            b_abs_str = f"{abs(b):.4g}"
            sign = "+" if b >= 0 else "-"
            equation_string = f"{str(self.target_var.symbol)} = {a_str} * ({str(best_fit_symbol)}) {sign} {b_abs_str}"
            return equation_string, best_diagnostics
        
        # No fit found
        return None

    def discover(self, data: pd.DataFrame, target_col: str, seed: int = 42) -> tuple[str, dict[str, float]]:
        """
        Runs the main BACON.3 passes.
        
        This is the main entry point that executes the layered
        discovery loop and attempts to find a closing relation.
        
        Args:
            data: A pandas DataFrame with input variables.
            target_col: The name of the column to use as the target variable.
            seed: A fixed seed for reproducibility and tie-breaking.
            
        Returns:
            A tuple containing:
                - The discovered symbolic law as a string (or a failure message).
                - A dictionary of residual diagnostics (e.g. R-squared, MSE, MAE).
        """
        
        np.random.seed(seed)
        self.logs = []
        
        self._initialise_variables(data, target_col)
        
        if self.target_var is None:
            self._log("Stop: target variable not found")
            return "No law found", {"R-squared": 0.0, "MSE": float("inf"), "MAE": float("inf")}

        self._log(f"Starting discovery. Target: '{str(self.target_var.symbol)}'. Seed: {seed}. Shape: {data.shape}")

        # main loop
        for i in range(self.max_depth):
            self._log(f"--- Layer {i+1} ---")
            
            vars_to_combine = self.variable_pool 
            
            new_composites = self._generate_composites(vars_to_combine)
            if not new_composites:
                self._log("Stop: no new composites generated")
                break
            
            promoted_this_layer: list[Variable] = []
            
            for composite in new_composites:
                
                if self._check_constancy(composite):
                    self._log(f"  -> Cue hit (Constancy): {str(composite.symbol)} is near-constant; promote")
                    promoted_this_layer.append(composite)
                    continue
                
                if self._check_monotonicity(composite):
                    self._log(f"  -> Cue hit (Monotonicity): {str(composite.symbol)} is monotonic with {self.target_var.symbol}; promote")
                    promoted_this_layer.append(composite)

            if not promoted_this_layer:
                self._log("Stop: no candidates met cues")
                break 
            
            self.variable_pool.extend(promoted_this_layer)
            self._log(f"Layer {i+1} complete. Promoted: {[str(v.symbol) for v in promoted_this_layer]}")
        
        self._log("--- Finding closing relation ---")
        
        # This will hold the {R2, MSE, MAE} dict
        self.final_diagnostics: dict[str, float] = {} 
        
        # Call the function
        fit_results = self._find_closing_relation()
        
        if fit_results:
            # Unpack the tuple here
            self.final_equation, self.final_diagnostics = fit_results
            self._log(f"Success: equation: {self.final_equation}")
            self._log(f"Diagnostics: {self.final_diagnostics}")
        else:
            self.final_equation = "No law found"
            self._log(f"Failed: {self.final_equation}")
            self.final_diagnostics = {"R-squared": 0.0, "MSE": float("inf"), "MAE": float("inf")}
        
        # Return both the equation and its diagnostics
        return self.final_equation, self.final_diagnostics