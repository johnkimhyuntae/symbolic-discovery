import numpy as np
import pandas as pd
import sympy
from dataclasses import dataclass
from sympy import Symbol, Expr, symbols
from typing import List, Tuple, Optional, Dict

from .utils import evaluate_equation_constancy, calculate_r2, fit_linear_model

@dataclass
class Variable:
    """
    Represents a symbolic variable, its data, and the variables it depends on.
    """
    symbol: Expr    
    values: np.ndarray
    # Track which independent variables this term still depends on
    dependencies: List[str] 

    def __repr__(self):
        return str(self.symbol)

class BACON1:
    """
    The 'Space of Laws' heuristic engine (Miller, 2024).
    Determines the relationship between two specific variables.
    """
    def __init__(self, epsilon: float, delta: float, c_val: float):
        self.epsilon = epsilon  # Threshold for linearity (1 - |r|) (Miller, 2024)
        self.delta = delta      # Threshold for constancy (Miller, 2024)
        self.c_val = c_val      # Threshold for zero intercept (Miller, 2024)

    def check(self, dependent: Variable, independent: Variable) -> List[Variable]:
        """
        Runs the BACON.1 heuristic flow: Constancy -> Linear -> Product/Ratio (Miller, 2024).
        Returns a list of candidate invariants (e.g., Y/X, Y*X, Y-mX).
        """
        candidates = []
        x = independent.values
        y = dependent.values
        
        # Avoid division by zero issues in heuristic checks
        safe_x = np.where(np.abs(x) < 1e-9, 1e-9, x)

        # 1. Linear Proportionality Check (Miller, 2024)
        # Calculate correlation coefficient r
        # We check both Y vs X and Y vs 1/X (inverse) to catch simple products/ratios quickly
        
        m, c, diagnostics = fit_linear_model(x, y)
        r2 = float(diagnostics.get("R-squared", 0.0))
        # BACON.1 uses the sign of the slope to pick the sign of the correlation.
        r = float(np.sign(m) * np.sqrt(max(r2, 0.0)))

        # Check linearity condition: 1 - |r| < epsilon (Miller, 2024)
        if (1 - abs(r)) < self.epsilon:
            # (m, c) already computed via shared helper
            
            # Check if intercept is effectively zero (Miller, 2024)
            # If |c / mean(y)| < c_val, treat as zero
            if abs(c / (np.mean(y) + 1e-9)) < self.c_val:
                # Intercept is zero: y = mx
                # If m > 0, Ratio is constant: y/x = m
                # If m < 0, Product might be relevant, but linear logic suggests y - mx = 0
                term = dependent.symbol / independent.symbol
                vals = y / safe_x
                candidates.append(Variable(term, vals, dependent.dependencies))
            else:
                # Intercept is non-zero: y - mx = c
                # Invariant is y - mx
                m_sym = float(f"{m:.4g}") # distinct constant
                term = dependent.symbol - m_sym * independent.symbol
                vals = y - m * x
                candidates.append(Variable(term, vals, dependent.dependencies))

        # 2. Product / Ratio Checks (Monotonicity) (Miller, 2024)
        # If not strictly linear, we check general increasing/decreasing trends
        # r > 0 -> Divide (Ratio)
        # r < 0 -> Multiply (Product)
        
        # We add these candidates regardless of strict linearity to allow BACON.3 
        # to layer them (e.g. creating PV to later find PV/T).
        
        # Product (X * Y)
        prod_term = dependent.symbol * independent.symbol
        prod_vals = y * x
        candidates.append(Variable(prod_term, prod_vals, dependent.dependencies))

        # Ratio (Y / X)
        div_term = dependent.symbol / independent.symbol
        div_vals = y / safe_x
        candidates.append(Variable(div_term, div_vals, dependent.dependencies))
        
        return candidates

class BACON7:
    """
    Miller's BACON.7: BACON.1 Heuristics + BACON.3 Layering + Averaging + Layer Methods.
    """
    def __init__(self, 
                 max_depth: int = 5,
                 initial_epsilon: float = 0.05,
                 initial_delta: float = 0.05,    
                 c_val: float = 0.05,
                 scale_factor: float = 1.2,
                 r2_threshold: float = 0.98,
                 verbose: bool = False):
        
        self.max_depth = max_depth
        self.epsilon = initial_epsilon
        self.delta = initial_delta # Used for "Sufficiently Constant" check (Miller, 2024)
        self.c_val = c_val
        self.scale_factor = scale_factor
        self.r2_threshold = r2_threshold
        self.verbose = verbose
        self.logs: List[str] = []

    def _log(self, message: str):
        self.logs.append(message)
        if self.verbose:
            if message:
                print(f"[BACON.7] {message}")
            else:
                print("")

    def _is_sufficiently_constant(self, var: Variable) -> Optional[float]:
        """
        Check if a variable is constant within threshold Delta (Miller, 2024).
        Equation (12): mean(1-D) < val < mean(1+D) for (1-D) proportion of data.
        """
        mean_val = np.mean(var.values)
        if abs(mean_val) < 1e-9: return None # Safety
        
        # Relaxed check: coeff of variation
        std_val = np.std(var.values)
        cov = std_val / abs(mean_val)
        
        if cov < self.delta:
            return float(mean_val)
        return None

    def _layer_method_min_mse(self, candidates: List[Variable]) -> Variable:
        """
        Implements the 'min_mse' layer selection method.
        Selects the candidate that creates the 'cleanest' variable (lowest variance/MSE relative to mean).
        """
        best_cand = None
        min_mse = float('inf')

        for cand in candidates:
            # Normalise to compare MSE across different scales (e.g. PV vs V/P)
            mean_val = np.mean(cand.values)
            if abs(mean_val) < 1e-9: continue
            
            # MSE of the normalised variable against 1.0 (checking constancy)
            # This is equivalent to checking coefficient of variation squared
            normalised_vals = cand.values / mean_val
            mse = np.mean((normalised_vals - 1.0) ** 2)
            
            if mse < min_mse:
                min_mse = mse
                best_cand = cand
                
        return best_cand

    def _average_and_reduce(self, 
                          current_dependent: Variable, 
                          consumed_independent: str, 
                          full_df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        The critical BACON.7 averaging step.
        1. Takes the calculated values of the new invariant.
        2. Groups the original dataframe by ALL remaining independent variables.
        3. Averages the invariant values within those groups.
        4. Returns the new reduced values and the reduced dataframe.
        """
        # Create a temp df with the new values and all potential grouping keys
        temp_df = full_df.copy()
        temp_df['__calculated__'] = current_dependent.values
        
        # Grouping keys are all columns except the one we just 'consumed' into the invariant
        group_keys = [c for c in full_df.columns if c != consumed_independent]
        
        if not group_keys:
            # If no variables left, we just average everything to a single point
            return np.array([np.mean(temp_df['__calculated__'])]), pd.DataFrame()

        # Group and mean (Miller, 2024)
        reduced_df = temp_df.groupby(group_keys)['__calculated__'].mean().reset_index()
        
        # The new values for the next layer
        new_values = reduced_df['__calculated__'].values
        
        # The new dataframe (smaller N)
        return new_values, reduced_df[group_keys]
    
    def _calculate_r2(self, equation_str: str, const_val: float, target_name: str) -> Tuple[float, float]:
        """
        Calculate R² and MSE for the discovered equation by evaluating the LHS.
        For equations like "P/(T*T³) = k", evaluate how constant P/(T*T³) actually is.
        """
        # Create evaluation dataframe with ALL variables (features + target)
        eval_df = self.original_X.copy()
        eval_df[target_name] = self.original_y
        
        # Add power terms for all variables
        all_vars = list(self.original_X.columns) + [target_name]
        for var in all_vars:
            if '²' not in var and '³' not in var:
                eval_df[f"{var}²"] = eval_df[var] ** 2
                eval_df[f"{var}³"] = eval_df[var] ** 3
        
        # Use shared utility function
        return evaluate_equation_constancy(equation_str, const_val, eval_df, target_name)

    def _compute_predictive_r2(self, equation_str: str, const_val: float, target_name: str) -> float:
        """
        Compute standard predictive R² by solving the equation for the target variable.
        
        For equations like "V/(I*R) = k", solve for V to get "V = k*I*R", 
        then compute R² = 1 - SS_res/SS_tot comparing predicted vs actual target values.
        
        This enables fair comparison with BACON.3's R² metric.
        
        Args:
            equation_str: Equation like "V/(I*R) = 1.0"
            const_val: The constant value (exact, not from formatted string)
            target_name: Name of target variable
            
        Returns:
            Standard predictive R² value
        """
        try:
            if " = " not in equation_str:
                return 0.0
            
            lhs_str, _ = equation_str.split(" = ", 1)
            
            # Parse the LHS expression into sympy
            # Replace unicode superscripts
            lhs_normalized = lhs_str.replace('²', '**2').replace('³', '**3')
            
            # Create sympy symbols for all variables in the dataset
            all_vars = list(self.original_X.columns) + [target_name]
            sym_dict = {var: Symbol(var) for var in all_vars}
            
            # Parse LHS expression
            lhs_expr = sympy.sympify(lhs_normalized, locals=sym_dict)
            target_sym = sym_dict[target_name]
            
            # Solve: LHS = const_val for target variable
            # e.g., V/(I*R) = k  →  V = k*I*R
            equation = sympy.Eq(lhs_expr, const_val)
            solutions = sympy.solve(equation, target_sym)
            
            if not solutions:
                return 0.0
            
            # Take the first real solution (usually only one for our laws)
            solution = solutions[0]
            
            # Convert to numerical function
            free_symbols = [sym for sym in solution.free_symbols if sym != target_sym]
            func = sympy.lambdify(free_symbols, solution, modules=['numpy'])
            
            # Generate predictions
            eval_df = self.original_X.copy()
            eval_df[target_name] = self.original_y
            
            # Get argument values in the same order as free_symbols
            args = [eval_df[str(sym)].values for sym in free_symbols]
            
            if len(args) == 0:
                # Constant prediction
                y_pred = np.full(len(self.original_y), float(solution))
            else:
                y_pred = func(*args)
            
            # Compute standard R²
            y_true = self.original_y if isinstance(self.original_y, np.ndarray) else self.original_y.values
            r2 = calculate_r2(y_true, y_pred)
            
            return float(r2)
            
        except Exception as e:
            # If we can't solve/evaluate, fall back to 0
            self._log(f"Warning: Could not compute predictive R²: {e}")
            return 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series, seed: Optional[int] = None) -> Tuple[Optional[str], Dict]:
        self.logs = []
        target_name = y.name if y.name else "y"
        
        # Store original data for R² calculation
        self.original_X = X.copy()
        self.original_y = y.values.copy()
        
        # We keep track of the full dataframe to perform averaging
        current_df = X.copy()
        current_vars = list(X.columns)
        
        # The "Dependent" variable evolves through the layers
        current_term = Variable(Symbol(target_name), y.values, current_vars)

        seed_str = f"{seed}" if seed is not None else "(unknown)"
        self._log(f"Starting discovery. Target: '{target_name}'. Seed: {seed_str}. Shape: {current_df.shape}")
        
        # Pre-generate polynomial powers for single-variable problems (T², T³, etc.)
        # Only generate for univariate problems to avoid combinatorial explosion
        if len(current_vars) == 1:
            var_name = current_vars[0]
            # Add squared term
            sq_name = f"{var_name}²"
            current_df[sq_name] = current_df[var_name] ** 2
            current_vars.append(sq_name)
            self._log(f"Generated power term: {sq_name}")
            
            # Add cubed term
            cube_name = f"{var_name}³"
            current_df[cube_name] = current_df[var_name] ** 3
            current_vars.append(cube_name)
            self._log(f"Generated power term: {cube_name}")

        for layer_idx in range(self.max_depth):
            self._log(f"--- Layer {layer_idx + 1} ---")
            
            # 1. Constancy Check (Miller, 2024)
            const_val = self._is_sufficiently_constant(current_term)
            if const_val is not None:
                self._log(f"  -> Cue hit (Constancy): {current_term.symbol} ≈ {const_val:.4g}")
                
                # Reconstruct equation: Term = Constant
                eq_str = f"{current_term.symbol} = {const_val:.4g}"
                
                # Calculate constancy R² for internal validation
                constancy_r2, mse = self._calculate_r2(eq_str, const_val, target_name)
                
                # Check if constancy R² meets threshold
                if constancy_r2 < self.r2_threshold:
                    self._log(f"  -> Reject: constancy R²={constancy_r2:.4f} < threshold={self.r2_threshold}")
                    return "No law found", {"r2": constancy_r2, "mse": mse}
                
                # Compute PREDICTIVE R² for fair comparison with BACON.3
                # This solves the equation for the target variable and computes standard R²
                predictive_r2 = self._compute_predictive_r2(eq_str, const_val, target_name)
                
                self._log(f"  -> Metrics: constancy R²={constancy_r2:.4f}, predictive R²={predictive_r2:.4f}")
                
                return eq_str, {
                    "constant": const_val, 
                    "final_term": str(current_term.symbol),
                    "r2": predictive_r2,  # Use predictive R² for reporting
                    "constancy_r2": constancy_r2,  # Keep constancy R² for diagnostics
                    "mse": mse
                }

            # 2. Search Space of Laws (Heuristics) (Miller, 2024)
            layer_candidates = []
            
            # We try to relate the current_term to each available independent variable
            # (In Miller's tree, this is checking branches)
            if not current_vars:
                self._log("Stop: no independent variables left to relate.")
                break
                
            bacon1 = BACON1(self.epsilon, self.delta, self.c_val)
            
            for indep_name in current_vars:
                indep_vals = current_df[indep_name].values
                indep_var = Variable(Symbol(indep_name), indep_vals, [])
                
                # Get candidates from BACON.1 (products, ratios, linear diffs)
                new_candidates = bacon1.check(current_term, indep_var)
                
                # Tag which variable was used so we can consume it
                for cand in new_candidates:
                    cand.consumed_var = indep_name 
                    layer_candidates.append(cand)

            if not layer_candidates:
                self._log("Stop: no valid relations found in this layer.")
                break

            # 3. Layer Selection (min_mse) 
            best_term = self._layer_method_min_mse(layer_candidates)
            
            if best_term is None:
                self._log("Stop: layer method failed to select a term.")
                break

            self._log(f"  -> Select: {best_term.symbol} (consume: {best_term.consumed_var})")

            # 4. Averaging & Reduction 
            # We consume the independent variable used in the relation
            # The dataset shrinks here
            new_vals, new_df = self._average_and_reduce(
                best_term, 
                best_term.consumed_var, 
                current_df
            )
            
            # Update State for next layer
            current_df = new_df
            current_vars.remove(best_term.consumed_var)
            
            # Update the dependent variable to be this new term
            current_term = Variable(
                best_term.symbol,
                new_vals,
                current_vars # Remaining dependencies
            )
            
            self._log(f"  -> Reduce: new dataset size after averaging: {len(current_df)}")
            
            # Increase tolerance slightly for deeper layers 
            self.epsilon *= self.scale_factor
            self.delta *= self.scale_factor

        self._log("Failed: no law found")
        return None, {}
    
    def discover(self, df: pd.DataFrame, target_col: str, seed: int = 42) -> Tuple[Optional[str], Dict]:
        """
        Wrapper for fit() to match the interface expected by runner.py.
        Separates features and target, then calls fit().
        """
        np.random.seed(seed)
        
        y = df[target_col]
        X = df.drop(columns=[target_col])
        
        eq, details = self.fit(X, y, seed=seed)
        
        # Convert keys to match BACON3 format for runner compatibility
        diagnostics = {
            "R-squared": details.get("r2", 0.0),
            "MSE": details.get("mse", float('inf'))
        }
        
        return eq, diagnostics