import numpy as np
import pandas as pd
import sympy
from dataclasses import dataclass
from sympy import Symbol, Expr, symbols
from typing import List, Tuple, Optional, Dict

from pandas.api.types import is_numeric_dtype
from scipy.stats import iqr

from ..utils import evaluate_equation_constancy, calculate_r2, fit_linear_model

@dataclass
class Variable:
    """
    Represents a symbolic variable, its data, and the variables it depends on.
    """
    symbol: Expr    
    values: np.ndarray
    # Track which independent variables this term still depends on
    dependencies: List[str] 

    # The variable "consumed" to form this term (set during search)
    consumed_var: Optional[str] = None

    def __post_init__(self):
        self.values = np.asarray(self.values)
        self.dependencies = list(self.dependencies) if self.dependencies is not None else []

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

        # Robust scale for intercept checks.
        y_scale = float(np.max(np.mean(y))) if y.size else 1.0

        # Prevent division by zero
        if y_scale < 1e-9:
            y_scale = 1e-9

        # Calculate correlation coefficient r

        m, c, diagnostics = fit_linear_model(x, y)
        r2 = float(diagnostics.get("R-squared", 0.0))
        # BACON.1 uses the sign of the slope to pick the sign of the correlation.
        r = float(np.sign(m) * np.sqrt(max(r2, 0.0)))

        # Check linearity condition: 1 - |r| < epsilon
        if (1 - abs(r)) < self.epsilon:
            # If |c / scale(y)| < c_val, treat as zero
            if abs(c / y_scale) < self.c_val:
                # Intercept is zero: y = mx
                # If m > 0, Ratio is constant: y/x = m
                # If m < 0, Product might be relevant, but linear logic suggests y - mx = 0
                term = dependent.symbol / independent.symbol  # type: ignore[operator]
                vals = y / safe_x
                candidates.append(Variable(term, vals, dependent.dependencies))
            else:
                # Intercept is non-zero: y - mx = c
                # Invariant is y - mx
                m_sym = float(f"{m:.4g}") # distinct constant
                term = dependent.symbol - m_sym * independent.symbol  # type: ignore[operator]
                vals = y - m * x
                candidates.append(Variable(term, vals, dependent.dependencies))
        
        # Product (X * Y)
        prod_term = dependent.symbol * independent.symbol  # type: ignore[operator]
        prod_vals = y * x
        candidates.append(Variable(prod_term, prod_vals, dependent.dependencies))

        # Ratio (Y / X)
        div_term = dependent.symbol / independent.symbol  # type: ignore[operator]
        div_vals = y / safe_x
        candidates.append(Variable(div_term, div_vals, dependent.dependencies))
        
        return candidates
    
    @staticmethod
    def filter_novel(candidates: List[Variable],
                    known_symbols: set) -> List[Variable]:
        """
        Novel relation check (Miller, 2024).
        Reject any candidate whose simplified SymPy expression matches
        an already-known symbol in the pool.
        """
        novel = []
        for cand in candidates:
            simplified = str(sympy.simplify(cand.symbol))
            if simplified in known_symbols:
                continue
            novel.append(cand)
        return novel

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
        mean(1-D) < val < mean(1+D) for (1-D) proportion of data.
        """
        # Use the same robust constancy scoring as BACON.3:
        #   robust_cv = IQR(values) / |median(values)|
        # This is less brittle when the mean is near 0 or when there are outliers.
        vals = np.asarray(var.values)
        mean_val = float(np.mean(vals))
        
        # Handle edge case where mean is extremely close to zero to prevent extreme bounds
        if abs(mean_val) < 1e-9:
            return None

        bound_1 = mean_val * (1 - self.delta)
        bound_2 = mean_val * (1 + self.delta)
        lower_bound = min(bound_1, bound_2)
        upper_bound = max(bound_1, bound_2)

        # Count how many values fall strictly within the bounds
        within_bounds = np.sum((vals > lower_bound) & (vals < upper_bound))
        proportion = float(within_bounds) / len(vals)

        # Check if the proportion meets the (1 - Delta) threshold
        if proportion >= (1 - self.delta):
            return mean_val
            
        return None

    def _layer_method_min_mse(self, candidates: List[Variable]) -> Optional[Variable]:
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
    
    def _layer_method_popular(self, candidates: List[Variable]) -> Optional[Variable]:
        """
        Implements the 'popular' layer selection method.
        Counts which invariant form appears most frequently across independent
        variables. On a tie, defers to min_mse.
        """
        if not candidates:
            return None

        # Classify each candidate by its structural form.
        # We use the SymPy expression with numeric coefficients replaced by a
        # placeholder so that e.g. "V - 0.96*P" and "V - 1.02*P" count as the
        # same form "V - _c*P".

        def _structural_key(expr: sympy.Expr) -> str:
            """Replace all pure-numeric atoms with a sentinel so that
            candidates that differ only in fitted constants hash together."""
            replaced = expr.xreplace(
                {a: sympy.Symbol("_c") for a in expr.atoms(sympy.Number)}
            )
            return str(replaced)

        key_to_cands: Dict[str, List[Variable]] = {}
        for cand in candidates:
            key = _structural_key(cand.symbol)
            key_to_cands.setdefault(key, []).append(cand)

        # Sort by (descending) popularity
        ranked = sorted(key_to_cands.items(), key=lambda kv: -len(kv[1]))
        top_count = len(ranked[0][1])

        # Collect all forms that share the top count
        tied_cands = []
        for key, cands in ranked:
            if len(cands) < top_count:
                break
            tied_cands.extend(cands)

        self._log(f"  Popular: top form(s) appeared {top_count}x "
                f"({len(tied_cands)} candidates in tie)")

        if len(tied_cands) == 1:
            return tied_cands[0]

        # Tie-break with min_mse
        return self._layer_method_min_mse(tied_cands)

    def _average_and_reduce(
        self,
        current_dependent: Variable,
        consumed_independent: str | List[str],
        full_df: pd.DataFrame,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        The BACON.7 averaging step.
        1. Takes the calculated values of the new invariant.
        2. Groups the original dataframe by ALL remaining independent variables.
        3. Averages the invariant values within those groups.
        4. Returns the new reduced values and the reduced dataframe.
        """
        # Create a temp df with the new values and all potential grouping keys
        temp_df = full_df.copy()
        temp_df['__calculated__'] = current_dependent.values
        
        # Grouping keys are all columns except the ones we just 'consumed' into the invariant.
        # Important: when we expand features (x, x², x³), consuming any member of that family
        # must exclude the entire family from grouping; otherwise averaging can't reduce.
        if isinstance(consumed_independent, str):
            consumed_set = {consumed_independent}
        else:
            consumed_set = set(consumed_independent)

        group_keys = [c for c in full_df.columns if c not in consumed_set]
        
        if not group_keys:
            # If no variables left, we just average everything to a single point
            return np.array([np.mean(temp_df['__calculated__'])]), pd.DataFrame()

        # Group and mean
        reduced_df = temp_df.groupby(group_keys)['__calculated__'].mean().reset_index()
        
        # The new values for the next layer
        new_values = np.asarray(reduced_df['__calculated__'].values)
        
        # The new dataframe
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
            # e.g. V/(I*R) = k -> V = k*I*R
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

    def discover(self, df: pd.DataFrame, target_col: str, seed: int = 42) -> Tuple[str, Dict]:
        """Run BACON.7 discovery on a single dataframe (Miller, 2024).

        This is the primary public entrypoint used by the experiment runner/tests.
        It expects the target column to be present in `df`.
        """
        np.random.seed(seed)

        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found in dataframe")

        self.logs = []
        target_name = str(target_col)

        y = df[target_col]
        X = df.drop(columns=[target_col])

        # Store original data for R² calculation
        self.original_X = X.copy()
        self.original_y = np.asarray(y.to_numpy(copy=True))

        # We keep track of the full dataframe to perform averaging
        current_df = X.copy()
        current_vars = list(X.columns)

        # In multivariate problems, expanded features (x, x², x³) must be treated as a family
        # when averaging; otherwise reduction is blocked. In univariate problems, consuming the
        # whole family prevents iterative discovery (e.g., building T*T³).
        consume_expansion_families = len(current_vars) > 1
        
        # The "Dependent" variable evolves through the layers
        current_term = Variable(Symbol(target_name), np.asarray(y.to_numpy(copy=True)), current_vars)

        self._log(f"Starting discovery. Target: '{target_name}'. Seed: {seed}. Shape: {current_df.shape}")
        
        # Feature expansion (Miller): precompute powers so a variable can be
        # effectively "consumed" via its power term in a single step after averaging.
        # We do this for ALL numeric variables (not just univariate problems).
        initial_cols = list(current_df.columns)
        expansion_families: Dict[str, set[str]] = {}

        def _base_name(col_name: str) -> str:
            if col_name.endswith('²') or col_name.endswith('³'):
                return col_name[:-1]
            return col_name

        def _family_for(col_name: str) -> set[str]:
            base = _base_name(col_name)
            return set(expansion_families.get(base, {base}))

        for var_name in initial_cols:
            if not is_numeric_dtype(current_df[var_name]):
                continue

            sq_name = f"{var_name}²"
            if sq_name not in current_df.columns:
                current_df[sq_name] = current_df[var_name] ** 2
                current_vars.append(sq_name)
                self._log(f"Generated power term: {sq_name}")

            cube_name = f"{var_name}³"
            if cube_name not in current_df.columns:
                current_df[cube_name] = current_df[var_name] ** 3
                current_vars.append(cube_name)
                self._log(f"Generated power term: {cube_name}")

            expansion_families[var_name] = {var_name, sq_name, cube_name}

        for layer_idx in range(self.max_depth):
            self._log(f"--- Layer {layer_idx + 1} ---")
            
            # 1. Constancy Check
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
                    return "No law found", {
                        "R-squared": float(constancy_r2),
                        "MSE": float(mse),
                        "constancy_r2": float(constancy_r2),
                    }
                
                # Compute PREDICTIVE R² for fair comparison with BACON.3
                # This solves the equation for the target variable and computes standard R²
                predictive_r2 = self._compute_predictive_r2(eq_str, const_val, target_name)
                
                self._log(f"  -> Metrics: constancy R²={constancy_r2:.4f}, predictive R²={predictive_r2:.4f}")

                diagnostics = {
                    "R-squared": float(predictive_r2),
                    "MSE": float(mse),
                    "constancy_r2": float(constancy_r2),
                    "constant": float(const_val),
                    "final_term": str(current_term.symbol),
                }
                return eq_str, diagnostics

            # 2. Search Space of Laws + Managing-layer retry
            max_retries = 3
            best_term = None
            saved_epsilon = self.epsilon
            saved_delta = self.delta

            for retry in range(max_retries):
                if retry > 0:
                    self._log(f"  Retry {retry}/{max_retries-1}: "
                              f"ε={self.epsilon:.4g}, δ={self.delta:.4g}")

                layer_candidates = []

                if not current_vars:
                    self._log("Stop: no independent variables left to relate.")
                    break

                bacon1 = BACON1(self.epsilon, self.delta, self.c_val)

                for indep_name in current_vars:
                    indep_vals = np.asarray(current_df[indep_name].values)
                    indep_var = Variable(Symbol(indep_name), indep_vals, [])
                    new_candidates = bacon1.check(current_term, indep_var)
                    for cand in new_candidates:
                        cand.consumed_var = indep_name
                        layer_candidates.append(cand)

                # Novel relation filter
                known_syms = {target_name} | set(current_vars)
                layer_candidates = BACON1.filter_novel(layer_candidates, known_syms)
                self._log(f"  After novel-relation filter: {len(layer_candidates)} candidates")

                if not layer_candidates:
                    # Scale up thresholds and retry
                    self.epsilon *= self.scale_factor
                    self.delta *= self.scale_factor
                    continue

                # Layer selection: popular, then min_mse tiebreak
                best_term = self._layer_method_popular(layer_candidates)
                if best_term is None:
                    best_term = self._layer_method_min_mse(layer_candidates)

                if best_term is not None:
                    # Restore pre-retry thresholds so the single
                    # end-of-layer scale-up is the only one that persists.
                    self.epsilon = saved_epsilon
                    self.delta = saved_delta
                    break

                # No term selected — scale up and retry
                self.epsilon *= self.scale_factor
                self.delta *= self.scale_factor

            if best_term is None:
                self._log("Stop: layer method failed after retries.")
                break

            # Best term is always tagged with the variable used to create it.
            assert best_term.consumed_var is not None

            self._log(f"  -> Select: {best_term.symbol} (consume: {best_term.consumed_var})")

            # 4. Averaging & Reduction 
            # We consume the independent variable used in the relation
            # The dataset shrinks here
            if consume_expansion_families:
                consumed_family = sorted(_family_for(best_term.consumed_var))
            else:
                consumed_family = [best_term.consumed_var]
            new_vals, new_df = self._average_and_reduce(
                best_term, 
                consumed_family, 
                current_df
            )
            
            # Update State for next layer
            current_df = new_df
            # Remove the whole consumed family from the remaining variables
            current_vars = [v for v in current_vars if v not in set(consumed_family)]
            
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
        return "No law found", {"R-squared": 0.0, "MSE": float("inf")}