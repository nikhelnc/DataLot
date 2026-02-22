"""
M9 - Bayesian Network Model

Réseau bayésien pour modéliser les dépendances conditionnelles entre numéros.
Utilise un DAG (graphe acyclique dirigé) pour capturer les relations.

Librairie: pgmpy (optionnel, fallback disponible)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings

try:
    from pgmpy.models import BayesianNetwork
    from pgmpy.estimators import HillClimbSearch, BicScore, BayesianEstimator
    from pgmpy.inference import VariableElimination
    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False
    warnings.warn("pgmpy not available. M9 Bayesian Network will use simplified fallback.")


class M9BayesianNetwork:
    """
    Bayesian Network model for lottery number prediction.
    
    Models conditional dependencies between numbers using a DAG.
    """
    
    def __init__(
        self, 
        rules: Dict, 
        structure_algo: str = 'hc',
        scoring: str = 'bic',
        max_parents: int = 3,
        significance_level: float = 0.05,
        n_top_numbers: int = 15
    ):
        """
        Initialize M9 Bayesian Network model.
        
        Args:
            rules: Game rules dictionary
            structure_algo: Structure learning algorithm ('hc', 'pc', 'mmhc')
            scoring: Scoring method ('bic', 'bdeu', 'k2')
            max_parents: Maximum parents per node
            significance_level: Significance level for independence tests
            n_top_numbers: Number of top numbers to model (for tractability)
        """
        self.rules = rules
        self.structure_algo = structure_algo
        self.scoring = scoring
        self.max_parents = max_parents
        self.significance_level = significance_level
        self.n_top_numbers = n_top_numbers
        
        # Parse game rules
        main_rules = rules.get("main", rules.get("numbers", {}))
        self.n_min = main_rules.get("min", 1)
        self.n_max = main_rules.get("max", 49)
        self.n_numbers = self.n_max - self.n_min + 1
        
        self.posterior = None
        self.model = None
        self.edges = []
        
    def _build_binary_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build binary presence/absence DataFrame."""
        n_draws = len(df)
        data = {}
        
        for num in range(self.n_min, self.n_max + 1):
            col_name = f"n{num}"
            data[col_name] = np.zeros(n_draws, dtype=int)
        
        for t, (_, row) in enumerate(df.iterrows()):
            for num in row["numbers"]:
                if self.n_min <= num <= self.n_max:
                    data[f"n{num}"][t] = 1
        
        return pd.DataFrame(data)
    
    def _fit_with_pgmpy(self, binary_df: pd.DataFrame):
        """Fit using pgmpy library."""
        # Select top N most frequent numbers for tractability
        freq = binary_df.sum().sort_values(ascending=False)
        top_cols = freq.head(self.n_top_numbers).index.tolist()
        reduced_df = binary_df[top_cols]
        
        # Structure learning
        try:
            hc = HillClimbSearch(reduced_df)
            best_model = hc.estimate(
                scoring_method=BicScore(reduced_df),
                max_indegree=self.max_parents
            )
            self.edges = list(best_model.edges())
            
            # Build and fit model
            self.model = BayesianNetwork(self.edges)
            self.model.fit(reduced_df, estimator=BayesianEstimator, prior_type='BDeu')
            
            # Inference
            inference = VariableElimination(self.model)
            
            # Get marginal probabilities
            marginals = {}
            for col in top_cols:
                try:
                    result = inference.query([col])
                    # Get probability of presence (value=1)
                    marginals[col] = float(result.values[1])
                except Exception:
                    marginals[col] = reduced_df[col].mean()
            
            return marginals
            
        except Exception as e:
            warnings.warn(f"pgmpy fitting failed: {e}")
            return None
    
    def _fit_fallback(self, binary_df: pd.DataFrame) -> Dict[str, float]:
        """Simplified fallback using pairwise correlations."""
        # Calculate pairwise correlations
        corr_matrix = binary_df.corr()
        
        # For each number, boost probability based on correlated numbers
        marginals = {}
        base_probs = binary_df.mean()
        
        for col in binary_df.columns:
            # Get correlated numbers
            correlations = corr_matrix[col].drop(col)
            
            # Weight by correlation strength
            boost = 0.0
            for other_col, corr in correlations.items():
                if abs(corr) > 0.1:  # Only significant correlations
                    boost += corr * base_probs[other_col]
            
            # Combine base probability with correlation boost
            prob = base_probs[col] + 0.1 * boost
            marginals[col] = max(0.01, min(0.99, prob))
        
        return marginals
    
    def fit(self, df: pd.DataFrame):
        """
        Train the model on historical draw data.
        
        Args:
            df: DataFrame with 'numbers' column containing lists of drawn numbers
        """
        n_draws = len(df)
        if n_draws < 30:
            self.posterior = {
                num: 1.0 / self.n_numbers
                for num in range(self.n_min, self.n_max + 1)
            }
            return
        
        # Build binary data
        binary_df = self._build_binary_data(df)
        
        # Fit model
        if PGMPY_AVAILABLE:
            marginals = self._fit_with_pgmpy(binary_df)
            if marginals is None:
                marginals = self._fit_fallback(binary_df)
        else:
            marginals = self._fit_fallback(binary_df)
        
        # Convert to full probability distribution
        probs = np.zeros(self.n_numbers)
        for num in range(self.n_min, self.n_max + 1):
            col_name = f"n{num}"
            if col_name in marginals:
                probs[num - self.n_min] = marginals[col_name]
            else:
                # Use base frequency
                probs[num - self.n_min] = binary_df[col_name].mean()
        
        # Normalize
        probs = probs / probs.sum()
        
        self.posterior = {
            num: float(probs[num - self.n_min])
            for num in range(self.n_min, self.n_max + 1)
        }
    
    def predict(self) -> Dict[str, float]:
        """
        Return probability distribution over all numbers.
        
        Returns:
            Dictionary mapping number (as string) to probability
        """
        if self.posterior is None:
            prob = 1.0 / self.n_numbers
            return {str(num): prob for num in range(self.n_min, self.n_max + 1)}
        
        return {str(num): float(prob) for num, prob in self.posterior.items()}
    
    def get_edges(self) -> List[Tuple[str, str]]:
        """Return learned DAG edges."""
        return self.edges
    
    def get_params(self) -> Dict:
        """Return model parameters for reproducibility."""
        return {
            "structure_algo": self.structure_algo,
            "scoring": self.scoring,
            "max_parents": self.max_parents,
            "significance_level": self.significance_level,
            "n_top_numbers": self.n_top_numbers,
            "n_edges": len(self.edges),
            "pgmpy_available": PGMPY_AVAILABLE
        }
