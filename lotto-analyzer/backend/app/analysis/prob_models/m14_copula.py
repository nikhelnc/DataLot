"""
M14 - Copula Model

Modélisation des dépendances entre numéros par copules.
Sépare les distributions marginales de la structure de dépendance.

Librairie: copulas ou pyvinecopulib (optionnel, fallback disponible)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import warnings

try:
    from copulas.multivariate import GaussianMultivariate
    COPULAS_AVAILABLE = True
except ImportError:
    COPULAS_AVAILABLE = False
    warnings.warn("copulas not available. M14 Copula will use simplified fallback.")


class M14Copula:
    """
    Copula model for lottery number prediction.
    
    Models dependencies between numbers by separating marginal
    distributions from the dependency structure.
    """
    
    def __init__(
        self, 
        rules: Dict, 
        copula_type: str = 'gaussian',
        n_simulations: int = 10000,
        marginal_method: str = 'empirical',
        selection_criterion: str = 'aic',
        n_groups: int = 5
    ):
        """
        Initialize M14 Copula model.
        
        Args:
            rules: Game rules dictionary
            copula_type: Type of copula ('gaussian', 'student_t', 'vine')
            n_simulations: Number of simulations for probability estimation
            marginal_method: Method for marginal estimation
            selection_criterion: Criterion for copula selection
            n_groups: Number of number groups for dimensionality reduction
        """
        self.rules = rules
        self.copula_type = copula_type
        self.n_simulations = n_simulations
        self.marginal_method = marginal_method
        self.selection_criterion = selection_criterion
        self.n_groups = n_groups
        
        # Parse game rules
        main_rules = rules.get("main", rules.get("numbers", {}))
        self.n_min = main_rules.get("min", 1)
        self.n_max = main_rules.get("max", 49)
        self.n_pick = main_rules.get("pick", 6)
        self.n_numbers = self.n_max - self.n_min + 1
        
        self.posterior = None
        self.copula_model = None
        
    def _build_group_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build grouped features for dimensionality reduction.
        Groups numbers into ranges and counts occurrences.
        """
        n_draws = len(df)
        group_size = self.n_numbers // self.n_groups
        
        features = {}
        for g in range(self.n_groups):
            start = self.n_min + g * group_size
            end = start + group_size if g < self.n_groups - 1 else self.n_max + 1
            features[f"group_{g}"] = np.zeros(n_draws)
        
        # Also track even/odd ratio and sum
        features["even_ratio"] = np.zeros(n_draws)
        features["sum_normalized"] = np.zeros(n_draws)
        
        for t, (_, row) in enumerate(df.iterrows()):
            numbers = row["numbers"]
            if len(numbers) == 0:
                continue
            
            # Count per group
            for num in numbers:
                if self.n_min <= num <= self.n_max:
                    g = min((num - self.n_min) // group_size, self.n_groups - 1)
                    features[f"group_{g}"][t] += 1
            
            # Even ratio
            features["even_ratio"][t] = sum(1 for n in numbers if n % 2 == 0) / len(numbers)
            
            # Normalized sum
            max_sum = self.n_pick * self.n_max
            features["sum_normalized"][t] = sum(numbers) / max_sum
        
        return pd.DataFrame(features)
    
    def _fit_with_copulas(self, features_df: pd.DataFrame) -> bool:
        """Fit using copulas library."""
        try:
            self.copula_model = GaussianMultivariate()
            self.copula_model.fit(features_df)
            return True
        except Exception as e:
            warnings.warn(f"Copula fitting failed: {e}")
            return False
    
    def _simulate_and_estimate(self, features_df: pd.DataFrame) -> np.ndarray:
        """Simulate from copula and estimate number probabilities."""
        probs = np.zeros(self.n_numbers)
        group_size = self.n_numbers // self.n_groups
        
        if COPULAS_AVAILABLE and self.copula_model is not None:
            # Sample from copula
            try:
                samples = self.copula_model.sample(self.n_simulations)
                
                # Convert group samples to number probabilities
                for _, sample in samples.iterrows():
                    for g in range(self.n_groups):
                        group_count = max(0, sample[f"group_{g}"])
                        start = self.n_min + g * group_size
                        end = start + group_size if g < self.n_groups - 1 else self.n_max + 1
                        
                        # Distribute probability within group
                        for num in range(start, end):
                            probs[num - self.n_min] += group_count / (end - start)
                
                probs /= self.n_simulations
                return probs
                
            except Exception:
                pass
        
        # Fallback: use empirical group frequencies
        group_means = features_df[[f"group_{g}" for g in range(self.n_groups)]].mean()
        
        for g in range(self.n_groups):
            start = self.n_min + g * group_size
            end = start + group_size if g < self.n_groups - 1 else self.n_max + 1
            group_prob = group_means[f"group_{g}"] / self.n_pick
            
            for num in range(start, end):
                probs[num - self.n_min] = group_prob / (end - start)
        
        return probs
    
    def _fit_fallback(self, df: pd.DataFrame) -> np.ndarray:
        """Simplified fallback using correlation-weighted frequencies."""
        # Build binary presence matrix
        n_draws = len(df)
        presence = np.zeros((n_draws, self.n_numbers))
        
        for t, (_, row) in enumerate(df.iterrows()):
            for num in row["numbers"]:
                if self.n_min <= num <= self.n_max:
                    presence[t, num - self.n_min] = 1
        
        # Calculate base frequencies
        base_freq = presence.mean(axis=0)
        
        # Calculate correlation matrix
        corr = np.corrcoef(presence.T)
        corr = np.nan_to_num(corr, nan=0.0)
        
        # Boost probabilities based on positive correlations
        probs = base_freq.copy()
        for i in range(self.n_numbers):
            for j in range(self.n_numbers):
                if i != j and corr[i, j] > 0.1:
                    probs[i] += 0.05 * corr[i, j] * base_freq[j]
        
        return probs
    
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
        
        # Build grouped features
        features_df = self._build_group_features(df)
        
        # Fit copula
        if COPULAS_AVAILABLE:
            success = self._fit_with_copulas(features_df)
            if success:
                probs = self._simulate_and_estimate(features_df)
            else:
                probs = self._fit_fallback(df)
        else:
            probs = self._fit_fallback(df)
        
        # Normalize
        probs = np.clip(probs, 0.001, None)
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
    
    def get_params(self) -> Dict:
        """Return model parameters for reproducibility."""
        return {
            "copula_type": self.copula_type,
            "n_simulations": self.n_simulations,
            "marginal_method": self.marginal_method,
            "selection_criterion": self.selection_criterion,
            "n_groups": self.n_groups,
            "copulas_available": COPULAS_AVAILABLE
        }
