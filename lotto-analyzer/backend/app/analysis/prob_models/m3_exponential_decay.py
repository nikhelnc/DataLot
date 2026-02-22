"""
M3 - Exponential Decay Model

Pondération temporelle exponentielle des tirages passés.
Contrairement à M2 (fenêtre fixe), M3 attribue un poids décroissant
exponentiellement à chaque tirage selon son ancienneté.

Formule: w(t) = exp(-λ * (T - t))
         P(i) = Σ_t [ w(t) * 𝟙{i ∈ tirage_t} ] / Σ_t w(t)
"""

import numpy as np
import pandas as pd
from typing import Dict


class M3ExponentialDecay:
    """
    Exponential Decay model for lottery number prediction.
    
    Assigns exponentially decaying weights to past draws,
    giving more importance to recent draws without a hard cutoff.
    """
    
    def __init__(
        self, 
        rules: Dict, 
        lambda_decay: float = 0.02,
        min_weight: float = 1e-6
    ):
        """
        Initialize M3 Exponential Decay model.
        
        Args:
            rules: Game rules dictionary
            lambda_decay: Decay rate (higher = faster decay, more weight on recent)
            min_weight: Minimum weight threshold (draws below are ignored)
        """
        self.rules = rules
        self.lambda_decay = lambda_decay
        self.min_weight = min_weight
        
        # Parse game rules
        main_rules = rules.get("main", rules.get("numbers", {}))
        self.n_min = main_rules.get("min", 1)
        self.n_max = main_rules.get("max", 49)
        
        self.posterior = None
        
    def fit(self, df: pd.DataFrame):
        """
        Train the model on historical draw data.
        
        Args:
            df: DataFrame with 'numbers' column containing lists of drawn numbers
        """
        n_draws = len(df)
        if n_draws == 0:
            self.posterior = None
            return
        
        # Initialize counts with exponential weights
        weighted_counts = np.zeros(self.n_max - self.n_min + 1)
        total_weight = 0.0
        
        # Calculate weights for each draw (most recent = highest weight)
        for t, (_, row) in enumerate(df.iterrows()):
            # t=0 is oldest, t=n_draws-1 is most recent
            age = n_draws - 1 - t  # age of draw (0 = most recent)
            weight = np.exp(-self.lambda_decay * age)
            
            # Skip if weight is too small
            if weight < self.min_weight:
                continue
            
            total_weight += weight
            
            # Add weighted counts for each number in this draw
            for num in row["numbers"]:
                if self.n_min <= num <= self.n_max:
                    weighted_counts[num - self.n_min] += weight
        
        # Include bonus numbers if available
        if "bonus_numbers" in df.columns:
            for t, (_, row) in enumerate(df.iterrows()):
                age = n_draws - 1 - t
                weight = np.exp(-self.lambda_decay * age)
                
                if weight < self.min_weight:
                    continue
                
                bonus_nums = row.get("bonus_numbers", [])
                if bonus_nums:
                    for num in bonus_nums:
                        if self.n_min <= num <= self.n_max:
                            weighted_counts[num - self.n_min] += weight
        
        # Normalize to probabilities
        if total_weight > 0:
            probs = weighted_counts / weighted_counts.sum()
        else:
            # Fallback to uniform
            probs = np.ones(self.n_max - self.n_min + 1) / (self.n_max - self.n_min + 1)
        
        # Store as dictionary
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
            n_range = self.n_max - self.n_min + 1
            prob = 1.0 / n_range
            return {str(num): prob for num in range(self.n_min, self.n_max + 1)}
        
        return {str(num): float(prob) for num, prob in self.posterior.items()}
    
    def get_params(self) -> Dict:
        """Return model parameters for reproducibility."""
        return {
            "lambda_decay": self.lambda_decay,
            "min_weight": self.min_weight
        }
