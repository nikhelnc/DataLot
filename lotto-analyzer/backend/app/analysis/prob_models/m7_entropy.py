"""
M7 - Entropy-Based Selection Model

Sélection basée sur l'entropie locale de chaque numéro.
Calcule l'entropie de Shannon sur des fenêtres glissantes.

Formule: H_i = -p_i * log2(p_i) - (1-p_i) * log2(1-p_i)
         score_i = H0_theorique - H_i
         P(i) ∝ softmax(score_i / temperature)
"""

import numpy as np
import pandas as pd
from typing import Dict


class M7Entropy:
    """
    Entropy-Based Selection model for lottery number prediction.
    
    Favors numbers with locally lower entropy (more predictable behavior)
    compared to the theoretical entropy under H0.
    """
    
    def __init__(
        self, 
        rules: Dict, 
        window_size: int = 30,
        alpha_threshold: float = 0.05,
        selection_mode: str = 'low_entropy',
        temperature: float = 1.0
    ):
        """
        Initialize M7 Entropy model.
        
        Args:
            rules: Game rules dictionary
            window_size: Window size for local entropy calculation
            alpha_threshold: Significance threshold for deviation from H0
            selection_mode: 'low_entropy' or 'high_entropy'
            temperature: Softmax temperature for probability conversion
        """
        self.rules = rules
        self.window_size = window_size
        self.alpha_threshold = alpha_threshold
        self.selection_mode = selection_mode
        self.temperature = temperature
        
        # Parse game rules
        main_rules = rules.get("main", rules.get("numbers", {}))
        self.n_min = main_rules.get("min", 1)
        self.n_max = main_rules.get("max", 49)
        self.n_pick = main_rules.get("pick", 6)
        self.n_numbers = self.n_max - self.n_min + 1
        
        self.posterior = None
        
    def _binary_entropy(self, p: float) -> float:
        """Calculate binary entropy H(p) = -p*log2(p) - (1-p)*log2(1-p)."""
        if p <= 0 or p >= 1:
            return 0.0
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    
    def fit(self, df: pd.DataFrame):
        """
        Train the model on historical draw data.
        
        Args:
            df: DataFrame with 'numbers' column containing lists of drawn numbers
        """
        n_draws = len(df)
        if n_draws < self.window_size:
            # Not enough data, use uniform
            self.posterior = {
                num: 1.0 / self.n_numbers
                for num in range(self.n_min, self.n_max + 1)
            }
            return
        
        # Build binary presence matrix
        presence = np.zeros((n_draws, self.n_numbers))
        for t, (_, row) in enumerate(df.iterrows()):
            for num in row["numbers"]:
                if self.n_min <= num <= self.n_max:
                    presence[t, num - self.n_min] = 1
        
        # Calculate theoretical entropy under H0
        p0 = self.n_pick / self.n_numbers
        h0 = self._binary_entropy(p0)
        
        # Calculate local entropy for each number (using last window)
        window_data = presence[-self.window_size:]
        local_freqs = window_data.mean(axis=0)
        
        # Calculate entropy for each number
        entropies = np.array([self._binary_entropy(p) for p in local_freqs])
        
        # Calculate scores (deviation from H0)
        if self.selection_mode == 'low_entropy':
            # Favor numbers with lower entropy (more predictable)
            scores = h0 - entropies
        else:
            # Favor numbers with higher entropy (more random)
            scores = entropies - h0
        
        # Convert to probabilities via softmax
        scores_scaled = scores / self.temperature
        exp_scores = np.exp(scores_scaled - scores_scaled.max())  # Numerical stability
        probs = exp_scores / exp_scores.sum()
        
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
            "window_size": self.window_size,
            "alpha_threshold": self.alpha_threshold,
            "selection_mode": self.selection_mode,
            "temperature": self.temperature
        }
