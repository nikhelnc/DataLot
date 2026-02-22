"""
M15 - Thompson Sampling Model

Bandit multi-bras bayésien pour la sélection de numéros.
Chaque numéro est traité comme un "bras" avec une distribution Beta(α, β).

Formule: α_i += 1 si numéro tiré, sinon β_i += 1
         θ_i ~ Beta(α_i, β_i)
         P(i) = E[θ_i] = α_i / (α_i + β_i)
"""

import numpy as np
import pandas as pd
from typing import Dict, List


class M15ThompsonSampling:
    """
    Thompson Sampling model for lottery number prediction.
    
    Treats each number as a multi-armed bandit arm with Beta prior.
    Provides natural exploration/exploitation balance.
    """
    
    def __init__(
        self, 
        rules: Dict, 
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        decay_factor: float = 1.0,
        n_samples: int = 1000
    ):
        """
        Initialize M15 Thompson Sampling model.
        
        Args:
            rules: Game rules dictionary
            prior_alpha: Initial α for Beta prior (1.0 = uniform)
            prior_beta: Initial β for Beta prior
            decay_factor: Decay applied to α, β each step (1.0 = no decay)
            n_samples: Number of samples for probability estimation
        """
        self.rules = rules
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.decay_factor = decay_factor
        self.n_samples = n_samples
        
        # Parse game rules
        main_rules = rules.get("main", rules.get("numbers", {}))
        self.n_min = main_rules.get("min", 1)
        self.n_max = main_rules.get("max", 49)
        self.n_numbers = self.n_max - self.n_min + 1
        
        # Initialize Beta parameters
        self.alphas = None
        self.betas = None
        self.posterior = None
        
    def fit(self, df: pd.DataFrame):
        """
        Train the model on historical draw data.
        
        Args:
            df: DataFrame with 'numbers' column containing lists of drawn numbers
        """
        # Initialize parameters
        self.alphas = np.full(self.n_numbers, self.prior_alpha)
        self.betas = np.full(self.n_numbers, self.prior_beta)
        
        # Update parameters for each draw
        for _, row in df.iterrows():
            # Apply decay
            if self.decay_factor < 1.0:
                self.alphas *= self.decay_factor
                self.betas *= self.decay_factor
            
            # Get drawn numbers
            drawn = set(row["numbers"])
            
            # Update alpha/beta for each number
            for num in range(self.n_min, self.n_max + 1):
                idx = num - self.n_min
                if num in drawn:
                    self.alphas[idx] += 1
                else:
                    self.betas[idx] += 1
            
            # Include bonus numbers
            if "bonus_numbers" in row and row["bonus_numbers"]:
                for num in row["bonus_numbers"]:
                    if self.n_min <= num <= self.n_max:
                        idx = num - self.n_min
                        self.alphas[idx] += 0.5  # Partial credit for bonus
        
        # Calculate posterior probabilities (expected value of Beta)
        expected = self.alphas / (self.alphas + self.betas)
        probs = expected / expected.sum()  # Normalize
        
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
    
    def sample_combination(self, n_pick: int) -> List[int]:
        """
        Sample a combination using Thompson Sampling.
        
        Args:
            n_pick: Number of numbers to select
            
        Returns:
            List of selected numbers
        """
        if self.alphas is None:
            # Not fitted, return random
            return list(np.random.choice(
                range(self.n_min, self.n_max + 1), 
                size=n_pick, 
                replace=False
            ))
        
        # Sample from each Beta distribution
        samples = np.random.beta(self.alphas, self.betas)
        
        # Select top n_pick
        top_indices = np.argsort(samples)[-n_pick:]
        return sorted([self.n_min + idx for idx in top_indices])
    
    def get_params(self) -> Dict:
        """Return model parameters for reproducibility."""
        return {
            "prior_alpha": self.prior_alpha,
            "prior_beta": self.prior_beta,
            "decay_factor": self.decay_factor,
            "n_samples": self.n_samples
        }
