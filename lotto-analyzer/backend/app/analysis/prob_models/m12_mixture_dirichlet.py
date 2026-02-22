"""
M12 - Mixture of Dirichlet Model

Mélange de distributions Dirichlet pour capturer l'hétérogénéité.
Chaque composante représente un "mode" potentiel de la loterie.

Estimation par algorithme EM.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy.special import digamma, gammaln


class M12MixtureDirichlet:
    """
    Mixture of Dirichlet model for lottery number prediction.
    
    Uses multiple Dirichlet components to capture heterogeneity
    in the lottery draw distribution.
    """
    
    def __init__(
        self, 
        rules: Dict, 
        n_components: int = 2,
        alpha_prior: float = 1.0,
        n_iter: int = 100,
        tol: float = 1e-4
    ):
        """
        Initialize M12 Mixture of Dirichlet model.
        
        Args:
            rules: Game rules dictionary
            n_components: Number of mixture components
            alpha_prior: Prior concentration for each component
            n_iter: Maximum EM iterations
            tol: Convergence tolerance
        """
        self.rules = rules
        self.n_components = n_components
        self.alpha_prior = alpha_prior
        self.n_iter = n_iter
        self.tol = tol
        
        # Parse game rules
        main_rules = rules.get("main", rules.get("numbers", {}))
        self.n_min = main_rules.get("min", 1)
        self.n_max = main_rules.get("max", 49)
        self.n_numbers = self.n_max - self.n_min + 1
        
        self.posterior = None
        self.weights = None  # Mixture weights
        self.alphas = None   # Dirichlet parameters for each component
        
    def _dirichlet_log_likelihood(self, x: np.ndarray, alpha: np.ndarray) -> float:
        """Calculate log-likelihood of x under Dirichlet(alpha)."""
        # x is a frequency vector (sums to 1)
        # Avoid log(0)
        x = np.clip(x, 1e-10, 1.0)
        x = x / x.sum()
        
        ll = gammaln(alpha.sum()) - gammaln(alpha).sum()
        ll += ((alpha - 1) * np.log(x)).sum()
        return ll
    
    def fit(self, df: pd.DataFrame):
        """
        Train the model on historical draw data using EM.
        
        Args:
            df: DataFrame with 'numbers' column containing lists of drawn numbers
        """
        n_draws = len(df)
        if n_draws < 10:
            # Not enough data, use uniform
            self.posterior = {
                num: 1.0 / self.n_numbers
                for num in range(self.n_min, self.n_max + 1)
            }
            return
        
        # Build frequency vectors for each draw
        freq_vectors = []
        for _, row in df.iterrows():
            freq = np.zeros(self.n_numbers)
            for num in row["numbers"]:
                if self.n_min <= num <= self.n_max:
                    freq[num - self.n_min] = 1
            # Normalize
            if freq.sum() > 0:
                freq = freq / freq.sum()
            else:
                freq = np.ones(self.n_numbers) / self.n_numbers
            freq_vectors.append(freq)
        
        X = np.array(freq_vectors)
        
        # Initialize components using K-means-like approach
        np.random.seed(42)
        indices = np.random.choice(n_draws, self.n_components, replace=False)
        self.alphas = np.array([X[i] * self.n_numbers + self.alpha_prior for i in indices])
        self.weights = np.ones(self.n_components) / self.n_components
        
        # EM algorithm
        prev_ll = -np.inf
        for iteration in range(self.n_iter):
            # E-step: compute responsibilities
            responsibilities = np.zeros((n_draws, self.n_components))
            for i in range(n_draws):
                for k in range(self.n_components):
                    ll = self._dirichlet_log_likelihood(X[i], self.alphas[k])
                    responsibilities[i, k] = np.log(self.weights[k] + 1e-10) + ll
                
                # Normalize (log-sum-exp trick)
                max_r = responsibilities[i].max()
                responsibilities[i] = np.exp(responsibilities[i] - max_r)
                responsibilities[i] /= responsibilities[i].sum()
            
            # M-step: update weights
            self.weights = responsibilities.sum(axis=0) / n_draws
            
            # M-step: update alphas (simplified - use weighted mean)
            for k in range(self.n_components):
                weighted_mean = (responsibilities[:, k:k+1] * X).sum(axis=0)
                weighted_mean /= responsibilities[:, k].sum() + 1e-10
                # Update alpha using method of moments
                self.alphas[k] = weighted_mean * self.n_numbers + self.alpha_prior
            
            # Check convergence
            total_ll = 0
            for i in range(n_draws):
                ll_i = 0
                for k in range(self.n_components):
                    ll_i += self.weights[k] * np.exp(
                        self._dirichlet_log_likelihood(X[i], self.alphas[k])
                    )
                total_ll += np.log(ll_i + 1e-10)
            
            if abs(total_ll - prev_ll) < self.tol:
                break
            prev_ll = total_ll
        
        # Compute posterior: weighted average of expected values
        expected = np.zeros(self.n_numbers)
        for k in range(self.n_components):
            alpha_sum = self.alphas[k].sum()
            expected += self.weights[k] * (self.alphas[k] / alpha_sum)
        
        # Normalize
        probs = expected / expected.sum()
        
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
            "n_components": self.n_components,
            "alpha_prior": self.alpha_prior,
            "n_iter": self.n_iter,
            "tol": self.tol,
            "weights": self.weights.tolist() if self.weights is not None else None
        }
