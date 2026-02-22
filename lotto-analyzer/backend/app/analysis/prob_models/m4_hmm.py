"""
M4 - Hidden Markov Model (HMM)

Modèle à états cachés pour capturer les changements de régime.
Chaque état a sa propre distribution de probabilité sur les numéros.

Librairie: hmmlearn (optionnel, fallback disponible)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import warnings

try:
    from hmmlearn import hmm
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False
    warnings.warn("hmmlearn not available. M4 HMM will use simplified fallback.")


class M4HMM:
    """
    Hidden Markov Model for lottery number prediction.
    
    Models lottery draws as coming from multiple latent regimes,
    each with its own probability distribution over numbers.
    """
    
    def __init__(
        self, 
        rules: Dict, 
        n_states: int = 3,
        n_iter: int = 100,
        tol: float = 1e-4,
        random_state: int = 42
    ):
        """
        Initialize M4 HMM model.
        
        Args:
            rules: Game rules dictionary
            n_states: Number of hidden states
            n_iter: Maximum iterations for Baum-Welch
            tol: Convergence tolerance
            random_state: Random seed for reproducibility
        """
        self.rules = rules
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        
        # Parse game rules
        main_rules = rules.get("main", rules.get("numbers", {}))
        self.n_min = main_rules.get("min", 1)
        self.n_max = main_rules.get("max", 49)
        self.n_pick = main_rules.get("pick", 6)
        self.n_numbers = self.n_max - self.n_min + 1
        
        self.posterior = None
        self.model = None
        self.emission_probs = None  # P(number | state)
        self.current_state_probs = None  # P(state | observations)
        
    def _build_observation_sequence(self, df: pd.DataFrame) -> np.ndarray:
        """
        Build observation sequence for HMM.
        Each observation is a feature vector representing the draw.
        """
        n_draws = len(df)
        # Use sum and spread as features (dimensionality reduction)
        observations = np.zeros((n_draws, 2))
        
        for t, (_, row) in enumerate(df.iterrows()):
            numbers = row["numbers"]
            if len(numbers) > 0:
                observations[t, 0] = np.mean(numbers) / self.n_max  # Normalized mean
                observations[t, 1] = np.std(numbers) / self.n_max if len(numbers) > 1 else 0
        
        return observations
    
    def _fit_with_hmmlearn(self, observations: np.ndarray, df: pd.DataFrame):
        """Fit using hmmlearn library."""
        # Use Gaussian HMM on the feature representation
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type='diag',
            n_iter=self.n_iter,
            tol=self.tol,
            random_state=self.random_state
        )
        
        # Try multiple initializations
        best_score = -np.inf
        best_model = None
        
        for init in range(5):
            try:
                model = hmm.GaussianHMM(
                    n_components=self.n_states,
                    covariance_type='diag',
                    n_iter=self.n_iter,
                    tol=self.tol,
                    random_state=self.random_state + init
                )
                model.fit(observations)
                score = model.score(observations)
                if score > best_score:
                    best_score = score
                    best_model = model
            except Exception:
                continue
        
        if best_model is None:
            return False
        
        self.model = best_model
        
        # Get state sequence
        state_sequence = self.model.predict(observations)
        
        # Compute emission probabilities for each state
        self.emission_probs = np.zeros((self.n_states, self.n_numbers))
        state_counts = np.zeros(self.n_states)
        
        for t, (_, row) in enumerate(df.iterrows()):
            state = state_sequence[t]
            state_counts[state] += 1
            for num in row["numbers"]:
                if self.n_min <= num <= self.n_max:
                    self.emission_probs[state, num - self.n_min] += 1
        
        # Normalize emission probabilities
        for s in range(self.n_states):
            if state_counts[s] > 0:
                self.emission_probs[s] /= self.emission_probs[s].sum() + 1e-10
            else:
                self.emission_probs[s] = np.ones(self.n_numbers) / self.n_numbers
        
        # Predict next state probabilities
        last_state = state_sequence[-1]
        self.current_state_probs = self.model.transmat_[last_state]
        
        return True
    
    def _fit_fallback(self, df: pd.DataFrame):
        """Simplified fallback without hmmlearn."""
        n_draws = len(df)
        
        # Simple clustering based on draw characteristics
        # Cluster by sum of numbers
        sums = []
        for _, row in df.iterrows():
            sums.append(sum(row["numbers"]))
        sums = np.array(sums)
        
        # K-means-like clustering
        percentiles = np.percentile(sums, np.linspace(0, 100, self.n_states + 1))
        state_sequence = np.digitize(sums, percentiles[1:-1])
        
        # Compute emission probabilities
        self.emission_probs = np.zeros((self.n_states, self.n_numbers))
        state_counts = np.zeros(self.n_states)
        
        for t, (_, row) in enumerate(df.iterrows()):
            state = min(state_sequence[t], self.n_states - 1)
            state_counts[state] += 1
            for num in row["numbers"]:
                if self.n_min <= num <= self.n_max:
                    self.emission_probs[state, num - self.n_min] += 1
        
        # Normalize
        for s in range(self.n_states):
            if self.emission_probs[s].sum() > 0:
                self.emission_probs[s] /= self.emission_probs[s].sum()
            else:
                self.emission_probs[s] = np.ones(self.n_numbers) / self.n_numbers
        
        # Estimate transition matrix
        trans_counts = np.zeros((self.n_states, self.n_states))
        for t in range(len(state_sequence) - 1):
            s1 = min(state_sequence[t], self.n_states - 1)
            s2 = min(state_sequence[t + 1], self.n_states - 1)
            trans_counts[s1, s2] += 1
        
        # Normalize transition matrix
        trans_matrix = trans_counts + 0.1  # Smoothing
        trans_matrix = trans_matrix / trans_matrix.sum(axis=1, keepdims=True)
        
        # Predict next state
        last_state = min(state_sequence[-1], self.n_states - 1)
        self.current_state_probs = trans_matrix[last_state]
    
    def fit(self, df: pd.DataFrame):
        """
        Train the model on historical draw data.
        
        Args:
            df: DataFrame with 'numbers' column containing lists of drawn numbers
        """
        n_draws = len(df)
        if n_draws < 20:
            self.posterior = {
                num: 1.0 / self.n_numbers
                for num in range(self.n_min, self.n_max + 1)
            }
            return
        
        observations = self._build_observation_sequence(df)
        
        if HMMLEARN_AVAILABLE:
            success = self._fit_with_hmmlearn(observations, df)
            if not success:
                self._fit_fallback(df)
        else:
            self._fit_fallback(df)
        
        # Compute final probabilities as weighted sum over states
        probs = np.zeros(self.n_numbers)
        for s in range(self.n_states):
            probs += self.current_state_probs[s] * self.emission_probs[s]
        
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
    
    def get_params(self) -> Dict:
        """Return model parameters for reproducibility."""
        return {
            "n_states": self.n_states,
            "n_iter": self.n_iter,
            "tol": self.tol,
            "random_state": self.random_state,
            "hmmlearn_available": HMMLEARN_AVAILABLE
        }
