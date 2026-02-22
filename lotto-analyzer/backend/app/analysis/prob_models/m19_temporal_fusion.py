"""
M19 - Temporal Fusion Model

Combines multiple temporal features at different time scales
using attention mechanisms to predict lottery numbers.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class M19Config:
    """Configuration for M19 model"""
    short_window: int = 10
    medium_window: int = 30
    long_window: int = 100
    attention_heads: int = 4
    temperature: float = 1.0
    decay_short: float = 0.1
    decay_medium: float = 0.03
    decay_long: float = 0.01


class M19TemporalFusion:
    """
    Temporal Fusion model for lottery prediction.
    
    Combines information from multiple time scales:
    - Short-term: Recent trends (last 10 draws)
    - Medium-term: Monthly patterns (last 30 draws)
    - Long-term: Historical baseline (last 100+ draws)
    
    Uses attention-like weighting to combine these scales
    based on their recent predictive performance.
    """
    
    model_id = "M19"
    model_name = "Temporal Fusion"
    model_type = "Multi-Scale Temporal"
    
    def __init__(self, config: Optional[M19Config] = None):
        self.config = config or M19Config()
        self.n_max = None
        self.k = None
        self._fitted = False
        self._scale_weights = None
        self._scale_probs = None
    
    def _compute_scale_probabilities(self, draws: List[List[int]], 
                                      window: int,
                                      decay: float) -> np.ndarray:
        """
        Compute probabilities for a specific time scale.
        
        Args:
            draws: Historical draws
            window: Window size for this scale
            decay: Exponential decay rate
            
        Returns:
            Probability array for each number
        """
        if len(draws) == 0:
            return np.ones(self.n_max) / self.n_max
        
        # Use at most 'window' recent draws
        recent = draws[-window:] if len(draws) > window else draws
        
        # Compute weighted frequencies
        freq = np.zeros(self.n_max)
        total_weight = 0
        
        for i, draw in enumerate(recent):
            # Weight decreases for older draws
            weight = np.exp(-decay * (len(recent) - i - 1))
            total_weight += weight
            
            for num in draw:
                if 1 <= num <= self.n_max:
                    freq[num - 1] += weight
        
        # Normalize
        if total_weight > 0:
            freq = freq / total_weight
        
        # Add smoothing
        freq = freq + 0.01
        probs = freq / freq.sum()
        
        return probs
    
    def _compute_attention_weights(self, draws: List[List[int]],
                                    scale_probs: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute attention weights for each time scale based on recent performance.
        
        Uses a simple validation approach: check how well each scale
        predicted the most recent draws.
        """
        if len(draws) < 20:
            # Not enough data, use equal weights
            return {"short": 0.33, "medium": 0.33, "long": 0.34}
        
        # Validate on last 10 draws
        validation_draws = draws[-10:]
        training_draws = draws[:-10]
        
        scores = {}
        
        for scale_name, window, decay in [
            ("short", self.config.short_window, self.config.decay_short),
            ("medium", self.config.medium_window, self.config.decay_medium),
            ("long", self.config.long_window, self.config.decay_long)
        ]:
            # Compute probabilities on training data
            probs = self._compute_scale_probabilities(training_draws, window, decay)
            
            # Score on validation data (log-likelihood)
            score = 0
            for draw in validation_draws:
                for num in draw:
                    if 1 <= num <= self.n_max:
                        score += np.log(probs[num - 1] + 1e-10)
            
            scores[scale_name] = score
        
        # Convert to weights via softmax
        max_score = max(scores.values())
        exp_scores = {k: np.exp((v - max_score) / self.config.temperature) 
                      for k, v in scores.items()}
        total = sum(exp_scores.values())
        
        weights = {k: v / total for k, v in exp_scores.items()}
        
        return weights
    
    def fit(self, draws: List[List[int]], n_max: int, k: int) -> None:
        """
        Fit the temporal fusion model.
        
        Args:
            draws: List of historical draws
            n_max: Maximum number in the pool
            k: Numbers per draw
        """
        self.n_max = n_max
        self.k = k
        self._last_draws = draws
        
        if len(draws) < 20:
            self._fitted = False
            return
        
        # Compute probabilities for each scale
        self._scale_probs = {
            "short": self._compute_scale_probabilities(
                draws, self.config.short_window, self.config.decay_short),
            "medium": self._compute_scale_probabilities(
                draws, self.config.medium_window, self.config.decay_medium),
            "long": self._compute_scale_probabilities(
                draws, self.config.long_window, self.config.decay_long)
        }
        
        # Compute attention weights
        self._scale_weights = self._compute_attention_weights(draws, self._scale_probs)
        
        self._fitted = True
    
    def predict_proba(self, draws: List[List[int]] = None) -> Dict[str, np.ndarray]:
        """
        Predict probabilities by fusing multiple time scales.
        """
        if draws is None:
            draws = getattr(self, '_last_draws', [])
        
        if not self._fitted or self._scale_probs is None:
            return self._fallback_predict(draws)
        
        # Recompute if draws changed
        if draws != self._last_draws:
            self.fit(draws, self.n_max, self.k)
        
        # Weighted combination of scales
        combined = np.zeros(self.n_max)
        
        for scale_name, probs in self._scale_probs.items():
            weight = self._scale_weights.get(scale_name, 0.33)
            combined += weight * probs
        
        # Normalize
        combined = combined / combined.sum()
        
        return {"main": combined}
    
    def _fallback_predict(self, draws: List[List[int]]) -> Dict[str, np.ndarray]:
        """Fallback prediction using simple frequency."""
        if not draws or self.n_max is None:
            n_max = self.n_max or 45
            return {"main": np.ones(n_max) / n_max}
        
        freq = np.zeros(self.n_max)
        for draw in draws:
            for num in draw:
                if 1 <= num <= self.n_max:
                    freq[num - 1] += 1
        
        freq = freq + 1
        probs = freq / freq.sum()
        
        return {"main": probs}
    
    def generate_combinations(self, n: int = 1,
                               draws: List[List[int]] = None) -> List[List[int]]:
        """Generate n combinations based on predicted probabilities."""
        probs = self.predict_proba(draws)["main"]
        
        combinations = []
        for _ in range(n):
            selected = np.random.choice(
                range(1, self.n_max + 1),
                size=self.k,
                replace=False,
                p=probs
            )
            combinations.append(sorted(selected.tolist()))
        
        return combinations
    
    def get_params(self) -> Dict[str, Any]:
        """Return model parameters."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "short_window": self.config.short_window,
            "medium_window": self.config.medium_window,
            "long_window": self.config.long_window,
            "attention_heads": self.config.attention_heads,
            "temperature": self.config.temperature
        }
    
    def get_scale_weights(self) -> Optional[Dict[str, float]]:
        """Get the learned attention weights for each time scale."""
        return self._scale_weights
    
    def get_scale_contributions(self) -> Optional[Dict[str, np.ndarray]]:
        """Get probability contributions from each time scale."""
        if self._scale_probs is None or self._scale_weights is None:
            return None
        
        contributions = {}
        for scale_name, probs in self._scale_probs.items():
            weight = self._scale_weights.get(scale_name, 0.33)
            contributions[scale_name] = probs * weight
        
        return contributions
