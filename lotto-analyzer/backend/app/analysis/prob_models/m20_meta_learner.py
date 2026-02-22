"""
M20 - Meta-Learner Adaptive Model

A meta-learning model that dynamically selects and weights
the best performing models based on recent performance.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class M20Config:
    """Configuration for M20 model"""
    validation_window: int = 20
    min_history: int = 50
    temperature: float = 1.0
    decay_factor: float = 0.95
    n_top_models: int = 5


class M20MetaLearner:
    """
    Meta-Learner that adaptively combines multiple prediction models.
    
    This model:
    1. Maintains a pool of base models (M1-M19)
    2. Evaluates each model's recent performance
    3. Dynamically weights models based on performance
    4. Combines predictions using learned weights
    
    The meta-learner adapts over time, giving more weight to
    models that have been performing well recently.
    """
    
    model_id = "M20"
    model_name = "Meta-Learner Adaptive"
    model_type = "Meta-Learning"
    
    def __init__(self, config: Optional[M20Config] = None, 
                 base_models: Optional[Dict[str, Any]] = None):
        self.config = config or M20Config()
        self.base_models = base_models or {}
        self.n_max = None
        self.k = None
        self._fitted = False
        self._model_weights = {}
        self._model_scores = {}
        self._performance_history = {}
    
    def _get_default_models(self) -> Dict[str, Any]:
        """
        Get default base models.
        Returns lightweight model implementations.
        """
        models = {}
        
        # M1 - Dirichlet
        models["M1"] = DirichletModel()
        
        # M2 - Windowed
        models["M2"] = WindowedModel()
        
        # M3 - Exponential Decay
        models["M3"] = ExponentialDecayModel()
        
        # M6 - Gaps
        models["M6"] = GapsModel()
        
        # M19 - Temporal Fusion (if available)
        try:
            from .m19_temporal_fusion import M19TemporalFusion
            models["M19"] = M19TemporalFusion()
        except ImportError:
            pass
        
        return models
    
    def _evaluate_model(self, model_name: str, model: Any,
                        training_draws: List[List[int]],
                        validation_draws: List[List[int]]) -> float:
        """
        Evaluate a model's performance on validation data.
        
        Returns log-likelihood score (higher is better).
        """
        try:
            # Fit model on training data
            if hasattr(model, 'fit'):
                model.fit(training_draws, self.n_max, self.k)
            
            # Get predictions
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(training_draws)
                if isinstance(probs, dict):
                    probs = probs.get("main", probs)
            else:
                return -1000  # Invalid model
            
            # Score on validation data
            score = 0
            for draw in validation_draws:
                for num in draw:
                    if 1 <= num <= self.n_max:
                        prob = probs[num - 1] if num - 1 < len(probs) else 0.01
                        score += np.log(max(prob, 1e-10))
            
            return score
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            return -1000
    
    def _compute_model_weights(self) -> Dict[str, float]:
        """
        Compute model weights based on performance scores.
        Uses softmax with temperature.
        """
        if not self._model_scores:
            return {}
        
        # Get top N models
        sorted_models = sorted(self._model_scores.items(), 
                               key=lambda x: x[1], reverse=True)
        top_models = sorted_models[:self.config.n_top_models]
        
        if not top_models:
            return {}
        
        # Softmax
        scores = np.array([s for _, s in top_models])
        max_score = scores.max()
        exp_scores = np.exp((scores - max_score) / self.config.temperature)
        weights = exp_scores / exp_scores.sum()
        
        return {name: float(w) for (name, _), w in zip(top_models, weights)}
    
    def fit(self, draws: List[List[int]], n_max: int, k: int) -> None:
        """
        Fit the meta-learner.
        
        Args:
            draws: List of historical draws
            n_max: Maximum number in the pool
            k: Numbers per draw
        """
        self.n_max = n_max
        self.k = k
        self._last_draws = draws
        
        if len(draws) < self.config.min_history:
            self._fitted = False
            return
        
        # Get base models
        if not self.base_models:
            self.base_models = self._get_default_models()
        
        # Split data
        val_size = self.config.validation_window
        training_draws = draws[:-val_size]
        validation_draws = draws[-val_size:]
        
        # Evaluate each model
        self._model_scores = {}
        for model_name, model in self.base_models.items():
            score = self._evaluate_model(
                model_name, model, training_draws, validation_draws
            )
            self._model_scores[model_name] = score
            
            # Update performance history
            if model_name not in self._performance_history:
                self._performance_history[model_name] = []
            self._performance_history[model_name].append(score)
        
        # Compute weights
        self._model_weights = self._compute_model_weights()
        
        # Fit all models on full data for prediction
        for model_name, model in self.base_models.items():
            if model_name in self._model_weights:
                try:
                    if hasattr(model, 'fit'):
                        model.fit(draws, n_max, k)
                except Exception:
                    pass
        
        self._fitted = True
    
    def predict_proba(self, draws: List[List[int]] = None) -> Dict[str, np.ndarray]:
        """
        Predict probabilities by combining weighted model predictions.
        """
        if draws is None:
            draws = getattr(self, '_last_draws', [])
        
        if not self._fitted or not self._model_weights:
            return self._fallback_predict(draws)
        
        # Combine predictions
        combined = np.zeros(self.n_max)
        total_weight = 0
        
        for model_name, weight in self._model_weights.items():
            if model_name not in self.base_models:
                continue
            
            model = self.base_models[model_name]
            
            try:
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(draws)
                    if isinstance(probs, dict):
                        probs = probs.get("main", np.ones(self.n_max) / self.n_max)
                    
                    combined += weight * probs
                    total_weight += weight
            except Exception:
                pass
        
        # Normalize
        if total_weight > 0:
            combined = combined / total_weight
        else:
            combined = np.ones(self.n_max) / self.n_max
        
        combined = combined / combined.sum()
        
        return {"main": combined}
    
    def _fallback_predict(self, draws: List[List[int]]) -> Dict[str, np.ndarray]:
        """Fallback prediction using frequency."""
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
            "validation_window": self.config.validation_window,
            "n_top_models": self.config.n_top_models,
            "temperature": self.config.temperature,
            "base_models": list(self.base_models.keys())
        }
    
    def get_model_weights(self) -> Dict[str, float]:
        """Get current model weights."""
        return self._model_weights.copy()
    
    def get_model_scores(self) -> Dict[str, float]:
        """Get model performance scores."""
        return self._model_scores.copy()
    
    def get_performance_history(self) -> Dict[str, List[float]]:
        """Get historical performance for each model."""
        return self._performance_history.copy()


# Lightweight base model implementations for the meta-learner

class DirichletModel:
    """Simple Dirichlet model (M1)."""
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.n_max = None
        self.freq = None
    
    def fit(self, draws: List[List[int]], n_max: int, k: int) -> None:
        self.n_max = n_max
        self.freq = np.zeros(n_max)
        for draw in draws:
            for num in draw:
                if 1 <= num <= n_max:
                    self.freq[num - 1] += 1
    
    def predict_proba(self, draws: List[List[int]] = None) -> Dict[str, np.ndarray]:
        if self.freq is None:
            return {"main": np.ones(self.n_max) / self.n_max}
        probs = (self.freq + self.alpha) / (self.freq.sum() + self.n_max * self.alpha)
        return {"main": probs}


class WindowedModel:
    """Simple windowed model (M2)."""
    
    def __init__(self, window: int = 50, shrinkage: float = 0.1):
        self.window = window
        self.shrinkage = shrinkage
        self.n_max = None
        self.global_freq = None
        self.window_freq = None
    
    def fit(self, draws: List[List[int]], n_max: int, k: int) -> None:
        self.n_max = n_max
        
        # Global frequency
        self.global_freq = np.zeros(n_max)
        for draw in draws:
            for num in draw:
                if 1 <= num <= n_max:
                    self.global_freq[num - 1] += 1
        self.global_freq = self.global_freq / len(draws) if draws else self.global_freq
        
        # Window frequency
        recent = draws[-self.window:] if len(draws) > self.window else draws
        self.window_freq = np.zeros(n_max)
        for draw in recent:
            for num in draw:
                if 1 <= num <= n_max:
                    self.window_freq[num - 1] += 1
        self.window_freq = self.window_freq / len(recent) if recent else self.window_freq
    
    def predict_proba(self, draws: List[List[int]] = None) -> Dict[str, np.ndarray]:
        if self.global_freq is None:
            return {"main": np.ones(self.n_max) / self.n_max}
        
        probs = self.shrinkage * self.global_freq + (1 - self.shrinkage) * self.window_freq
        probs = probs / probs.sum()
        return {"main": probs}


class ExponentialDecayModel:
    """Simple exponential decay model (M3)."""
    
    def __init__(self, decay: float = 0.02):
        self.decay = decay
        self.n_max = None
        self.weighted_freq = None
    
    def fit(self, draws: List[List[int]], n_max: int, k: int) -> None:
        self.n_max = n_max
        self.weighted_freq = np.zeros(n_max)
        
        total_weight = 0
        for i, draw in enumerate(draws):
            weight = np.exp(-self.decay * (len(draws) - i - 1))
            total_weight += weight
            for num in draw:
                if 1 <= num <= n_max:
                    self.weighted_freq[num - 1] += weight
        
        if total_weight > 0:
            self.weighted_freq = self.weighted_freq / total_weight
    
    def predict_proba(self, draws: List[List[int]] = None) -> Dict[str, np.ndarray]:
        if self.weighted_freq is None:
            return {"main": np.ones(self.n_max) / self.n_max}
        
        probs = self.weighted_freq + 0.001
        probs = probs / probs.sum()
        return {"main": probs}


class GapsModel:
    """Simple gaps model (M6)."""
    
    def __init__(self):
        self.n_max = None
        self.gap_scores = None
    
    def fit(self, draws: List[List[int]], n_max: int, k: int) -> None:
        self.n_max = n_max
        self.gap_scores = np.zeros(n_max)
        
        for num in range(1, n_max + 1):
            # Find last appearance
            last_idx = -1
            for i in range(len(draws) - 1, -1, -1):
                if num in draws[i]:
                    last_idx = i
                    break
            
            current_gap = len(draws) - last_idx - 1 if last_idx >= 0 else len(draws)
            
            # Find mean gap
            appearances = [i for i, d in enumerate(draws) if num in d]
            if len(appearances) >= 2:
                gaps = [appearances[j+1] - appearances[j] for j in range(len(appearances)-1)]
                mean_gap = np.mean(gaps)
            else:
                mean_gap = len(draws) / max(1, len(appearances))
            
            # Gap ratio
            self.gap_scores[num - 1] = current_gap / mean_gap if mean_gap > 0 else 1.0
    
    def predict_proba(self, draws: List[List[int]] = None) -> Dict[str, np.ndarray]:
        if self.gap_scores is None:
            return {"main": np.ones(self.n_max) / self.n_max}
        
        # Higher gap ratio = more "overdue"
        probs = self.gap_scores + 0.1
        probs = probs / probs.sum()
        return {"main": probs}
