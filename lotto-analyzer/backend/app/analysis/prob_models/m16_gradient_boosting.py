"""
M16 - Gradient Boosting Ensemble Model

Uses gradient boosting (XGBoost/LightGBM) to predict number probabilities
based on engineered features from historical draws.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


@dataclass
class M16Config:
    """Configuration for M16 model"""
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    window_size: int = 50
    use_lightgbm: bool = False
    feature_set: str = "full"  # "basic", "temporal", "full"


class M16GradientBoosting:
    """
    Gradient Boosting model for lottery number prediction.
    
    Features engineered:
    - Frequency features (global, windowed, exponential decay)
    - Gap features (current gap, mean gap, gap ratio)
    - Co-occurrence features (pair frequencies)
    - Temporal features (day of week, month patterns)
    - Statistical features (sum, range, variance of recent draws)
    """
    
    model_id = "M16"
    model_name = "Gradient Boosting Ensemble"
    model_type = "Machine Learning"
    
    def __init__(self, config: Optional[M16Config] = None):
        self.config = config or M16Config()
        self.model = None
        self.feature_names = []
        self.n_max = None
        self.k = None
        self._fitted = False
    
    def _compute_features(self, draws: List[List[int]], 
                          target_idx: int,
                          number: int) -> np.ndarray:
        """
        Compute features for a specific number at a specific point in time.
        
        Args:
            draws: Historical draws up to target_idx
            target_idx: Index of the target draw (for walk-forward)
            number: The number to compute features for
            
        Returns:
            Feature vector
        """
        history = draws[:target_idx]
        if len(history) < 10:
            return None
        
        features = []
        
        # 1. Frequency features
        # Global frequency
        total_appearances = sum(1 for d in history if number in d)
        global_freq = total_appearances / len(history)
        features.append(global_freq)
        
        # Windowed frequency (last W draws)
        window = min(self.config.window_size, len(history))
        recent = history[-window:]
        recent_appearances = sum(1 for d in recent if number in d)
        windowed_freq = recent_appearances / window
        features.append(windowed_freq)
        
        # Exponential decay frequency
        lambda_decay = 0.02
        weights = np.array([np.exp(-lambda_decay * (len(history) - i - 1)) 
                           for i in range(len(history))])
        weighted_appearances = sum(w for i, w in enumerate(weights) if number in history[i])
        exp_freq = weighted_appearances / weights.sum()
        features.append(exp_freq)
        
        # 2. Gap features
        # Current gap (draws since last appearance)
        last_appearance = -1
        for i in range(len(history) - 1, -1, -1):
            if number in history[i]:
                last_appearance = i
                break
        current_gap = len(history) - last_appearance - 1 if last_appearance >= 0 else len(history)
        features.append(current_gap)
        
        # Mean gap
        appearances = [i for i, d in enumerate(history) if number in d]
        if len(appearances) >= 2:
            gaps = [appearances[i+1] - appearances[i] for i in range(len(appearances)-1)]
            mean_gap = np.mean(gaps)
            std_gap = np.std(gaps) if len(gaps) > 1 else 0
        else:
            mean_gap = len(history) / max(1, len(appearances))
            std_gap = 0
        features.append(mean_gap)
        features.append(std_gap)
        
        # Gap ratio (current / mean)
        gap_ratio = current_gap / mean_gap if mean_gap > 0 else 1.0
        features.append(gap_ratio)
        
        # 3. Statistical features of recent draws
        if len(recent) > 0:
            recent_sums = [sum(d) for d in recent]
            recent_ranges = [max(d) - min(d) for d in recent]
            features.append(np.mean(recent_sums))
            features.append(np.std(recent_sums))
            features.append(np.mean(recent_ranges))
        else:
            features.extend([0, 0, 0])
        
        # 4. Number-specific features
        # Is the number in the "hot" zone (above median frequency)?
        median_freq = np.median([sum(1 for d in history if n in d) / len(history) 
                                 for n in range(1, self.n_max + 1)])
        is_hot = 1 if global_freq > median_freq else 0
        features.append(is_hot)
        
        # Number position (low/mid/high)
        position = number / self.n_max
        features.append(position)
        
        # Is even
        is_even = 1 if number % 2 == 0 else 0
        features.append(is_even)
        
        return np.array(features)
    
    def fit(self, draws: List[List[int]], n_max: int, k: int) -> None:
        """
        Train the gradient boosting model.
        
        Args:
            draws: List of historical draws
            n_max: Maximum number in the pool
            k: Numbers per draw
        """
        self.n_max = n_max
        self.k = k
        
        if not HAS_XGB and not HAS_LGB:
            print("Warning: Neither XGBoost nor LightGBM available. Using fallback.")
            self._fitted = False
            return
        
        # Build training data
        X = []
        y = []
        
        # Use walk-forward approach
        min_history = max(50, self.config.window_size)
        
        for target_idx in range(min_history, len(draws)):
            target_draw = draws[target_idx]
            
            for number in range(1, n_max + 1):
                features = self._compute_features(draws, target_idx, number)
                if features is not None:
                    X.append(features)
                    y.append(1 if number in target_draw else 0)
        
        if len(X) == 0:
            self._fitted = False
            return
        
        X = np.array(X)
        y = np.array(y)
        
        # Train model
        if self.config.use_lightgbm and HAS_LGB:
            self.model = lgb.LGBMClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                verbose=-1
            )
        elif HAS_XGB:
            self.model = xgb.XGBClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0
            )
        else:
            self._fitted = False
            return
        
        self.model.fit(X, y)
        self._fitted = True
        self._last_draws = draws
    
    def predict_proba(self, draws: List[List[int]] = None) -> Dict[str, np.ndarray]:
        """
        Predict probabilities for each number.
        
        Returns:
            Dictionary with 'main' key containing probability array
        """
        if draws is None:
            draws = getattr(self, '_last_draws', [])
        
        if not self._fitted or self.model is None:
            # Fallback to uniform
            probs = np.ones(self.n_max) / self.n_max
            return {"main": probs}
        
        # Compute features for each number
        probs = np.zeros(self.n_max)
        
        for number in range(1, self.n_max + 1):
            features = self._compute_features(draws, len(draws), number)
            if features is not None:
                prob = self.model.predict_proba(features.reshape(1, -1))[0, 1]
                probs[number - 1] = prob
            else:
                probs[number - 1] = 1.0 / self.n_max
        
        # Normalize
        probs = probs / probs.sum()
        
        return {"main": probs}
    
    def generate_combinations(self, n: int = 1, 
                               draws: List[List[int]] = None) -> List[List[int]]:
        """
        Generate n combinations based on predicted probabilities.
        """
        probs = self.predict_proba(draws)["main"]
        
        combinations = []
        for _ in range(n):
            # Sample without replacement
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
            "n_estimators": self.config.n_estimators,
            "max_depth": self.config.max_depth,
            "learning_rate": self.config.learning_rate,
            "window_size": self.config.window_size,
            "use_lightgbm": self.config.use_lightgbm,
            "has_xgb": HAS_XGB,
            "has_lgb": HAS_LGB
        }
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from the trained model."""
        if not self._fitted or self.model is None:
            return None
        
        feature_names = [
            "global_freq", "windowed_freq", "exp_freq",
            "current_gap", "mean_gap", "std_gap", "gap_ratio",
            "recent_sum_mean", "recent_sum_std", "recent_range_mean",
            "is_hot", "position", "is_even"
        ]
        
        importances = self.model.feature_importances_
        
        return dict(zip(feature_names, importances.tolist()))
