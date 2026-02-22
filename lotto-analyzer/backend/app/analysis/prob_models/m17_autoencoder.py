"""
M17 - Autoencoder Anomaly Detection Model

Uses an autoencoder neural network to detect anomalous patterns
in lottery draws and predict numbers based on reconstruction error.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TF = True
except ImportError:
    HAS_TF = False


@dataclass
class M17Config:
    """Configuration for M17 model"""
    encoding_dim: int = 16
    hidden_layers: List[int] = None
    epochs: int = 50
    batch_size: int = 32
    sequence_length: int = 20
    anomaly_threshold: float = 0.1
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [32, 16]


class M17Autoencoder:
    """
    Autoencoder-based anomaly detection for lottery prediction.
    
    The model learns to reconstruct "normal" lottery draw patterns.
    Numbers with high reconstruction error are considered anomalous
    and may be more likely to appear (deviation from expected pattern).
    
    Architecture:
    - Encoder: Compresses draw sequences into latent space
    - Decoder: Reconstructs the original sequence
    - Anomaly score: Based on reconstruction error
    """
    
    model_id = "M17"
    model_name = "Autoencoder Anomaly"
    model_type = "Deep Learning"
    
    def __init__(self, config: Optional[M17Config] = None):
        self.config = config or M17Config()
        self.autoencoder = None
        self.encoder = None
        self.n_max = None
        self.k = None
        self._fitted = False
        self._reconstruction_errors = None
    
    def _build_model(self, input_dim: int) -> None:
        """Build the autoencoder architecture."""
        if not HAS_TF:
            return
        
        # Encoder
        input_layer = keras.layers.Input(shape=(input_dim,))
        x = input_layer
        
        for units in self.config.hidden_layers:
            x = keras.layers.Dense(units, activation='relu')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.2)(x)
        
        # Latent space
        latent = keras.layers.Dense(self.config.encoding_dim, activation='relu', 
                                     name='latent')(x)
        
        # Decoder
        x = latent
        for units in reversed(self.config.hidden_layers):
            x = keras.layers.Dense(units, activation='relu')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.2)(x)
        
        # Output
        output_layer = keras.layers.Dense(input_dim, activation='sigmoid')(x)
        
        # Models
        self.autoencoder = keras.Model(input_layer, output_layer)
        self.encoder = keras.Model(input_layer, latent)
        
        self.autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy'
        )
    
    def _prepare_sequences(self, draws: List[List[int]]) -> np.ndarray:
        """
        Convert draws to binary matrix sequences.
        
        Each draw is represented as a binary vector of length n_max.
        """
        n_draws = len(draws)
        X = np.zeros((n_draws, self.n_max))
        
        for i, draw in enumerate(draws):
            for num in draw:
                if 1 <= num <= self.n_max:
                    X[i, num - 1] = 1
        
        return X
    
    def _create_sliding_windows(self, X: np.ndarray) -> np.ndarray:
        """Create sliding window sequences for training."""
        n_samples = len(X) - self.config.sequence_length + 1
        if n_samples <= 0:
            return X
        
        # Flatten sequences into single vectors
        window_size = self.config.sequence_length * self.n_max
        windows = np.zeros((n_samples, window_size))
        
        for i in range(n_samples):
            windows[i] = X[i:i + self.config.sequence_length].flatten()
        
        return windows
    
    def fit(self, draws: List[List[int]], n_max: int, k: int) -> None:
        """
        Train the autoencoder model.
        
        Args:
            draws: List of historical draws
            n_max: Maximum number in the pool
            k: Numbers per draw
        """
        self.n_max = n_max
        self.k = k
        
        if not HAS_TF:
            print("Warning: TensorFlow not available. Using fallback.")
            self._fitted = False
            self._last_draws = draws
            return
        
        if len(draws) < self.config.sequence_length + 10:
            print("Warning: Not enough draws for autoencoder training.")
            self._fitted = False
            self._last_draws = draws
            return
        
        # Prepare data
        X = self._prepare_sequences(draws)
        X_windows = self._create_sliding_windows(X)
        
        # Build model
        input_dim = X_windows.shape[1]
        self._build_model(input_dim)
        
        # Train
        self.autoencoder.fit(
            X_windows, X_windows,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_split=0.1,
            verbose=0
        )
        
        # Compute reconstruction errors for each number
        self._compute_number_errors(draws)
        
        self._fitted = True
        self._last_draws = draws
    
    def _compute_number_errors(self, draws: List[List[int]]) -> None:
        """Compute reconstruction error contribution for each number."""
        if not HAS_TF or self.autoencoder is None:
            return
        
        X = self._prepare_sequences(draws)
        X_windows = self._create_sliding_windows(X)
        
        # Get reconstructions
        X_reconstructed = self.autoencoder.predict(X_windows, verbose=0)
        
        # Compute per-number errors
        errors = np.abs(X_windows - X_reconstructed)
        
        # Reshape to get per-number errors
        n_samples = errors.shape[0]
        errors_reshaped = errors.reshape(n_samples, self.config.sequence_length, self.n_max)
        
        # Average error per number across all samples and time steps
        self._reconstruction_errors = errors_reshaped.mean(axis=(0, 1))
    
    def predict_proba(self, draws: List[List[int]] = None) -> Dict[str, np.ndarray]:
        """
        Predict probabilities based on reconstruction errors.
        
        Numbers with higher reconstruction error are considered
        more "anomalous" and potentially more likely to appear.
        """
        if draws is None:
            draws = getattr(self, '_last_draws', [])
        
        if not self._fitted or self._reconstruction_errors is None:
            # Fallback to frequency-based
            return self._fallback_predict(draws)
        
        # Use reconstruction errors as anomaly scores
        # Higher error = more anomalous = potentially more likely
        errors = self._reconstruction_errors
        
        # Combine with frequency information
        freq = np.zeros(self.n_max)
        for draw in draws:
            for num in draw:
                if 1 <= num <= self.n_max:
                    freq[num - 1] += 1
        freq = freq / len(draws) if draws else freq
        
        # Combine: weight anomaly score with frequency
        # Numbers that are anomalous AND have reasonable frequency
        combined = errors * (freq + 0.1)  # Add small constant to avoid zeros
        
        # Normalize
        probs = combined / combined.sum()
        
        return {"main": probs}
    
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
        
        # Add smoothing
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
            "encoding_dim": self.config.encoding_dim,
            "hidden_layers": self.config.hidden_layers,
            "epochs": self.config.epochs,
            "sequence_length": self.config.sequence_length,
            "has_tensorflow": HAS_TF
        }
    
    def get_latent_representation(self, draws: List[List[int]]) -> Optional[np.ndarray]:
        """Get the latent space representation of draws."""
        if not self._fitted or self.encoder is None:
            return None
        
        X = self._prepare_sequences(draws)
        X_windows = self._create_sliding_windows(X)
        
        return self.encoder.predict(X_windows, verbose=0)
    
    def get_anomaly_scores(self) -> Optional[np.ndarray]:
        """Get anomaly scores for each number."""
        return self._reconstruction_errors
