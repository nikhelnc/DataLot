"""
M11 LSTM Hybrid Model

Architecture hybride combinant:
- LSTM Bidirectionnel pour capturer la séquentialité temporelle
- Mécanisme d'Attention (Multi-Head) pour l'importance relative des tirages passés
- Embeddings pour les relations latentes entre numéros
- Méta-features (somme, écart-type, pairs/impairs, etc.)

Basé sur l'architecture décrite dans cdc/08_LTSM.md
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings

# Suppress TensorFlow warnings and force CPU usage
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU

try:
    import tensorflow as tf
    # Force CPU usage
    tf.config.set_visible_devices([], 'GPU')
    from tensorflow.keras import layers, models, Input
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    warnings.warn("TensorFlow not available. M11 LSTM model will use fallback mode.")


class M11LSTMHybrid:
    """
    LSTM Hybrid model with Attention mechanism and Embeddings.
    
    Predicts a probability heatmap for all possible numbers.
    """
    
    def __init__(
        self, 
        rules: Dict, 
        sequence_length: int = 50,
        embedding_dim: int = 32,
        lstm_units: int = 64,
        attention_heads: int = 2,
        dropout_rate: float = 0.3,
        epochs: int = 50,
        batch_size: int = 16,
        verbose: int = 0
    ):
        """
        Initialize M11 LSTM Hybrid model.
        
        Args:
            rules: Game rules dictionary
            sequence_length: Number of past draws to use as input
            embedding_dim: Dimension of number embeddings
            lstm_units: Number of LSTM units
            attention_heads: Number of attention heads
            dropout_rate: Dropout rate for regularization
            epochs: Training epochs
            batch_size: Training batch size
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
        """
        self.rules = rules
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.attention_heads = attention_heads
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        
        # Parse game rules
        main_rules = rules.get("main", rules.get("numbers", {}))
        self.n_min = main_rules.get("min", 1)
        self.n_max = main_rules.get("max", 49)
        self.n_pick = main_rules.get("pick", 6)
        self.num_balls = self.n_max - self.n_min + 1
        
        # Bonus rules
        bonus_rules = rules.get("bonus", {})
        self.bonus_enabled = bonus_rules.get("enabled", False)
        self.bonus_min = bonus_rules.get("min", 1)
        self.bonus_max = bonus_rules.get("max", 10)
        self.bonus_pick = bonus_rules.get("pick", 1)
        
        self.model = None
        self.posterior = None
        self.is_fitted = False
        
    def _compute_meta_features(self, numbers: List[int]) -> np.ndarray:
        """
        Compute meta-features for a single draw.
        
        Features:
        - Sum of numbers (normalized)
        - Standard deviation (normalized)
        - Ratio of even numbers
        - Ratio of low numbers (< median)
        - Range (max - min, normalized)
        """
        if not numbers or len(numbers) == 0:
            return np.zeros(5)
        
        arr = np.array(numbers)
        median = (self.n_min + self.n_max) / 2
        max_sum = self.n_pick * self.n_max
        max_range = self.n_max - self.n_min
        
        features = [
            np.sum(arr) / max_sum,  # Normalized sum
            np.std(arr) / (self.n_max / 2) if len(arr) > 1 else 0,  # Normalized std
            np.sum(arr % 2 == 0) / len(arr),  # Even ratio
            np.sum(arr < median) / len(arr),  # Low numbers ratio
            (np.max(arr) - np.min(arr)) / max_range if len(arr) > 1 else 0  # Normalized range
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _prepare_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare input sequences and targets for training.
        
        Returns:
            input_seq: Shape (n_samples, sequence_length, n_pick)
            input_meta: Shape (n_samples, sequence_length, n_features)
            targets: Shape (n_samples, num_balls) - binary multi-label
        """
        numbers_list = df["numbers"].tolist()
        n_draws = len(numbers_list)
        
        if n_draws <= self.sequence_length:
            # Not enough data
            return None, None, None
        
        n_samples = n_draws - self.sequence_length
        n_features = 5  # Number of meta-features
        
        input_seq = np.zeros((n_samples, self.sequence_length, self.n_pick), dtype=np.int32)
        input_meta = np.zeros((n_samples, self.sequence_length, n_features), dtype=np.float32)
        targets = np.zeros((n_samples, self.num_balls), dtype=np.float32)
        
        for i in range(n_samples):
            # Input: sequence of past draws
            for j in range(self.sequence_length):
                draw_idx = i + j
                numbers = numbers_list[draw_idx]
                
                # Pad or truncate to n_pick
                padded = list(numbers[:self.n_pick])
                while len(padded) < self.n_pick:
                    padded.append(0)
                
                input_seq[i, j, :] = padded
                input_meta[i, j, :] = self._compute_meta_features(numbers)
            
            # Target: the next draw (multi-label binary)
            target_draw = numbers_list[i + self.sequence_length]
            for num in target_draw:
                if self.n_min <= num <= self.n_max:
                    targets[i, num - self.n_min] = 1.0
        
        return input_seq, input_meta, targets
    
    def _build_model(self) -> 'tf.keras.Model':
        """
        Build the LSTM Hybrid model architecture.
        """
        if not TENSORFLOW_AVAILABLE:
            return None
        
        # Input 1: Sequence of numbers
        input_seq = Input(shape=(self.sequence_length, self.n_pick), name='input_sequence')
        
        # Embedding layer for each number
        # +1 for padding (0), numbers are 1-indexed
        x = layers.TimeDistributed(
            layers.Embedding(input_dim=self.num_balls + 2, output_dim=self.embedding_dim)
        )(input_seq)
        
        # Flatten embeddings per timestep
        x = layers.TimeDistributed(layers.Flatten())(x)
        
        # Input 2: Meta-features
        input_meta = Input(shape=(self.sequence_length, 5), name='input_meta')
        
        # Concatenate embeddings and meta-features
        combined = layers.Concatenate()([x, input_meta])
        
        # Bidirectional LSTM
        lstm_out = layers.Bidirectional(
            layers.LSTM(self.lstm_units, return_sequences=True, dropout=self.dropout_rate)
        )(combined)
        
        # Multi-Head Attention
        attention = layers.MultiHeadAttention(
            num_heads=self.attention_heads, 
            key_dim=self.lstm_units
        )(lstm_out, lstm_out)
        
        # Residual connection and normalization
        x = layers.LayerNormalization()(lstm_out + attention)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Output: probability for each ball (sigmoid for multi-label)
        output = layers.Dense(self.num_balls, activation='sigmoid', name='output_probs')(x)
        
        model = models.Model(inputs=[input_seq, input_meta], outputs=output)
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def fit(self, df: pd.DataFrame):
        """
        Train the model on historical draw data.
        
        Args:
            df: DataFrame with 'numbers' column containing lists of drawn numbers
        """
        if not TENSORFLOW_AVAILABLE:
            # Fallback: use frequency-based estimation
            self._fit_fallback(df)
            return
        
        # Prepare data
        input_seq, input_meta, targets = self._prepare_sequences(df)
        
        if input_seq is None or len(input_seq) < 10:
            # Not enough data for LSTM, use fallback
            self._fit_fallback(df)
            return
        
        # Build model
        self.model = self._build_model()
        
        # Early stopping
        early_stop = EarlyStopping(
            monitor='loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Train
        try:
            self.model.fit(
                [input_seq, input_meta],
                targets,
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=[early_stop],
                verbose=self.verbose,
                validation_split=0.1
            )
            
            # Get predictions for the last sequence
            last_seq = input_seq[-1:] if len(input_seq) > 0 else None
            last_meta = input_meta[-1:] if len(input_meta) > 0 else None
            
            if last_seq is not None:
                probs = self.model.predict([last_seq, last_meta], verbose=0)[0]
                self.posterior = {
                    num: float(probs[num - self.n_min])
                    for num in range(self.n_min, self.n_max + 1)
                }
            else:
                self._fit_fallback(df)
                
            self.is_fitted = True
            
        except Exception as e:
            warnings.warn(f"LSTM training failed: {e}. Using fallback.")
            self._fit_fallback(df)
    
    def _fit_fallback(self, df: pd.DataFrame):
        """
        Fallback fitting using frequency-based estimation.
        """
        all_numbers = []
        for numbers in df["numbers"]:
            all_numbers.extend(numbers)
        
        counts = pd.Series(all_numbers).value_counts()
        total = len(all_numbers)
        
        self.posterior = {}
        for num in range(self.n_min, self.n_max + 1):
            count = counts.get(num, 0)
            self.posterior[num] = count / total if total > 0 else 1.0 / self.num_balls
        
        self.is_fitted = True
    
    def predict(self) -> Dict[str, float]:
        """
        Return probability distribution over all numbers.
        
        Returns:
            Dictionary mapping number (as string) to probability
        """
        if self.posterior is None:
            # Return uniform distribution
            prob = 1.0 / self.num_balls
            return {str(num): prob for num in range(self.n_min, self.n_max + 1)}
        
        return {str(num): float(prob) for num, prob in self.posterior.items()}
    
    def predict_next(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Predict probabilities for the next draw given recent history.
        
        Args:
            df: DataFrame with recent draws
            
        Returns:
            Dictionary mapping number (as string) to probability
        """
        if not TENSORFLOW_AVAILABLE or self.model is None:
            return self.predict()
        
        # Prepare the last sequence
        numbers_list = df["numbers"].tolist()
        
        if len(numbers_list) < self.sequence_length:
            return self.predict()
        
        # Take the last sequence_length draws
        recent = numbers_list[-self.sequence_length:]
        
        input_seq = np.zeros((1, self.sequence_length, self.n_pick), dtype=np.int32)
        input_meta = np.zeros((1, self.sequence_length, 5), dtype=np.float32)
        
        for j, numbers in enumerate(recent):
            padded = list(numbers[:self.n_pick])
            while len(padded) < self.n_pick:
                padded.append(0)
            input_seq[0, j, :] = padded
            input_meta[0, j, :] = self._compute_meta_features(numbers)
        
        # Predict
        probs = self.model.predict([input_seq, input_meta], verbose=0)[0]
        
        return {
            str(num): float(probs[num - self.n_min])
            for num in range(self.n_min, self.n_max + 1)
        }
    
    def get_top_n(self, n: int = 6) -> List[int]:
        """
        Get the top N most probable numbers.
        
        Args:
            n: Number of top predictions to return
            
        Returns:
            List of top N numbers sorted by probability
        """
        if self.posterior is None:
            return list(range(self.n_min, self.n_min + n))
        
        sorted_nums = sorted(self.posterior.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:n]]
