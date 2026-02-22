"""
M13 - Spectral / Fourier Analysis Model

Analyse spectrale pour détecter des périodicités dans les tirages.
Utilise la FFT sur les séries binaires de présence/absence.

Formule: X_i(f) = FFT(x_i(t))
         PSD_i(f) = |X_i(f)|² / N
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class M13Spectral:
    """
    Spectral/Fourier Analysis model for lottery number prediction.
    
    Detects periodicities in number occurrence patterns using FFT.
    """
    
    def __init__(
        self, 
        rules: Dict, 
        min_frequency: float = 0.01,
        significance_threshold: float = 0.01,
        n_harmonics: int = 3,
        detrend: bool = True,
        temperature: float = 1.0
    ):
        """
        Initialize M13 Spectral model.
        
        Args:
            rules: Game rules dictionary
            min_frequency: Minimum frequency to analyze
            significance_threshold: Threshold for peak detection
            n_harmonics: Number of harmonics to retain for prediction
            detrend: Whether to remove linear trend before FFT
            temperature: Softmax temperature for probability conversion
        """
        self.rules = rules
        self.min_frequency = min_frequency
        self.significance_threshold = significance_threshold
        self.n_harmonics = n_harmonics
        self.detrend = detrend
        self.temperature = temperature
        
        # Parse game rules
        main_rules = rules.get("main", rules.get("numbers", {}))
        self.n_min = main_rules.get("min", 1)
        self.n_max = main_rules.get("max", 49)
        self.n_pick = main_rules.get("pick", 6)
        self.n_numbers = self.n_max - self.n_min + 1
        
        self.posterior = None
        self.significant_frequencies = {}
        
    def _fisher_g_test(self, psd: np.ndarray) -> float:
        """
        Fisher's g-test for periodicity detection.
        Returns the g statistic (max(PSD) / sum(PSD)).
        """
        if psd.sum() == 0:
            return 0.0
        return psd.max() / psd.sum()
    
    def _detrend_signal(self, signal: np.ndarray) -> np.ndarray:
        """Remove linear trend from signal."""
        n = len(signal)
        t = np.arange(n)
        # Fit linear trend
        slope = np.cov(t, signal)[0, 1] / (np.var(t) + 1e-10)
        intercept = signal.mean() - slope * t.mean()
        trend = slope * t + intercept
        return signal - trend
    
    def fit(self, df: pd.DataFrame):
        """
        Train the model on historical draw data.
        
        Args:
            df: DataFrame with 'numbers' column containing lists of drawn numbers
        """
        n_draws = len(df)
        if n_draws < 32:  # Need enough data for meaningful FFT
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
        
        # Analyze each number's time series
        predictions = np.zeros(self.n_numbers)
        
        for num_idx in range(self.n_numbers):
            signal = presence[:, num_idx]
            
            # Detrend if requested
            if self.detrend:
                signal = self._detrend_signal(signal)
            
            # Apply FFT
            fft_result = np.fft.rfft(signal)
            psd = np.abs(fft_result) ** 2 / n_draws
            freqs = np.fft.rfftfreq(n_draws)
            
            # Filter by minimum frequency
            valid_mask = freqs >= self.min_frequency
            if not valid_mask.any():
                predictions[num_idx] = 0.5
                continue
            
            psd_valid = psd[valid_mask]
            freqs_valid = freqs[valid_mask]
            fft_valid = fft_result[valid_mask]
            
            # Fisher's g-test for significance
            g_stat = self._fisher_g_test(psd_valid)
            
            # If significant periodicity detected, use harmonic extrapolation
            if g_stat > self.significance_threshold and len(psd_valid) > 0:
                # Get top harmonics
                top_indices = np.argsort(psd_valid)[-self.n_harmonics:]
                
                # Extrapolate to next time step
                t_next = n_draws
                prediction = 0.0
                for idx in top_indices:
                    amp = np.abs(fft_valid[idx]) / n_draws
                    phase = np.angle(fft_valid[idx])
                    freq = freqs_valid[idx]
                    prediction += amp * np.cos(2 * np.pi * freq * t_next + phase)
                
                # Normalize to [0, 1]
                predictions[num_idx] = 0.5 + prediction
            else:
                # No significant periodicity, use mean frequency
                predictions[num_idx] = presence[:, num_idx].mean()
        
        # Clip and normalize
        predictions = np.clip(predictions, 0.01, 0.99)
        
        # Convert to probabilities via softmax
        scores = predictions / self.temperature
        exp_scores = np.exp(scores - scores.max())
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
            "min_frequency": self.min_frequency,
            "significance_threshold": self.significance_threshold,
            "n_harmonics": self.n_harmonics,
            "detrend": self.detrend,
            "temperature": self.temperature
        }
