"""
M8 - Changepoint Detection Model

Détection de ruptures structurelles dans les séries de fréquences.
Utilise PELT (Pruned Exact Linear Time) pour identifier les points de rupture.
Après détection, recalcule les probabilités sur le segment post-rupture.

Librairie: ruptures
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False


class M8Changepoint:
    """
    Changepoint Detection model for lottery number prediction.
    
    Identifies structural breaks in frequency series and uses only
    post-breakpoint data for prediction.
    """
    
    def __init__(
        self, 
        rules: Dict, 
        model_type: str = 'l2',
        min_segment_size: int = 20,
        penalty: str = 'bic',
        cusum_threshold: float = 4.0
    ):
        """
        Initialize M8 Changepoint model.
        
        Args:
            rules: Game rules dictionary
            model_type: Cost model for PELT ('l1', 'l2', 'rbf', 'normal')
            min_segment_size: Minimum segment size between breakpoints
            penalty: Penalty for PELT ('bic', 'aic', or numeric value)
            cusum_threshold: Threshold for CUSUM detection (in std devs)
        """
        self.rules = rules
        self.model_type = model_type
        self.min_segment_size = min_segment_size
        self.penalty = penalty
        self.cusum_threshold = cusum_threshold
        
        # Parse game rules
        main_rules = rules.get("main", rules.get("numbers", {}))
        self.n_min = main_rules.get("min", 1)
        self.n_max = main_rules.get("max", 49)
        self.n_pick = main_rules.get("pick", 6)
        self.n_numbers = self.n_max - self.n_min + 1
        
        self.posterior = None
        self.breakpoints = []
        self.last_breakpoint = 0
        
    def _detect_breakpoints(self, signal: np.ndarray) -> List[int]:
        """Detect breakpoints using PELT algorithm."""
        if not RUPTURES_AVAILABLE:
            return []
        
        if len(signal) < 2 * self.min_segment_size:
            return []
        
        try:
            algo = rpt.Pelt(model=self.model_type, min_size=self.min_segment_size)
            algo.fit(signal)
            
            # Determine penalty
            if self.penalty == 'bic':
                pen = np.log(len(signal)) * signal.var()
            elif self.penalty == 'aic':
                pen = 2 * signal.var()
            else:
                pen = float(self.penalty)
            
            breakpoints = algo.predict(pen=pen)
            # Remove the last element (always equals len(signal))
            return [bp for bp in breakpoints if bp < len(signal)]
        except Exception:
            return []
    
    def fit(self, df: pd.DataFrame):
        """
        Train the model on historical draw data.
        
        Args:
            df: DataFrame with 'numbers' column containing lists of drawn numbers
        """
        n_draws = len(df)
        if n_draws < self.min_segment_size:
            # Not enough data, use uniform
            self.posterior = {
                num: 1.0 / self.n_numbers
                for num in range(self.n_min, self.n_max + 1)
            }
            return
        
        # Build cumulative frequency signal
        cumsum = np.zeros((n_draws, self.n_numbers))
        for t, (_, row) in enumerate(df.iterrows()):
            if t > 0:
                cumsum[t] = cumsum[t - 1].copy()
            for num in row["numbers"]:
                if self.n_min <= num <= self.n_max:
                    cumsum[t, num - self.n_min] += 1
        
        # Detect global breakpoint using sum of frequencies
        total_freq = cumsum.sum(axis=1)
        
        # Normalize to get rate
        time_idx = np.arange(1, n_draws + 1)
        rate = total_freq / time_idx
        
        # Detect breakpoints
        if RUPTURES_AVAILABLE and len(rate) >= 2 * self.min_segment_size:
            self.breakpoints = self._detect_breakpoints(rate.reshape(-1, 1))
            if self.breakpoints:
                self.last_breakpoint = max(bp for bp in self.breakpoints if bp < n_draws)
            else:
                self.last_breakpoint = 0
        else:
            self.last_breakpoint = 0
        
        # Use only data after last breakpoint
        if self.last_breakpoint > 0 and self.last_breakpoint < n_draws:
            df_segment = df.iloc[self.last_breakpoint:]
        else:
            df_segment = df
        
        # Calculate frequencies on the segment
        counts = np.zeros(self.n_numbers)
        for _, row in df_segment.iterrows():
            for num in row["numbers"]:
                if self.n_min <= num <= self.n_max:
                    counts[num - self.n_min] += 1
        
        # Normalize to probabilities
        total = counts.sum()
        if total > 0:
            probs = counts / total
        else:
            probs = np.ones(self.n_numbers) / self.n_numbers
        
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
    
    def get_breakpoints(self) -> List[int]:
        """Return detected breakpoints."""
        return self.breakpoints
    
    def get_params(self) -> Dict:
        """Return model parameters for reproducibility."""
        return {
            "model_type": self.model_type,
            "min_segment_size": self.min_segment_size,
            "penalty": self.penalty,
            "cusum_threshold": self.cusum_threshold,
            "last_breakpoint": self.last_breakpoint,
            "breakpoints": self.breakpoints
        }
