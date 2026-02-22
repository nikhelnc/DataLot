import numpy as np
import pandas as pd
from typing import Dict, Any


class WalkForwardEvaluator:
    def __init__(self, df: pd.DataFrame, rules: Dict, model):
        self.df = df
        self.rules = rules
        self.model = model
        # Support both old (numbers) and new (main) structure
        main_rules = rules.get("main", rules.get("numbers", {}))
        self.n_min = main_rules.get("min", 1)
        self.n_max = main_rules.get("max", 49)
        self.n_count = main_rules.get("pick", main_rules.get("count", 6))

    def evaluate(self) -> Dict[str, Any]:
        min_train = 30
        if len(self.df) < min_train + 10:
            return {
                "warning": "Insufficient data for walk-forward evaluation",
                "brier_score": 0.0,
                "ece": 0.0,
            }
        
        predictions = []
        actuals = []
        
        for i in range(min_train, len(self.df)):
            train_df = self.df.iloc[:i]
            test_draw = self.df.iloc[i]
            
            self.model.fit(train_df)
            probs = self.model.predict()
            
            actual_numbers = set(test_draw["numbers"])
            
            pred_vector = []
            actual_vector = []
            for num in range(self.n_min, self.n_max + 1):
                pred_vector.append(probs.get(str(num), 0.0))
                actual_vector.append(1.0 if num in actual_numbers else 0.0)
            
            predictions.append(pred_vector)
            actuals.append(actual_vector)
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        brier_score = self._calculate_brier(predictions, actuals)
        ece = self._calculate_ece(predictions, actuals)
        
        from app.analysis.prob_models.m0_baseline import M0Baseline
        baseline_model = M0Baseline(self.rules)
        baseline_probs = baseline_model.predict()
        baseline_vector = [baseline_probs[str(num)] for num in range(self.n_min, self.n_max + 1)]
        baseline_predictions = np.tile(baseline_vector, (len(actuals), 1))
        baseline_brier = self._calculate_brier(baseline_predictions, actuals)
        
        lift = baseline_brier - brier_score
        
        return {
            "brier_score": float(brier_score),
            "baseline_brier": float(baseline_brier),
            "lift": float(lift),
            "ece": float(ece),
            "n_predictions": len(predictions),
        }

    def _calculate_brier(self, predictions: np.ndarray, actuals: np.ndarray) -> float:
        return np.mean((predictions - actuals) ** 2)

    def _calculate_ece(self, predictions: np.ndarray, actuals: np.ndarray, n_bins: int = 10) -> float:
        predictions_flat = predictions.flatten()
        actuals_flat = actuals.flatten()
        
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            bin_mask = (predictions_flat >= bin_edges[i]) & (predictions_flat < bin_edges[i + 1])
            if i == n_bins - 1:
                bin_mask = (predictions_flat >= bin_edges[i]) & (predictions_flat <= bin_edges[i + 1])
            
            if np.sum(bin_mask) > 0:
                bin_preds = predictions_flat[bin_mask]
                bin_actuals = actuals_flat[bin_mask]
                
                avg_pred = np.mean(bin_preds)
                avg_actual = np.mean(bin_actuals)
                
                ece += np.abs(avg_pred - avg_actual) * np.sum(bin_mask) / len(predictions_flat)
        
        return ece
