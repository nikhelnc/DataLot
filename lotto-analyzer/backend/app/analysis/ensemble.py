import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve


class EnsembleStacking:
    def __init__(self, rules: Dict, models: Dict):
        """
        models: dict with model names as keys and fitted model objects as values
        Each model should have a predict() method that returns number_probs
        """
        self.rules = rules
        # Support both old (numbers) and new (main) structure
        main_rules = rules.get("main", rules.get("numbers", {}))
        self.n_min = main_rules.get("min", 1)
        self.n_max = main_rules.get("max", 49)
        self.n_range = self.n_max - self.n_min + 1
        self.models = models
        self.df = None
        self.weights = None
        self.stacked_probs = None
        self.results = None

    def fit(self, df: pd.DataFrame, validation_split: float = 0.3):
        """
        Fit ensemble by optimizing weights to minimize Brier score
        """
        self.df = df
        
        # Split data for meta-learning
        split_idx = int(len(df) * (1 - validation_split))
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]
        
        if len(val_df) < 10:
            # Not enough validation data, use simple averaging
            self.weights = {name: 1.0 / len(self.models) for name in self.models.keys()}
            self.results = self._evaluate_ensemble(df)
            return self
        
        # Get predictions from each model on validation set
        model_predictions = {}
        for name, model in self.models.items():
            if hasattr(model, 'predict'):
                preds = model.predict()
                model_predictions[name] = preds
        
        if len(model_predictions) == 0:
            self.weights = {}
            self.results = {"error": "No valid models for ensemble"}
            return self
        
        # Optimize weights using validation set
        self.weights = self._optimize_weights(val_df, model_predictions)
        
        # Evaluate on full dataset
        self.results = self._evaluate_ensemble(df)
        
        return self

    def _optimize_weights(self, val_df: pd.DataFrame, model_predictions: Dict) -> Dict:
        """
        Find optimal weights to minimize Brier score on validation set
        Using simple grid search over weight space
        """
        model_names = list(model_predictions.keys())
        n_models = len(model_names)
        
        if n_models == 1:
            return {model_names[0]: 1.0}
        
        # Simple approach: equal weights (can be improved with optimization)
        # For now, use inverse Brier score as weights
        weights = {}
        brier_scores = {}
        
        for name, probs in model_predictions.items():
            brier = self._calculate_brier(val_df, probs)
            brier_scores[name] = brier
        
        # Inverse Brier as weights (lower Brier = higher weight)
        total_inv_brier = sum(1.0 / (b + 1e-6) for b in brier_scores.values())
        for name, brier in brier_scores.items():
            weights[name] = (1.0 / (brier + 1e-6)) / total_inv_brier
        
        return weights

    def _calculate_brier(self, df: pd.DataFrame, probs: Dict) -> float:
        """Calculate Brier score for given probabilities"""
        brier_sum = 0
        n_predictions = 0
        
        for _, row in df.iterrows():
            numbers = list(row["numbers"]) if row["numbers"] else []
            if "bonus_numbers" in row and row["bonus_numbers"]:
                numbers.extend(row["bonus_numbers"])
            
            for num in range(self.n_min, self.n_max + 1):
                prob = probs.get(str(num), 1.0 / self.n_range)
                actual = 1 if num in numbers else 0
                brier_sum += (prob - actual) ** 2
                n_predictions += 1
        
        return brier_sum / n_predictions if n_predictions > 0 else 1.0

    def _evaluate_ensemble(self, df: pd.DataFrame) -> Dict:
        """Evaluate ensemble performance"""
        # Combine predictions using weights
        combined_probs = {}
        
        for num in range(self.n_min, self.n_max + 1):
            weighted_prob = 0
            for name, weight in self.weights.items():
                if name in self.models and hasattr(self.models[name], 'predict'):
                    model_probs = self.models[name].predict()
                    prob = model_probs.get(str(num), 1.0 / self.n_range)
                    weighted_prob += weight * prob
            combined_probs[str(num)] = weighted_prob
        
        # Normalize to sum to k/N (expected number of numbers per draw)
        # Support both old (numbers.count) and new (main.pick) structure
        main_rules = self.rules.get("main", self.rules.get("numbers", {}))
        k = main_rules.get("pick", main_rules.get("count", 6))
        bonus_rules = self.rules.get("bonus", {})
        if bonus_rules.get("enabled"):
            k += bonus_rules.get("pick", bonus_rules.get("count", 1))
        
        total_prob = sum(combined_probs.values())
        if total_prob > 0:
            normalization_factor = (k / self.n_range) * self.n_range / total_prob
            combined_probs = {num: prob * normalization_factor for num, prob in combined_probs.items()}
        
        self.stacked_probs = combined_probs
        
        # Calculate metrics
        brier_score = self._calculate_brier(df, combined_probs)
        
        # Baseline Brier
        baseline_probs = {str(num): 1.0 / self.n_range for num in range(self.n_min, self.n_max + 1)}
        baseline_brier = self._calculate_brier(df, baseline_probs)
        
        # Lift
        lift = baseline_brier / brier_score if brier_score > 0 else 1.0
        
        # Calibration curve
        calibration_data = self._calculate_calibration(df, combined_probs)
        
        return {
            "weights": {name: float(weight) for name, weight in self.weights.items()},
            "combined_probs": combined_probs,
            "brier_score": float(brier_score),
            "baseline_brier": float(baseline_brier),
            "delta_brier": float(brier_score - baseline_brier),
            "lift": float(lift),
            "calibration": calibration_data,
            "n_models": len(self.weights)
        }

    def _calculate_calibration(self, df: pd.DataFrame, probs: Dict) -> Dict:
        """Calculate calibration curve data"""
        # Collect predictions and outcomes
        predictions = []
        outcomes = []
        
        for _, row in df.iterrows():
            numbers = list(row["numbers"]) if row["numbers"] else []
            if "bonus_numbers" in row and row["bonus_numbers"]:
                numbers.extend(row["bonus_numbers"])
            
            for num in range(self.n_min, self.n_max + 1):
                prob = probs.get(str(num), 1.0 / self.n_range)
                actual = 1 if num in numbers else 0
                predictions.append(prob)
                outcomes.append(actual)
        
        if len(predictions) < 10:
            return {"bins": [], "pred_probs": [], "obs_freqs": [], "ece": None}
        
        # Calculate calibration curve
        try:
            obs_freqs, pred_probs = calibration_curve(outcomes, predictions, n_bins=10, strategy='uniform')
            
            # Expected Calibration Error (ECE)
            ece = np.mean(np.abs(obs_freqs - pred_probs))
            
            return {
                "bins": list(range(len(pred_probs))),
                "pred_probs": pred_probs.tolist(),
                "obs_freqs": obs_freqs.tolist(),
                "ece": float(ece)
            }
        except:
            return {"bins": [], "pred_probs": [], "obs_freqs": [], "ece": None}

    def predict(self) -> Dict:
        """Return stacked probabilities"""
        if self.stacked_probs is None:
            return {str(num): 1.0 / self.n_range for num in range(self.n_min, self.n_max + 1)}
        return self.stacked_probs

    def get_results(self) -> Dict:
        if self.results is None:
            return {
                "error": "Model not fitted",
                "warnings": ["Call fit() before get_results()"]
            }
        
        if "error" in self.results:
            return {
                "method": "M10_Ensemble",
                "error": self.results["error"],
                "warnings": ["No valid models to ensemble"]
            }
        
        return {
            "method": "M10_Ensemble",
            "explain": "Ensemble stacking : combine les prédictions de M0, M1, M2 avec des poids optimisés pour minimiser le Brier score",
            "number_probs": self.results["combined_probs"],
            "baseline_probs": {str(num): 1.0 / self.n_range for num in range(self.n_min, self.n_max + 1)},
            "ensemble": self.results,
            "evaluation": {
                "brier_score": self.results["brier_score"],
                "baseline_brier": self.results["baseline_brier"],
                "lift": self.results["lift"],
                "ece": self.results["calibration"].get("ece")
            },
            "warnings": self._generate_warnings(),
            "charts": {
                "weights": {
                    "models": list(self.results["weights"].keys()),
                    "weights": list(self.results["weights"].values())
                },
                "calibration": self.results["calibration"]
            }
        }

    def _generate_warnings(self) -> List[str]:
        warnings = []
        
        if self.results["n_models"] < 2:
            warnings.append("Moins de 2 modèles disponibles : l'ensemble n'apporte pas de bénéfice")
        
        if abs(self.results["delta_brier"]) < 0.0001:
            warnings.append("Gain négligeable vs baseline (ΔBrier < 0.0001)")
        
        if self.results["lift"] < 1.01:
            warnings.append("Lift très faible (<1.01) : l'ensemble n'améliore pas significativement la baseline")
        
        ece = self.results["calibration"].get("ece")
        if ece is not None and ece > 0.05:
            warnings.append(f"Calibration médiocre (ECE = {ece:.3f}) : les probabilités ne sont pas bien calibrées")
        
        return warnings
