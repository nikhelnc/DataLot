import numpy as np
import pandas as pd
from typing import Dict, Any, List
from scipy.stats import entropy
import ruptures as rpt


class AnomalyDetector:
    def __init__(self, df: pd.DataFrame, rules: Dict[str, Any]):
        self.df = df
        self.rules = rules
        # Support both old (numbers) and new (main) structure
        main_rules = rules.get("main", rules.get("numbers", {}))
        self.n_min = main_rules.get("min", 1)
        self.n_max = main_rules.get("max", 49)
        self.n_count = main_rules.get("pick", main_rules.get("count", 6))
        self.warnings = []

    def detect_all(self) -> Dict[str, Any]:
        return {
            "drift": self._detect_drift(),
            "change_points": self._detect_change_points(),
            "outliers": self._detect_outliers(),
        }

    def _detect_drift(self) -> Dict[str, Any]:
        window_size = min(200, len(self.df) // 2)
        
        if len(self.df) < 100:
            self.warnings.append("Dataset too small for reliable drift detection")
            return {"warning": "Insufficient data"}
        
        n_range = self.n_max - self.n_min + 1
        uniform_probs = np.ones(n_range) / n_range
        
        psi_values = []
        kl_values = []
        dates = []
        
        for i in range(window_size, len(self.df), 50):
            window_numbers = []
            for numbers in self.df["numbers"].iloc[i - window_size : i]:
                window_numbers.extend(numbers)
            
            freq_counts = pd.Series(window_numbers).value_counts()
            observed = np.array([freq_counts.get(num, 0) for num in range(self.n_min, self.n_max + 1)])
            observed_probs = observed / observed.sum() if observed.sum() > 0 else uniform_probs
            
            psi = self._calculate_psi(observed_probs, uniform_probs)
            kl = entropy(observed_probs, uniform_probs)
            
            psi_values.append(psi)
            kl_values.append(kl)
            dates.append(self.df["draw_date"].iloc[i].isoformat())
        
        kl_mean, kl_std = self._monte_carlo_kl_reference(n_simulations=1000)
        z_kl_values = [(kl - kl_mean) / kl_std if kl_std > 0 else 0 for kl in kl_values]
        
        return {
            "psi_series": [{"date": d, "value": float(v)} for d, v in zip(dates, psi_values)],
            "kl_series": [{"date": d, "value": float(v)} for d, v in zip(dates, kl_values)],
            "z_kl_series": [{"date": d, "value": float(v)} for d, v in zip(dates, z_kl_values)],
            "max_psi": float(max(psi_values)) if psi_values else 0,
            "max_z_kl": float(max(z_kl_values)) if z_kl_values else 0,
        }

    def _calculate_psi(self, actual: np.ndarray, expected: np.ndarray) -> float:
        psi = 0
        for a, e in zip(actual, expected):
            if a > 0 and e > 0:
                psi += (a - e) * np.log(a / e)
        return psi

    def _monte_carlo_kl_reference(self, n_simulations: int = 1000) -> tuple:
        n_range = self.n_max - self.n_min + 1
        uniform_probs = np.ones(n_range) / n_range
        window_size = min(200, len(self.df) // 2)
        n_draws = window_size
        
        kl_samples = []
        np.random.seed(42)
        
        for _ in range(n_simulations):
            simulated = np.random.choice(range(self.n_min, self.n_max + 1), size=n_draws * self.n_count, replace=True)
            freq_counts = pd.Series(simulated).value_counts()
            observed = np.array([freq_counts.get(num, 0) for num in range(self.n_min, self.n_max + 1)])
            observed_probs = observed / observed.sum()
            kl = entropy(observed_probs, uniform_probs)
            kl_samples.append(kl)
        
        return np.mean(kl_samples), np.std(kl_samples)

    def _detect_change_points(self) -> Dict[str, Any]:
        if len(self.df) < 50:
            return {"warning": "Insufficient data for change point detection"}
        
        entropies = []
        window_size = 20
        
        for i in range(window_size, len(self.df)):
            window_numbers = []
            for numbers in self.df["numbers"].iloc[i - window_size : i]:
                window_numbers.extend(numbers)
            
            freq_counts = pd.Series(window_numbers).value_counts()
            n_range = self.n_max - self.n_min + 1
            observed = np.array([freq_counts.get(num, 0) for num in range(self.n_min, self.n_max + 1)])
            observed_probs = observed / observed.sum() if observed.sum() > 0 else np.ones(n_range) / n_range
            
            ent = entropy(observed_probs, base=2)
            entropies.append(ent)
        
        signal = np.array(entropies).reshape(-1, 1)
        
        try:
            algo = rpt.Pelt(model="rbf").fit(signal)
            change_points = algo.predict(pen=3)
            
            change_point_dates = []
            for cp in change_points[:-1]:
                if cp + window_size < len(self.df):
                    change_point_dates.append(self.df["draw_date"].iloc[cp + window_size].isoformat())
            
            return {
                "method": "PELT",
                "change_points": change_point_dates,
                "count": len(change_point_dates),
            }
        except Exception as e:
            return {"error": str(e)}

    def _detect_outliers(self) -> Dict[str, Any]:
        sums = [sum(numbers) for numbers in self.df["numbers"]]
        
        median = np.median(sums)
        mad = np.median([abs(s - median) for s in sums])
        
        if mad == 0:
            return {"warning": "MAD is zero, cannot compute robust z-scores"}
        
        z_scores = [(s - median) / (1.4826 * mad) for s in sums]
        
        outliers = []
        for idx, z in enumerate(z_scores):
            if abs(z) > 3:
                outliers.append({
                    "date": self.df["draw_date"].iloc[idx].isoformat(),
                    "numbers": self.df["numbers"].iloc[idx],
                    "sum": sums[idx],
                    "z_score": float(z),
                })
        
        return {
            "method": "MAD z-score",
            "threshold": 3,
            "outliers": outliers,
            "count": len(outliers),
        }

    def generate_alerts(self) -> List[Dict[str, Any]]:
        alerts = []
        anomalies = self.detect_all()
        
        if "drift" in anomalies and "max_psi" in anomalies["drift"]:
            max_psi = anomalies["drift"]["max_psi"]
            if max_psi >= 0.50:
                alerts.append({
                    "severity": "high",
                    "score": 5,
                    "message": f"High PSI drift detected: {max_psi:.3f}",
                    "evidence_json": {
                        "type": "DRIFT",
                        "metric": "PSI",
                        "value": max_psi,
                        "thresholds": {"low": 0.10, "medium": 0.25, "high": 0.50},
                    },
                })
            elif max_psi >= 0.25:
                alerts.append({
                    "severity": "medium",
                    "score": 3,
                    "message": f"Medium PSI drift detected: {max_psi:.3f}",
                    "evidence_json": {
                        "type": "DRIFT",
                        "metric": "PSI",
                        "value": max_psi,
                        "thresholds": {"low": 0.10, "medium": 0.25, "high": 0.50},
                    },
                })
        
        if "drift" in anomalies and "max_z_kl" in anomalies["drift"]:
            max_z_kl = anomalies["drift"]["max_z_kl"]
            if max_z_kl >= 4:
                alerts.append({
                    "severity": "high",
                    "score": 5,
                    "message": f"High KL divergence z-score: {max_z_kl:.2f}",
                    "evidence_json": {
                        "type": "DRIFT",
                        "metric": "z_KL",
                        "value": max_z_kl,
                        "thresholds": {"low": 2, "medium": 3, "high": 4},
                    },
                })
            elif max_z_kl >= 3:
                alerts.append({
                    "severity": "medium",
                    "score": 3,
                    "message": f"Medium KL divergence z-score: {max_z_kl:.2f}",
                    "evidence_json": {
                        "type": "DRIFT",
                        "metric": "z_KL",
                        "value": max_z_kl,
                        "thresholds": {"low": 2, "medium": 3, "high": 4},
                    },
                })
        
        if "change_points" in anomalies and anomalies["change_points"].get("count", 0) > 0:
            count = anomalies["change_points"]["count"]
            if count >= 3:
                alerts.append({
                    "severity": "high",
                    "score": 4,
                    "message": f"Multiple change points detected: {count}",
                    "evidence_json": {
                        "type": "CHANGE_POINT",
                        "count": count,
                        "dates": anomalies["change_points"]["change_points"],
                    },
                })
        
        if "outliers" in anomalies and anomalies["outliers"].get("count", 0) > 0:
            count = anomalies["outliers"]["count"]
            if count > len(self.df) * 0.05:
                alerts.append({
                    "severity": "medium",
                    "score": 3,
                    "message": f"High number of outliers: {count}",
                    "evidence_json": {
                        "type": "OUTLIER",
                        "count": count,
                        "samples": anomalies["outliers"]["outliers"][:5],
                    },
                })
        
        return alerts

    def get_warnings(self) -> List[str]:
        return self.warnings
