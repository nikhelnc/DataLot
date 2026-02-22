"""
Fraud score calculator.
Aggregates all fraud detection tests into a single risk score.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class FraudScoreResult:
    """Result of fraud score calculation"""
    score: float  # 0-100, higher = more suspicious
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    n_alerts: int
    alerts_by_severity: Dict[str, int]
    category_scores: Dict[str, float]
    interpretation: str
    details: Dict[str, Any]


class FraudScoreCalculator:
    """
    Calculates a composite fraud risk score from multiple test categories.
    
    Score ranges:
    - 0-25: LOW risk (normal)
    - 25-50: MEDIUM risk (some anomalies)
    - 50-75: HIGH risk (significant anomalies)
    - 75-100: CRITICAL risk (strong fraud indicators)
    """
    
    # Category weights
    CATEGORY_WEIGHTS = {
        "dispersion": 0.25,
        "benford": 0.15,
        "clustering": 0.30,
        "jackpot": 0.30
    }
    
    # Severity scores
    SEVERITY_SCORES = {
        "INFO": 0,
        "WARNING": 25,
        "HIGH": 50,
        "CRITICAL": 100
    }
    
    def __init__(self):
        pass
    
    def _compute_category_score(self, test_results: Dict[str, Any]) -> float:
        """
        Compute fraud score for a single category.
        
        Returns score 0-100 based on test failures and severities.
        """
        if not test_results or "tests" not in test_results:
            return 0.0
        
        tests = test_results["tests"]
        if not tests:
            return 0.0
        
        # Score based on failed tests and their severity
        total_score = 0.0
        n_tests = len(tests)
        
        for test_name, test_data in tests.items():
            if not test_data.get("passed", True):
                severity = test_data.get("severity", "WARNING")
                total_score += self.SEVERITY_SCORES.get(severity, 25)
            else:
                # Even passed tests with low p-values contribute slightly
                p_value = test_data.get("p_value", 1.0)
                if p_value < 0.05:
                    total_score += 10
                elif p_value < 0.1:
                    total_score += 5
        
        # Normalize by number of tests
        return min(100, total_score / n_tests * 2) if n_tests > 0 else 0.0
    
    def _count_alerts(self, test_results: Dict[str, Any]) -> Dict[str, int]:
        """Count alerts by severity from test results"""
        counts = {"INFO": 0, "WARNING": 0, "HIGH": 0, "CRITICAL": 0}
        
        if not test_results or "tests" not in test_results:
            return counts
        
        for test_data in test_results["tests"].values():
            if not test_data.get("passed", True):
                severity = test_data.get("severity", "WARNING")
                counts[severity] = counts.get(severity, 0) + 1
        
        return counts
    
    def compute_score(self,
                      dispersion_results: Optional[Dict[str, Any]] = None,
                      benford_results: Optional[Dict[str, Any]] = None,
                      clustering_results: Optional[Dict[str, Any]] = None,
                      jackpot_results: Optional[Dict[str, Any]] = None) -> FraudScoreResult:
        """
        Compute composite fraud risk score.
        
        Args:
            dispersion_results: Results from dispersion tests
            benford_results: Results from Benford tests
            clustering_results: Results from clustering tests
            jackpot_results: Results from jackpot fraud tests
            
        Returns:
            FraudScoreResult with score, risk level, and details
        """
        category_scores = {}
        alerts_by_severity = {"INFO": 0, "WARNING": 0, "HIGH": 0, "CRITICAL": 0}
        
        # Compute score for each category
        all_results = {
            "dispersion": dispersion_results,
            "benford": benford_results,
            "clustering": clustering_results,
            "jackpot": jackpot_results
        }
        
        active_weights = {}
        
        for category, results in all_results.items():
            if results and results.get("tests"):
                category_scores[category] = self._compute_category_score(results)
                active_weights[category] = self.CATEGORY_WEIGHTS[category]
                
                # Count alerts
                cat_alerts = self._count_alerts(results)
                for sev, count in cat_alerts.items():
                    alerts_by_severity[sev] += count
            else:
                category_scores[category] = 0.0
        
        # Normalize weights for active categories
        total_weight = sum(active_weights.values())
        if total_weight > 0:
            normalized_weights = {k: v / total_weight for k, v in active_weights.items()}
        else:
            normalized_weights = {}
        
        # Compute weighted score
        final_score = sum(
            category_scores.get(cat, 0) * normalized_weights.get(cat, 0)
            for cat in self.CATEGORY_WEIGHTS
        )
        
        # Determine risk level
        if final_score >= 75:
            risk_level = "CRITICAL"
        elif final_score >= 50:
            risk_level = "HIGH"
        elif final_score >= 25:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # Count total alerts
        n_alerts = sum(alerts_by_severity.values())
        
        # Generate interpretation
        if risk_level == "CRITICAL":
            interpretation = "CRITICAL: Multiple strong fraud indicators detected. Immediate investigation recommended."
        elif risk_level == "HIGH":
            interpretation = "HIGH: Significant statistical anomalies detected. Further analysis recommended."
        elif risk_level == "MEDIUM":
            interpretation = "MEDIUM: Some anomalies detected. May warrant monitoring."
        else:
            interpretation = "LOW: No significant anomalies detected. Data appears consistent with random draws."
        
        return FraudScoreResult(
            score=final_score,
            risk_level=risk_level,
            n_alerts=n_alerts,
            alerts_by_severity=alerts_by_severity,
            category_scores=category_scores,
            interpretation=interpretation,
            details={
                "active_categories": list(active_weights.keys()),
                "normalized_weights": normalized_weights,
                "computation_timestamp": datetime.utcnow().isoformat()
            }
        )
    
    def generate_alert_summary(self,
                               dispersion_results: Optional[Dict[str, Any]] = None,
                               benford_results: Optional[Dict[str, Any]] = None,
                               clustering_results: Optional[Dict[str, Any]] = None,
                               jackpot_results: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Generate a list of all alerts from test results.
        
        Returns list of alert dictionaries sorted by severity.
        """
        alerts = []
        
        all_results = {
            "Dispersion": dispersion_results,
            "Benford": benford_results,
            "Clustering": clustering_results,
            "Jackpot": jackpot_results
        }
        
        for category, results in all_results.items():
            if not results or "tests" not in results:
                continue
            
            for test_name, test_data in results["tests"].items():
                if not test_data.get("passed", True):
                    alerts.append({
                        "category": category,
                        "test_name": test_name,
                        "severity": test_data.get("severity", "WARNING"),
                        "p_value": test_data.get("p_value"),
                        "statistic": test_data.get("statistic"),
                        "description": test_data.get("description", ""),
                        "details": test_data.get("details", {})
                    })
        
        # Sort by severity (CRITICAL first)
        severity_order = {"CRITICAL": 0, "HIGH": 1, "WARNING": 2, "INFO": 3}
        alerts.sort(key=lambda x: severity_order.get(x["severity"], 4))
        
        return alerts
