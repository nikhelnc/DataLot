"""
Regression Discontinuity Design (RDD) analysis for jackpot thresholds.
Tests for discontinuities in draw characteristics at jackpot thresholds.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RDDResult:
    """Result of RDD analysis"""
    name: str
    discontinuity: float
    p_value: float
    significant: bool
    alpha: float = 0.05
    description: str = ""
    details: Dict[str, Any] = None


class RDDAnalyzer:
    """
    Regression Discontinuity Design analysis for jackpot thresholds.
    
    RDD is a quasi-experimental method that tests for discontinuities
    at known thresholds. In lottery context, we test if draw characteristics
    change abruptly at certain jackpot levels.
    """
    
    def __init__(self, alpha: float = 0.05, bandwidth_factor: float = 0.2):
        """
        Initialize RDD analyzer.
        
        Args:
            alpha: Significance level
            bandwidth_factor: Fraction of data range to use as bandwidth
        """
        self.alpha = alpha
        self.bandwidth_factor = bandwidth_factor
    
    def _local_linear_regression(self, x: np.ndarray, y: np.ndarray,
                                  cutoff: float, bandwidth: float) -> Tuple[float, float]:
        """
        Perform local linear regression on both sides of cutoff.
        
        Returns: (left_intercept, right_intercept) at the cutoff
        """
        # Left side
        left_mask = (x < cutoff) & (x >= cutoff - bandwidth)
        if np.sum(left_mask) < 5:
            left_intercept = np.mean(y[x < cutoff]) if np.any(x < cutoff) else 0
        else:
            x_left = x[left_mask] - cutoff
            y_left = y[left_mask]
            slope, intercept, _, _, _ = stats.linregress(x_left, y_left)
            left_intercept = intercept
        
        # Right side
        right_mask = (x >= cutoff) & (x <= cutoff + bandwidth)
        if np.sum(right_mask) < 5:
            right_intercept = np.mean(y[x >= cutoff]) if np.any(x >= cutoff) else 0
        else:
            x_right = x[right_mask] - cutoff
            y_right = y[right_mask]
            slope, intercept, _, _, _ = stats.linregress(x_right, y_right)
            right_intercept = intercept
        
        return left_intercept, right_intercept
    
    def analyze_threshold(self, jackpots: np.ndarray, outcomes: np.ndarray,
                          threshold: float, outcome_name: str = "outcome") -> RDDResult:
        """
        Analyze discontinuity at a specific threshold.
        
        Args:
            jackpots: Array of jackpot amounts
            outcomes: Array of outcome values (e.g., draw sums)
            threshold: The threshold to test
            outcome_name: Name of the outcome variable
            
        Returns:
            RDDResult with discontinuity estimate and significance
        """
        if len(jackpots) < 30:
            return RDDResult(
                name=f"RDD at {threshold:,.0f}",
                discontinuity=0.0,
                p_value=1.0,
                significant=False,
                alpha=self.alpha,
                description="Insufficient data"
            )
        
        # Calculate bandwidth
        bandwidth = (jackpots.max() - jackpots.min()) * self.bandwidth_factor
        
        # Local linear regression
        left_intercept, right_intercept = self._local_linear_regression(
            jackpots, outcomes, threshold, bandwidth
        )
        
        discontinuity = right_intercept - left_intercept
        
        # Bootstrap for p-value
        n_bootstrap = 1000
        bootstrap_discs = []
        
        for _ in range(n_bootstrap):
            # Resample
            indices = np.random.choice(len(jackpots), size=len(jackpots), replace=True)
            boot_jackpots = jackpots[indices]
            boot_outcomes = outcomes[indices]
            
            left_int, right_int = self._local_linear_regression(
                boot_jackpots, boot_outcomes, threshold, bandwidth
            )
            bootstrap_discs.append(right_int - left_int)
        
        bootstrap_discs = np.array(bootstrap_discs)
        
        # Two-tailed p-value
        p_value = 2 * min(
            np.mean(bootstrap_discs >= discontinuity),
            np.mean(bootstrap_discs <= discontinuity)
        )
        p_value = max(p_value, 1 / n_bootstrap)
        
        return RDDResult(
            name=f"RDD at {threshold:,.0f}",
            discontinuity=float(discontinuity),
            p_value=float(p_value),
            significant=p_value < self.alpha,
            alpha=self.alpha,
            description=f"Tests for discontinuity in {outcome_name} at jackpot = {threshold:,.0f}",
            details={
                "threshold": float(threshold),
                "bandwidth": float(bandwidth),
                "left_intercept": float(left_intercept),
                "right_intercept": float(right_intercept),
                "n_left": int(np.sum(jackpots < threshold)),
                "n_right": int(np.sum(jackpots >= threshold)),
                "bootstrap_std": float(np.std(bootstrap_discs))
            }
        )
    
    def find_optimal_threshold(self, jackpots: np.ndarray, outcomes: np.ndarray,
                                n_candidates: int = 10) -> Tuple[float, RDDResult]:
        """
        Find the threshold with the largest discontinuity.
        
        Args:
            jackpots: Array of jackpot amounts
            outcomes: Array of outcome values
            n_candidates: Number of candidate thresholds to test
            
        Returns:
            Tuple of (optimal_threshold, RDDResult)
        """
        # Generate candidate thresholds (percentiles)
        percentiles = np.linspace(20, 80, n_candidates)
        thresholds = np.percentile(jackpots, percentiles)
        
        best_result = None
        best_threshold = None
        max_abs_disc = 0
        
        for threshold in thresholds:
            result = self.analyze_threshold(jackpots, outcomes, threshold)
            if abs(result.discontinuity) > max_abs_disc:
                max_abs_disc = abs(result.discontinuity)
                best_result = result
                best_threshold = threshold
        
        return best_threshold, best_result
    
    def analyze_multiple_outcomes(self, draws: List[List[int]],
                                   jackpots: List[float],
                                   threshold: float) -> Dict[str, Any]:
        """
        Analyze discontinuities for multiple draw characteristics.
        
        Args:
            draws: List of draws
            jackpots: List of jackpot amounts
            threshold: Threshold to test
            
        Returns:
            Dictionary with results for each outcome
        """
        valid_pairs = [(d, j) for d, j in zip(draws, jackpots) if j and j > 0]
        
        if len(valid_pairs) < 30:
            return {
                "threshold": threshold,
                "n_draws": len(valid_pairs),
                "error": "Insufficient data"
            }
        
        jackpot_arr = np.array([j for _, j in valid_pairs])
        
        # Different outcomes to test
        outcomes = {
            "sum": np.array([sum(d) for d, _ in valid_pairs]),
            "range": np.array([max(d) - min(d) for d, _ in valid_pairs]),
            "variance": np.array([np.var(d) for d, _ in valid_pairs]),
            "min_number": np.array([min(d) for d, _ in valid_pairs]),
            "max_number": np.array([max(d) for d, _ in valid_pairs])
        }
        
        results = {
            "threshold": float(threshold),
            "n_draws": len(valid_pairs),
            "outcomes": {}
        }
        
        for name, outcome_arr in outcomes.items():
            rdd_result = self.analyze_threshold(jackpot_arr, outcome_arr, threshold, name)
            results["outcomes"][name] = {
                "discontinuity": rdd_result.discontinuity,
                "p_value": rdd_result.p_value,
                "significant": rdd_result.significant,
                "details": rdd_result.details
            }
        
        # Summary
        n_significant = sum(1 for r in results["outcomes"].values() if r["significant"])
        results["summary"] = {
            "n_outcomes_tested": len(outcomes),
            "n_significant": n_significant,
            "any_significant": n_significant > 0
        }
        
        return results
    
    def run_full_analysis(self, draws: List[List[int]],
                          jackpots: List[float],
                          thresholds: List[float] = None) -> Dict[str, Any]:
        """
        Run full RDD analysis.
        
        Args:
            draws: List of draws
            jackpots: List of jackpot amounts
            thresholds: List of thresholds to test (auto-detected if None)
            
        Returns:
            Complete RDD analysis results
        """
        valid_pairs = [(d, j) for d, j in zip(draws, jackpots) if j and j > 0]
        
        if len(valid_pairs) < 50:
            return {
                "n_draws": len(valid_pairs),
                "error": "Insufficient data for RDD analysis (need >= 50 draws)"
            }
        
        jackpot_arr = np.array([j for _, j in valid_pairs])
        sum_arr = np.array([sum(d) for d, _ in valid_pairs])
        
        # Auto-detect thresholds if not provided
        if thresholds is None:
            thresholds = list(np.percentile(jackpot_arr, [25, 50, 75, 90]))
        
        results = {
            "n_draws": len(valid_pairs),
            "jackpot_range": [float(jackpot_arr.min()), float(jackpot_arr.max())],
            "thresholds_tested": [float(t) for t in thresholds],
            "threshold_results": {},
            "optimal_threshold": None
        }
        
        # Test each threshold
        for threshold in thresholds:
            results["threshold_results"][f"{threshold:.0f}"] = self.analyze_multiple_outcomes(
                [d for d, _ in valid_pairs],
                [j for _, j in valid_pairs],
                threshold
            )
        
        # Find optimal threshold
        opt_threshold, opt_result = self.find_optimal_threshold(jackpot_arr, sum_arr)
        results["optimal_threshold"] = {
            "threshold": float(opt_threshold),
            "discontinuity": opt_result.discontinuity,
            "p_value": opt_result.p_value,
            "significant": opt_result.significant
        }
        
        return results
