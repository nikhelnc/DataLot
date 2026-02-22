"""
Dispersion tests for fraud detection.
Tests for over-dispersion and under-dispersion in lottery draws.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from collections import Counter


@dataclass
class TestResult:
    """Result of a single statistical test"""
    name: str
    statistic: float
    p_value: float
    passed: bool
    alpha: float = 0.01
    description: str = ""
    severity: str = "INFO"  # INFO, WARNING, HIGH, CRITICAL
    details: Dict[str, Any] = None


class DispersionTests:
    """
    Tests for detecting abnormal dispersion patterns in lottery draws.
    
    Over-dispersion: Numbers appear more spread out than expected (possible manipulation)
    Under-dispersion: Numbers cluster more than expected (possible bias)
    """
    
    def __init__(self, alpha: float = 0.01):
        """
        Initialize dispersion tests.
        
        Args:
            alpha: Significance level
        """
        self.alpha = alpha
    
    def variance_ratio_test(self, draws: List[List[int]], n_max: int) -> TestResult:
        """
        Test if the variance of number frequencies matches expected variance.
        
        Under uniform distribution, frequency variance should follow a known distribution.
        Excess variance suggests manipulation; deficit suggests over-uniformity.
        
        Method: Chi-squared test on variance ratio
        """
        # Count frequency of each number
        frequencies = np.zeros(n_max)
        total_balls = 0
        
        for draw in draws:
            for num in draw:
                if 1 <= num <= n_max:
                    frequencies[num - 1] += 1
                    total_balls += 1
        
        if total_balls == 0:
            return TestResult(
                name="Variance Ratio",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="No data available"
            )
        
        n_draws = len(draws)
        k = len(draws[0]) if draws else 0
        
        # Expected frequency per number
        expected_freq = n_draws * k / n_max
        
        # Observed variance
        observed_var = np.var(frequencies, ddof=1)
        
        # Expected variance under uniform distribution
        # Var(X) = n*p*(1-p) for binomial, but with without-replacement correction
        p = k / n_max
        expected_var = n_draws * p * (1 - p) * (n_max - k) / (n_max - 1)
        
        # Variance ratio
        if expected_var > 0:
            variance_ratio = observed_var / expected_var
        else:
            variance_ratio = 1.0
        
        # Chi-squared test: (n-1)*s²/σ² ~ χ²(n-1)
        chi2_stat = (n_max - 1) * variance_ratio
        
        # Two-tailed test
        p_low = stats.chi2.cdf(chi2_stat, n_max - 1)
        p_high = 1 - p_low
        p_value = 2 * min(p_low, p_high)
        
        # Determine severity
        if p_value < 0.001:
            severity = "CRITICAL"
        elif p_value < 0.01:
            severity = "HIGH"
        elif p_value < 0.05:
            severity = "WARNING"
        else:
            severity = "INFO"
        
        # Interpretation
        if variance_ratio > 1.5:
            interpretation = "Over-dispersed: frequencies more variable than expected"
        elif variance_ratio < 0.5:
            interpretation = "Under-dispersed: frequencies suspiciously uniform"
        else:
            interpretation = "Normal dispersion"
        
        return TestResult(
            name="Variance Ratio",
            statistic=float(variance_ratio),
            p_value=float(p_value),
            passed=p_value >= self.alpha,
            alpha=self.alpha,
            description="Tests if frequency variance matches expected variance",
            severity=severity,
            details={
                "observed_variance": float(observed_var),
                "expected_variance": float(expected_var),
                "variance_ratio": float(variance_ratio),
                "chi2_statistic": float(chi2_stat),
                "interpretation": interpretation,
                "n_draws": n_draws,
                "n_max": n_max
            }
        )
    
    def index_of_dispersion_test(self, draws: List[List[int]], n_max: int) -> TestResult:
        """
        Index of Dispersion (Variance-to-Mean Ratio) test.
        
        For Poisson-distributed counts, VMR ≈ 1.
        VMR > 1 indicates over-dispersion (clustering)
        VMR < 1 indicates under-dispersion (regularity)
        """
        frequencies = np.zeros(n_max)
        
        for draw in draws:
            for num in draw:
                if 1 <= num <= n_max:
                    frequencies[num - 1] += 1
        
        mean_freq = np.mean(frequencies)
        var_freq = np.var(frequencies, ddof=1)
        
        if mean_freq == 0:
            return TestResult(
                name="Index of Dispersion",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="No data available"
            )
        
        vmr = var_freq / mean_freq
        
        # Under Poisson, (n-1)*VMR ~ χ²(n-1)
        chi2_stat = (n_max - 1) * vmr
        
        # Two-tailed test
        p_low = stats.chi2.cdf(chi2_stat, n_max - 1)
        p_high = 1 - p_low
        p_value = 2 * min(p_low, p_high)
        
        if p_value < 0.001:
            severity = "CRITICAL"
        elif p_value < 0.01:
            severity = "HIGH"
        elif p_value < 0.05:
            severity = "WARNING"
        else:
            severity = "INFO"
        
        return TestResult(
            name="Index of Dispersion",
            statistic=float(vmr),
            p_value=float(p_value),
            passed=p_value >= self.alpha,
            alpha=self.alpha,
            description="Tests variance-to-mean ratio (should be ~1 for random)",
            severity=severity,
            details={
                "variance": float(var_freq),
                "mean": float(mean_freq),
                "vmr": float(vmr),
                "chi2_statistic": float(chi2_stat),
                "interpretation": "over-dispersed" if vmr > 1.5 else ("under-dispersed" if vmr < 0.5 else "normal")
            }
        )
    
    def sum_dispersion_test(self, draws: List[List[int]], n_max: int) -> TestResult:
        """
        Test dispersion of draw sums.
        
        The sum of k numbers drawn from {1,...,N} should follow a known distribution.
        Abnormal dispersion in sums can indicate manipulation.
        """
        if not draws:
            return TestResult(
                name="Sum Dispersion",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="No data available"
            )
        
        k = len(draws[0])
        sums = np.array([sum(draw) for draw in draws])
        
        # Theoretical mean and variance
        expected_mean = k * (n_max + 1) / 2
        expected_var = k * (n_max + 1) * (n_max - k) / 12
        
        observed_mean = np.mean(sums)
        observed_var = np.var(sums, ddof=1)
        
        # Variance ratio
        if expected_var > 0:
            var_ratio = observed_var / expected_var
        else:
            var_ratio = 1.0
        
        # F-test for variance comparison
        n = len(draws)
        f_stat = var_ratio
        
        # Two-tailed F-test
        p_low = stats.f.cdf(f_stat, n - 1, n - 1)
        p_high = 1 - p_low
        p_value = 2 * min(p_low, p_high)
        
        if p_value < 0.001:
            severity = "CRITICAL"
        elif p_value < 0.01:
            severity = "HIGH"
        elif p_value < 0.05:
            severity = "WARNING"
        else:
            severity = "INFO"
        
        return TestResult(
            name="Sum Dispersion",
            statistic=float(var_ratio),
            p_value=float(p_value),
            passed=p_value >= self.alpha,
            alpha=self.alpha,
            description="Tests if sum variance matches theoretical expectation",
            severity=severity,
            details={
                "observed_mean": float(observed_mean),
                "expected_mean": float(expected_mean),
                "observed_variance": float(observed_var),
                "expected_variance": float(expected_var),
                "variance_ratio": float(var_ratio),
                "n_draws": len(draws)
            }
        )
    
    def gap_dispersion_test(self, draws: List[List[int]]) -> TestResult:
        """
        Test dispersion of gaps between consecutive numbers in sorted draws.
        
        Abnormal gap patterns can indicate manipulation.
        """
        if not draws:
            return TestResult(
                name="Gap Dispersion",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="No data available"
            )
        
        # Collect all gaps
        all_gaps = []
        for draw in draws:
            sorted_draw = sorted(draw)
            for i in range(len(sorted_draw) - 1):
                all_gaps.append(sorted_draw[i + 1] - sorted_draw[i])
        
        if not all_gaps:
            return TestResult(
                name="Gap Dispersion",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="No gaps to analyze"
            )
        
        all_gaps = np.array(all_gaps)
        
        mean_gap = np.mean(all_gaps)
        var_gap = np.var(all_gaps, ddof=1)
        
        # VMR for gaps
        vmr = var_gap / mean_gap if mean_gap > 0 else 1.0
        
        # Coefficient of variation
        cv = np.std(all_gaps) / mean_gap if mean_gap > 0 else 0.0
        
        # Simple test: compare to expected CV under uniformity
        # For geometric-like distribution, CV ≈ 1
        # Significant deviation indicates abnormality
        
        # Bootstrap for p-value
        n_bootstrap = 1000
        bootstrap_cvs = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(all_gaps, size=len(all_gaps), replace=True)
            boot_cv = np.std(sample) / np.mean(sample) if np.mean(sample) > 0 else 0
            bootstrap_cvs.append(boot_cv)
        
        bootstrap_cvs = np.array(bootstrap_cvs)
        p_value = np.mean(np.abs(bootstrap_cvs - np.mean(bootstrap_cvs)) >= np.abs(cv - np.mean(bootstrap_cvs)))
        
        if p_value < 0.001:
            severity = "CRITICAL"
        elif p_value < 0.01:
            severity = "HIGH"
        elif p_value < 0.05:
            severity = "WARNING"
        else:
            severity = "INFO"
        
        return TestResult(
            name="Gap Dispersion",
            statistic=float(cv),
            p_value=float(p_value),
            passed=p_value >= self.alpha,
            alpha=self.alpha,
            description="Tests if gap distribution is consistent with randomness",
            severity=severity,
            details={
                "mean_gap": float(mean_gap),
                "variance_gap": float(var_gap),
                "vmr": float(vmr),
                "coefficient_of_variation": float(cv),
                "n_gaps": len(all_gaps)
            }
        )
    
    def run_all_tests(self, draws: List[List[int]], n_max: int) -> Dict[str, Any]:
        """
        Run all dispersion tests.
        
        Args:
            draws: List of draws
            n_max: Maximum number in pool
            
        Returns:
            Dictionary with all test results
        """
        results = {
            "n_draws": len(draws),
            "n_max": n_max,
            "alpha": self.alpha,
            "tests": {}
        }
        
        tests = [
            self.variance_ratio_test(draws, n_max),
            self.index_of_dispersion_test(draws, n_max),
            self.sum_dispersion_test(draws, n_max),
            self.gap_dispersion_test(draws)
        ]
        
        n_passed = 0
        max_severity = "INFO"
        severity_order = {"INFO": 0, "WARNING": 1, "HIGH": 2, "CRITICAL": 3}
        
        for test in tests:
            results["tests"][test.name] = {
                "statistic": test.statistic,
                "p_value": test.p_value,
                "passed": test.passed,
                "description": test.description,
                "severity": test.severity,
                "details": test.details
            }
            if test.passed:
                n_passed += 1
            if severity_order.get(test.severity, 0) > severity_order.get(max_severity, 0):
                max_severity = test.severity
        
        results["summary"] = {
            "n_tests": len(tests),
            "n_passed": n_passed,
            "pass_rate": n_passed / len(tests) if tests else 0.0,
            "overall_passed": n_passed == len(tests),
            "max_severity": max_severity
        }
        
        return results
