"""
Benford's Law tests for fraud detection.
Tests if leading digits follow Benford's distribution.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Any
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
    severity: str = "INFO"
    details: Dict[str, Any] = None


class BenfordTests:
    """
    Benford's Law tests for lottery draw analysis.
    
    While lottery numbers themselves don't follow Benford's Law (they're uniform),
    derived quantities like sums, gaps, and frequencies often do.
    """
    
    # Benford's Law expected probabilities for first digit
    BENFORD_PROBS = {
        1: 0.301,
        2: 0.176,
        3: 0.125,
        4: 0.097,
        5: 0.079,
        6: 0.067,
        7: 0.058,
        8: 0.051,
        9: 0.046
    }
    
    def __init__(self, alpha: float = 0.01):
        """
        Initialize Benford tests.
        
        Args:
            alpha: Significance level
        """
        self.alpha = alpha
    
    def _get_first_digit(self, n: int) -> int:
        """Get the first digit of a positive integer"""
        if n <= 0:
            return 0
        while n >= 10:
            n //= 10
        return n
    
    def _chi_squared_benford(self, observed_counts: Dict[int, int], n_total: int) -> tuple:
        """
        Compute chi-squared statistic against Benford's distribution.
        
        Returns: (chi2_stat, p_value)
        """
        chi2 = 0.0
        for digit in range(1, 10):
            observed = observed_counts.get(digit, 0)
            expected = n_total * self.BENFORD_PROBS[digit]
            if expected > 0:
                chi2 += (observed - expected) ** 2 / expected
        
        p_value = 1 - stats.chi2.cdf(chi2, df=8)  # 9 digits - 1 = 8 df
        return chi2, p_value
    
    def sum_first_digit_test(self, draws: List[List[int]]) -> TestResult:
        """
        Test if first digits of draw sums follow Benford's Law.
        
        Draw sums are derived quantities that often follow Benford's Law.
        """
        if not draws:
            return TestResult(
                name="Sum First Digit (Benford)",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="No data available"
            )
        
        sums = [sum(draw) for draw in draws]
        first_digits = [self._get_first_digit(s) for s in sums if s > 0]
        
        if len(first_digits) < 50:
            return TestResult(
                name="Sum First Digit (Benford)",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="Insufficient data for Benford test (need >= 50 values)"
            )
        
        digit_counts = Counter(first_digits)
        chi2, p_value = self._chi_squared_benford(digit_counts, len(first_digits))
        
        # Compute observed vs expected
        observed_probs = {d: digit_counts.get(d, 0) / len(first_digits) for d in range(1, 10)}
        
        if p_value < 0.001:
            severity = "CRITICAL"
        elif p_value < 0.01:
            severity = "HIGH"
        elif p_value < 0.05:
            severity = "WARNING"
        else:
            severity = "INFO"
        
        return TestResult(
            name="Sum First Digit (Benford)",
            statistic=float(chi2),
            p_value=float(p_value),
            passed=p_value >= self.alpha,
            alpha=self.alpha,
            description="Tests if first digits of draw sums follow Benford's Law",
            severity=severity,
            details={
                "n_values": len(first_digits),
                "observed_distribution": {str(k): v for k, v in observed_probs.items()},
                "expected_distribution": {str(k): v for k, v in self.BENFORD_PROBS.items()},
                "chi2_statistic": float(chi2)
            }
        )
    
    def gap_first_digit_test(self, draws: List[List[int]]) -> TestResult:
        """
        Test if first digits of gaps follow Benford's Law.
        """
        if not draws:
            return TestResult(
                name="Gap First Digit (Benford)",
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
                gap = sorted_draw[i + 1] - sorted_draw[i]
                if gap > 0:
                    all_gaps.append(gap)
        
        if len(all_gaps) < 50:
            return TestResult(
                name="Gap First Digit (Benford)",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="Insufficient gaps for Benford test"
            )
        
        first_digits = [self._get_first_digit(g) for g in all_gaps]
        digit_counts = Counter(first_digits)
        chi2, p_value = self._chi_squared_benford(digit_counts, len(first_digits))
        
        observed_probs = {d: digit_counts.get(d, 0) / len(first_digits) for d in range(1, 10)}
        
        if p_value < 0.001:
            severity = "CRITICAL"
        elif p_value < 0.01:
            severity = "HIGH"
        elif p_value < 0.05:
            severity = "WARNING"
        else:
            severity = "INFO"
        
        return TestResult(
            name="Gap First Digit (Benford)",
            statistic=float(chi2),
            p_value=float(p_value),
            passed=p_value >= self.alpha,
            alpha=self.alpha,
            description="Tests if first digits of gaps follow Benford's Law",
            severity=severity,
            details={
                "n_gaps": len(all_gaps),
                "observed_distribution": {str(k): v for k, v in observed_probs.items()},
                "expected_distribution": {str(k): v for k, v in self.BENFORD_PROBS.items()},
                "chi2_statistic": float(chi2)
            }
        )
    
    def frequency_first_digit_test(self, draws: List[List[int]], n_max: int) -> TestResult:
        """
        Test if first digits of number frequencies follow Benford's Law.
        
        This test requires many draws to have meaningful frequencies.
        """
        if len(draws) < 100:
            return TestResult(
                name="Frequency First Digit (Benford)",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="Insufficient draws for frequency Benford test (need >= 100)"
            )
        
        # Count frequencies
        frequencies = Counter()
        for draw in draws:
            for num in draw:
                frequencies[num] += 1
        
        # Get first digits of frequencies
        first_digits = [self._get_first_digit(f) for f in frequencies.values() if f > 0]
        
        if len(first_digits) < 20:
            return TestResult(
                name="Frequency First Digit (Benford)",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="Insufficient unique frequencies for Benford test"
            )
        
        digit_counts = Counter(first_digits)
        chi2, p_value = self._chi_squared_benford(digit_counts, len(first_digits))
        
        observed_probs = {d: digit_counts.get(d, 0) / len(first_digits) for d in range(1, 10)}
        
        if p_value < 0.001:
            severity = "CRITICAL"
        elif p_value < 0.01:
            severity = "HIGH"
        elif p_value < 0.05:
            severity = "WARNING"
        else:
            severity = "INFO"
        
        return TestResult(
            name="Frequency First Digit (Benford)",
            statistic=float(chi2),
            p_value=float(p_value),
            passed=p_value >= self.alpha,
            alpha=self.alpha,
            description="Tests if first digits of number frequencies follow Benford's Law",
            severity=severity,
            details={
                "n_frequencies": len(first_digits),
                "observed_distribution": {str(k): v for k, v in observed_probs.items()},
                "expected_distribution": {str(k): v for k, v in self.BENFORD_PROBS.items()},
                "chi2_statistic": float(chi2)
            }
        )
    
    def jackpot_first_digit_test(self, jackpots: List[float]) -> TestResult:
        """
        Test if first digits of jackpot amounts follow Benford's Law.
        
        Jackpot amounts are naturally occurring financial data that should
        follow Benford's Law if not manipulated.
        """
        valid_jackpots = [j for j in jackpots if j and j > 0]
        
        if len(valid_jackpots) < 50:
            return TestResult(
                name="Jackpot First Digit (Benford)",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="Insufficient jackpot data for Benford test (need >= 50)"
            )
        
        first_digits = [self._get_first_digit(int(j)) for j in valid_jackpots]
        digit_counts = Counter(first_digits)
        chi2, p_value = self._chi_squared_benford(digit_counts, len(first_digits))
        
        observed_probs = {d: digit_counts.get(d, 0) / len(first_digits) for d in range(1, 10)}
        
        if p_value < 0.001:
            severity = "CRITICAL"
        elif p_value < 0.01:
            severity = "HIGH"
        elif p_value < 0.05:
            severity = "WARNING"
        else:
            severity = "INFO"
        
        return TestResult(
            name="Jackpot First Digit (Benford)",
            statistic=float(chi2),
            p_value=float(p_value),
            passed=p_value >= self.alpha,
            alpha=self.alpha,
            description="Tests if first digits of jackpot amounts follow Benford's Law",
            severity=severity,
            details={
                "n_jackpots": len(valid_jackpots),
                "observed_distribution": {str(k): v for k, v in observed_probs.items()},
                "expected_distribution": {str(k): v for k, v in self.BENFORD_PROBS.items()},
                "chi2_statistic": float(chi2)
            }
        )
    
    def run_all_tests(self, draws: List[List[int]], n_max: int,
                      jackpots: List[float] = None) -> Dict[str, Any]:
        """
        Run all Benford tests.
        
        Args:
            draws: List of draws
            n_max: Maximum number in pool
            jackpots: Optional list of jackpot amounts
            
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
            self.sum_first_digit_test(draws),
            self.gap_first_digit_test(draws),
            self.frequency_first_digit_test(draws, n_max)
        ]
        
        if jackpots:
            tests.append(self.jackpot_first_digit_test(jackpots))
        
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
