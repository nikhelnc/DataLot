"""
Jackpot-related fraud tests.
Tests for correlations between jackpot amounts and draw outcomes.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


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


class JackpotFraudTests:
    """
    Tests for detecting suspicious correlations between jackpot amounts
    and draw outcomes.
    
    These tests check if draws are influenced by the jackpot size,
    which would indicate manipulation.
    """
    
    def __init__(self, alpha: float = 0.01):
        """
        Initialize jackpot fraud tests.
        
        Args:
            alpha: Significance level
        """
        self.alpha = alpha
    
    def jackpot_sum_correlation_test(self, draws: List[List[int]], 
                                      jackpots: List[float]) -> TestResult:
        """
        Test for correlation between jackpot amount and draw sum.
        
        If draws are manipulated based on jackpot size, we might see
        a correlation between jackpot and draw characteristics.
        """
        # Filter to draws with valid jackpot data
        valid_pairs = [(d, j) for d, j in zip(draws, jackpots) if j and j > 0]
        
        if len(valid_pairs) < 30:
            return TestResult(
                name="Jackpot-Sum Correlation",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="Insufficient jackpot data (need >= 30 draws with jackpot)"
            )
        
        sums = [sum(d) for d, _ in valid_pairs]
        jackpot_values = [j for _, j in valid_pairs]
        
        # Spearman correlation (robust to outliers)
        corr, p_value = stats.spearmanr(jackpot_values, sums)
        
        if p_value < 0.001 and abs(corr) > 0.3:
            severity = "CRITICAL"
        elif p_value < 0.01 and abs(corr) > 0.2:
            severity = "HIGH"
        elif p_value < 0.05:
            severity = "WARNING"
        else:
            severity = "INFO"
        
        return TestResult(
            name="Jackpot-Sum Correlation",
            statistic=float(corr),
            p_value=float(p_value),
            passed=p_value >= self.alpha or abs(corr) < 0.15,
            alpha=self.alpha,
            description="Tests for correlation between jackpot amount and draw sum",
            severity=severity,
            details={
                "n_draws": len(valid_pairs),
                "spearman_correlation": float(corr),
                "correlation_direction": "positive" if corr > 0 else "negative",
                "mean_jackpot": float(np.mean(jackpot_values)),
                "mean_sum": float(np.mean(sums))
            }
        )
    
    def jackpot_frequency_correlation_test(self, draws: List[List[int]], 
                                            jackpots: List[float],
                                            n_max: int) -> TestResult:
        """
        Test if certain numbers appear more often during high jackpots.
        
        Compares number frequencies in high-jackpot vs low-jackpot draws.
        """
        valid_pairs = [(d, j) for d, j in zip(draws, jackpots) if j and j > 0]
        
        if len(valid_pairs) < 50:
            return TestResult(
                name="Jackpot-Frequency Correlation",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="Insufficient jackpot data (need >= 50 draws)"
            )
        
        # Split into high and low jackpot groups
        jackpot_values = [j for _, j in valid_pairs]
        median_jackpot = np.median(jackpot_values)
        
        high_jackpot_draws = [d for d, j in valid_pairs if j >= median_jackpot]
        low_jackpot_draws = [d for d, j in valid_pairs if j < median_jackpot]
        
        # Count frequencies in each group
        high_freq = np.zeros(n_max)
        low_freq = np.zeros(n_max)
        
        for draw in high_jackpot_draws:
            for num in draw:
                if 1 <= num <= n_max:
                    high_freq[num - 1] += 1
        
        for draw in low_jackpot_draws:
            for num in draw:
                if 1 <= num <= n_max:
                    low_freq[num - 1] += 1
        
        # Normalize to proportions
        high_prop = high_freq / high_freq.sum() if high_freq.sum() > 0 else high_freq
        low_prop = low_freq / low_freq.sum() if low_freq.sum() > 0 else low_freq
        
        # Chi-squared test for homogeneity
        # Create contingency table
        contingency = np.array([high_freq, low_freq])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        
        # Find most different numbers
        diff = high_prop - low_prop
        most_different = []
        sorted_indices = np.argsort(np.abs(diff))[::-1]
        for idx in sorted_indices[:5]:
            most_different.append({
                "number": int(idx + 1),
                "high_jackpot_prop": float(high_prop[idx]),
                "low_jackpot_prop": float(low_prop[idx]),
                "difference": float(diff[idx])
            })
        
        if p_value < 0.001:
            severity = "CRITICAL"
        elif p_value < 0.01:
            severity = "HIGH"
        elif p_value < 0.05:
            severity = "WARNING"
        else:
            severity = "INFO"
        
        return TestResult(
            name="Jackpot-Frequency Correlation",
            statistic=float(chi2),
            p_value=float(p_value),
            passed=p_value >= self.alpha,
            alpha=self.alpha,
            description="Tests if number frequencies differ between high/low jackpot draws",
            severity=severity,
            details={
                "n_high_jackpot": len(high_jackpot_draws),
                "n_low_jackpot": len(low_jackpot_draws),
                "median_jackpot": float(median_jackpot),
                "chi2_statistic": float(chi2),
                "most_different_numbers": most_different
            }
        )
    
    def rollover_pattern_test(self, draws: List[List[int]], 
                               jackpots: List[float],
                               rollovers: List[bool]) -> TestResult:
        """
        Test if draw patterns differ after rollovers.
        
        Compares draws after rollover vs draws after win.
        """
        valid_data = [
            (d, j, r) for d, j, r in zip(draws, jackpots, rollovers)
            if j is not None
        ]
        
        if len(valid_data) < 30:
            return TestResult(
                name="Rollover Pattern",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="Insufficient rollover data"
            )
        
        rollover_draws = [d for d, _, r in valid_data if r]
        non_rollover_draws = [d for d, _, r in valid_data if not r]
        
        if len(rollover_draws) < 10 or len(non_rollover_draws) < 10:
            return TestResult(
                name="Rollover Pattern",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="Insufficient draws in one category"
            )
        
        # Compare sum distributions
        rollover_sums = [sum(d) for d in rollover_draws]
        non_rollover_sums = [sum(d) for d in non_rollover_draws]
        
        # Mann-Whitney U test (non-parametric)
        u_stat, p_value = stats.mannwhitneyu(rollover_sums, non_rollover_sums, alternative='two-sided')
        
        if p_value < 0.001:
            severity = "CRITICAL"
        elif p_value < 0.01:
            severity = "HIGH"
        elif p_value < 0.05:
            severity = "WARNING"
        else:
            severity = "INFO"
        
        return TestResult(
            name="Rollover Pattern",
            statistic=float(u_stat),
            p_value=float(p_value),
            passed=p_value >= self.alpha,
            alpha=self.alpha,
            description="Tests if draw patterns differ after rollovers",
            severity=severity,
            details={
                "n_rollover_draws": len(rollover_draws),
                "n_non_rollover_draws": len(non_rollover_draws),
                "mean_sum_rollover": float(np.mean(rollover_sums)),
                "mean_sum_non_rollover": float(np.mean(non_rollover_sums)),
                "mann_whitney_u": float(u_stat)
            }
        )
    
    def must_be_won_test(self, draws: List[List[int]], 
                          must_be_won: List[bool],
                          n_max: int) -> TestResult:
        """
        Test if "must be won" draws differ from regular draws.
        
        In some lotteries, if jackpot reaches a cap, it "must be won".
        This test checks if these draws show different patterns.
        """
        valid_data = [(d, m) for d, m in zip(draws, must_be_won) if m is not None]
        
        mbw_draws = [d for d, m in valid_data if m]
        regular_draws = [d for d, m in valid_data if not m]
        
        if len(mbw_draws) < 5:
            return TestResult(
                name="Must-Be-Won Pattern",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="Insufficient 'must be won' draws for analysis"
            )
        
        if len(regular_draws) < 20:
            return TestResult(
                name="Must-Be-Won Pattern",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="Insufficient regular draws for comparison"
            )
        
        # Compare frequency distributions
        mbw_freq = np.zeros(n_max)
        regular_freq = np.zeros(n_max)
        
        for draw in mbw_draws:
            for num in draw:
                if 1 <= num <= n_max:
                    mbw_freq[num - 1] += 1
        
        for draw in regular_draws:
            for num in draw:
                if 1 <= num <= n_max:
                    regular_freq[num - 1] += 1
        
        # Normalize
        mbw_prop = mbw_freq / mbw_freq.sum() if mbw_freq.sum() > 0 else mbw_freq
        regular_prop = regular_freq / regular_freq.sum() if regular_freq.sum() > 0 else regular_freq
        
        # KL divergence (asymmetric, so use symmetric version)
        epsilon = 1e-10
        kl_div = 0.5 * (
            np.sum(mbw_prop * np.log((mbw_prop + epsilon) / (regular_prop + epsilon))) +
            np.sum(regular_prop * np.log((regular_prop + epsilon) / (mbw_prop + epsilon)))
        )
        
        # Permutation test for significance
        n_permutations = 1000
        all_draws = mbw_draws + regular_draws
        observed_kl = kl_div
        
        perm_kls = []
        for _ in range(n_permutations):
            perm = np.random.permutation(len(all_draws))
            perm_mbw = [all_draws[i] for i in perm[:len(mbw_draws)]]
            perm_regular = [all_draws[i] for i in perm[len(mbw_draws):]]
            
            perm_mbw_freq = np.zeros(n_max)
            perm_regular_freq = np.zeros(n_max)
            
            for draw in perm_mbw:
                for num in draw:
                    if 1 <= num <= n_max:
                        perm_mbw_freq[num - 1] += 1
            
            for draw in perm_regular:
                for num in draw:
                    if 1 <= num <= n_max:
                        perm_regular_freq[num - 1] += 1
            
            perm_mbw_prop = perm_mbw_freq / perm_mbw_freq.sum() if perm_mbw_freq.sum() > 0 else perm_mbw_freq
            perm_regular_prop = perm_regular_freq / perm_regular_freq.sum() if perm_regular_freq.sum() > 0 else perm_regular_freq
            
            perm_kl = 0.5 * (
                np.sum(perm_mbw_prop * np.log((perm_mbw_prop + epsilon) / (perm_regular_prop + epsilon))) +
                np.sum(perm_regular_prop * np.log((perm_regular_prop + epsilon) / (perm_mbw_prop + epsilon)))
            )
            perm_kls.append(perm_kl)
        
        p_value = (np.sum(np.array(perm_kls) >= observed_kl) + 1) / (n_permutations + 1)
        
        if p_value < 0.001:
            severity = "CRITICAL"
        elif p_value < 0.01:
            severity = "HIGH"
        elif p_value < 0.05:
            severity = "WARNING"
        else:
            severity = "INFO"
        
        return TestResult(
            name="Must-Be-Won Pattern",
            statistic=float(kl_div),
            p_value=float(p_value),
            passed=p_value >= self.alpha,
            alpha=self.alpha,
            description="Tests if 'must be won' draws differ from regular draws",
            severity=severity,
            details={
                "n_must_be_won": len(mbw_draws),
                "n_regular": len(regular_draws),
                "kl_divergence": float(kl_div),
                "n_permutations": n_permutations
            }
        )
    
    def run_all_tests(self, draws: List[List[int]], n_max: int,
                      jackpots: List[float] = None,
                      rollovers: List[bool] = None,
                      must_be_won: List[bool] = None) -> Dict[str, Any]:
        """
        Run all jackpot fraud tests.
        
        Args:
            draws: List of draws
            n_max: Maximum number in pool
            jackpots: List of jackpot amounts (optional)
            rollovers: List of rollover flags (optional)
            must_be_won: List of must-be-won flags (optional)
            
        Returns:
            Dictionary with all test results
        """
        results = {
            "n_draws": len(draws),
            "n_max": n_max,
            "alpha": self.alpha,
            "has_jackpot_data": jackpots is not None and any(j for j in jackpots if j),
            "tests": {}
        }
        
        tests = []
        
        if jackpots:
            tests.append(self.jackpot_sum_correlation_test(draws, jackpots))
            tests.append(self.jackpot_frequency_correlation_test(draws, jackpots, n_max))
            
            if rollovers:
                tests.append(self.rollover_pattern_test(draws, jackpots, rollovers))
        
        if must_be_won:
            tests.append(self.must_be_won_test(draws, must_be_won, n_max))
        
        if not tests:
            results["summary"] = {
                "n_tests": 0,
                "message": "No jackpot data available for testing"
            }
            return results
        
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
