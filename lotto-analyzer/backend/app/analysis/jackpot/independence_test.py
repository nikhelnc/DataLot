"""
Jackpot independence tests.
Tests if draw outcomes are independent of jackpot amounts.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Any, Tuple, Optional
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
    details: Dict[str, Any] = None


class JackpotIndependenceTest:
    """
    Tests for independence between jackpot amounts and draw outcomes.
    
    A fair lottery should have draw outcomes completely independent
    of the jackpot size.
    """
    
    def __init__(self, alpha: float = 0.01):
        """
        Initialize jackpot independence tests.
        
        Args:
            alpha: Significance level
        """
        self.alpha = alpha
    
    def sum_correlation_test(self, draws: List[List[int]], 
                              jackpots: List[float]) -> TestResult:
        """
        Test correlation between jackpot amount and draw sum.
        
        Uses Spearman correlation (robust to outliers).
        """
        valid_pairs = [(d, j) for d, j in zip(draws, jackpots) if j and j > 0]
        
        if len(valid_pairs) < 30:
            return TestResult(
                name="Jackpot-Sum Independence",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="Insufficient data (need >= 30 draws with jackpot)"
            )
        
        sums = [sum(d) for d, _ in valid_pairs]
        jackpot_values = [j for _, j in valid_pairs]
        
        corr, p_value = stats.spearmanr(jackpot_values, sums)
        
        return TestResult(
            name="Jackpot-Sum Independence",
            statistic=float(corr),
            p_value=float(p_value),
            passed=p_value >= self.alpha,
            alpha=self.alpha,
            description="Tests if draw sums are independent of jackpot amount",
            details={
                "n_draws": len(valid_pairs),
                "spearman_correlation": float(corr),
                "mean_jackpot": float(np.mean(jackpot_values)),
                "std_jackpot": float(np.std(jackpot_values)),
                "mean_sum": float(np.mean(sums)),
                "std_sum": float(np.std(sums))
            }
        )
    
    def range_correlation_test(self, draws: List[List[int]], 
                                jackpots: List[float]) -> TestResult:
        """
        Test correlation between jackpot amount and draw range.
        
        Range = max(draw) - min(draw)
        """
        valid_pairs = [(d, j) for d, j in zip(draws, jackpots) if j and j > 0]
        
        if len(valid_pairs) < 30:
            return TestResult(
                name="Jackpot-Range Independence",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="Insufficient data"
            )
        
        ranges = [max(d) - min(d) for d, _ in valid_pairs]
        jackpot_values = [j for _, j in valid_pairs]
        
        corr, p_value = stats.spearmanr(jackpot_values, ranges)
        
        return TestResult(
            name="Jackpot-Range Independence",
            statistic=float(corr),
            p_value=float(p_value),
            passed=p_value >= self.alpha,
            alpha=self.alpha,
            description="Tests if draw ranges are independent of jackpot amount",
            details={
                "n_draws": len(valid_pairs),
                "spearman_correlation": float(corr),
                "mean_range": float(np.mean(ranges)),
                "std_range": float(np.std(ranges))
            }
        )
    
    def variance_correlation_test(self, draws: List[List[int]], 
                                   jackpots: List[float]) -> TestResult:
        """
        Test correlation between jackpot amount and draw variance.
        """
        valid_pairs = [(d, j) for d, j in zip(draws, jackpots) if j and j > 0]
        
        if len(valid_pairs) < 30:
            return TestResult(
                name="Jackpot-Variance Independence",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="Insufficient data"
            )
        
        variances = [np.var(d) for d, _ in valid_pairs]
        jackpot_values = [j for _, j in valid_pairs]
        
        corr, p_value = stats.spearmanr(jackpot_values, variances)
        
        return TestResult(
            name="Jackpot-Variance Independence",
            statistic=float(corr),
            p_value=float(p_value),
            passed=p_value >= self.alpha,
            alpha=self.alpha,
            description="Tests if draw variances are independent of jackpot amount",
            details={
                "n_draws": len(valid_pairs),
                "spearman_correlation": float(corr),
                "mean_variance": float(np.mean(variances))
            }
        )
    
    def number_frequency_independence_test(self, draws: List[List[int]], 
                                            jackpots: List[float],
                                            n_max: int) -> TestResult:
        """
        Test if number frequencies differ between high and low jackpot draws.
        
        Uses chi-squared test for homogeneity.
        """
        valid_pairs = [(d, j) for d, j in zip(draws, jackpots) if j and j > 0]
        
        if len(valid_pairs) < 50:
            return TestResult(
                name="Number Frequency Independence",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="Insufficient data (need >= 50 draws)"
            )
        
        # Split by median jackpot
        jackpot_values = [j for _, j in valid_pairs]
        median_jackpot = np.median(jackpot_values)
        
        high_draws = [d for d, j in valid_pairs if j >= median_jackpot]
        low_draws = [d for d, j in valid_pairs if j < median_jackpot]
        
        # Count frequencies
        high_freq = np.zeros(n_max)
        low_freq = np.zeros(n_max)
        
        for draw in high_draws:
            for num in draw:
                if 1 <= num <= n_max:
                    high_freq[num - 1] += 1
        
        for draw in low_draws:
            for num in draw:
                if 1 <= num <= n_max:
                    low_freq[num - 1] += 1
        
        # Chi-squared test
        contingency = np.array([high_freq, low_freq])
        
        # Remove columns with zero counts
        nonzero_cols = (contingency.sum(axis=0) > 0)
        contingency = contingency[:, nonzero_cols]
        
        if contingency.shape[1] < 2:
            return TestResult(
                name="Number Frequency Independence",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="Insufficient variation in data"
            )
        
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        
        return TestResult(
            name="Number Frequency Independence",
            statistic=float(chi2),
            p_value=float(p_value),
            passed=p_value >= self.alpha,
            alpha=self.alpha,
            description="Tests if number frequencies are independent of jackpot level",
            details={
                "n_high_jackpot": len(high_draws),
                "n_low_jackpot": len(low_draws),
                "median_jackpot": float(median_jackpot),
                "chi2_statistic": float(chi2),
                "degrees_of_freedom": int(dof)
            }
        )
    
    def jackpot_quartile_test(self, draws: List[List[int]], 
                               jackpots: List[float],
                               n_max: int) -> TestResult:
        """
        Test if draw characteristics differ across jackpot quartiles.
        
        Uses Kruskal-Wallis test (non-parametric ANOVA).
        """
        valid_pairs = [(d, j) for d, j in zip(draws, jackpots) if j and j > 0]
        
        if len(valid_pairs) < 40:
            return TestResult(
                name="Jackpot Quartile Independence",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="Insufficient data (need >= 40 draws)"
            )
        
        # Split into quartiles
        jackpot_values = np.array([j for _, j in valid_pairs])
        quartiles = np.percentile(jackpot_values, [25, 50, 75])
        
        groups = [[], [], [], []]
        for d, j in valid_pairs:
            draw_sum = sum(d)
            if j < quartiles[0]:
                groups[0].append(draw_sum)
            elif j < quartiles[1]:
                groups[1].append(draw_sum)
            elif j < quartiles[2]:
                groups[2].append(draw_sum)
            else:
                groups[3].append(draw_sum)
        
        # Filter out empty groups
        groups = [g for g in groups if len(g) >= 5]
        
        if len(groups) < 2:
            return TestResult(
                name="Jackpot Quartile Independence",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="Insufficient groups for comparison"
            )
        
        # Kruskal-Wallis test
        h_stat, p_value = stats.kruskal(*groups)
        
        return TestResult(
            name="Jackpot Quartile Independence",
            statistic=float(h_stat),
            p_value=float(p_value),
            passed=p_value >= self.alpha,
            alpha=self.alpha,
            description="Tests if draw sums differ across jackpot quartiles",
            details={
                "n_groups": len(groups),
                "group_sizes": [len(g) for g in groups],
                "group_means": [float(np.mean(g)) for g in groups],
                "quartile_thresholds": [float(q) for q in quartiles],
                "h_statistic": float(h_stat)
            }
        )
    
    def run_all_tests(self, draws: List[List[int]], jackpots: List[float],
                      n_max: int) -> Dict[str, Any]:
        """
        Run all jackpot independence tests.
        
        Args:
            draws: List of draws
            jackpots: List of jackpot amounts
            n_max: Maximum number in pool
            
        Returns:
            Dictionary with all test results
        """
        valid_jackpots = [j for j in jackpots if j and j > 0]
        
        results = {
            "n_draws": len(draws),
            "n_draws_with_jackpot": len(valid_jackpots),
            "n_max": n_max,
            "alpha": self.alpha,
            "jackpot_stats": {
                "min": float(min(valid_jackpots)) if valid_jackpots else None,
                "max": float(max(valid_jackpots)) if valid_jackpots else None,
                "mean": float(np.mean(valid_jackpots)) if valid_jackpots else None,
                "median": float(np.median(valid_jackpots)) if valid_jackpots else None
            },
            "tests": {}
        }
        
        tests = [
            self.sum_correlation_test(draws, jackpots),
            self.range_correlation_test(draws, jackpots),
            self.variance_correlation_test(draws, jackpots),
            self.number_frequency_independence_test(draws, jackpots, n_max),
            self.jackpot_quartile_test(draws, jackpots, n_max)
        ]
        
        n_passed = 0
        for test in tests:
            results["tests"][test.name] = {
                "statistic": test.statistic,
                "p_value": test.p_value,
                "passed": test.passed,
                "description": test.description,
                "details": test.details
            }
            if test.passed:
                n_passed += 1
        
        results["summary"] = {
            "n_tests": len(tests),
            "n_passed": n_passed,
            "pass_rate": n_passed / len(tests) if tests else 0.0,
            "overall_passed": n_passed == len(tests),
            "conclusion": "INDEPENDENT" if n_passed == len(tests) else "POTENTIAL_DEPENDENCY"
        }
        
        return results
