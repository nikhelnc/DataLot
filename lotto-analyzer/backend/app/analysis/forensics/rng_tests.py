"""
RNG (Random Number Generator) tests for detecting software-based generator vulnerabilities.
Tests for LSB bias, modulo bias, periodicity, and birthday paradox violations.
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
    details: Dict[str, Any] = None


class RNGTests:
    """
    Tests targeting vulnerabilities in pseudo-random number generators.
    
    These tests can detect poorly designed PRNGs even without emission order data.
    """
    
    def __init__(self, alpha: float = 0.01, n_permutations: int = 10000):
        """
        Initialize RNG tests.
        
        Args:
            alpha: Significance level
            n_permutations: Number of permutations for permutation tests
        """
        self.alpha = alpha
        self.n_permutations = n_permutations
    
    def lsb_bias_test(self, draws: List[List[int]]) -> TestResult:
        """
        Test for Least Significant Bit bias.
        
        A well-designed RNG should produce numbers with uniformly distributed LSBs.
        Poorly designed PRNGs often have biased low-order bits.
        
        Method: Chi-squared test on LSB distribution
        """
        # Extract all numbers
        all_numbers = []
        for draw in draws:
            all_numbers.extend(draw)
        
        if not all_numbers:
            return TestResult(
                name="LSB Bias",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="No data available"
            )
        
        # Count LSBs (0 or 1)
        lsb_counts = [0, 0]
        for num in all_numbers:
            lsb_counts[num % 2] += 1
        
        # Chi-squared test (expected 50-50)
        expected = len(all_numbers) / 2
        chi2 = sum((obs - expected) ** 2 / expected for obs in lsb_counts)
        p_value = 1 - stats.chi2.cdf(chi2, df=1)
        
        return TestResult(
            name="LSB Bias",
            statistic=float(chi2),
            p_value=float(p_value),
            passed=p_value >= self.alpha,
            alpha=self.alpha,
            description="Tests if least significant bits are uniformly distributed",
            details={
                "n_even": int(lsb_counts[0]),
                "n_odd": int(lsb_counts[1]),
                "proportion_even": float(lsb_counts[0] / len(all_numbers)),
                "expected_proportion": 0.5,
                "total_numbers": len(all_numbers)
            }
        )
    
    def modulo_bias_test(self, draws: List[List[int]], n_max: int) -> TestResult:
        """
        Test for modulo bias.
        
        When a RNG generates numbers in [0, M) and then applies modulo N,
        if M is not a multiple of N, some remainders are more likely.
        
        Method: Chi-squared test on modulo distribution for various moduli
        """
        all_numbers = []
        for draw in draws:
            all_numbers.extend(draw)
        
        if not all_numbers:
            return TestResult(
                name="Modulo Bias",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="No data available"
            )
        
        # Test for common moduli that might reveal bias
        test_moduli = [2, 3, 4, 5, 7, 8, 10]
        modulo_results = {}
        min_p_value = 1.0
        
        for mod in test_moduli:
            if mod > n_max:
                continue
                
            # Count remainders
            remainder_counts = Counter(num % mod for num in all_numbers)
            observed = [remainder_counts.get(i, 0) for i in range(mod)]
            expected = len(all_numbers) / mod
            
            # Chi-squared test
            chi2 = sum((obs - expected) ** 2 / expected for obs in observed)
            p_value = 1 - stats.chi2.cdf(chi2, df=mod - 1)
            
            modulo_results[mod] = {
                "chi2": float(chi2),
                "p_value": float(p_value),
                "passed": p_value >= self.alpha,
                "observed": observed,
                "expected": float(expected)
            }
            
            if p_value < min_p_value:
                min_p_value = p_value
        
        # Combined assessment using Bonferroni correction
        n_tests = len(modulo_results)
        corrected_alpha = self.alpha / n_tests if n_tests > 0 else self.alpha
        overall_passed = all(r["p_value"] >= corrected_alpha for r in modulo_results.values())
        
        return TestResult(
            name="Modulo Bias",
            statistic=float(min_p_value),
            p_value=float(min_p_value),
            passed=overall_passed,
            alpha=self.alpha,
            description="Tests for modulo bias from truncated RNG output",
            details={
                "modulo_results": modulo_results,
                "min_p_value": float(min_p_value),
                "bonferroni_alpha": float(corrected_alpha),
                "n_tests": n_tests
            }
        )
    
    def periodicity_test(self, draws: List[List[int]], 
                         period_candidates: List[int] = None) -> TestResult:
        """
        Test for periodicity in draw sequences.
        
        A PRNG with short period will produce repeating patterns.
        
        Method: Autocorrelation analysis at candidate periods
        
        Args:
            draws: List of draws
            period_candidates: Periods to test (default: [52, 104, 260, 520])
        """
        if period_candidates is None:
            period_candidates = [52, 104, 260, 520]
        
        if len(draws) < max(period_candidates) + 10:
            return TestResult(
                name="Periodicity",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="Insufficient data for periodicity test"
            )
        
        # Convert draws to feature vectors (sum of numbers)
        sums = np.array([sum(draw) for draw in draws])
        
        # Compute autocorrelation at each candidate period
        period_results = {}
        max_autocorr = 0.0
        
        for period in period_candidates:
            if period >= len(sums):
                continue
            
            # Compute autocorrelation
            n = len(sums)
            mean = np.mean(sums)
            var = np.var(sums)
            
            if var == 0:
                autocorr = 0.0
            else:
                autocorr = np.mean((sums[:-period] - mean) * (sums[period:] - mean)) / var
            
            # Under null hypothesis, autocorrelation ~ N(0, 1/n)
            z_score = autocorr * np.sqrt(n - period)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            
            period_results[period] = {
                "autocorrelation": float(autocorr),
                "z_score": float(z_score),
                "p_value": float(p_value),
                "passed": p_value >= self.alpha
            }
            
            if abs(autocorr) > abs(max_autocorr):
                max_autocorr = autocorr
        
        # Overall assessment
        min_p = min(r["p_value"] for r in period_results.values()) if period_results else 1.0
        
        return TestResult(
            name="Periodicity",
            statistic=float(max_autocorr),
            p_value=float(min_p),
            passed=min_p >= self.alpha,
            alpha=self.alpha,
            description="Tests for periodic patterns in draw sequences",
            details={
                "period_results": period_results,
                "max_autocorrelation": float(max_autocorr),
                "n_draws": len(draws)
            }
        )
    
    def birthday_paradox_test(self, draws: List[List[int]], n_max: int, k: int) -> TestResult:
        """
        Birthday Paradox Test.
        
        Tests whether the number of identical draws matches the expected
        frequency under true randomness.
        
        For k numbers drawn from N without replacement, the probability of
        collision (identical draw) follows a known distribution.
        
        Args:
            draws: List of draws
            n_max: Maximum number in pool
            k: Numbers per draw
        """
        n_draws = len(draws)
        
        if n_draws < 50:
            return TestResult(
                name="Birthday Paradox",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="Insufficient draws for birthday test"
            )
        
        # Convert draws to tuples for hashing
        draw_tuples = [tuple(sorted(draw)) for draw in draws]
        
        # Count collisions (identical draws)
        draw_counts = Counter(draw_tuples)
        n_collisions = sum(1 for count in draw_counts.values() if count > 1)
        n_collision_pairs = sum(count * (count - 1) // 2 for count in draw_counts.values())
        
        # Expected collisions under uniform distribution
        # Number of possible draws: C(n_max, k)
        from math import comb
        n_possible = comb(n_max, k)
        
        # Expected number of collision pairs: n_draws * (n_draws - 1) / (2 * n_possible)
        expected_pairs = n_draws * (n_draws - 1) / (2 * n_possible)
        
        # Poisson approximation for collision pairs
        if expected_pairs > 0:
            p_value = 1 - stats.poisson.cdf(n_collision_pairs - 1, expected_pairs)
        else:
            p_value = 1.0 if n_collision_pairs == 0 else 0.0
        
        return TestResult(
            name="Birthday Paradox",
            statistic=float(n_collision_pairs),
            p_value=float(p_value),
            passed=p_value >= self.alpha,
            alpha=self.alpha,
            description="Tests if draw collisions match expected frequency",
            details={
                "n_draws": n_draws,
                "n_unique_draws": len(draw_counts),
                "n_collisions": int(n_collisions),
                "n_collision_pairs": int(n_collision_pairs),
                "expected_collision_pairs": float(expected_pairs),
                "n_possible_draws": int(n_possible)
            }
        )
    
    def transition_matrix_test(self, draws: List[List[int]], n_max: int) -> TestResult:
        """
        Transition Matrix Test.
        
        Tests for sequential dependencies between consecutive draws.
        A PRNG with memory would show non-uniform transition probabilities.
        
        Method: Chi-squared test on transition frequencies
        """
        if len(draws) < 100:
            return TestResult(
                name="Transition Matrix",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="Insufficient draws for transition test"
            )
        
        # Compute transitions between consecutive draws
        # Use sum of draw as a proxy for draw state
        n_bins = 10
        sums = [sum(draw) for draw in draws]
        
        # Bin the sums
        min_sum, max_sum = min(sums), max(sums)
        bin_width = (max_sum - min_sum) / n_bins if max_sum > min_sum else 1
        
        def get_bin(s):
            if bin_width == 0:
                return 0
            b = int((s - min_sum) / bin_width)
            return min(b, n_bins - 1)
        
        binned = [get_bin(s) for s in sums]
        
        # Count transitions
        transition_counts = np.zeros((n_bins, n_bins))
        for i in range(len(binned) - 1):
            transition_counts[binned[i], binned[i + 1]] += 1
        
        # Expected under independence: row_sum * col_sum / total
        row_sums = transition_counts.sum(axis=1)
        col_sums = transition_counts.sum(axis=0)
        total = transition_counts.sum()
        
        if total == 0:
            return TestResult(
                name="Transition Matrix",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="No transitions to analyze"
            )
        
        # Chi-squared statistic
        chi2 = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                expected = row_sums[i] * col_sums[j] / total
                if expected > 0:
                    chi2 += (transition_counts[i, j] - expected) ** 2 / expected
        
        df = (n_bins - 1) ** 2
        p_value = 1 - stats.chi2.cdf(chi2, df)
        
        return TestResult(
            name="Transition Matrix",
            statistic=float(chi2),
            p_value=float(p_value),
            passed=p_value >= self.alpha,
            alpha=self.alpha,
            description="Tests for sequential dependencies between draws",
            details={
                "chi2": float(chi2),
                "df": int(df),
                "n_bins": n_bins,
                "n_transitions": int(total)
            }
        )
    
    def run_all_tests(self, draws: List[List[int]], n_max: int, k: int,
                      period_candidates: List[int] = None) -> Dict[str, Any]:
        """
        Run all RNG tests.
        
        Args:
            draws: List of draws
            n_max: Maximum number in pool
            k: Numbers per draw
            period_candidates: Periods to test for periodicity
            
        Returns:
            Dictionary with all test results
        """
        results = {
            "n_draws": len(draws),
            "n_max": n_max,
            "k": k,
            "alpha": self.alpha,
            "tests": {}
        }
        
        tests = [
            self.lsb_bias_test(draws),
            self.modulo_bias_test(draws, n_max),
            self.periodicity_test(draws, period_candidates),
            self.birthday_paradox_test(draws, n_max, k),
            self.transition_matrix_test(draws, n_max)
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
            "overall_passed": n_passed == len(tests)
        }
        
        return results
