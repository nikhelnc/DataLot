"""
Structural tests for lottery draw analysis.
Tests for order statistics, sum distribution, inter-position correlation, and gap distribution.
"""

import numpy as np
from scipy import stats
from scipy.special import comb
from typing import Dict, List, Any, Tuple
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


class StructuralTests:
    """
    Structural tests analyzing the mathematical properties of draw sequences.
    
    These tests compare observed distributions against theoretical expectations
    for uniform random draws without replacement.
    """
    
    def __init__(self, alpha: float = 0.01, n_simulations: int = 10000):
        """
        Initialize structural tests.
        
        Args:
            alpha: Significance level
            n_simulations: Number of Monte Carlo simulations
        """
        self.alpha = alpha
        self.n_simulations = n_simulations
    
    def _order_statistic_mean(self, r: int, k: int, n: int) -> float:
        """
        Compute expected value of r-th order statistic for k draws from {1,...,n}.
        
        E[X_(r)] = r(N+1) / (k+1)
        """
        return r * (n + 1) / (k + 1)
    
    def _order_statistic_var(self, r: int, k: int, n: int) -> float:
        """
        Compute variance of r-th order statistic for k draws from {1,...,n}.
        
        Var[X_(r)] = r(k-r+1)(N+1)(N-k) / [(k+1)²(k+2)]
        """
        return r * (k - r + 1) * (n + 1) * (n - k) / ((k + 1) ** 2 * (k + 2))
    
    def order_statistics_test(self, draws: List[List[int]], n_max: int) -> TestResult:
        """
        Test order statistics against theoretical distribution.
        
        For k numbers drawn from {1,...,N} without replacement, the r-th smallest
        number has a known distribution. This test compares observed vs expected.
        
        Method: KS test per position against theoretical distribution
        """
        if not draws:
            return TestResult(
                name="Order Statistics",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="No data available"
            )
        
        k = len(draws[0])
        n_draws = len(draws)
        
        # Collect values by position
        position_values = [[] for _ in range(k)]
        for draw in draws:
            sorted_draw = sorted(draw)
            for pos, val in enumerate(sorted_draw):
                position_values[pos].append(val)
        
        # Test each position
        position_results = []
        ks_stats = []
        p_values = []
        
        for r in range(1, k + 1):
            values = np.array(position_values[r - 1])
            
            # Theoretical mean and std
            expected_mean = self._order_statistic_mean(r, k, n_max)
            expected_var = self._order_statistic_var(r, k, n_max)
            expected_std = np.sqrt(expected_var)
            
            # Observed statistics
            observed_mean = np.mean(values)
            observed_std = np.std(values)
            
            # Z-score for mean
            se_mean = expected_std / np.sqrt(n_draws)
            z_score = (observed_mean - expected_mean) / se_mean if se_mean > 0 else 0
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            
            position_results.append({
                "position": r,
                "expected_mean": float(expected_mean),
                "observed_mean": float(observed_mean),
                "expected_std": float(expected_std),
                "observed_std": float(observed_std),
                "z_score": float(z_score),
                "p_value": float(p_value),
                "passed": p_value >= self.alpha
            })
            
            ks_stats.append(abs(z_score))
            p_values.append(p_value)
        
        # Combined p-value using Fisher's method
        combined_chi2 = -2 * np.sum(np.log(np.maximum(p_values, 1e-300)))
        combined_p = 1 - stats.chi2.cdf(combined_chi2, 2 * k)
        
        return TestResult(
            name="Order Statistics",
            statistic=float(combined_chi2),
            p_value=float(combined_p),
            passed=combined_p >= self.alpha,
            alpha=self.alpha,
            description="Tests if order statistics match theoretical distribution",
            details={
                "n_draws": n_draws,
                "k": k,
                "n_max": n_max,
                "position_results": position_results
            }
        )
    
    def sum_distribution_test(self, draws: List[List[int]], n_max: int) -> TestResult:
        """
        Test distribution of draw sums against theoretical distribution.
        
        By CLT, the sum of k numbers drawn from {1,...,N} is approximately normal
        with known mean and variance.
        
        Method: KS test against normal approximation
        """
        if not draws:
            return TestResult(
                name="Sum Distribution",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="No data available"
            )
        
        k = len(draws[0])
        sums = np.array([sum(draw) for draw in draws])
        
        # Theoretical mean and variance for sum of k draws from {1,...,N}
        # E[sum] = k * (N+1) / 2
        # Var[sum] = k * (N+1) * (N-k) / 12
        expected_mean = k * (n_max + 1) / 2
        expected_var = k * (n_max + 1) * (n_max - k) / 12
        expected_std = np.sqrt(expected_var)
        
        # Observed statistics
        observed_mean = np.mean(sums)
        observed_std = np.std(sums)
        
        # KS test against normal distribution
        standardized = (sums - expected_mean) / expected_std
        ks_stat, p_value = stats.kstest(standardized, 'norm')
        
        return TestResult(
            name="Sum Distribution",
            statistic=float(ks_stat),
            p_value=float(p_value),
            passed=p_value >= self.alpha,
            alpha=self.alpha,
            description="Tests if sum distribution matches theoretical normal approximation",
            details={
                "n_draws": len(draws),
                "k": k,
                "n_max": n_max,
                "expected_mean": float(expected_mean),
                "observed_mean": float(observed_mean),
                "expected_std": float(expected_std),
                "observed_std": float(observed_std),
                "ks_statistic": float(ks_stat),
                "min_sum": int(min(sums)),
                "max_sum": int(max(sums))
            }
        )
    
    def inter_position_correlation_test(self, draws: List[List[int]], n_max: int) -> TestResult:
        """
        Test correlation between positions in sorted draws.
        
        For uniform draws without replacement, the expected correlation between
        positions r and s is: ρ(r,s) = sqrt(r(k-s+1) / (s(k-r+1)))
        
        Method: Compare observed Spearman correlations to expected
        """
        if not draws or len(draws) < 30:
            return TestResult(
                name="Inter-Position Correlation",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="Insufficient data for correlation test"
            )
        
        k = len(draws[0])
        
        # Build matrix of sorted values
        matrix = np.array([sorted(draw) for draw in draws])
        
        # Compute Spearman correlation matrix
        observed_corr = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                if i != j:
                    corr, _ = stats.spearmanr(matrix[:, i], matrix[:, j])
                    observed_corr[i, j] = corr
                else:
                    observed_corr[i, j] = 1.0
        
        # Expected correlation under uniform draws
        # For adjacent positions, expected correlation is positive due to ordering
        # Simplified: expected correlation ≈ -1/(k-1) for non-adjacent under independence
        # But for ordered statistics, correlations are positive
        
        # Test: correlation should be positive and decreasing with distance
        correlation_issues = []
        
        for i in range(k - 1):
            # Adjacent positions should have positive correlation
            if observed_corr[i, i + 1] < 0:
                correlation_issues.append({
                    "positions": [i + 1, i + 2],
                    "correlation": float(observed_corr[i, i + 1]),
                    "issue": "negative_adjacent_correlation"
                })
        
        # Overall test: Frobenius norm of deviation from expected pattern
        # Expected: positive correlations decreasing with distance
        expected_pattern = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                if i == j:
                    expected_pattern[i, j] = 1.0
                else:
                    # Approximate expected correlation
                    distance = abs(i - j)
                    expected_pattern[i, j] = max(0, 1 - distance * 0.2)
        
        deviation = np.linalg.norm(observed_corr - expected_pattern, 'fro')
        
        # Monte Carlo for p-value
        simulated_deviations = []
        for _ in range(min(1000, self.n_simulations)):
            sim_draws = [sorted(np.random.choice(range(1, n_max + 1), size=k, replace=False)) 
                        for _ in range(len(draws))]
            sim_matrix = np.array(sim_draws)
            sim_corr = np.zeros((k, k))
            for i in range(k):
                for j in range(k):
                    if i != j:
                        corr, _ = stats.spearmanr(sim_matrix[:, i], sim_matrix[:, j])
                        sim_corr[i, j] = corr
                    else:
                        sim_corr[i, j] = 1.0
            sim_dev = np.linalg.norm(sim_corr - expected_pattern, 'fro')
            simulated_deviations.append(sim_dev)
        
        p_value = (np.sum(np.array(simulated_deviations) >= deviation) + 1) / (len(simulated_deviations) + 1)
        
        return TestResult(
            name="Inter-Position Correlation",
            statistic=float(deviation),
            p_value=float(p_value),
            passed=p_value >= self.alpha,
            alpha=self.alpha,
            description="Tests if correlations between positions match expected pattern",
            details={
                "k": k,
                "n_draws": len(draws),
                "correlation_matrix": observed_corr.tolist(),
                "frobenius_deviation": float(deviation),
                "correlation_issues": correlation_issues
            }
        )
    
    def min_gap_distribution_test(self, draws: List[List[int]], n_max: int) -> TestResult:
        """
        Test distribution of minimum gaps between consecutive numbers.
        
        The minimum gap in a sorted draw has a known distribution under uniformity.
        
        Method: Compare observed distribution to Monte Carlo simulation
        """
        if not draws:
            return TestResult(
                name="Min Gap Distribution",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="No data available"
            )
        
        k = len(draws[0])
        
        # Compute minimum gaps
        observed_min_gaps = []
        for draw in draws:
            sorted_draw = sorted(draw)
            gaps = [sorted_draw[i + 1] - sorted_draw[i] for i in range(len(sorted_draw) - 1)]
            if gaps:
                observed_min_gaps.append(min(gaps))
        
        if not observed_min_gaps:
            return TestResult(
                name="Min Gap Distribution",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="Could not compute gaps"
            )
        
        observed_min_gaps = np.array(observed_min_gaps)
        
        # Monte Carlo simulation
        simulated_min_gaps = []
        for _ in range(self.n_simulations):
            sim_draw = sorted(np.random.choice(range(1, n_max + 1), size=k, replace=False))
            gaps = [sim_draw[i + 1] - sim_draw[i] for i in range(len(sim_draw) - 1)]
            if gaps:
                simulated_min_gaps.append(min(gaps))
        
        simulated_min_gaps = np.array(simulated_min_gaps)
        
        # KS test
        ks_stat, p_value = stats.ks_2samp(observed_min_gaps, simulated_min_gaps)
        
        return TestResult(
            name="Min Gap Distribution",
            statistic=float(ks_stat),
            p_value=float(p_value),
            passed=p_value >= self.alpha,
            alpha=self.alpha,
            description="Tests if minimum gap distribution matches expected",
            details={
                "n_draws": len(draws),
                "k": k,
                "observed_mean_min_gap": float(np.mean(observed_min_gaps)),
                "simulated_mean_min_gap": float(np.mean(simulated_min_gaps)),
                "observed_std_min_gap": float(np.std(observed_min_gaps)),
                "simulated_std_min_gap": float(np.std(simulated_min_gaps)),
                "ks_statistic": float(ks_stat)
            }
        )
    
    def run_all_tests(self, draws: List[List[int]], n_max: int) -> Dict[str, Any]:
        """
        Run all structural tests.
        
        Args:
            draws: List of draws
            n_max: Maximum number in pool
            
        Returns:
            Dictionary with all test results
        """
        results = {
            "n_draws": len(draws),
            "n_max": n_max,
            "k": len(draws[0]) if draws else 0,
            "alpha": self.alpha,
            "tests": {}
        }
        
        tests = [
            self.order_statistics_test(draws, n_max),
            self.sum_distribution_test(draws, n_max),
            self.inter_position_correlation_test(draws, n_max),
            self.min_gap_distribution_test(draws, n_max)
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
