"""
Physical bias tests for lottery ball machines.
Tests for weight bias, position bias, thermal drift, adjacency effects, and mutual information.
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
    details: Dict[str, Any] = None


class PhysicalTests:
    """
    Tests targeting mechanical anomalies documented in physical ball machines.
    
    These tests require emission order data (the temporal sequence in which
    balls are drawn, before sorting).
    """
    
    def __init__(self, alpha: float = 0.01, n_permutations: int = 10000):
        """
        Initialize physical tests.
        
        Args:
            alpha: Significance level
            n_permutations: Number of permutations for permutation tests
        """
        self.alpha = alpha
        self.n_permutations = n_permutations
    
    def weight_bias_test(self, draws: List[List[int]], n_max: int) -> TestResult:
        """
        Test for weight bias in ball selection.
        
        Hypothesis: Lighter balls (higher numbers = less ink) are drawn more often.
        
        Method: Linear regression of frequency ~ number
        
        Args:
            draws: List of draws (sorted numbers)
            n_max: Maximum number in pool
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
                name="Weight Bias",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="No data available"
            )
        
        # Normalize to proportions
        proportions = frequencies / total_balls
        numbers = np.arange(1, n_max + 1)
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(numbers, proportions)
        
        # Determine bias direction
        if slope > 0:
            bias_direction = "high_numbers_favored"
        elif slope < 0:
            bias_direction = "low_numbers_favored"
        else:
            bias_direction = "no_bias"
        
        return TestResult(
            name="Weight Bias",
            statistic=float(slope),
            p_value=float(p_value),
            passed=p_value >= self.alpha,
            alpha=self.alpha,
            description="Tests if ball weight (ink amount) affects selection probability",
            details={
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_value ** 2),
                "std_error": float(std_err),
                "bias_direction": bias_direction,
                "n_draws": len(draws),
                "total_balls": int(total_balls)
            }
        )
    
    def emission_position_bias_test(self, emission_orders: List[List[int]], n_max: int) -> TestResult:
        """
        Test for position bias in emission order.
        
        By De Finetti's exchangeability theorem, in a fair draw without replacement,
        the marginal distribution at each position should be identical (uniform).
        Any deviation indicates mechanical bias.
        
        Method: Chi-squared test per emission position
        
        Args:
            emission_orders: List of emission orders (temporal sequence of balls)
            n_max: Maximum number in pool
        """
        if not emission_orders or not emission_orders[0]:
            return TestResult(
                name="Emission Position Bias",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="No emission order data available"
            )
        
        k = len(emission_orders[0])  # Number of balls per draw
        n_draws = len(emission_orders)
        
        # Count frequency of each number at each position
        position_counts = np.zeros((k, n_max))
        
        for emission in emission_orders:
            for pos, num in enumerate(emission):
                if 1 <= num <= n_max and pos < k:
                    position_counts[pos, num - 1] += 1
        
        # Expected frequency (uniform)
        expected = n_draws / n_max
        
        # Chi-squared test for each position
        chi2_stats = []
        p_values = []
        
        for pos in range(k):
            observed = position_counts[pos]
            chi2, p = stats.chisquare(observed, f_exp=[expected] * n_max)
            chi2_stats.append(float(chi2))
            p_values.append(float(p))
        
        # Combined p-value using Fisher's method
        combined_chi2 = -2 * np.sum(np.log(np.maximum(p_values, 1e-300)))
        combined_p = 1 - stats.chi2.cdf(combined_chi2, 2 * k)
        
        # Find most biased position
        min_p_idx = np.argmin(p_values)
        
        return TestResult(
            name="Emission Position Bias",
            statistic=float(combined_chi2),
            p_value=float(combined_p),
            passed=combined_p >= self.alpha,
            alpha=self.alpha,
            description="Tests if number distribution varies by emission position",
            details={
                "n_positions": k,
                "n_draws": n_draws,
                "chi2_by_position": chi2_stats,
                "p_values_by_position": p_values,
                "most_biased_position": int(min_p_idx + 1),
                "most_biased_p_value": float(p_values[min_p_idx]),
                "position_counts": position_counts.tolist()
            }
        )
    
    def thermal_drift_test(self, emission_orders: List[List[int]], n_max: int) -> TestResult:
        """
        Test for thermal drift during drawing.
        
        Hypothesis: The machine heats up during drawing, changing probabilities
        as emissions progress.
        
        Method: Linear regression of emitted number ~ emission position,
        with permutation test for significance.
        
        Args:
            emission_orders: List of emission orders
            n_max: Maximum number in pool
        """
        if not emission_orders or not emission_orders[0]:
            return TestResult(
                name="Thermal Drift",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="No emission order data available"
            )
        
        k = len(emission_orders[0])
        
        # Flatten: for each emission, record (position, number)
        positions = []
        numbers = []
        
        for emission in emission_orders:
            for pos, num in enumerate(emission):
                positions.append(pos + 1)
                numbers.append(num)
        
        positions = np.array(positions)
        numbers = np.array(numbers)
        
        # Observed slope
        slope, intercept, r_value, p_value_param, std_err = stats.linregress(positions, numbers)
        
        # Permutation test
        observed_slope = abs(slope)
        n_extreme = 0
        
        for _ in range(self.n_permutations):
            perm_numbers = np.random.permutation(numbers)
            perm_slope, _, _, _, _ = stats.linregress(positions, perm_numbers)
            if abs(perm_slope) >= observed_slope:
                n_extreme += 1
        
        p_value_perm = (n_extreme + 1) / (self.n_permutations + 1)
        
        # Compute mean number by position
        mean_by_position = []
        for pos in range(1, k + 1):
            mask = positions == pos
            if np.any(mask):
                mean_by_position.append(float(np.mean(numbers[mask])))
            else:
                mean_by_position.append(None)
        
        return TestResult(
            name="Thermal Drift",
            statistic=float(slope),
            p_value=float(p_value_perm),
            passed=p_value_perm >= self.alpha,
            alpha=self.alpha,
            description="Tests if emitted numbers trend with emission position (thermal effect)",
            details={
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_value ** 2),
                "p_value_parametric": float(p_value_param),
                "p_value_permutation": float(p_value_perm),
                "n_permutations": self.n_permutations,
                "mean_by_position": mean_by_position,
                "n_emissions": len(emission_orders)
            }
        )
    
    def adjacency_avoidance_test(self, emission_orders: List[List[int]], n_max: int,
                                  n_simulations: int = 50000) -> TestResult:
        """
        Test for adjacency effects in consecutive emissions.
        
        Hypothesis: Physically adjacent balls (consecutive numbers) tend to
        attract or repel each other.
        
        Method: Compare frequency of adjacent pairs (i, i+1) emitted consecutively
        vs expected frequency from Monte Carlo simulation.
        
        Args:
            emission_orders: List of emission orders
            n_max: Maximum number in pool
            n_simulations: Number of Monte Carlo simulations
        """
        if not emission_orders or not emission_orders[0]:
            return TestResult(
                name="Adjacency Avoidance",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="No emission order data available"
            )
        
        k = len(emission_orders[0])
        n_draws = len(emission_orders)
        
        # Count observed adjacent pairs in consecutive positions
        observed_adjacent = 0
        total_consecutive_pairs = 0
        
        for emission in emission_orders:
            for i in range(len(emission) - 1):
                total_consecutive_pairs += 1
                if abs(emission[i] - emission[i + 1]) == 1:
                    observed_adjacent += 1
        
        # Monte Carlo simulation
        simulated_counts = []
        
        for _ in range(n_simulations):
            sim_adjacent = 0
            for _ in range(n_draws):
                # Simulate random draw without replacement
                sim_emission = np.random.choice(range(1, n_max + 1), size=k, replace=False)
                for i in range(len(sim_emission) - 1):
                    if abs(sim_emission[i] - sim_emission[i + 1]) == 1:
                        sim_adjacent += 1
            simulated_counts.append(sim_adjacent)
        
        simulated_counts = np.array(simulated_counts)
        expected = np.mean(simulated_counts)
        std_sim = np.std(simulated_counts)
        
        # Two-tailed p-value
        n_extreme = np.sum(np.abs(simulated_counts - expected) >= abs(observed_adjacent - expected))
        p_value = (n_extreme + 1) / (n_simulations + 1)
        
        # Effect direction
        if observed_adjacent > expected:
            effect = "attraction"
        elif observed_adjacent < expected:
            effect = "repulsion"
        else:
            effect = "none"
        
        return TestResult(
            name="Adjacency Avoidance",
            statistic=float(observed_adjacent - expected),
            p_value=float(p_value),
            passed=p_value >= self.alpha,
            alpha=self.alpha,
            description="Tests if adjacent numbers attract or repel in consecutive emissions",
            details={
                "observed_adjacent_pairs": int(observed_adjacent),
                "expected_adjacent_pairs": float(expected),
                "std_simulated": float(std_sim),
                "z_score": float((observed_adjacent - expected) / std_sim) if std_sim > 0 else 0.0,
                "effect_direction": effect,
                "n_simulations": n_simulations,
                "total_consecutive_pairs": int(total_consecutive_pairs)
            }
        )
    
    def mutual_information_test(self, emission_orders: List[List[int]], n_max: int,
                                 n_permutations: int = 1000) -> TestResult:
        """
        Test for intra-draw dependency using mutual information.
        
        Hypothesis: The ball emitted at position p is influenced by the ball
        at position p-1, beyond the without-replacement constraint.
        
        Method: Normalized mutual information between consecutive positions,
        with null distribution from permutation.
        
        Args:
            emission_orders: List of emission orders
            n_max: Maximum number in pool
            n_permutations: Number of permutations for null distribution
        """
        if not emission_orders or len(emission_orders[0]) < 2:
            return TestResult(
                name="Mutual Information",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="Insufficient emission order data"
            )
        
        k = len(emission_orders[0])
        
        # Collect consecutive pairs
        prev_balls = []
        curr_balls = []
        
        for emission in emission_orders:
            for i in range(1, len(emission)):
                prev_balls.append(emission[i - 1])
                curr_balls.append(emission[i])
        
        prev_balls = np.array(prev_balls)
        curr_balls = np.array(curr_balls)
        
        def compute_mi(x: np.ndarray, y: np.ndarray, n_bins: int) -> float:
            """Compute mutual information between two arrays"""
            # Create joint histogram
            joint_hist, _, _ = np.histogram2d(x, y, bins=n_bins)
            joint_prob = joint_hist / joint_hist.sum()
            
            # Marginal probabilities
            p_x = joint_prob.sum(axis=1)
            p_y = joint_prob.sum(axis=0)
            
            # Mutual information
            mi = 0.0
            for i in range(n_bins):
                for j in range(n_bins):
                    if joint_prob[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                        mi += joint_prob[i, j] * np.log2(joint_prob[i, j] / (p_x[i] * p_y[j]))
            
            return mi
        
        # Compute observed MI
        n_bins = min(20, n_max)
        observed_mi = compute_mi(prev_balls, curr_balls, n_bins)
        
        # Permutation test
        permuted_mis = []
        for _ in range(n_permutations):
            perm_curr = np.random.permutation(curr_balls)
            perm_mi = compute_mi(prev_balls, perm_curr, n_bins)
            permuted_mis.append(perm_mi)
        
        permuted_mis = np.array(permuted_mis)
        
        # P-value (one-tailed, testing for excess MI)
        p_value = (np.sum(permuted_mis >= observed_mi) + 1) / (n_permutations + 1)
        
        return TestResult(
            name="Mutual Information",
            statistic=float(observed_mi),
            p_value=float(p_value),
            passed=p_value >= self.alpha,
            alpha=self.alpha,
            description="Tests for dependency between consecutive emission positions",
            details={
                "observed_mi": float(observed_mi),
                "mean_permuted_mi": float(np.mean(permuted_mis)),
                "std_permuted_mi": float(np.std(permuted_mis)),
                "n_permutations": n_permutations,
                "n_pairs": len(prev_balls)
            }
        )
    
    def run_all_tests(self, draws: List[List[int]], emission_orders: Optional[List[List[int]]],
                      n_max: int) -> Dict[str, Any]:
        """
        Run all physical bias tests.
        
        Args:
            draws: List of draws (sorted numbers)
            emission_orders: List of emission orders (may be None)
            n_max: Maximum number in pool
            
        Returns:
            Dictionary with all test results
        """
        results = {
            "n_draws": len(draws),
            "n_max": n_max,
            "has_emission_order": emission_orders is not None and len(emission_orders) > 0,
            "alpha": self.alpha,
            "tests": {}
        }
        
        # Weight bias test (uses sorted draws)
        weight_result = self.weight_bias_test(draws, n_max)
        results["tests"][weight_result.name] = {
            "statistic": weight_result.statistic,
            "p_value": weight_result.p_value,
            "passed": weight_result.passed,
            "description": weight_result.description,
            "details": weight_result.details
        }
        
        # Tests requiring emission order
        if emission_orders and len(emission_orders) > 0:
            emission_tests = [
                self.emission_position_bias_test(emission_orders, n_max),
                self.thermal_drift_test(emission_orders, n_max),
                self.adjacency_avoidance_test(emission_orders, n_max),
                self.mutual_information_test(emission_orders, n_max)
            ]
            
            for test in emission_tests:
                results["tests"][test.name] = {
                    "statistic": test.statistic,
                    "p_value": test.p_value,
                    "passed": test.passed,
                    "description": test.description,
                    "details": test.details
                }
        
        # Summary
        all_tests = list(results["tests"].values())
        n_passed = sum(1 for t in all_tests if t["passed"])
        
        results["summary"] = {
            "n_tests": len(all_tests),
            "n_passed": n_passed,
            "pass_rate": n_passed / len(all_tests) if all_tests else 0.0,
            "overall_passed": n_passed == len(all_tests)
        }
        
        return results
