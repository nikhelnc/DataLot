"""
Clustering tests for fraud detection.
Tests for abnormal clustering patterns in lottery draws.
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform
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
    severity: str = "INFO"
    details: Dict[str, Any] = None


class ClusteringTests:
    """
    Tests for detecting abnormal clustering in lottery draws.
    
    Clustering can indicate:
    - Manipulation (draws too similar)
    - Generator bias (certain patterns favored)
    - Data quality issues (duplicates, errors)
    """
    
    def __init__(self, alpha: float = 0.01, n_simulations: int = 1000):
        """
        Initialize clustering tests.
        
        Args:
            alpha: Significance level
            n_simulations: Number of Monte Carlo simulations
        """
        self.alpha = alpha
        self.n_simulations = n_simulations
    
    def _jaccard_distance(self, set1: set, set2: set) -> float:
        """Compute Jaccard distance between two sets"""
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return 1 - (intersection / union) if union > 0 else 0
    
    def _hamming_distance(self, draw1: List[int], draw2: List[int]) -> int:
        """Compute number of different elements between two draws"""
        return len(set(draw1) ^ set(draw2))
    
    def duplicate_detection_test(self, draws: List[List[int]]) -> TestResult:
        """
        Test for exact duplicate draws.
        
        While duplicates can occur by chance, too many indicate problems.
        """
        if not draws:
            return TestResult(
                name="Duplicate Detection",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="No data available"
            )
        
        # Convert to tuples for hashing
        draw_tuples = [tuple(sorted(draw)) for draw in draws]
        draw_counts = Counter(draw_tuples)
        
        n_duplicates = sum(1 for count in draw_counts.values() if count > 1)
        n_duplicate_instances = sum(count - 1 for count in draw_counts.values() if count > 1)
        
        # Expected duplicates under uniformity (birthday problem)
        # For large pools, expected duplicates ≈ 0 for reasonable n_draws
        # Use Poisson approximation
        from math import comb
        k = len(draws[0]) if draws else 0
        n_max = max(max(d) for d in draws) if draws else 45
        n_possible = comb(n_max, k)
        
        n = len(draws)
        expected_pairs = n * (n - 1) / (2 * n_possible)
        
        # Poisson test
        if expected_pairs > 0:
            p_value = 1 - stats.poisson.cdf(n_duplicate_instances - 1, expected_pairs)
        else:
            p_value = 0.0 if n_duplicate_instances > 0 else 1.0
        
        if n_duplicate_instances > 5:
            severity = "CRITICAL"
        elif n_duplicate_instances > 2:
            severity = "HIGH"
        elif n_duplicate_instances > 0:
            severity = "WARNING"
        else:
            severity = "INFO"
        
        return TestResult(
            name="Duplicate Detection",
            statistic=float(n_duplicate_instances),
            p_value=float(p_value),
            passed=n_duplicate_instances <= 1,  # Allow 1 duplicate by chance
            alpha=self.alpha,
            description="Tests for exact duplicate draws",
            severity=severity,
            details={
                "n_draws": len(draws),
                "n_unique_draws": len(draw_counts),
                "n_duplicated_patterns": n_duplicates,
                "n_duplicate_instances": n_duplicate_instances,
                "expected_duplicates": float(expected_pairs),
                "duplicate_draws": [
                    {"draw": list(draw), "count": count}
                    for draw, count in draw_counts.items() if count > 1
                ][:10]  # Limit to first 10
            }
        )
    
    def near_duplicate_test(self, draws: List[List[int]], threshold: int = 1) -> TestResult:
        """
        Test for near-duplicate draws (differing by only 1-2 numbers).
        
        Too many near-duplicates can indicate manipulation or bias.
        
        Args:
            draws: List of draws
            threshold: Maximum Hamming distance to consider as near-duplicate
        """
        if len(draws) < 10:
            return TestResult(
                name="Near-Duplicate Detection",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="Insufficient draws for near-duplicate test"
            )
        
        # Count near-duplicates
        n_near_duplicates = 0
        near_duplicate_pairs = []
        
        for i in range(len(draws)):
            for j in range(i + 1, len(draws)):
                dist = self._hamming_distance(draws[i], draws[j])
                if dist <= threshold * 2:  # Hamming counts both additions and removals
                    n_near_duplicates += 1
                    if len(near_duplicate_pairs) < 10:
                        near_duplicate_pairs.append({
                            "draw1_idx": i,
                            "draw2_idx": j,
                            "distance": dist
                        })
        
        # Monte Carlo for expected near-duplicates
        k = len(draws[0])
        n_max = max(max(d) for d in draws)
        
        simulated_counts = []
        for _ in range(min(100, self.n_simulations)):  # Reduced for speed
            sim_draws = [
                sorted(np.random.choice(range(1, n_max + 1), size=k, replace=False).tolist())
                for _ in range(len(draws))
            ]
            sim_near_dup = 0
            for i in range(len(sim_draws)):
                for j in range(i + 1, min(i + 50, len(sim_draws))):  # Sample pairs
                    dist = self._hamming_distance(sim_draws[i], sim_draws[j])
                    if dist <= threshold * 2:
                        sim_near_dup += 1
            simulated_counts.append(sim_near_dup)
        
        expected = np.mean(simulated_counts)
        std_sim = np.std(simulated_counts)
        
        if std_sim > 0:
            z_score = (n_near_duplicates - expected) / std_sim
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        else:
            p_value = 1.0 if n_near_duplicates == expected else 0.0
        
        if p_value < 0.001:
            severity = "CRITICAL"
        elif p_value < 0.01:
            severity = "HIGH"
        elif p_value < 0.05:
            severity = "WARNING"
        else:
            severity = "INFO"
        
        return TestResult(
            name="Near-Duplicate Detection",
            statistic=float(n_near_duplicates),
            p_value=float(p_value),
            passed=p_value >= self.alpha,
            alpha=self.alpha,
            description=f"Tests for draws differing by <= {threshold} number(s)",
            severity=severity,
            details={
                "n_near_duplicates": n_near_duplicates,
                "expected_near_duplicates": float(expected),
                "threshold": threshold,
                "sample_pairs": near_duplicate_pairs
            }
        )
    
    def temporal_clustering_test(self, draws: List[List[int]], 
                                  window_size: int = 10) -> TestResult:
        """
        Test for temporal clustering (similar draws appearing close in time).
        
        Args:
            draws: List of draws in chronological order
            window_size: Size of sliding window
        """
        if len(draws) < window_size * 2:
            return TestResult(
                name="Temporal Clustering",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="Insufficient draws for temporal clustering test"
            )
        
        # Compute average Jaccard distance within windows
        window_distances = []
        
        for start in range(0, len(draws) - window_size, window_size // 2):
            window = draws[start:start + window_size]
            distances = []
            for i in range(len(window)):
                for j in range(i + 1, len(window)):
                    dist = self._jaccard_distance(set(window[i]), set(window[j]))
                    distances.append(dist)
            if distances:
                window_distances.append(np.mean(distances))
        
        if len(window_distances) < 5:
            return TestResult(
                name="Temporal Clustering",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="Insufficient windows for temporal clustering test"
            )
        
        # Test for variance in window distances
        # High variance suggests some periods have more similar draws
        observed_var = np.var(window_distances)
        observed_mean = np.mean(window_distances)
        
        # Monte Carlo for expected variance
        k = len(draws[0])
        n_max = max(max(d) for d in draws)
        
        simulated_vars = []
        for _ in range(min(100, self.n_simulations)):
            sim_draws = [
                sorted(np.random.choice(range(1, n_max + 1), size=k, replace=False).tolist())
                for _ in range(len(draws))
            ]
            sim_window_distances = []
            for start in range(0, len(sim_draws) - window_size, window_size // 2):
                window = sim_draws[start:start + window_size]
                distances = []
                for i in range(len(window)):
                    for j in range(i + 1, len(window)):
                        dist = self._jaccard_distance(set(window[i]), set(window[j]))
                        distances.append(dist)
                if distances:
                    sim_window_distances.append(np.mean(distances))
            if sim_window_distances:
                simulated_vars.append(np.var(sim_window_distances))
        
        if simulated_vars:
            p_value = np.mean(np.array(simulated_vars) >= observed_var)
        else:
            p_value = 1.0
        
        if p_value < 0.001:
            severity = "CRITICAL"
        elif p_value < 0.01:
            severity = "HIGH"
        elif p_value < 0.05:
            severity = "WARNING"
        else:
            severity = "INFO"
        
        return TestResult(
            name="Temporal Clustering",
            statistic=float(observed_var),
            p_value=float(p_value),
            passed=p_value >= self.alpha,
            alpha=self.alpha,
            description="Tests for periods with unusually similar draws",
            severity=severity,
            details={
                "window_size": window_size,
                "n_windows": len(window_distances),
                "mean_window_distance": float(observed_mean),
                "variance_window_distance": float(observed_var),
                "window_distances": [float(d) for d in window_distances]
            }
        )
    
    def number_co_occurrence_test(self, draws: List[List[int]], n_max: int) -> TestResult:
        """
        Test for abnormal co-occurrence patterns.
        
        Some number pairs appearing together too often or too rarely.
        """
        if len(draws) < 50:
            return TestResult(
                name="Number Co-occurrence",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="Insufficient draws for co-occurrence test"
            )
        
        k = len(draws[0])
        
        # Build co-occurrence matrix
        cooccurrence = np.zeros((n_max, n_max))
        for draw in draws:
            for i, num1 in enumerate(draw):
                for num2 in draw[i + 1:]:
                    if 1 <= num1 <= n_max and 1 <= num2 <= n_max:
                        cooccurrence[num1 - 1, num2 - 1] += 1
                        cooccurrence[num2 - 1, num1 - 1] += 1
        
        # Expected co-occurrence under uniformity
        # E[co-occur(i,j)] = n_draws * C(k,2) * 2 / C(n_max,2) for i≠j
        from math import comb
        n_draws = len(draws)
        expected_cooccur = n_draws * comb(k, 2) * 2 / comb(n_max, 2)
        
        # Extract upper triangle (excluding diagonal)
        upper_tri = cooccurrence[np.triu_indices(n_max, k=1)]
        
        # Chi-squared test
        chi2 = np.sum((upper_tri - expected_cooccur) ** 2 / expected_cooccur)
        df = len(upper_tri) - 1
        p_value = 1 - stats.chi2.cdf(chi2, df)
        
        # Find most anomalous pairs
        deviations = (upper_tri - expected_cooccur) / np.sqrt(expected_cooccur)
        indices = np.triu_indices(n_max, k=1)
        
        anomalous_pairs = []
        sorted_indices = np.argsort(np.abs(deviations))[::-1]
        for idx in sorted_indices[:10]:
            i, j = indices[0][idx], indices[1][idx]
            anomalous_pairs.append({
                "numbers": [int(i + 1), int(j + 1)],
                "observed": int(cooccurrence[i, j]),
                "expected": float(expected_cooccur),
                "z_score": float(deviations[idx])
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
            name="Number Co-occurrence",
            statistic=float(chi2),
            p_value=float(p_value),
            passed=p_value >= self.alpha,
            alpha=self.alpha,
            description="Tests for abnormal number pair frequencies",
            severity=severity,
            details={
                "expected_cooccurrence": float(expected_cooccur),
                "chi2_statistic": float(chi2),
                "df": int(df),
                "most_anomalous_pairs": anomalous_pairs
            }
        )
    
    def run_all_tests(self, draws: List[List[int]], n_max: int) -> Dict[str, Any]:
        """
        Run all clustering tests.
        
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
            self.duplicate_detection_test(draws),
            self.near_duplicate_test(draws),
            self.temporal_clustering_test(draws),
            self.number_co_occurrence_test(draws, n_max)
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
