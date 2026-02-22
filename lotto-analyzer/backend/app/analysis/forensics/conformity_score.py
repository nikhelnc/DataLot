"""
Conformity score calculator for generator forensics.
Computes a global conformity score with Monte Carlo confidence intervals.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import hashlib
import json


@dataclass
class ConformityResult:
    """Result of conformity score calculation"""
    score: float
    ci_low: float
    ci_high: float
    n_simulations: int
    interpretation: str
    category_scores: Dict[str, float]
    details: Dict[str, Any]


class ConformityScoreCalculator:
    """
    Calculates a global conformity score from forensic test results.
    
    The score is calibrated via Monte Carlo simulation to determine
    the expected range for a truly random generator.
    """
    
    # Category weights (must sum to 1.0)
    CATEGORY_WEIGHTS = {
        "nist_tests": 0.25,
        "physical_tests": 0.35,
        "rng_tests": 0.25,
        "structural_tests": 0.15
    }
    
    def __init__(self, n_simulations: int = 1000, seed: int = 42):
        """
        Initialize conformity score calculator.
        
        Args:
            n_simulations: Number of Monte Carlo simulations for CI
            seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.seed = seed
    
    def _compute_category_score(self, test_results: Dict[str, Any]) -> float:
        """
        Compute score for a single category of tests.
        
        Score = proportion of tests passed
        """
        if not test_results or "tests" not in test_results:
            return 1.0  # No tests = assume conforming
        
        tests = test_results["tests"]
        if not tests:
            return 1.0
        
        n_passed = sum(1 for t in tests.values() if t.get("passed", False))
        return n_passed / len(tests)
    
    def compute_score(self, 
                      nist_results: Optional[Dict[str, Any]] = None,
                      physical_results: Optional[Dict[str, Any]] = None,
                      rng_results: Optional[Dict[str, Any]] = None,
                      structural_results: Optional[Dict[str, Any]] = None) -> float:
        """
        Compute weighted conformity score from test results.
        
        Args:
            nist_results: Results from NIST tests
            physical_results: Results from physical bias tests
            rng_results: Results from RNG tests
            structural_results: Results from structural tests
            
        Returns:
            Weighted conformity score between 0 and 1
        """
        category_scores = {}
        
        # Compute score for each category
        if nist_results:
            category_scores["nist_tests"] = self._compute_category_score(nist_results)
        else:
            category_scores["nist_tests"] = 1.0
            
        if physical_results:
            category_scores["physical_tests"] = self._compute_category_score(physical_results)
        else:
            category_scores["physical_tests"] = 1.0
            
        if rng_results:
            category_scores["rng_tests"] = self._compute_category_score(rng_results)
        else:
            category_scores["rng_tests"] = 1.0
            
        if structural_results:
            category_scores["structural_tests"] = self._compute_category_score(structural_results)
        else:
            category_scores["structural_tests"] = 1.0
        
        # Weighted average
        score = sum(
            self.CATEGORY_WEIGHTS[cat] * category_scores[cat]
            for cat in self.CATEGORY_WEIGHTS
        )
        
        return score, category_scores
    
    def compute_monte_carlo_ci(self, n_draws: int, n_max: int, k: int,
                                has_emission_order: bool = False,
                                alpha: float = 0.05) -> Tuple[float, float, List[float]]:
        """
        Compute confidence interval for conformity score via Monte Carlo.
        
        Generates synthetic random datasets and computes the conformity score
        for each, then returns the empirical confidence interval.
        
        Args:
            n_draws: Number of draws in the dataset
            n_max: Maximum number in pool
            k: Numbers per draw
            has_emission_order: Whether emission order data is available
            alpha: Significance level for CI (default 0.05 for 95% CI)
            
        Returns:
            Tuple of (ci_low, ci_high, all_scores)
        """
        from .nist_tests import NISTTests
        from .physical_tests import PhysicalTests
        from .rng_tests import RNGTests
        from .structural_tests import StructuralTests
        
        np.random.seed(self.seed)
        
        scores = []
        
        for _ in range(self.n_simulations):
            # Generate random draws
            draws = [
                sorted(np.random.choice(range(1, n_max + 1), size=k, replace=False).tolist())
                for _ in range(n_draws)
            ]
            
            # Generate emission orders if needed
            emission_orders = None
            if has_emission_order:
                emission_orders = [
                    np.random.permutation(draw).tolist()
                    for draw in draws
                ]
            
            # Run tests
            nist = NISTTests(alpha=0.01)
            physical = PhysicalTests(alpha=0.01, n_permutations=100)  # Reduced for speed
            rng = RNGTests(alpha=0.01, n_permutations=100)
            structural = StructuralTests(alpha=0.01, n_simulations=100)
            
            nist_results = nist.run_all_tests(draws, n_max)
            physical_results = physical.run_all_tests(draws, emission_orders, n_max)
            rng_results = rng.run_all_tests(draws, n_max, k)
            structural_results = structural.run_all_tests(draws, n_max)
            
            score, _ = self.compute_score(
                nist_results, physical_results, rng_results, structural_results
            )
            scores.append(score)
        
        scores = np.array(scores)
        ci_low = np.percentile(scores, 100 * alpha / 2)
        ci_high = np.percentile(scores, 100 * (1 - alpha / 2))
        
        return ci_low, ci_high, scores.tolist()
    
    def compute_full_result(self,
                            nist_results: Optional[Dict[str, Any]] = None,
                            physical_results: Optional[Dict[str, Any]] = None,
                            rng_results: Optional[Dict[str, Any]] = None,
                            structural_results: Optional[Dict[str, Any]] = None,
                            n_draws: int = 100,
                            n_max: int = 45,
                            k: int = 7,
                            has_emission_order: bool = False,
                            compute_ci: bool = True) -> ConformityResult:
        """
        Compute full conformity result with score, CI, and interpretation.
        
        Args:
            nist_results: Results from NIST tests
            physical_results: Results from physical bias tests
            rng_results: Results from RNG tests
            structural_results: Results from structural tests
            n_draws: Number of draws (for CI computation)
            n_max: Maximum number in pool
            k: Numbers per draw
            has_emission_order: Whether emission order data is available
            compute_ci: Whether to compute Monte Carlo CI (slow)
            
        Returns:
            ConformityResult with score, CI, and interpretation
        """
        # Compute score
        score, category_scores = self.compute_score(
            nist_results, physical_results, rng_results, structural_results
        )
        
        # Compute CI if requested
        if compute_ci:
            ci_low, ci_high, sim_scores = self.compute_monte_carlo_ci(
                n_draws, n_max, k, has_emission_order
            )
        else:
            # Default CI based on typical values
            ci_low, ci_high = 0.70, 1.0
            sim_scores = []
        
        # Interpretation
        if score >= ci_low:
            if score <= ci_high:
                interpretation = "CONFORMING: Score within expected range for random generator"
            else:
                interpretation = "SUSPICIOUS: Score higher than expected (possible over-uniformity)"
        else:
            interpretation = "NON-CONFORMING: Score below expected range (bias detected)"
        
        # Detailed breakdown
        details = {
            "category_weights": self.CATEGORY_WEIGHTS,
            "n_simulations": self.n_simulations if compute_ci else 0,
            "seed": self.seed,
            "simulated_scores_summary": {
                "mean": float(np.mean(sim_scores)) if sim_scores else None,
                "std": float(np.std(sim_scores)) if sim_scores else None,
                "min": float(np.min(sim_scores)) if sim_scores else None,
                "max": float(np.max(sim_scores)) if sim_scores else None
            } if sim_scores else None
        }
        
        return ConformityResult(
            score=score,
            ci_low=ci_low,
            ci_high=ci_high,
            n_simulations=self.n_simulations if compute_ci else 0,
            interpretation=interpretation,
            category_scores=category_scores,
            details=details
        )
    
    @staticmethod
    def compute_dataset_hash(draws: List[List[int]]) -> str:
        """
        Compute SHA-256 hash of the dataset for reproducibility tracking.
        
        Args:
            draws: List of draws
            
        Returns:
            SHA-256 hash string
        """
        data_str = json.dumps(draws, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
