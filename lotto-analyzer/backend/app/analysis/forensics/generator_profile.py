"""
Generator profiler - orchestrates all forensic tests and produces a complete profile.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import hashlib
import json

from .nist_tests import NISTTests


def _convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert numpy types to Python native types for JSON serialization.
    """
    if isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, np.bool8)):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.str_):
        return str(obj)
    else:
        return obj


from .physical_tests import PhysicalTests
from .rng_tests import RNGTests
from .structural_tests import StructuralTests
from .conformity_score import ConformityScoreCalculator, ConformityResult


class GeneratorProfiler:
    """
    Orchestrates all forensic tests and produces a complete generator profile.
    
    This is the main entry point for forensic analysis.
    """
    
    APP_VERSION = "2.0.0"
    
    def __init__(self, 
                 alpha: float = 0.01,
                 n_permutations: int = 10000,
                 n_simulations: int = 1000,
                 seed: int = 42):
        """
        Initialize generator profiler.
        
        Args:
            alpha: Significance level for all tests
            n_permutations: Number of permutations for permutation tests
            n_simulations: Number of Monte Carlo simulations
            seed: Random seed for reproducibility
        """
        self.alpha = alpha
        self.n_permutations = n_permutations
        self.n_simulations = n_simulations
        self.seed = seed
        
        # Initialize test modules
        self.nist_tests = NISTTests(alpha=alpha)
        self.physical_tests = PhysicalTests(alpha=alpha, n_permutations=n_permutations)
        self.rng_tests = RNGTests(alpha=alpha, n_permutations=n_permutations)
        self.structural_tests = StructuralTests(alpha=alpha, n_simulations=n_simulations)
        self.conformity_calculator = ConformityScoreCalculator(
            n_simulations=n_simulations, seed=seed
        )
    
    def _compute_dataset_hash(self, draws: List[List[int]], 
                               emission_orders: Optional[List[List[int]]] = None) -> str:
        """Compute SHA-256 hash of the dataset"""
        data = {
            "draws": draws,
            "emission_orders": emission_orders
        }
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def _detect_generator_type(self, 
                                physical_results: Dict[str, Any],
                                rng_results: Dict[str, Any]) -> str:
        """
        Attempt to detect the type of generator based on test results.
        
        Returns: 'physical', 'rng', 'hybrid', or 'unknown'
        """
        physical_pass_rate = physical_results.get("summary", {}).get("pass_rate", 1.0)
        rng_pass_rate = rng_results.get("summary", {}).get("pass_rate", 1.0)
        
        # Heuristics for generator type detection
        has_emission_data = physical_results.get("has_emission_order", False)
        
        if not has_emission_data:
            # Without emission data, we can't distinguish physical from RNG
            return "unknown"
        
        # Check for physical bias signatures
        physical_tests = physical_results.get("tests", {})
        has_weight_bias = not physical_tests.get("Weight Bias", {}).get("passed", True)
        has_position_bias = not physical_tests.get("Emission Position Bias", {}).get("passed", True)
        has_thermal_drift = not physical_tests.get("Thermal Drift", {}).get("passed", True)
        
        # Check for RNG signatures
        rng_tests_data = rng_results.get("tests", {})
        has_lsb_bias = not rng_tests_data.get("LSB Bias", {}).get("passed", True)
        has_modulo_bias = not rng_tests_data.get("Modulo Bias", {}).get("passed", True)
        has_periodicity = not rng_tests_data.get("Periodicity", {}).get("passed", True)
        
        physical_signals = sum([has_weight_bias, has_position_bias, has_thermal_drift])
        rng_signals = sum([has_lsb_bias, has_modulo_bias, has_periodicity])
        
        if physical_signals > 0 and rng_signals == 0:
            return "physical"
        elif rng_signals > 0 and physical_signals == 0:
            return "rng"
        elif physical_signals > 0 and rng_signals > 0:
            return "hybrid"
        else:
            return "unknown"
    
    def run_full_profile(self,
                         draws: List[List[int]],
                         n_max: int,
                         k: int,
                         emission_orders: Optional[List[List[int]]] = None,
                         bonus_draws: Optional[List[List[int]]] = None,
                         bonus_emission_orders: Optional[List[List[int]]] = None,
                         period_start: Optional[datetime] = None,
                         period_end: Optional[datetime] = None,
                         compute_ci: bool = True,
                         nist_block_size: int = 128,
                         nist_serial_length: int = 3,
                         nist_entropy_length: int = 10,
                         period_candidates: Optional[List[int]] = None,
                         progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Run complete forensic profile on the dataset.
        
        Args:
            draws: List of draws (sorted numbers)
            n_max: Maximum number in main pool
            k: Numbers per draw
            emission_orders: List of emission orders (optional)
            bonus_draws: List of bonus draws (optional)
            bonus_emission_orders: List of bonus emission orders (optional)
            period_start: Start date of analysis period
            period_end: End date of analysis period
            compute_ci: Whether to compute Monte Carlo CI (slow)
            nist_block_size: Block size for NIST block frequency test
            nist_serial_length: Pattern length for NIST serial test
            nist_entropy_length: Block length for NIST approximate entropy test
            period_candidates: Periods to test for periodicity
            progress_callback: Optional callback function(step: str, progress: int, total: int)
            
        Returns:
            Complete forensic profile dictionary
        """
        start_time = datetime.utcnow()
        
        def report_progress(step: str, progress: int, total: int = 100):
            if progress_callback:
                progress_callback(step, progress, total)
        
        report_progress("Initialisation", 0, 100)
        
        # Compute dataset hash
        dataset_hash = self._compute_dataset_hash(draws, emission_orders)
        report_progress("Tests NIST", 5, 100)
        
        # Run NIST tests
        nist_results = self.nist_tests.run_all_tests(
            draws, n_max,
            block_size=nist_block_size,
            serial_length=nist_serial_length,
            entropy_length=nist_entropy_length
        )
        report_progress("Tests physiques", 25, 100)
        
        # Run physical tests
        physical_results = self.physical_tests.run_all_tests(
            draws, emission_orders, n_max
        )
        report_progress("Tests RNG", 40, 100)
        
        # Run RNG tests
        rng_results = self.rng_tests.run_all_tests(
            draws, n_max, k,
            period_candidates=period_candidates
        )
        report_progress("Tests structurels", 55, 100)
        
        # Run structural tests
        structural_results = self.structural_tests.run_all_tests(draws, n_max)
        report_progress("Score de conformité", 70, 100)
        
        # Compute conformity score
        has_emission = emission_orders is not None and len(emission_orders) > 0
        conformity_result = self.conformity_calculator.compute_full_result(
            nist_results=nist_results,
            physical_results=physical_results,
            rng_results=rng_results,
            structural_results=structural_results,
            n_draws=len(draws),
            n_max=n_max,
            k=k,
            has_emission_order=has_emission,
            compute_ci=compute_ci
        )
        report_progress("Détection type générateur", 95, 100)
        
        # Detect generator type
        generator_type = self._detect_generator_type(physical_results, rng_results)
        report_progress("Terminé", 100, 100)
        
        end_time = datetime.utcnow()
        
        # Build profile
        profile = {
            "computed_at": end_time.isoformat(),
            "computation_time_seconds": (end_time - start_time).total_seconds(),
            
            # Period
            "period_start": period_start.isoformat() if period_start else None,
            "period_end": period_end.isoformat() if period_end else None,
            "n_draws": len(draws),
            
            # Conformity score
            "conformity_score": conformity_result.score,
            "conformity_ci_low": conformity_result.ci_low,
            "conformity_ci_high": conformity_result.ci_high,
            "conformity_n_simulations": conformity_result.n_simulations,
            "conformity_interpretation": conformity_result.interpretation,
            "category_scores": conformity_result.category_scores,
            
            # Generator type
            "generator_type": generator_type,
            
            # Test results by category
            "standard_tests": {
                "chi_squared": None,  # Placeholder for standard χ² tests
                "runs": None,
                "independence": None
            },
            "nist_tests": nist_results,
            "physical_tests": physical_results,
            "rng_tests": rng_results,
            "structural_tests": structural_results,
            
            # Reproducibility
            "dataset_hash": dataset_hash,
            "app_version": self.APP_VERSION,
            "params": {
                "alpha": self.alpha,
                "n_permutations": self.n_permutations,
                "n_simulations": self.n_simulations,
                "nist_block_size": nist_block_size,
                "nist_serial_length": nist_serial_length,
                "nist_entropy_length": nist_entropy_length,
                "period_candidates": period_candidates
            },
            "seed": self.seed,
            
            # Summary
            "summary": {
                "overall_passed": conformity_result.score >= conformity_result.ci_low,
                "n_tests_total": (
                    nist_results.get("summary", {}).get("n_tests", 0) +
                    physical_results.get("summary", {}).get("n_tests", 0) +
                    rng_results.get("summary", {}).get("n_tests", 0) +
                    structural_results.get("summary", {}).get("n_tests", 0)
                ),
                "n_tests_passed": (
                    nist_results.get("summary", {}).get("n_passed", 0) +
                    physical_results.get("summary", {}).get("n_passed", 0) +
                    rng_results.get("summary", {}).get("n_passed", 0) +
                    structural_results.get("summary", {}).get("n_passed", 0)
                ),
                "has_emission_order": has_emission,
                "n_draws_with_emission": len(emission_orders) if emission_orders else 0
            }
        }
        
        # Convert all numpy types to Python native types for JSON serialization
        return _convert_numpy_types(profile)
    
    def run_quick_profile(self,
                          draws: List[List[int]],
                          n_max: int,
                          k: int,
                          emission_orders: Optional[List[List[int]]] = None) -> Dict[str, Any]:
        """
        Run a quick forensic profile (no Monte Carlo CI, reduced permutations).
        
        Useful for real-time feedback during data exploration.
        """
        # Temporarily reduce simulation counts
        original_n_perm = self.n_permutations
        original_n_sim = self.n_simulations
        
        self.n_permutations = 100
        self.n_simulations = 100
        self.physical_tests.n_permutations = 100
        self.rng_tests.n_permutations = 100
        self.structural_tests.n_simulations = 100
        
        try:
            profile = self.run_full_profile(
                draws=draws,
                n_max=n_max,
                k=k,
                emission_orders=emission_orders,
                compute_ci=False
            )
            profile["is_quick_profile"] = True
        finally:
            # Restore original values
            self.n_permutations = original_n_perm
            self.n_simulations = original_n_sim
            self.physical_tests.n_permutations = original_n_perm
            self.rng_tests.n_permutations = original_n_perm
            self.structural_tests.n_simulations = original_n_sim
        
        return profile
