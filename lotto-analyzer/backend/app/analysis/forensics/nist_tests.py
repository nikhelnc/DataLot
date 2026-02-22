"""
NIST SP 800-22 adapted tests for lottery sequences.
Transforms lottery draws into binary sequences and applies standard randomness tests.
"""

import numpy as np
from scipy import stats
from scipy.special import gammaincc, erfc
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


class NISTTests:
    """
    NIST SP 800-22 adapted tests for lottery draw sequences.
    
    The draws are converted to binary sequences via one-hot encoding:
    - Each draw -> binary vector of size N (1 if number present, 0 otherwise)
    - Concatenation of all draws -> sequence of length T × N bits
    """
    
    def __init__(self, alpha: float = 0.01):
        """
        Initialize NIST tests.
        
        Args:
            alpha: Significance level (default 0.01 per NIST recommendation)
        """
        self.alpha = alpha
    
    def _draws_to_binary(self, draws: List[List[int]], n_max: int) -> np.ndarray:
        """
        Convert draws to binary sequence via one-hot encoding.
        
        Args:
            draws: List of draws, each draw is a list of numbers
            n_max: Maximum number in the pool (e.g., 45 for Oz Lotto)
            
        Returns:
            Binary numpy array of shape (len(draws) * n_max,)
        """
        binary = []
        for draw in draws:
            one_hot = np.zeros(n_max, dtype=int)
            for num in draw:
                if 1 <= num <= n_max:
                    one_hot[num - 1] = 1
            binary.extend(one_hot)
        return np.array(binary)
    
    def frequency_monobit_test(self, binary_seq: np.ndarray) -> TestResult:
        """
        NIST Frequency (Monobit) Test.
        
        Tests whether the proportion of ones and zeros in the sequence
        is approximately equal, as expected for a random sequence.
        
        H0: The sequence is random (50% ones, 50% zeros)
        """
        n = len(binary_seq)
        if n == 0:
            return TestResult(
                name="Frequency Monobit",
                statistic=0.0,
                p_value=0.0,
                passed=False,
                alpha=self.alpha,
                description="Empty sequence"
            )
        
        # Count ones and compute S_n
        s_n = 2 * np.sum(binary_seq) - n  # S_n = sum(2*X_i - 1)
        s_obs = abs(s_n) / np.sqrt(n)
        
        # P-value using complementary error function
        p_value = erfc(s_obs / np.sqrt(2))
        
        return TestResult(
            name="Frequency Monobit",
            statistic=s_obs,
            p_value=p_value,
            passed=p_value >= self.alpha,
            alpha=self.alpha,
            description="Tests if proportion of 1s and 0s is approximately equal",
            details={
                "n_bits": n,
                "n_ones": int(np.sum(binary_seq)),
                "n_zeros": int(n - np.sum(binary_seq)),
                "proportion_ones": float(np.sum(binary_seq) / n),
                "s_obs": float(s_obs)
            }
        )
    
    def runs_test(self, binary_seq: np.ndarray) -> TestResult:
        """
        NIST Runs Test.
        
        Tests whether the number of runs (uninterrupted sequences of identical bits)
        is as expected for a random sequence.
        
        H0: The sequence is random
        """
        n = len(binary_seq)
        if n < 100:
            return TestResult(
                name="Runs Test",
                statistic=0.0,
                p_value=0.0,
                passed=False,
                alpha=self.alpha,
                description="Sequence too short (need >= 100 bits)"
            )
        
        # Pre-test: check if frequency test passes
        pi = np.sum(binary_seq) / n
        tau = 2 / np.sqrt(n)
        
        if abs(pi - 0.5) >= tau:
            return TestResult(
                name="Runs Test",
                statistic=0.0,
                p_value=0.0,
                passed=False,
                alpha=self.alpha,
                description="Pre-test failed: frequency too unbalanced",
                details={"pi": float(pi), "tau": float(tau)}
            )
        
        # Count runs
        v_n = 1  # Start with 1 run
        for i in range(1, n):
            if binary_seq[i] != binary_seq[i-1]:
                v_n += 1
        
        # Compute test statistic
        numerator = abs(v_n - 2 * n * pi * (1 - pi))
        denominator = 2 * np.sqrt(2 * n) * pi * (1 - pi)
        
        if denominator == 0:
            p_value = 0.0
        else:
            p_value = erfc(numerator / denominator)
        
        return TestResult(
            name="Runs Test",
            statistic=float(v_n),
            p_value=p_value,
            passed=p_value >= self.alpha,
            alpha=self.alpha,
            description="Tests if number of runs is as expected for random sequence",
            details={
                "n_bits": n,
                "n_runs": int(v_n),
                "expected_runs": float(2 * n * pi * (1 - pi) + 1),
                "pi": float(pi)
            }
        )
    
    def block_frequency_test(self, binary_seq: np.ndarray, block_size: int = 128) -> TestResult:
        """
        NIST Frequency Test within a Block.
        
        Tests whether the frequency of ones in M-bit blocks is approximately M/2.
        
        Args:
            binary_seq: Binary sequence
            block_size: Size of each block (M)
        """
        n = len(binary_seq)
        n_blocks = n // block_size
        
        if n_blocks < 1:
            return TestResult(
                name="Block Frequency",
                statistic=0.0,
                p_value=0.0,
                passed=False,
                alpha=self.alpha,
                description=f"Sequence too short for block size {block_size}"
            )
        
        # Compute proportion of ones in each block
        chi_squared = 0.0
        block_proportions = []
        
        for i in range(n_blocks):
            block = binary_seq[i * block_size:(i + 1) * block_size]
            pi_i = np.sum(block) / block_size
            block_proportions.append(float(pi_i))
            chi_squared += (pi_i - 0.5) ** 2
        
        chi_squared *= 4 * block_size
        
        # P-value using incomplete gamma function
        p_value = gammaincc(n_blocks / 2, chi_squared / 2)
        
        return TestResult(
            name="Block Frequency",
            statistic=float(chi_squared),
            p_value=float(p_value),
            passed=p_value >= self.alpha,
            alpha=self.alpha,
            description=f"Tests frequency uniformity in {block_size}-bit blocks",
            details={
                "n_blocks": n_blocks,
                "block_size": block_size,
                "chi_squared": float(chi_squared),
                "mean_proportion": float(np.mean(block_proportions)),
                "std_proportion": float(np.std(block_proportions))
            }
        )
    
    def serial_test(self, binary_seq: np.ndarray, pattern_length: int = 3) -> TestResult:
        """
        NIST Serial Test.
        
        Tests whether the frequency of all possible m-bit patterns is approximately equal.
        
        Args:
            binary_seq: Binary sequence
            pattern_length: Length of patterns to test (m)
        """
        n = len(binary_seq)
        m = pattern_length
        
        if n < m:
            return TestResult(
                name="Serial Test",
                statistic=0.0,
                p_value=0.0,
                passed=False,
                alpha=self.alpha,
                description=f"Sequence too short for pattern length {m}"
            )
        
        def psi_squared(seq: np.ndarray, m: int) -> float:
            """Compute psi^2 for pattern length m"""
            if m == 0:
                return 0.0
            
            n = len(seq)
            # Extend sequence cyclically
            extended = np.concatenate([seq, seq[:m-1]])
            
            # Count patterns
            pattern_counts = {}
            for i in range(n):
                pattern = tuple(extended[i:i+m])
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
            # Compute psi^2
            psi_sq = (2**m / n) * sum(c**2 for c in pattern_counts.values()) - n
            return psi_sq
        
        psi_m = psi_squared(binary_seq, m)
        psi_m1 = psi_squared(binary_seq, m - 1) if m > 1 else 0
        psi_m2 = psi_squared(binary_seq, m - 2) if m > 2 else 0
        
        delta_psi = psi_m - psi_m1
        delta2_psi = psi_m - 2 * psi_m1 + psi_m2
        
        # P-values
        p_value1 = gammaincc(2**(m-2), delta_psi / 2)
        p_value2 = gammaincc(2**(m-3), delta2_psi / 2) if m > 2 else 1.0
        
        # Use the minimum p-value
        p_value = min(p_value1, p_value2)
        
        return TestResult(
            name="Serial Test",
            statistic=float(delta_psi),
            p_value=float(p_value),
            passed=p_value >= self.alpha,
            alpha=self.alpha,
            description=f"Tests uniformity of {m}-bit patterns",
            details={
                "pattern_length": m,
                "psi_m": float(psi_m),
                "delta_psi": float(delta_psi),
                "delta2_psi": float(delta2_psi),
                "p_value1": float(p_value1),
                "p_value2": float(p_value2)
            }
        )
    
    def approximate_entropy_test(self, binary_seq: np.ndarray, block_length: int = 10) -> TestResult:
        """
        NIST Approximate Entropy Test.
        
        Compares the frequency of overlapping blocks of two consecutive lengths
        against the expected result for a random sequence.
        
        Args:
            binary_seq: Binary sequence
            block_length: Block length m
        """
        n = len(binary_seq)
        m = block_length
        
        if n < m + 5:
            return TestResult(
                name="Approximate Entropy",
                statistic=0.0,
                p_value=0.0,
                passed=False,
                alpha=self.alpha,
                description=f"Sequence too short for block length {m}"
            )
        
        def phi(seq: np.ndarray, m: int) -> float:
            """Compute phi for block length m"""
            n = len(seq)
            extended = np.concatenate([seq, seq[:m-1]])
            
            pattern_counts = {}
            for i in range(n):
                pattern = tuple(extended[i:i+m])
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
            # Compute phi
            phi_val = 0.0
            for count in pattern_counts.values():
                c_i = count / n
                if c_i > 0:
                    phi_val += c_i * np.log(c_i)
            
            return phi_val
        
        phi_m = phi(binary_seq, m)
        phi_m1 = phi(binary_seq, m + 1)
        
        ap_en = phi_m - phi_m1
        chi_squared = 2 * n * (np.log(2) - ap_en)
        
        p_value = gammaincc(2**(m-1), chi_squared / 2)
        
        return TestResult(
            name="Approximate Entropy",
            statistic=float(ap_en),
            p_value=float(p_value),
            passed=p_value >= self.alpha,
            alpha=self.alpha,
            description=f"Tests regularity of {m}-bit patterns",
            details={
                "block_length": m,
                "phi_m": float(phi_m),
                "phi_m1": float(phi_m1),
                "approximate_entropy": float(ap_en),
                "chi_squared": float(chi_squared)
            }
        )
    
    def run_all_tests(self, draws: List[List[int]], n_max: int, 
                      block_size: int = 128, serial_length: int = 3,
                      entropy_length: int = 10) -> Dict[str, Any]:
        """
        Run all NIST tests on the draw sequence.
        
        Args:
            draws: List of draws
            n_max: Maximum number in pool
            block_size: Block size for block frequency test
            serial_length: Pattern length for serial test
            entropy_length: Block length for approximate entropy test
            
        Returns:
            Dictionary with all test results
        """
        binary_seq = self._draws_to_binary(draws, n_max)
        
        results = {
            "n_draws": len(draws),
            "n_bits": len(binary_seq),
            "n_max": n_max,
            "alpha": self.alpha,
            "tests": {}
        }
        
        # Run each test
        tests = [
            self.frequency_monobit_test(binary_seq),
            self.runs_test(binary_seq),
            self.block_frequency_test(binary_seq, block_size),
            self.serial_test(binary_seq, serial_length),
            self.approximate_entropy_test(binary_seq, entropy_length)
        ]
        
        n_passed = 0
        for test in tests:
            results["tests"][test.name] = {
                "statistic": test.statistic,
                "p_value": test.p_value,
                "passed": test.passed,
                "alpha": test.alpha,
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
