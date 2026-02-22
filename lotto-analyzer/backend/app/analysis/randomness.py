import numpy as np
import pandas as pd
from typing import Dict, Any, List
from scipy.stats import chisquare, kstest
from statsmodels.stats.multitest import multipletests


class RandomnessTests:
    def __init__(self, df: pd.DataFrame, rules: Dict[str, Any]):
        self.df = df
        self.rules = rules
        # Support both old (numbers) and new (main) structure
        main_rules = rules.get("main", rules.get("numbers", {}))
        self.n_min = main_rules.get("min", 1)
        self.n_max = main_rules.get("max", 49)
        self.n_count = main_rules.get("pick", main_rules.get("count", 6))
        self.warnings = []

    def run_all_tests(self) -> Dict[str, Any]:
        tests = {
            "uniformity": self._test_uniformity(),
            "independence": self._test_independence(),
            "meta_test": self._meta_test(),
        }
        
        all_pvalues = []
        test_names = []
        
        if "chi2_numbers" in tests["uniformity"]:
            all_pvalues.append(tests["uniformity"]["chi2_numbers"]["p_value"])
            test_names.append("chi2_numbers")
        
        for test_name, test_result in tests["independence"].items():
            if "p_value" in test_result:
                all_pvalues.append(test_result["p_value"])
                test_names.append(test_name)
        
        if len(all_pvalues) > 1:
            reject, pvals_corrected, _, _ = multipletests(all_pvalues, method="fdr_bh")
            tests["fdr_correction"] = {
                "method": "Benjamini-Hochberg",
                "corrected_pvalues": {
                    name: float(pval) for name, pval in zip(test_names, pvals_corrected)
                },
                "rejected": {name: bool(rej) for name, rej in zip(test_names, reject)},
            }
        
        return tests

    def _test_uniformity(self) -> Dict[str, Any]:
        all_numbers = []
        for numbers in self.df["numbers"]:
            all_numbers.extend(numbers)
        
        freq_counts = pd.Series(all_numbers).value_counts()
        n_range = self.n_max - self.n_min + 1
        expected_freq = len(all_numbers) / n_range
        
        observed = [freq_counts.get(num, 0) for num in range(self.n_min, self.n_max + 1)]
        expected = [expected_freq] * n_range
        
        chi2_stat, p_value = chisquare(observed, expected)
        
        result = {
            "chi2_numbers": {
                "statistic": float(chi2_stat),
                "p_value": float(p_value),
                "df": n_range - 1,
                "interpretation": "Reject H0: non-uniform" if p_value < 0.05 else "Fail to reject H0: uniform",
            }
        }
        
        if self.rules.get("bonus", {}).get("enabled"):
            # Flatten bonus_numbers arrays into a single list
            bonus_values = []
            for bonus_nums in self.df["bonus_numbers"]:
                if bonus_nums and len(bonus_nums) > 0:
                    bonus_values.extend(bonus_nums)
            
            if len(bonus_values) > 0:
                bonus_counts = pd.Series(bonus_values).value_counts()
                bonus_min = self.rules["bonus"]["min"]
                bonus_max = self.rules["bonus"]["max"]
                bonus_range = bonus_max - bonus_min + 1
                bonus_expected = len(bonus_values) / bonus_range
                
                bonus_observed = [bonus_counts.get(num, 0) for num in range(bonus_min, bonus_max + 1)]
                bonus_exp = [bonus_expected] * bonus_range
                
                chi2_bonus, p_bonus = chisquare(bonus_observed, bonus_exp)
                result["chi2_bonus"] = {
                    "statistic": float(chi2_bonus),
                    "p_value": float(p_bonus),
                    "df": bonus_range - 1,
                    "interpretation": "Reject H0: non-uniform" if p_bonus < 0.05 else "Fail to reject H0: uniform",
                }
        
        return result

    def _test_independence(self) -> Dict[str, Any]:
        even_counts = []
        for numbers in self.df["numbers"]:
            even_counts.append(sum(1 for n in numbers if n % 2 == 0))
        
        runs = self._runs_test(even_counts, np.median(even_counts))
        
        sums = [sum(numbers) for numbers in self.df["numbers"]]
        runs_sum = self._runs_test(sums, np.median(sums))
        
        return {
            "runs_even": runs,
            "runs_sum": runs_sum,
        }

    def _runs_test(self, series: List[float], threshold: float) -> Dict[str, Any]:
        binary = [1 if x > threshold else 0 for x in series]
        
        runs = 1
        for i in range(1, len(binary)):
            if binary[i] != binary[i - 1]:
                runs += 1
        
        n1 = sum(binary)
        n2 = len(binary) - n1
        
        if n1 == 0 or n2 == 0:
            return {
                "runs": runs,
                "expected_runs": 0,
                "z_score": 0,
                "p_value": 1.0,
                "interpretation": "Insufficient variation for runs test",
            }
        
        expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
        variance_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
        
        if variance_runs > 0:
            z_score = (runs - expected_runs) / np.sqrt(variance_runs)
            p_value = 2 * (1 - np.abs(z_score) / np.sqrt(2 * np.pi))
        else:
            z_score = 0
            p_value = 1.0
        
        return {
            "runs": runs,
            "expected_runs": expected_runs,
            "z_score": float(z_score),
            "p_value": float(p_value),
            "interpretation": "Reject H0: not independent" if p_value < 0.05 else "Fail to reject H0: independent",
        }

    def _meta_test(self) -> Dict[str, Any]:
        uniformity = self._test_uniformity()
        independence = self._test_independence()
        
        p_values = []
        if "chi2_numbers" in uniformity:
            p_values.append(uniformity["chi2_numbers"]["p_value"])
        if "chi2_bonus" in uniformity:
            p_values.append(uniformity["chi2_bonus"]["p_value"])
        for test in independence.values():
            if "p_value" in test:
                p_values.append(test["p_value"])
        
        if len(p_values) < 2:
            return {"warning": "Insufficient tests for meta-analysis"}
        
        ks_stat, ks_pvalue = kstest(p_values, "uniform")
        
        return {
            "ks_test": {
                "statistic": float(ks_stat),
                "p_value": float(ks_pvalue),
                "interpretation": "P-values deviate from uniform" if ks_pvalue < 0.05 else "P-values consistent with uniform",
            },
            "note": "Under H0, p-values should be uniformly distributed",
        }

    def get_warnings(self) -> List[str]:
        if len(self.df) < 50:
            self.warnings.append("Small sample size may affect test power")
        return self.warnings
