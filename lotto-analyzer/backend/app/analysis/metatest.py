import numpy as np
import pandas as pd
from typing import Dict, List
from scipy.stats import kstest, uniform, chi2


class MetaTestAnalysis:
    def __init__(self, rules: Dict):
        self.rules = rules
        # Support both old (numbers) and new (main) structure
        main_rules = rules.get("main", rules.get("numbers", {}))
        self.n_min = main_rules.get("min", 1)
        self.n_max = main_rules.get("max", 49)
        self.n_range = self.n_max - self.n_min + 1
        self.df = None
        self.results = None

    def fit(self, df: pd.DataFrame, p_values_from_tests: Dict = None):
        """
        Analyze p-values from various statistical tests.
        p_values_from_tests: dict with test names as keys and lists of p-values as values
        """
        self.df = df
        self.p_values_from_tests = p_values_from_tests or {}
        self.results = self._analyze_pvalues()
        return self

    def _analyze_pvalues(self) -> Dict:
        # Collect all p-values from randomness tests
        all_pvalues = []
        pvalue_sources = []
        
        for test_name, pvals in self.p_values_from_tests.items():
            if pvals and len(pvals) > 0:
                all_pvalues.extend(pvals)
                pvalue_sources.extend([test_name] * len(pvals))
        
        if len(all_pvalues) == 0:
            # Generate synthetic p-values from uniformity test
            all_pvalues = self._generate_uniformity_pvalues()
            pvalue_sources = ["uniformity_test"] * len(all_pvalues)
        
        # QQ plot data (compare to uniform [0,1])
        sorted_pvals = np.sort(all_pvalues)
        theoretical_quantiles = np.linspace(0, 1, len(sorted_pvals))
        
        # KS test: are p-values uniformly distributed?
        ks_stat, ks_pval = kstest(all_pvalues, 'uniform')
        
        # Count significant p-values at different thresholds
        sig_001 = sum(1 for p in all_pvalues if p < 0.01)
        sig_005 = sum(1 for p in all_pvalues if p < 0.05)
        sig_010 = sum(1 for p in all_pvalues if p < 0.10)
        
        expected_001 = len(all_pvalues) * 0.01
        expected_005 = len(all_pvalues) * 0.05
        expected_010 = len(all_pvalues) * 0.10
        
        # Temporal drift: split p-values by time periods
        if len(self.df) >= 20:
            n_periods = 4
            period_size = len(self.df) // n_periods
            period_pvals = []
            
            for i in range(n_periods):
                start_idx = i * period_size
                end_idx = (i + 1) * period_size if i < n_periods - 1 else len(self.df)
                
                # Get p-values for this period (simplified: use uniformity test)
                period_df = self.df.iloc[start_idx:end_idx]
                period_pvals.append({
                    "period": i + 1,
                    "start_draw": int(start_idx),
                    "end_draw": int(end_idx),
                    "mean_pval": float(np.mean(all_pvalues[start_idx:min(end_idx, len(all_pvalues))])) if end_idx <= len(all_pvalues) else None
                })
        else:
            period_pvals = []
        
        # Verdict
        if ks_pval < 0.01:
            verdict = "p-values_not_uniform"
            interpretation = "Les p-values ne sont PAS uniformément distribuées (KS p < 0.01) : possible biais dans les tests"
        elif ks_pval < 0.05:
            verdict = "p-values_suspicious"
            interpretation = "Les p-values sont suspectes (KS p < 0.05) : vérifier les hypothèses des tests"
        else:
            verdict = "p-values_ok"
            interpretation = "Les p-values sont compatibles avec une distribution uniforme : pas de biais détecté"
        
        return {
            "n_pvalues": len(all_pvalues),
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pval),
            "verdict": verdict,
            "interpretation": interpretation,
            "qq_plot": {
                "theoretical": theoretical_quantiles.tolist(),
                "observed": sorted_pvals.tolist()
            },
            "significance_counts": {
                "p_001": {"observed": sig_001, "expected": expected_001, "delta": sig_001 - expected_001},
                "p_005": {"observed": sig_005, "expected": expected_005, "delta": sig_005 - expected_005},
                "p_010": {"observed": sig_010, "expected": expected_010, "delta": sig_010 - expected_010}
            },
            "temporal_drift": period_pvals,
            "pvalue_sources": list(set(pvalue_sources))
        }

    def _generate_uniformity_pvalues(self) -> List[float]:
        """Generate p-values from chi-square uniformity tests on each number"""
        pvalues = []
        
        for num in range(self.n_min, self.n_max + 1):
            # Count occurrences of this number
            count = 0
            for _, row in self.df.iterrows():
                numbers = list(row["numbers"]) if row["numbers"] else []
                if "bonus_numbers" in row and row["bonus_numbers"]:
                    numbers.extend(row["bonus_numbers"])
                if num in numbers:
                    count += 1
            
            # Expected count
            # Support both old (numbers.count) and new (main.pick) structure
            main_rules = self.rules.get("main", self.rules.get("numbers", {}))
            k = main_rules.get("pick", main_rules.get("count", 6))
            bonus_rules = self.rules.get("bonus", {})
            if bonus_rules.get("enabled"):
                k += bonus_rules.get("pick", bonus_rules.get("count", 1))
            expected = len(self.df) * k / self.n_range
            
            # Chi-square test
            if expected > 0:
                chi2_stat = ((count - expected) ** 2) / expected
                pval = 1 - chi2.cdf(chi2_stat, df=1)
                pvalues.append(pval)
        
        return pvalues

    def get_results(self) -> Dict:
        if self.results is None:
            return {
                "error": "Model not fitted",
                "warnings": ["Call fit() before get_results()"]
            }
        
        return {
            "method": "M9_MetaTest",
            "explain": "Meta-analyse des p-values : vérifie si les tests statistiques produisent des p-values uniformes (attendu sous H0) ou biaisées",
            "metatest": self.results,
            "warnings": self._generate_warnings(),
            "charts": {
                "qq_plot": self.results["qq_plot"],
                "temporal_drift": {
                    "periods": [p["period"] for p in self.results["temporal_drift"]],
                    "mean_pvals": [p["mean_pval"] for p in self.results["temporal_drift"] if p["mean_pval"] is not None]
                }
            }
        }

    def _generate_warnings(self) -> List[str]:
        warnings = []
        
        if self.results["n_pvalues"] < 10:
            warnings.append("Peu de p-values disponibles (<10) : le test KS peut être peu puissant")
        
        if self.results["verdict"] == "p-values_not_uniform":
            warnings.append("⚠️ P-values non uniformes : possible biais dans les tests ou données non aléatoires")
        elif self.results["verdict"] == "p-values_suspicious":
            warnings.append("⚠️ P-values suspectes : vérifier les hypothèses des tests statistiques")
        
        # Check for excess of significant results
        sig_005 = self.results["significance_counts"]["p_005"]
        if sig_005["delta"] > sig_005["expected"] * 0.5:
            warnings.append(f"Excès de résultats significatifs (p<0.05) : {sig_005['observed']} observés vs {sig_005['expected']:.1f} attendus")
        
        return warnings
