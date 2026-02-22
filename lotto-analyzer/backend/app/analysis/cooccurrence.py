import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests


class CooccurrenceAnalysis:
    def __init__(self, rules: Dict):
        self.rules = rules
        # Support both old (numbers) and new (main) structure
        main_rules = rules.get("main", rules.get("numbers", {}))
        self.n_min = main_rules.get("min", 1)
        self.n_max = main_rules.get("max", 49)
        self.n_range = self.n_max - self.n_min + 1
        self.df = None
        self.results = None

    def fit(self, df: pd.DataFrame):
        self.df = df
        self.results = self._analyze_cooccurrence()
        return self

    def _analyze_cooccurrence(self) -> Dict:
        # Build observed co-occurrence matrix
        cooc_matrix = np.zeros((self.n_range, self.n_range))
        
        for _, row in self.df.iterrows():
            numbers = row["numbers"]
            if not numbers or len(numbers) == 0:
                continue
            
            # Include bonus numbers if available
            if "bonus_numbers" in row and row["bonus_numbers"] and len(row["bonus_numbers"]) > 0:
                numbers = list(numbers) + list(row["bonus_numbers"])
            
            # Count co-occurrences
            for i, num1 in enumerate(numbers):
                for num2 in numbers[i+1:]:
                    idx1 = num1 - self.n_min
                    idx2 = num2 - self.n_min
                    if 0 <= idx1 < self.n_range and 0 <= idx2 < self.n_range:
                        cooc_matrix[idx1, idx2] += 1
                        cooc_matrix[idx2, idx1] += 1  # Symmetric
        
        # Calculate expected co-occurrence (without replacement)
        n_draws = len(self.df)
        # Support both old (numbers.count) and new (main.pick) structure
        main_rules = self.rules.get("main", self.rules.get("numbers", {}))
        k = main_rules.get("pick", main_rules.get("count", 6))  # Numbers per draw
        
        # Include bonus in count if enabled
        bonus_rules = self.rules.get("bonus", {})
        if bonus_rules.get("enabled"):
            k += bonus_rules.get("pick", bonus_rules.get("count", 1))
        
        # Expected: P(both in same draw) = (k choose 2) / (N choose 2) * n_draws
        # For sampling without replacement: P(i and j both drawn) = k(k-1) / (N(N-1))
        expected_prob = (k * (k - 1)) / (self.n_range * (self.n_range - 1))
        expected_matrix = np.full((self.n_range, self.n_range), expected_prob * n_draws)
        np.fill_diagonal(expected_matrix, 0)  # No self-cooccurrence
        
        # Calculate delta and p-values
        delta_matrix = cooc_matrix - expected_matrix
        p_values = []
        pairs = []
        
        for i in range(self.n_range):
            for j in range(i+1, self.n_range):
                obs = cooc_matrix[i, j]
                exp = expected_matrix[i, j]
                
                if exp > 0:
                    # Chi-square test for this pair
                    contingency = np.array([
                        [obs, n_draws - obs],
                        [exp, n_draws - exp]
                    ])
                    try:
                        _, p_val, _, _ = chi2_contingency(contingency)
                        p_values.append(p_val)
                        pairs.append({
                            "num1": i + self.n_min,
                            "num2": j + self.n_min,
                            "observed": float(obs),
                            "expected": float(exp),
                            "delta": float(delta_matrix[i, j]),
                            "p_value": float(p_val)
                        })
                    except:
                        pass
        
        # FDR correction
        if len(p_values) > 0:
            _, p_values_corrected, _, _ = multipletests(p_values, method='fdr_bh')
            for i, pair in enumerate(pairs):
                pair["p_value_fdr"] = float(p_values_corrected[i])
        
        # Sort pairs by absolute delta
        pairs.sort(key=lambda x: abs(x["delta"]), reverse=True)
        
        # Top pairs (most over-represented)
        top_pairs = [p for p in pairs if p["delta"] > 0][:20]
        
        # Bottom pairs (most under-represented)
        bottom_pairs = [p for p in pairs if p["delta"] < 0][:20]
        
        return {
            "cooc_matrix": cooc_matrix.tolist(),
            "expected_matrix": expected_matrix.tolist(),
            "delta_matrix": delta_matrix.tolist(),
            "top_pairs": top_pairs,
            "bottom_pairs": bottom_pairs,
            "all_pairs": pairs,
            "n_draws": n_draws,
            "k": k,
            "expected_prob": float(expected_prob)
        }

    def get_results(self) -> Dict:
        if self.results is None:
            return {
                "error": "Model not fitted",
                "warnings": ["Call fit() before get_results()"]
            }
        
        return {
            "method": "M5_Cooccurrence",
            "explain": "Analyse des co-occurrences de numéros : quels numéros apparaissent souvent ensemble vs attendu théorique (sans remise)",
            "cooccurrence": self.results,
            "warnings": self._generate_warnings(),
            "charts": {
                "heatmap_observed": {
                    "data": self.results["cooc_matrix"],
                    "labels": list(range(self.n_min, self.n_max + 1))
                },
                "heatmap_delta": {
                    "data": self.results["delta_matrix"],
                    "labels": list(range(self.n_min, self.n_max + 1))
                }
            }
        }

    def _generate_warnings(self) -> List[str]:
        warnings = []
        
        if self.results["n_draws"] < 100:
            warnings.append("Dataset petit (<100 tirages) : les p-values peuvent être peu fiables")
        
        # Check if any pair is significantly different after FDR
        significant_pairs = [p for p in self.results["all_pairs"] if p.get("p_value_fdr", 1.0) < 0.05]
        if len(significant_pairs) == 0:
            warnings.append("Aucune paire significativement différente de l'attendu (FDR < 0.05)")
        else:
            warnings.append(f"{len(significant_pairs)} paires significativement différentes de l'attendu (FDR < 0.05)")
        
        return warnings
