import numpy as np
import pandas as pd
from typing import Dict, List
from scipy.stats import kstest, chisquare
from collections import defaultdict


class GapsStreaksAnalysis:
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
        self.df = df.sort_values('draw_date')
        self.results = self._analyze_gaps_and_streaks()
        return self

    def _analyze_gaps_and_streaks(self) -> Dict:
        # Track last occurrence of each number
        last_occurrence = {}
        gaps_by_number = defaultdict(list)
        streaks_by_number = defaultdict(list)
        current_streak = defaultdict(int)
        
        all_numbers = []
        for _, row in self.df.iterrows():
            numbers = list(row["numbers"]) if row["numbers"] else []
            
            # Include bonus numbers
            if "bonus_numbers" in row and row["bonus_numbers"]:
                numbers.extend(row["bonus_numbers"])
            
            all_numbers.append(set(numbers))
        
        # Calculate gaps (time between occurrences)
        for draw_idx, numbers_set in enumerate(all_numbers):
            for num in range(self.n_min, self.n_max + 1):
                if num in numbers_set:
                    # Number appeared
                    if num in last_occurrence:
                        gap = draw_idx - last_occurrence[num] - 1
                        gaps_by_number[num].append(gap)
                    last_occurrence[num] = draw_idx
                    current_streak[num] += 1
                else:
                    # Number didn't appear
                    if current_streak[num] > 0:
                        streaks_by_number[num].append(current_streak[num])
                        current_streak[num] = 0
        
        # Finalize streaks
        for num, streak in current_streak.items():
            if streak > 0:
                streaks_by_number[num].append(streak)
        
        # Calculate statistics per number
        number_stats = []
        # Support both old (numbers.count) and new (main.pick) structure
        main_rules = self.rules.get("main", self.rules.get("numbers", {}))
        k = main_rules.get("pick", main_rules.get("count", 6))
        bonus_rules = self.rules.get("bonus", {})
        if bonus_rules.get("enabled"):
            k += bonus_rules.get("pick", bonus_rules.get("count", 1))
        
        # Expected gap: geometric distribution with p = k/N
        p_appear = k / self.n_range
        expected_gap = (1 / p_appear) - 1
        
        for num in range(self.n_min, self.n_max + 1):
            gaps = gaps_by_number.get(num, [])
            streaks = streaks_by_number.get(num, [])
            
            if len(gaps) > 0:
                mean_gap = np.mean(gaps)
                std_gap = np.std(gaps)
                max_gap = np.max(gaps)
                
                # KS test against geometric distribution
                if len(gaps) >= 5:
                    # Generate expected geometric samples
                    expected_gaps = np.random.geometric(p_appear, len(gaps)) - 1
                    ks_stat, ks_pval = kstest(gaps, expected_gaps)
                else:
                    ks_stat, ks_pval = None, None
            else:
                mean_gap = None
                std_gap = None
                max_gap = None
                ks_stat, ks_pval = None, None
            
            if len(streaks) > 0:
                mean_streak = np.mean(streaks)
                max_streak = np.max(streaks)
            else:
                mean_streak = None
                max_streak = None
            
            number_stats.append({
                "number": num,
                "n_gaps": len(gaps),
                "mean_gap": float(mean_gap) if mean_gap is not None else None,
                "std_gap": float(std_gap) if std_gap is not None else None,
                "max_gap": int(max_gap) if max_gap is not None else None,
                "expected_gap": float(expected_gap),
                "delta_gap": float(mean_gap - expected_gap) if mean_gap is not None else None,
                "ks_stat": float(ks_stat) if ks_stat is not None else None,
                "ks_pval": float(ks_pval) if ks_pval is not None else None,
                "n_streaks": len(streaks),
                "mean_streak": float(mean_streak) if mean_streak is not None else None,
                "max_streak": int(max_streak) if max_streak is not None else None
            })
        
        # Sort by delta_gap (most atypical)
        number_stats_sorted = sorted(
            [s for s in number_stats if s["delta_gap"] is not None],
            key=lambda x: abs(x["delta_gap"]),
            reverse=True
        )
        
        # Global gap distribution
        all_gaps = []
        for gaps in gaps_by_number.values():
            all_gaps.extend(gaps)
        
        if len(all_gaps) > 0:
            gap_histogram = np.histogram(all_gaps, bins=20)
            gap_hist_data = {
                "counts": gap_histogram[0].tolist(),
                "bins": gap_histogram[1].tolist()
            }
        else:
            gap_hist_data = {"counts": [], "bins": []}
        
        return {
            "number_stats": number_stats,
            "top_atypical": number_stats_sorted[:10],
            "expected_gap": float(expected_gap),
            "p_appear": float(p_appear),
            "gap_histogram": gap_hist_data,
            "n_draws": len(self.df)
        }

    def get_results(self) -> Dict:
        if self.results is None:
            return {
                "error": "Model not fitted",
                "warnings": ["Call fit() before get_results()"]
            }
        
        return {
            "method": "M6_GapsStreaks",
            "explain": "Analyse des gaps (temps entre occurrences) et streaks (séquences consécutives) pour détecter des numéros avec comportement atypique",
            "gaps_streaks": self.results,
            "warnings": self._generate_warnings(),
            "charts": {
                "gap_histogram": self.results["gap_histogram"],
                "top_atypical_numbers": {
                    "numbers": [s["number"] for s in self.results["top_atypical"]],
                    "delta_gaps": [s["delta_gap"] for s in self.results["top_atypical"]],
                    "expected": [self.results["expected_gap"]] * len(self.results["top_atypical"])
                }
            }
        }

    def _generate_warnings(self) -> List[str]:
        warnings = []
        
        if self.results["n_draws"] < 50:
            warnings.append("Dataset petit (<50 tirages) : les statistiques de gaps peuvent être peu fiables")
        
        # Check for numbers with significantly different gaps
        significant_numbers = [
            s for s in self.results["number_stats"]
            if s["ks_pval"] is not None and s["ks_pval"] < 0.05
        ]
        
        if len(significant_numbers) == 0:
            warnings.append("Aucun numéro avec distribution de gaps significativement différente de l'attendu")
        else:
            warnings.append(f"{len(significant_numbers)} numéros avec gaps atypiques (KS p < 0.05)")
        
        return warnings
