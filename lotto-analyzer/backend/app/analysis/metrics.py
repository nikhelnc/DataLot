import numpy as np
import pandas as pd
from typing import Dict, Any, List
from scipy.stats import entropy


class MetricsCalculator:
    def __init__(self, df: pd.DataFrame, rules: Dict[str, Any]):
        self.df = df
        self.rules = rules
        # Support both old (numbers) and new (main) structure
        main_rules = rules.get("main", rules.get("numbers", {}))
        self.n_min = main_rules.get("min", 1)
        self.n_max = main_rules.get("max", 49)
        self.n_count = main_rules.get("pick", main_rules.get("count", 6))
        self.warnings = []

    def calculate_all(self) -> Dict[str, Any]:
        return {
            "frequencies": self._calculate_frequencies(),
            "structure": self._calculate_structure(),
            "gaps": self._calculate_gaps(),
            "quality": self._calculate_quality(),
        }

    def _calculate_frequencies(self) -> Dict[str, Any]:
        all_numbers = []
        for numbers in self.df["numbers"]:
            all_numbers.extend(numbers)
        
        freq_counts = pd.Series(all_numbers).value_counts().sort_index()
        n_draws = len(self.df)
        n_range = self.n_max - self.n_min + 1
        expected_count = (n_draws * self.n_count) / n_range
        
        frequencies = {}
        z_scores = {}
        
        for num in range(self.n_min, self.n_max + 1):
            observed = freq_counts.get(num, 0)
            frequencies[str(num)] = int(observed)
            
            std_dev = np.sqrt(expected_count * (1 - self.n_count / n_range))
            z_scores[str(num)] = (observed - expected_count) / std_dev if std_dev > 0 else 0
        
        freq_array = [freq_counts.get(num, 0) for num in range(self.n_min, self.n_max + 1)]
        uniform_probs = np.ones(n_range) / n_range
        observed_probs = np.array(freq_array) / sum(freq_array) if sum(freq_array) > 0 else uniform_probs
        
        shannon_entropy = entropy(observed_probs, base=2)
        max_entropy = np.log2(n_range)
        kl_divergence = entropy(observed_probs, uniform_probs)
        
        return {
            "counts": frequencies,
            "expected_count": expected_count,
            "z_scores": z_scores,
            "entropy": shannon_entropy,
            "max_entropy": max_entropy,
            "kl_divergence": kl_divergence,
        }

    def _calculate_structure(self) -> Dict[str, Any]:
        sums = []
        means = []
        variances = []
        even_counts = []
        consecutive_counts = []
        ranges = []
        
        for numbers in self.df["numbers"]:
            sums.append(sum(numbers))
            means.append(np.mean(numbers))
            variances.append(np.var(numbers))
            even_counts.append(sum(1 for n in numbers if n % 2 == 0))
            
            sorted_nums = sorted(numbers)
            consec = sum(1 for i in range(len(sorted_nums) - 1) if sorted_nums[i + 1] - sorted_nums[i] == 1)
            consecutive_counts.append(consec)
            
            ranges.append(max(numbers) - min(numbers))
        
        return {
            "sum": {
                "mean": np.mean(sums),
                "std": np.std(sums),
                "min": min(sums),
                "max": max(sums),
            },
            "mean": {
                "mean": np.mean(means),
                "std": np.std(means),
            },
            "variance": {
                "mean": np.mean(variances),
                "std": np.std(variances),
            },
            "even_count": {
                "mean": np.mean(even_counts),
                "std": np.std(even_counts),
            },
            "consecutive_count": {
                "mean": np.mean(consecutive_counts),
                "max": max(consecutive_counts),
            },
            "range": {
                "mean": np.mean(ranges),
                "std": np.std(ranges),
            },
        }

    def _calculate_gaps(self) -> Dict[str, Any]:
        gaps_by_number = {}
        
        for num in range(self.n_min, self.n_max + 1):
            last_seen = -1
            gaps = []
            
            for idx, numbers in enumerate(self.df["numbers"]):
                if num in numbers:
                    if last_seen >= 0:
                        gaps.append(idx - last_seen)
                    last_seen = idx
            
            if gaps:
                gaps_by_number[str(num)] = {
                    "mean": np.mean(gaps),
                    "median": np.median(gaps),
                    "max": max(gaps),
                }
        
        return gaps_by_number

    def _calculate_quality(self) -> Dict[str, Any]:
        n_draws = len(self.df)
        
        if n_draws < 30:
            self.warnings.append("Dataset too small (< 30 draws) for robust statistics")
        
        return {
            "total_draws": n_draws,
            "date_range": {
                "from": self.df["draw_date"].min().isoformat(),
                "to": self.df["draw_date"].max().isoformat(),
            },
        }

    def get_warnings(self) -> List[str]:
        return self.warnings
