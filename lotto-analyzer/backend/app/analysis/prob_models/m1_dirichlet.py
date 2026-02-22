import numpy as np
import pandas as pd
from typing import Dict


class M1Dirichlet:
    def __init__(self, rules: Dict, alpha: float = 1.0):
        self.rules = rules
        # Support both old (numbers) and new (main) structure
        main_rules = rules.get("main", rules.get("numbers", {}))
        self.n_min = main_rules.get("min", 1)
        self.n_max = main_rules.get("max", 49)
        self.alpha = alpha
        self.posterior = None

    def fit(self, df: pd.DataFrame):
        all_numbers = []
        for numbers in df["numbers"]:
            all_numbers.extend(numbers)
        
        # Include bonus numbers if available
        if "bonus_numbers" in df.columns:
            for bonus_nums in df["bonus_numbers"]:
                if bonus_nums and len(bonus_nums) > 0:
                    all_numbers.extend(bonus_nums)
        
        counts = pd.Series(all_numbers).value_counts()
        n_range = self.n_max - self.n_min + 1
        
        self.posterior = {}
        total_alpha = self.alpha * n_range
        total_count = len(all_numbers)
        
        for num in range(self.n_min, self.n_max + 1):
            count = counts.get(num, 0)
            self.posterior[num] = (self.alpha + count) / (total_alpha + total_count)

    def predict(self) -> Dict[str, float]:
        if self.posterior is None:
            n_range = self.n_max - self.n_min + 1
            prob = 1.0 / n_range
            return {str(num): prob for num in range(self.n_min, self.n_max + 1)}
        
        return {str(num): float(prob) for num, prob in self.posterior.items()}
