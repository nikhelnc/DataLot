import pandas as pd
from typing import Dict


class M2Windowed:
    def __init__(self, rules: Dict, window_size: int = 100, lambda_shrink: float = 0.3):
        self.rules = rules
        # Support both old (numbers) and new (main) structure
        main_rules = rules.get("main", rules.get("numbers", {}))
        self.n_min = main_rules.get("min", 1)
        self.n_max = main_rules.get("max", 49)
        self.window_size = window_size
        self.lambda_shrink = lambda_shrink
        self.probs = None

    def fit(self, df: pd.DataFrame):
        window_df = df.tail(self.window_size)
        
        all_numbers = []
        for numbers in window_df["numbers"]:
            all_numbers.extend(numbers)
        
        # Include bonus numbers if available
        if "bonus_numbers" in window_df.columns:
            for bonus_nums in window_df["bonus_numbers"]:
                if bonus_nums and len(bonus_nums) > 0:
                    all_numbers.extend(bonus_nums)
        
        counts = pd.Series(all_numbers).value_counts()
        n_range = self.n_max - self.n_min + 1
        total = len(all_numbers)
        
        uniform_prob = 1.0 / n_range
        
        self.probs = {}
        for num in range(self.n_min, self.n_max + 1):
            window_prob = counts.get(num, 0) / total if total > 0 else uniform_prob
            self.probs[num] = self.lambda_shrink * window_prob + (1 - self.lambda_shrink) * uniform_prob

    def predict(self) -> Dict[str, float]:
        if self.probs is None:
            n_range = self.n_max - self.n_min + 1
            prob = 1.0 / n_range
            return {str(num): prob for num in range(self.n_min, self.n_max + 1)}
        
        return {str(num): float(prob) for num, prob in self.probs.items()}
