import pandas as pd
from typing import Dict


class M0Baseline:
    def __init__(self, rules: Dict):
        self.rules = rules
        # Support both old (numbers) and new (main) structure
        main_rules = rules.get("main", rules.get("numbers", {}))
        self.n_min = main_rules.get("min", 1)
        self.n_max = main_rules.get("max", 49)

    def fit(self, df: pd.DataFrame):
        pass

    def predict(self) -> Dict[str, float]:
        n_range = self.n_max - self.n_min + 1
        prob = 1.0 / n_range
        return {str(num): prob for num in range(self.n_min, self.n_max + 1)}
