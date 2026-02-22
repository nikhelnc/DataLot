import numpy as np
import pandas as pd
from typing import Dict, Any, List

from app.analysis.prob_models.m0_baseline import M0Baseline
from app.analysis.prob_models.m1_dirichlet import M1Dirichlet
from app.analysis.prob_models.m2_windowed import M2Windowed
from app.analysis.evaluation.walk_forward import WalkForwardEvaluator


class ModelRunner:
    def __init__(self, df: pd.DataFrame, rules: Dict[str, Any]):
        self.df = df
        self.rules = rules
        self.warnings = []

    def run_models(self, model_names: List[str]) -> Dict[str, Any]:
        if len(self.df) < 30:
            self.warnings.append("Dataset too small for reliable probability estimation")
        
        results = {}
        
        for model_name in model_names:
            if model_name == "M0":
                model = M0Baseline(self.rules)
                results["M0"] = self._run_single_model(model, "M0_baseline")
            elif model_name == "M1":
                model = M1Dirichlet(self.rules, alpha=1.0)
                results["M1"] = self._run_single_model(model, "M1_dirichlet")
            elif model_name == "M2":
                model = M2Windowed(self.rules, window_size=100, lambda_shrink=0.3)
                results["M2"] = self._run_single_model(model, "M2_windowed")
        
        return results

    def _run_single_model(self, model, method_id: str) -> Dict[str, Any]:
        evaluator = WalkForwardEvaluator(self.df, self.rules, model)
        evaluation = evaluator.evaluate()
        
        model.fit(self.df)
        probs = model.predict()
        baseline_probs = M0Baseline(self.rules).predict()
        
        # Support both old (numbers.count) and new (main.pick) structure
        main_rules = self.rules.get("main", self.rules.get("numbers", {}))
        n_count = main_rules.get("pick", main_rules.get("count", 6))
        top_numbers = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:n_count]
        
        warnings = []
        if evaluation["brier_score"] >= evaluation.get("baseline_brier", 0.2):
            warnings.append("No statistically significant lift vs baseline")
        
        return {
            "method_id": method_id,
            "number_probs": probs,
            "baseline_probs": baseline_probs,
            "top_numbers": [int(num) for num, _ in top_numbers],
            "evaluation": evaluation,
            "warnings": warnings,
        }

    def get_warnings(self) -> List[str]:
        return self.warnings
