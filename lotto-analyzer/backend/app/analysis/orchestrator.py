from typing import Dict, Any, List
import pandas as pd

from app.db.models import Game, Draw
from app.analysis.metrics import MetricsCalculator
from app.analysis.randomness import RandomnessTests
from app.analysis.anomalies import AnomalyDetector
from app.analysis.prob_models.model_runner import ModelRunner
from app.analysis.cooccurrence import CooccurrenceAnalysis
from app.analysis.gaps_streaks import GapsStreaksAnalysis
from app.analysis.metatest import MetaTestAnalysis
from app.analysis.ensemble import EnsembleStacking
from app.analysis.backtest import WalkForwardBacktest


class AnalysisOrchestrator:
    def __init__(self, game: Game, draws: List[Draw]):
        self.game = game
        self.draws = draws
        self.df = self._prepare_dataframe()

    def _prepare_dataframe(self) -> pd.DataFrame:
        data = []
        for draw in self.draws:
            data.append(
                {
                    "draw_date": draw.draw_date,
                    "numbers": draw.numbers,
                    "bonus_numbers": draw.bonus_numbers if draw.bonus_numbers else [],
                }
            )
        return pd.DataFrame(data)

    def run(self, analysis_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if analysis_name == "descriptive_v1":
            return self._run_descriptive(params)
        elif analysis_name == "randomness_tests_v1":
            return self._run_randomness_tests(params)
        elif analysis_name == "anomaly_detection_v1":
            return self._run_anomaly_detection(params)
        elif analysis_name == "forecast_probabilities_v1":
            return self._run_probabilities(params)
        elif analysis_name == "advanced_models_v1":
            return self._run_advanced_models(params)
        elif analysis_name == "full_analysis_v1":
            return self._run_full_analysis(params)
        elif analysis_name == "backtest_validation":
            return self._run_backtest(params)
        else:
            raise ValueError(f"Unknown analysis: {analysis_name}")

    def _run_descriptive(self, params: Dict[str, Any]) -> Dict[str, Any]:
        calculator = MetricsCalculator(self.df, self.game.rules_json)
        metrics = calculator.calculate_all()
        
        return {
            "summary": "Descriptive statistics completed",
            "metrics": metrics,
            "warnings": calculator.get_warnings(),
        }

    def _run_randomness_tests(self, params: Dict[str, Any]) -> Dict[str, Any]:
        tester = RandomnessTests(self.df, self.game.rules_json)
        tests = tester.run_all_tests()
        
        return {
            "summary": "Randomness tests completed",
            "tests": tests,
            "warnings": tester.get_warnings(),
        }

    def _run_anomaly_detection(self, params: Dict[str, Any]) -> Dict[str, Any]:
        detector = AnomalyDetector(self.df, self.game.rules_json)
        anomalies = detector.detect_all()
        alerts = detector.generate_alerts()
        
        return {
            "summary": "Anomaly detection completed",
            "anomalies": anomalies,
            "alerts": alerts,
            "warnings": detector.get_warnings(),
        }

    def _run_probabilities(self, params: Dict[str, Any]) -> Dict[str, Any]:
        runner = ModelRunner(self.df, self.game.rules_json)
        results = runner.run_models(params.get("models", ["M0", "M1", "M2"]))
        
        return {
            "summary": "Probability models completed",
            "probabilities": results,
            "warnings": runner.get_warnings(),
        }

    def _run_advanced_models(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run advanced models M5, M6, M9, M10"""
        results = {}
        all_warnings = []
        
        models_to_run = params.get("models", ["M5", "M6", "M9", "M10"])
        
        # M5 - Co-occurrence
        if "M5" in models_to_run:
            m5 = CooccurrenceAnalysis(self.game.rules_json)
            m5.fit(self.df)
            results["M5"] = m5.get_results()
            all_warnings.extend(results["M5"].get("warnings", []))
        
        # M6 - Gaps & Streaks
        if "M6" in models_to_run:
            m6 = GapsStreaksAnalysis(self.game.rules_json)
            m6.fit(self.df)
            results["M6"] = m6.get_results()
            all_warnings.extend(results["M6"].get("warnings", []))
        
        # M9 - Meta-test (needs p-values from randomness tests)
        if "M9" in models_to_run:
            # Run randomness tests to get p-values
            randomness = RandomnessTests(self.df, self.game.rules_json)
            randomness_results = randomness.run_all_tests()
            
            # Extract p-values
            p_values_dict = {}
            for test_name, test_result in randomness_results.items():
                if "p_value" in test_result:
                    p_values_dict[test_name] = [test_result["p_value"]]
            
            m9 = MetaTestAnalysis(self.game.rules_json)
            m9.fit(self.df, p_values_dict)
            results["M9"] = m9.get_results()
            all_warnings.extend(results["M9"].get("warnings", []))
        
        # M10 - Ensemble (needs M0, M1, M2 fitted)
        if "M10" in models_to_run:
            # Get base models
            runner = ModelRunner(self.df, self.game.rules_json)
            base_results = runner.run_models(["M0", "M1", "M2"])
            
            # Create model objects for ensemble
            fitted_models = {}
            for model_name in ["M0", "M1", "M2"]:
                if model_name in base_results:
                    # Create a simple wrapper that has predict() method
                    class ModelWrapper:
                        def __init__(self, probs):
                            self.probs = probs
                        def predict(self):
                            return self.probs
                    
                    fitted_models[model_name] = ModelWrapper(
                        base_results[model_name].get("number_probs", {})
                    )
            
            if len(fitted_models) >= 2:
                m10 = EnsembleStacking(self.game.rules_json, fitted_models)
                m10.fit(self.df)
                results["M10"] = m10.get_results()
                all_warnings.extend(results["M10"].get("warnings", []))
            else:
                results["M10"] = {
                    "error": "Not enough base models for ensemble",
                    "warnings": ["Need at least 2 base models (M0, M1, M2) for ensemble"]
                }
        
        return {
            "summary": "Advanced models completed",
            "advanced_models": results,
            "warnings": all_warnings,
        }

    def _run_full_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        descriptive = self._run_descriptive(params)
        randomness = self._run_randomness_tests(params)
        anomalies = self._run_anomaly_detection(params)
        probabilities = self._run_probabilities(params)
        
        all_warnings = []
        all_warnings.extend(descriptive.get("warnings", []))
        all_warnings.extend(randomness.get("warnings", []))
        all_warnings.extend(anomalies.get("warnings", []))
        all_warnings.extend(probabilities.get("warnings", []))
        
        return {
            "summary": "Full analysis completed",
            "metrics": descriptive.get("metrics"),
            "tests": randomness.get("tests"),
            "anomalies": anomalies.get("anomalies"),
            "alerts": anomalies.get("alerts", []),
            "probabilities": probabilities.get("probabilities"),
            "warnings": list(set(all_warnings)),
        }

    def _run_backtest(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run walk-forward backtesting on models"""
        n_test_draws = params.get("n_test_draws", 20)
        top_n = params.get("top_n", 10)
        n_combinations = params.get("n_combinations", 10)
        max_common_main = params.get("max_common_main", 2)
        max_common_bonus = params.get("max_common_bonus", 1)
        models = params.get("models", ["M0", "M1", "M2", "M5", "M6", "M10"])
        
        backtester = WalkForwardBacktest(
            draws=self.draws,
            game=self.game,
            n_test_draws=n_test_draws,
            top_n=top_n,
            n_combinations=n_combinations,
            max_common_main=max_common_main,
            max_common_bonus=max_common_bonus
        )
        
        results = backtester.run_backtest(models=models)
        
        return {
            "summary": "Backtest validation completed",
            "backtest": results,
            "warnings": results.get("warnings", [])
        }
