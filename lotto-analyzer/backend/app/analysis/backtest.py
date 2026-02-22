"""
Walk-forward backtesting module for evaluating Top N predictions.

This module implements a walk-forward validation strategy where:
1. For each test draw, only historical data before that draw is used
2. Each model generates its Top N predictions
3. Predictions are compared against actual drawn numbers
4. Hit rates and statistics are computed
"""

import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session

from app.db.models import Draw, Game
from app.analysis.prob_models.m0_baseline import M0Baseline
from app.analysis.prob_models.m1_dirichlet import M1Dirichlet
from app.analysis.prob_models.m2_windowed import M2Windowed
from app.analysis.prob_models.m3_exponential_decay import M3ExponentialDecay
from app.analysis.prob_models.m4_hmm import M4HMM
from app.analysis.prob_models.m7_entropy import M7Entropy
from app.analysis.prob_models.m8_changepoint import M8Changepoint
from app.analysis.prob_models.m9_bayesian_network import M9BayesianNetwork
from app.analysis.prob_models.m11_lstm_hybrid import M11LSTMHybrid
from app.analysis.prob_models.m12_mixture_dirichlet import M12MixtureDirichlet
from app.analysis.prob_models.m13_spectral import M13Spectral
from app.analysis.prob_models.m14_copula import M14Copula
from app.analysis.prob_models.m15_thompson_sampling import M15ThompsonSampling
from app.analysis.prob_models.m16_gradient_boosting import M16GradientBoosting
from app.analysis.prob_models.m17_autoencoder import M17Autoencoder
from app.analysis.prob_models.m18_graph_neural import M18GraphNeural
from app.analysis.prob_models.m19_temporal_fusion import M19TemporalFusion
from app.analysis.prob_models.m20_meta_learner import M20MetaLearner
from app.analysis.cooccurrence import CooccurrenceAnalysis
from app.analysis.gaps_streaks import GapsStreaksAnalysis
from app.analysis.ensemble import EnsembleStacking


class WalkForwardBacktest:
    """
    Walk-forward backtesting for lottery prediction models.
    
    For each test draw:
    - Uses only draws before the test date
    - Generates Top N predictions from each model
    - Compares predictions with actual drawn numbers
    - Computes hit rates and statistics
    """
    
    def __init__(
        self,
        draws: List[Draw],
        game: Game,
        n_test_draws: int = 20,
        top_n: int = 10,
        n_combinations: int = 10,
        max_common_main: int = 2,
        max_common_bonus: int = 1
    ):
        """
        Initialize backtester.
        
        Args:
            draws: All available draws (sorted by date ascending)
            game: Game configuration
            n_test_draws: Number of recent draws to test
            top_n: Number of top predictions to generate per model
            n_combinations: Number of combinations for anti-consensus model
            max_common_main: Max main numbers in common between combinations (ANTI2)
            max_common_bonus: Max bonus numbers in common between combinations (ANTI2)
        """
        self.draws = sorted(draws, key=lambda d: d.draw_date)
        self.game = game
        self.n_test_draws = n_test_draws
        self.top_n = top_n
        self.n_combinations = n_combinations
        self.max_common_main = max_common_main
        self.max_common_bonus = max_common_bonus
        self.rules = game.rules_json
        
        # Extract game configuration
        self._parse_game_rules()
    
    def _get_prize_division(self, n_main_hits: int, n_bonus_hits: int) -> Optional[int]:
        """
        Determine which prize division was won based on hits.
        
        Args:
            n_main_hits: Number of main numbers matched
            n_bonus_hits: Number of supplementary/bonus numbers matched
            
        Returns:
            Division number (1 = jackpot) or None if no prize
        """
        prize_divisions = self.rules.get('prize_divisions', [])
        if not prize_divisions:
            return None
        
        # Sort by division number (ascending) to check from best to worst
        sorted_divisions = sorted(prize_divisions, key=lambda d: d.get('division', 999))
        
        for div in sorted_divisions:
            required_main = div.get('main_numbers', 0)
            required_supp = div.get('supplementary', 0)
            
            if n_main_hits >= required_main and n_bonus_hits >= required_supp:
                # Check exact match for main numbers (not just >=)
                if n_main_hits == required_main:
                    return div.get('division')
        
        return None
    
    def _parse_game_rules(self):
        """
        Parse game rules to extract main and bonus configuration.
        Supports different lottery types:
        - Same pool: bonus numbers come from same pool as main (Oz Lotto, TattsLotto)
        - Separate pool: bonus numbers come from different pool (Powerball)
        
        Key distinction:
        - 'pick': how many numbers the PLAYER chooses
        - 'drawn': how many numbers are DRAWN in the lottery
        """
        # Main numbers configuration
        main_rules = self.rules.get('main', {})
        numbers_rules = self.rules.get('numbers', {})
        
        # Player picks vs drawn numbers
        self.main_pick = main_rules.get('pick', numbers_rules.get('count', self.top_n))
        self.main_drawn = main_rules.get('drawn', self.main_pick)  # Default to pick if not specified
        self.main_min = main_rules.get('min', numbers_rules.get('min', 1))
        self.main_max = main_rules.get('max', numbers_rules.get('max', 49))
        
        # Bonus numbers configuration
        bonus_rules = self.rules.get('bonus', {})
        self.bonus_enabled = bonus_rules.get('enabled', False)
        self.bonus_pick = bonus_rules.get('pick', 0) if self.bonus_enabled else 0  # 0 for Oz Lotto
        self.bonus_drawn = bonus_rules.get('drawn', self.bonus_pick) if self.bonus_enabled else 0
        self.bonus_separate_pool = bonus_rules.get('separate_pool', False)
        
        # If separate pool, bonus has its own min/max
        if self.bonus_separate_pool:
            self.bonus_min = bonus_rules.get('min', 1)
            self.bonus_max = bonus_rules.get('max', 20)
        else:
            # Same pool as main numbers
            self.bonus_min = self.main_min
            self.bonus_max = self.main_max
        
        print(f"[RULES] Main: pick={self.main_pick}, drawn={self.main_drawn} from {self.main_min}-{self.main_max}")
        print(f"[RULES] Bonus: enabled={self.bonus_enabled}, pick={self.bonus_pick}, drawn={self.bonus_drawn}, separate_pool={self.bonus_separate_pool}")
        if self.bonus_separate_pool:
            print(f"[RULES] Bonus pool: {self.bonus_min}-{self.bonus_max}")
        
    def run_backtest(self, models: List[str] = None) -> Dict[str, Any]:
        """
        Run walk-forward backtest on specified models.
        
        Args:
            models: List of model names to test (e.g., ['M0', 'M1', 'M2'])
                   If None, tests all available models
        
        Returns:
            Dictionary with backtest results per model
        """
        if models is None:
            models = ['M0', 'M1', 'M2', 'M5', 'M6', 'M10', 'ANTI']
        
        if len(self.draws) < self.n_test_draws + 10:
            return {
                'error': f'Insufficient data: need at least {self.n_test_draws + 10} draws, got {len(self.draws)}'
            }
        
        # Get test draws (last n_test_draws)
        test_draws = self.draws[-self.n_test_draws:]
        
        results = {
            'config': {
                'n_test_draws': self.n_test_draws,
                'top_n': self.top_n,
                'models_tested': models,
                'total_draws_available': len(self.draws)
            },
            'models': {}
        }
        
        # Test each model (except ANTI first)
        standard_models = [m for m in models if m != 'ANTI']
        for model_name in standard_models:
            print(f"Backtesting {model_name}...")
            model_results = self._backtest_model(model_name, test_draws)
            results['models'][model_name] = model_results
        
        # Test ANTI-Consensus model if requested
        if 'ANTI' in models:
            print(f"Backtesting ANTI-Consensus...")
            anti_results = self._backtest_anti_consensus(test_draws, results['models'])
            results['models']['ANTI'] = anti_results
        
        # Test ANTI2-Consensus model (with diversity constraint) if requested
        if 'ANTI2' in models:
            print(f"Backtesting ANTI2-Consensus (diversity: max {self.max_common_main} main, {self.max_common_bonus} bonus in common)...")
            anti2_results = self._backtest_anti_consensus_v2(test_draws, results['models'])
            results['models']['ANTI2'] = anti2_results
        
        # Compute comparative statistics
        results['comparison'] = self._compute_comparison(results['models'])
        
        return results
    
    def _backtest_model(self, model_name: str, test_draws: List[Draw]) -> Dict[str, Any]:
        """
        Backtest a single model on test draws.
        
        Args:
            model_name: Name of the model (e.g., 'M1')
            test_draws: List of draws to test on
        
        Returns:
            Dictionary with model backtest results
        """
        predictions = []
        total_main_hits = 0
        total_main_possible = 0
        total_bonus_hits = 0
        total_bonus_possible = 0
        
        for i, test_draw in enumerate(test_draws):
            # Get training data: all draws before this test draw
            train_draws = [d for d in self.draws if d.draw_date < test_draw.draw_date]
            
            if len(train_draws) < 5:
                continue
            
            # Generate Top N predictions using only training data
            predictions_dict = self._get_top_n_predictions(model_name, train_draws)
            predicted_main = predictions_dict['main_numbers']
            predicted_bonus = predictions_dict['bonus_numbers']
            
            # Get actual drawn numbers
            actual_main = set(test_draw.numbers)
            actual_bonus = set(test_draw.bonus_numbers) if test_draw.bonus_numbers else set()
            
            # Compute hits for main numbers
            main_hits = actual_main.intersection(set(predicted_main))
            n_main_hits = len(main_hits)
            n_main_drawn = len(actual_main)
            
            # Compute hits for bonus numbers
            bonus_hits = actual_bonus.intersection(set(predicted_bonus))
            n_bonus_hits = len(bonus_hits)
            n_bonus_drawn = len(actual_bonus)
            
            total_main_hits += n_main_hits
            total_main_possible += n_main_drawn
            total_bonus_hits += n_bonus_hits
            total_bonus_possible += n_bonus_drawn
            
            # Store prediction details
            predictions.append({
                'draw_date': test_draw.draw_date.isoformat(),
                'draw_id': str(test_draw.id),
                'predicted_main': predicted_main,
                'predicted_bonus': predicted_bonus,
                'actual_main': list(actual_main),
                'actual_bonus': list(actual_bonus),
                'main_hits': list(main_hits),
                'bonus_hits': list(bonus_hits),
                'n_main_hits': n_main_hits,
                'n_main_drawn': n_main_drawn,
                'n_bonus_hits': n_bonus_hits,
                'n_bonus_drawn': n_bonus_drawn,
                'main_hit_rate': n_main_hits / n_main_drawn if n_main_drawn > 0 else 0,
                'bonus_hit_rate': n_bonus_hits / n_bonus_drawn if n_bonus_drawn > 0 else 0,
                'training_size': len(train_draws)
            })
        
        # Compute aggregate statistics
        main_hit_rates = [p['main_hit_rate'] for p in predictions]
        bonus_hit_rates = [p['bonus_hit_rate'] for p in predictions if p['n_bonus_drawn'] > 0]
        
        avg_main_hit_rate = sum(main_hit_rates) / len(main_hit_rates) if main_hit_rates else 0
        avg_bonus_hit_rate = sum(bonus_hit_rates) / len(bonus_hit_rates) if bonus_hit_rates else 0
        
        # Expected hit rate for random selection (using correct pool sizes)
        main_pool_size = self.main_max - self.main_min + 1
        bonus_pool_size = self.bonus_max - self.bonus_min + 1 if self.bonus_enabled else 0
        expected_main_hit_rate = (self.main_pick / main_pool_size) if main_pool_size > 0 else 0
        expected_bonus_hit_rate = (self.bonus_pick / bonus_pool_size) if bonus_pool_size > 0 and self.bonus_pick > 0 else 0
        
        # Overall hit rate (weighted average)
        total_hits = total_main_hits + total_bonus_hits
        total_possible = total_main_possible + total_bonus_possible
        avg_hit_rate = total_hits / total_possible if total_possible > 0 else 0
        
        return {
            'model_name': model_name,
            'predictions': predictions,
            'statistics': {
                'total_main_hits': total_main_hits,
                'total_main_possible': total_main_possible,
                'total_bonus_hits': total_bonus_hits,
                'total_bonus_possible': total_bonus_possible,
                'total_hits': total_hits,
                'total_possible': total_possible,
                'avg_main_hit_rate': avg_main_hit_rate,
                'avg_bonus_hit_rate': avg_bonus_hit_rate,
                'avg_hit_rate': avg_hit_rate,
                'max_hit_rate': max(main_hit_rates) if main_hit_rates else 0,
                'expected_main_hit_rate': expected_main_hit_rate,
                'expected_bonus_hit_rate': expected_bonus_hit_rate,
                'lift_vs_random': avg_main_hit_rate / expected_main_hit_rate if expected_main_hit_rate > 0 else 0,
                'best_main_hit_rate': max(main_hit_rates) if main_hit_rates else 0,
                'worst_main_hit_rate': min(main_hit_rates) if main_hit_rates else 0,
                'best_bonus_hit_rate': max(bonus_hit_rates) if bonus_hit_rates else 0,
                'worst_bonus_hit_rate': min(bonus_hit_rates) if bonus_hit_rates else 0
            }
        }
    
    def _get_top_n_predictions(self, model_name: str, train_draws: List[Draw]) -> Dict[str, List[int]]:
        """
        Generate Top N predictions from a model using training data.
        
        Args:
            model_name: Name of the model
            train_draws: Training draws (before test date)
        
        Returns:
            Dictionary with 'main_numbers' and 'bonus_numbers' lists
        """
        # Prepare DataFrame from training draws
        df = self._prepare_dataframe(train_draws)
        
        try:
            if model_name == 'M0':
                # M0 Baseline: random selection (true baseline for comparison)
                import random
                all_main_numbers = list(range(self.main_min, self.main_max + 1))
                main_numbers = sorted(random.sample(all_main_numbers, min(self.main_pick, len(all_main_numbers))))
                
                bonus_numbers = []
                if self.bonus_enabled and self.bonus_pick > 0:
                    if self.bonus_separate_pool:
                        all_bonus_numbers = list(range(self.bonus_min, self.bonus_max + 1))
                        bonus_numbers = sorted(random.sample(all_bonus_numbers, min(self.bonus_pick, len(all_bonus_numbers))))
                    else:
                        # Same pool: pick from remaining numbers
                        remaining = [n for n in all_main_numbers if n not in main_numbers]
                        bonus_numbers = sorted(random.sample(remaining, min(self.bonus_pick, len(remaining))))
                
                return {'main_numbers': main_numbers, 'bonus_numbers': bonus_numbers}
                
            elif model_name == 'M1':
                model = M1Dirichlet(self.rules)
                model.fit(df)
                probs = model.predict()
                
            elif model_name == 'M2':
                model = M2Windowed(self.rules, window_size=50, lambda_shrink=0.1)
                model.fit(df)
                probs = model.predict()
                
            elif model_name == 'M5':
                # Co-occurrence: use top numbers from most over-represented pairs
                analysis = CooccurrenceAnalysis(self.rules)
                analysis.fit(df)
                results = analysis.get_results()
                top_pairs = results['cooccurrence']['top_pairs'][:20]
                
                # Count frequency of numbers in top pairs
                number_freq = {}
                for pair in top_pairs:
                    for num in [pair['num1'], pair['num2']]:
                        number_freq[num] = number_freq.get(num, 0) + pair['delta']
                
                # Sort by frequency and take top N for main and bonus
                sorted_nums = sorted(number_freq.items(), key=lambda x: x[1], reverse=True)
                all_predicted = [num for num, _ in sorted_nums]
                
                # Filter for main pool
                main_pool_predicted = [n for n in all_predicted if self.main_min <= n <= self.main_max]
                main_numbers = main_pool_predicted[:self.main_pick]
                
                # Handle bonus
                if self.bonus_enabled and self.bonus_pick > 0:
                    if self.bonus_separate_pool:
                        bonus_pool_predicted = [n for n in all_predicted if self.bonus_min <= n <= self.bonus_max]
                        if len(bonus_pool_predicted) < self.bonus_pick:
                            bonus_pool_predicted = list(range(self.bonus_min, self.bonus_min + self.bonus_pick))
                        bonus_numbers = bonus_pool_predicted[:self.bonus_pick]
                    else:
                        bonus_numbers = main_pool_predicted[self.main_pick:self.main_pick + self.bonus_pick]
                else:
                    bonus_numbers = []
                
                return {'main_numbers': main_numbers, 'bonus_numbers': bonus_numbers}
                
            elif model_name == 'M6':
                # Gaps: use numbers with largest positive gaps (overdue)
                analysis = GapsStreaksAnalysis(self.rules)
                analysis.fit(df)
                results = analysis.get_results()
                top_atypical = results['gaps_streaks']['top_atypical']
                
                # Filter for positive gaps and sort
                overdue = [s for s in top_atypical if s.get('delta_gap', 0) > 0]
                overdue.sort(key=lambda x: x.get('delta_gap', 0), reverse=True)
                all_predicted = [s['number'] for s in overdue]
                
                # Filter for main pool
                main_pool_predicted = [n for n in all_predicted if self.main_min <= n <= self.main_max]
                main_numbers = main_pool_predicted[:self.main_pick]
                
                # Handle bonus
                if self.bonus_enabled and self.bonus_pick > 0:
                    if self.bonus_separate_pool:
                        bonus_pool_predicted = [n for n in all_predicted if self.bonus_min <= n <= self.bonus_max]
                        if len(bonus_pool_predicted) < self.bonus_pick:
                            bonus_pool_predicted = list(range(self.bonus_min, self.bonus_min + self.bonus_pick))
                        bonus_numbers = bonus_pool_predicted[:self.bonus_pick]
                    else:
                        bonus_numbers = main_pool_predicted[self.main_pick:self.main_pick + self.bonus_pick]
                else:
                    bonus_numbers = []
                
                return {'main_numbers': main_numbers, 'bonus_numbers': bonus_numbers}
                
            elif model_name == 'M10':
                # Ensemble: need M0, M1, M2 first
                m0_model = M0Baseline(self.rules)
                m0_model.fit(df)
                m0 = m0_model.predict()
                
                m1_model = M1Dirichlet(self.rules)
                m1_model.fit(df)
                m1 = m1_model.predict()
                
                m2_model = M2Windowed(self.rules)
                m2_model.fit(df)
                m2 = m2_model.predict()
                
                # Create model wrappers for ensemble
                class ModelWrapper:
                    def __init__(self, probs):
                        self.probs = probs
                    def predict(self):
                        return self.probs
                
                fitted_models = {
                    'M0': ModelWrapper(m0),
                    'M1': ModelWrapper(m1),
                    'M2': ModelWrapper(m2)
                }
                
                ensemble = EnsembleStacking(self.rules, fitted_models)
                ensemble.fit(df)
                results = ensemble.get_results()
                probs = results['number_probs']
            
            elif model_name == 'M3':
                # M3 Exponential Decay
                model = M3ExponentialDecay(self.rules, lambda_decay=0.02)
                model.fit(df)
                probs = model.predict()
            
            elif model_name == 'M4':
                # M4 HMM (Hidden Markov Model)
                model = M4HMM(self.rules, n_states=3)
                model.fit(df)
                probs = model.predict()
            
            elif model_name == 'M7':
                # M7 Entropy-Based Selection
                model = M7Entropy(self.rules, window_size=30)
                model.fit(df)
                probs = model.predict()
            
            elif model_name == 'M8':
                # M8 Changepoint Detection
                model = M8Changepoint(self.rules)
                model.fit(df)
                probs = model.predict()
            
            elif model_name == 'M9':
                # M9 Bayesian Network
                model = M9BayesianNetwork(self.rules)
                model.fit(df)
                probs = model.predict()
            
            elif model_name == 'M11':
                # M11 LSTM Hybrid: Deep learning model with attention
                model = M11LSTMHybrid(
                    self.rules,
                    sequence_length=min(50, len(df) - 1),
                    epochs=30,
                    verbose=0
                )
                model.fit(df)
                probs = model.predict()
            
            elif model_name == 'M12':
                # M12 Mixture of Dirichlet
                model = M12MixtureDirichlet(self.rules, n_components=2)
                model.fit(df)
                probs = model.predict()
            
            elif model_name == 'M13':
                # M13 Spectral/Fourier Analysis
                model = M13Spectral(self.rules)
                model.fit(df)
                probs = model.predict()
            
            elif model_name == 'M14':
                # M14 Copula Model
                model = M14Copula(self.rules)
                model.fit(df)
                probs = model.predict()
            
            elif model_name == 'M15':
                # M15 Thompson Sampling
                model = M15ThompsonSampling(self.rules)
                model.fit(df)
                probs = model.predict()
            
            elif model_name == 'M16':
                # M16 Gradient Boosting - uses different interface
                draws_list = [list(d.numbers) for d in train_draws]
                model = M16GradientBoosting()
                model.fit(draws_list, self.main_max, self.main_pick)
                probs_array = model.predict_proba()["main"]
                # Convert to dict format
                probs = {i + 1: float(probs_array[i]) for i in range(len(probs_array))}
            
            elif model_name == 'M17':
                # M17 Autoencoder Anomaly - uses different interface
                draws_list = [list(d.numbers) for d in train_draws]
                model = M17Autoencoder()
                model.fit(draws_list, self.main_max, self.main_pick)
                probs_array = model.predict_proba()["main"]
                probs = {i + 1: float(probs_array[i]) for i in range(len(probs_array))}
            
            elif model_name == 'M18':
                # M18 Graph Neural Network - uses different interface
                draws_list = [list(d.numbers) for d in train_draws]
                model = M18GraphNeural()
                model.fit(draws_list, self.main_max, self.main_pick)
                probs_array = model.predict_proba()["main"]
                probs = {i + 1: float(probs_array[i]) for i in range(len(probs_array))}
            
            elif model_name == 'M19':
                # M19 Temporal Fusion - uses different interface
                draws_list = [list(d.numbers) for d in train_draws]
                model = M19TemporalFusion()
                model.fit(draws_list, self.main_max, self.main_pick)
                probs_array = model.predict_proba()["main"]
                probs = {i + 1: float(probs_array[i]) for i in range(len(probs_array))}
            
            elif model_name == 'M20':
                # M20 Meta-Learner - uses different interface
                draws_list = [list(d.numbers) for d in train_draws]
                model = M20MetaLearner()
                model.fit(draws_list, self.main_max, self.main_pick)
                probs_array = model.predict_proba()["main"]
                probs = {i + 1: float(probs_array[i]) for i in range(len(probs_array))}
                
            else:
                # Unknown model, return uniform random
                import random
                all_main_numbers = list(range(self.main_min, self.main_max + 1))
                main_numbers = sorted(random.sample(all_main_numbers, min(self.main_pick, len(all_main_numbers))))
                bonus_numbers = []
                if self.bonus_enabled and self.bonus_pick > 0:
                    if self.bonus_separate_pool:
                        all_bonus_numbers = list(range(self.bonus_min, self.bonus_max + 1))
                        bonus_numbers = sorted(random.sample(all_bonus_numbers, min(self.bonus_pick, len(all_bonus_numbers))))
                    else:
                        remaining = [n for n in all_main_numbers if n not in main_numbers]
                        bonus_numbers = sorted(random.sample(remaining, min(self.bonus_pick, len(remaining))))
                return {'main_numbers': main_numbers, 'bonus_numbers': bonus_numbers}
            
            # For probability-based models, extract top N for main and bonus
            if model_name in ['M0', 'M1', 'M2', 'M3', 'M4', 'M7', 'M8', 'M9', 'M10', 'M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20']:
                sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                all_predicted = [int(num) for num, _ in sorted_probs]
                
                # Filter for main pool numbers only
                main_pool_predicted = [n for n in all_predicted if self.main_min <= n <= self.main_max]
                main_numbers = main_pool_predicted[:self.main_pick]
                
                # Handle bonus numbers
                if self.bonus_enabled and self.bonus_pick > 0:
                    if self.bonus_separate_pool:
                        # Separate pool (Powerball): filter for bonus pool numbers
                        bonus_pool_predicted = [n for n in all_predicted if self.bonus_min <= n <= self.bonus_max]
                        # If model doesn't predict bonus pool numbers, use top from bonus range
                        if len(bonus_pool_predicted) < self.bonus_pick:
                            # Fallback: use numbers from bonus pool based on frequency in main predictions
                            bonus_pool_predicted = list(range(self.bonus_min, self.bonus_min + self.bonus_pick))
                        bonus_numbers = bonus_pool_predicted[:self.bonus_pick]
                    else:
                        # Same pool: take next numbers after main
                        bonus_numbers = main_pool_predicted[self.main_pick:self.main_pick + self.bonus_pick]
                else:
                    bonus_numbers = []
                
                return {'main_numbers': main_numbers, 'bonus_numbers': bonus_numbers}
            
        except Exception as e:
            print(f"Error in {model_name}: {e}")
            # Fallback: return default numbers from correct pools
            main_numbers = list(range(self.main_min, self.main_min + self.main_pick))
            if self.bonus_enabled and self.bonus_pick > 0:
                bonus_numbers = list(range(self.bonus_min, self.bonus_min + self.bonus_pick))
            else:
                bonus_numbers = []
            return {'main_numbers': main_numbers, 'bonus_numbers': bonus_numbers}
        
        # Fallback at end of function
        main_numbers = list(range(self.main_min, self.main_min + self.main_pick))
        if self.bonus_enabled and self.bonus_pick > 0:
            bonus_numbers = list(range(self.bonus_min, self.bonus_min + self.bonus_pick))
        else:
            bonus_numbers = []
        return {'main_numbers': main_numbers, 'bonus_numbers': bonus_numbers}
    
    def _backtest_anti_consensus(self, test_draws: List[Draw], other_models_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Backtest anti-consensus model: predict numbers NOT predicted by other models.
        Generates multiple random combinations and selects the best one.
        
        Args:
            test_draws: List of draws to test on
            other_models_results: Results from other models to extract their predictions
        
        Returns:
            Dictionary with anti-consensus backtest results
        """
        import random
        
        predictions = []
        total_main_hits = 0
        total_main_possible = 0
        total_bonus_hits = 0
        total_bonus_possible = 0
        
        # Use parsed properties
        all_main_numbers = set(range(self.main_min, self.main_max + 1))
        all_bonus_numbers = set(range(self.bonus_min, self.bonus_max + 1)) if self.bonus_enabled else set()
        
        for i, test_draw in enumerate(test_draws):
            # Collect all numbers predicted by other models for this draw
            predicted_main_by_others = set()
            predicted_bonus_by_others = set()
            for model_name, model_data in other_models_results.items():
                if 'predictions' in model_data and i < len(model_data['predictions']):
                    pred = model_data['predictions'][i]
                    predicted_main_by_others.update(pred.get('predicted_main', []))
                    predicted_bonus_by_others.update(pred.get('predicted_bonus', []))
            
            # Get numbers NOT predicted by any model (from correct pools)
            non_predicted_main = list(all_main_numbers - predicted_main_by_others)
            
            if self.bonus_separate_pool:
                # Separate pool: bonus numbers come from different pool
                non_predicted_bonus = list(all_bonus_numbers - predicted_bonus_by_others)
            else:
                # Same pool: bonus numbers come from main pool (excluding main predictions)
                non_predicted_bonus = non_predicted_main.copy()
            
            if len(non_predicted_main) < self.main_pick:
                # Not enough non-predicted main numbers, skip this draw
                continue
            
            if self.bonus_enabled and self.bonus_pick > 0 and len(non_predicted_bonus) < self.bonus_pick:
                # Not enough non-predicted bonus numbers, skip this draw
                continue
            
            # Generate n_combinations random combinations
            combinations = []
            for _ in range(self.n_combinations):
                combo_main = sorted(random.sample(non_predicted_main, self.main_pick))
                if self.bonus_enabled and self.bonus_pick > 0:
                    if self.bonus_separate_pool:
                        combo_bonus = sorted(random.sample(non_predicted_bonus, self.bonus_pick))
                    else:
                        # Same pool: exclude main numbers from bonus selection
                        available_for_bonus = [n for n in non_predicted_bonus if n not in combo_main]
                        if len(available_for_bonus) >= self.bonus_pick:
                            combo_bonus = sorted(random.sample(available_for_bonus, self.bonus_pick))
                        else:
                            combo_bonus = sorted(available_for_bonus[:self.bonus_pick])
                else:
                    combo_bonus = []
                combinations.append({'main': combo_main, 'bonus': combo_bonus})
            
            # Get actual drawn numbers
            actual_main = set(test_draw.numbers)
            actual_bonus = set(test_draw.bonus_numbers) if test_draw.bonus_numbers else set()
            
            # Evaluate all combinations and store details for each
            all_combos_details = []
            best_combo_main = combinations[0]['main'] if combinations else []
            best_combo_bonus = combinations[0]['bonus'] if combinations else []
            best_main_hits = 0
            best_bonus_hits = 0
            best_total_score = 0
            
            for combo in combinations:
                main_set = set(combo['main'])
                bonus_set = set(combo['bonus'])
                
                main_hits = actual_main.intersection(main_set)
                bonus_hits = actual_bonus.intersection(bonus_set)
                
                n_main_hits = len(main_hits)
                n_bonus_hits = len(bonus_hits)
                
                main_hit_rate = n_main_hits / len(actual_main) if len(actual_main) > 0 else 0
                bonus_hit_rate = n_bonus_hits / len(actual_bonus) if len(actual_bonus) > 0 else 0
                
                # Total score for ranking (weighted by importance)
                total_score = n_main_hits + (n_bonus_hits * 0.5)
                
                # Calculate prize division for this combination
                # For bonus hits, we check against actual_bonus (supplementary numbers)
                division = self._get_prize_division(n_main_hits, n_bonus_hits)
                
                # Store details for this combination
                all_combos_details.append({
                    'main_combination': combo['main'],
                    'bonus_combination': combo['bonus'],
                    'main_hits': list(main_hits),
                    'bonus_hits': list(bonus_hits),
                    'n_main_hits': n_main_hits,
                    'n_bonus_hits': n_bonus_hits,
                    'main_hit_rate': main_hit_rate,
                    'bonus_hit_rate': bonus_hit_rate,
                    'total_score': total_score,
                    'division': division
                })
                
                if total_score > best_total_score:
                    best_total_score = total_score
                    best_combo_main = combo['main']
                    best_combo_bonus = combo['bonus']
                    best_main_hits = n_main_hits
                    best_bonus_hits = n_bonus_hits
            
            total_main_hits += best_main_hits
            total_main_possible += len(actual_main)
            total_bonus_hits += best_bonus_hits
            total_bonus_possible += len(actual_bonus)
            
            # Sort combinations by total_score (descending)
            all_combos_details.sort(key=lambda x: x['total_score'], reverse=True)
            
            # Calculate division summary for this draw
            division_summary = {}
            for combo_detail in all_combos_details:
                div = combo_detail.get('division')
                if div is not None:
                    division_summary[f'div_{div}'] = division_summary.get(f'div_{div}', 0) + 1
            
            # Store prediction details with all combinations
            pred_dict = {
                'draw_date': test_draw.draw_date.isoformat(),
                'draw_id': str(test_draw.id),
                'predicted_main': best_combo_main,
                'predicted_bonus': best_combo_bonus,
                'actual_main': list(actual_main),
                'actual_bonus': list(actual_bonus),
                'main_hits': list(actual_main.intersection(set(best_combo_main))),
                'bonus_hits': list(actual_bonus.intersection(set(best_combo_bonus))),
                'n_main_hits': best_main_hits,
                'n_main_drawn': len(actual_main),
                'n_bonus_hits': best_bonus_hits,
                'n_bonus_drawn': len(actual_bonus),
                'main_hit_rate': best_main_hits / len(actual_main) if len(actual_main) > 0 else 0,
                'bonus_hit_rate': best_bonus_hits / len(actual_bonus) if len(actual_bonus) > 0 else 0,
                'n_combinations_tested': self.n_combinations,
                'n_non_predicted_main': len(non_predicted_main),
                'n_non_predicted_bonus': len(non_predicted_bonus) if self.bonus_enabled else 0,
                'all_combinations': all_combos_details,
                'division_summary': division_summary
            }
            print(f"[ANTI DEBUG] Created prediction with {len(all_combos_details)} combinations")
            predictions.append(pred_dict)
        
        # Compute aggregate statistics
        main_hit_rates = [p['main_hit_rate'] for p in predictions]
        bonus_hit_rates = [p['bonus_hit_rate'] for p in predictions if p['n_bonus_drawn'] > 0]
        
        avg_main_hit_rate = sum(main_hit_rates) / len(main_hit_rates) if main_hit_rates else 0
        avg_bonus_hit_rate = sum(bonus_hit_rates) / len(bonus_hit_rates) if bonus_hit_rates else 0
        
        # Expected hit rate for random selection (using correct pool sizes)
        main_pool_size = self.main_max - self.main_min + 1
        bonus_pool_size = self.bonus_max - self.bonus_min + 1 if self.bonus_enabled else 0
        expected_main_hit_rate = (self.main_pick / main_pool_size) if main_pool_size > 0 else 0
        expected_bonus_hit_rate = (self.bonus_pick / bonus_pool_size) if bonus_pool_size > 0 and self.bonus_pick > 0 else 0
        
        # Overall hit rate
        total_hits = total_main_hits + total_bonus_hits
        total_possible = total_main_possible + total_bonus_possible
        avg_hit_rate = total_hits / total_possible if total_possible > 0 else 0
        
        return {
            'model_name': 'ANTI',
            'predictions': predictions,
            'statistics': {
                'total_tests': len(predictions),
                'total_main_hits': total_main_hits,
                'total_main_possible': total_main_possible,
                'total_bonus_hits': total_bonus_hits,
                'total_bonus_possible': total_bonus_possible,
                'total_hits': total_hits,
                'total_possible': total_possible,
                'avg_main_hit_rate': avg_main_hit_rate,
                'avg_bonus_hit_rate': avg_bonus_hit_rate,
                'avg_hit_rate': avg_hit_rate,
                'max_hit_rate': max(main_hit_rates) if main_hit_rates else 0,
                'expected_main_hit_rate': expected_main_hit_rate,
                'expected_bonus_hit_rate': expected_bonus_hit_rate,
                'lift_vs_random': avg_main_hit_rate / expected_main_hit_rate if expected_main_hit_rate > 0 else 0,
                'best_main_hit_rate': max(main_hit_rates) if main_hit_rates else 0,
                'worst_main_hit_rate': min(main_hit_rates) if main_hit_rates else 0,
                'best_bonus_hit_rate': max(bonus_hit_rates) if bonus_hit_rates else 0,
                'worst_bonus_hit_rate': min(bonus_hit_rates) if bonus_hit_rates else 0,
                'n_combinations_per_draw': self.n_combinations
            }
        }
    
    def _backtest_anti_consensus_v2(self, test_draws: List[Draw], other_models_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Backtest anti-consensus v2 model with diversity constraint.
        Similar to ANTI but ensures combinations don't share too many numbers.
        
        Args:
            test_draws: List of draws to test on
            other_models_results: Results from other models to extract their predictions
        
        Returns:
            Dictionary with anti-consensus v2 backtest results
        """
        import random
        
        predictions = []
        total_main_hits = 0
        total_main_possible = 0
        total_bonus_hits = 0
        total_bonus_possible = 0
        
        all_main_numbers = set(range(self.main_min, self.main_max + 1))
        all_bonus_numbers = set(range(self.bonus_min, self.bonus_max + 1)) if self.bonus_enabled else set()
        
        for i, test_draw in enumerate(test_draws):
            # Collect all numbers predicted by other models for this draw
            predicted_main_by_others = set()
            predicted_bonus_by_others = set()
            for model_name, model_data in other_models_results.items():
                if model_name in ['ANTI', 'ANTI2']:
                    continue  # Skip other anti-consensus models
                if 'predictions' in model_data and i < len(model_data['predictions']):
                    pred = model_data['predictions'][i]
                    predicted_main_by_others.update(pred.get('predicted_main', []))
                    predicted_bonus_by_others.update(pred.get('predicted_bonus', []))
            
            # Get numbers NOT predicted by any model
            non_predicted_main = list(all_main_numbers - predicted_main_by_others)
            
            if self.bonus_separate_pool:
                non_predicted_bonus = list(all_bonus_numbers - predicted_bonus_by_others)
            else:
                non_predicted_bonus = non_predicted_main.copy()
            
            if len(non_predicted_main) < self.main_pick:
                continue
            
            if self.bonus_enabled and self.bonus_pick > 0 and len(non_predicted_bonus) < self.bonus_pick:
                continue
            
            # Generate combinations with diversity constraint
            combinations = []
            used_bonus_numbers = set()  # Track used bonus numbers when max_common_bonus=0
            max_attempts_per_combo = 100  # Max attempts per combination
            
            # Phase 1: Try to generate diverse combinations
            for combo_idx in range(self.n_combinations):
                found = False
                
                # Determine available bonus numbers for this combination
                if self.bonus_enabled and self.bonus_pick > 0 and self.max_common_bonus == 0:
                    # When max_common_bonus=0, exclude already used bonus numbers
                    available_bonus_pool = [n for n in non_predicted_bonus if n not in used_bonus_numbers]
                else:
                    available_bonus_pool = non_predicted_bonus.copy() if non_predicted_bonus else []
                
                for attempt in range(max_attempts_per_combo):
                    # Generate a candidate combination
                    combo_main = sorted(random.sample(non_predicted_main, self.main_pick))
                    combo_bonus = []
                    
                    if self.bonus_enabled and self.bonus_pick > 0:
                        if self.bonus_separate_pool:
                            # For separate pool, use available_bonus_pool
                            if len(available_bonus_pool) >= self.bonus_pick:
                                combo_bonus = sorted(random.sample(available_bonus_pool, self.bonus_pick))
                            elif available_bonus_pool:
                                # Not enough unique bonus numbers left, use what's available
                                combo_bonus = sorted(available_bonus_pool[:self.bonus_pick])
                            else:
                                # Fallback: use from full pool if no unique available
                                combo_bonus = sorted(random.sample(non_predicted_bonus, self.bonus_pick))
                        else:
                            available_for_bonus = [n for n in available_bonus_pool if n not in combo_main]
                            if len(available_for_bonus) >= self.bonus_pick:
                                combo_bonus = sorted(random.sample(available_for_bonus, self.bonus_pick))
                            elif available_for_bonus:
                                combo_bonus = sorted(available_for_bonus[:self.bonus_pick])
                            else:
                                # Fallback
                                fallback_bonus = [n for n in non_predicted_bonus if n not in combo_main]
                                if fallback_bonus:
                                    combo_bonus = sorted(random.sample(fallback_bonus, min(self.bonus_pick, len(fallback_bonus))))
                    
                    # Check diversity constraint against all existing combinations
                    is_diverse = True
                    for existing in combinations:
                        # Count common main numbers
                        common_main = len(set(combo_main) & set(existing['main']))
                        if common_main > self.max_common_main:
                            is_diverse = False
                            break
                        
                        # Count common bonus numbers (if applicable)
                        if self.bonus_enabled and self.bonus_pick > 0 and combo_bonus and existing['bonus']:
                            common_bonus = len(set(combo_bonus) & set(existing['bonus']))
                            if common_bonus > self.max_common_bonus:
                                is_diverse = False
                                break
                    
                    if is_diverse:
                        combinations.append({'main': combo_main, 'bonus': combo_bonus})
                        # Track used bonus numbers
                        if combo_bonus:
                            used_bonus_numbers.update(combo_bonus)
                        found = True
                        break
                
                # Phase 2: If we couldn't find a diverse combo, add one anyway but still try to respect bonus constraint
                if not found:
                    combo_main = sorted(random.sample(non_predicted_main, self.main_pick))
                    combo_bonus = []
                    if self.bonus_enabled and self.bonus_pick > 0:
                        if self.bonus_separate_pool:
                            # Try to use unused bonus first
                            if len(available_bonus_pool) >= self.bonus_pick:
                                combo_bonus = sorted(random.sample(available_bonus_pool, self.bonus_pick))
                            else:
                                # Must reuse some bonus numbers
                                combo_bonus = sorted(random.sample(non_predicted_bonus, self.bonus_pick))
                        else:
                            available_for_bonus = [n for n in available_bonus_pool if n not in combo_main]
                            if len(available_for_bonus) >= self.bonus_pick:
                                combo_bonus = sorted(random.sample(available_for_bonus, self.bonus_pick))
                            else:
                                fallback_bonus = [n for n in non_predicted_bonus if n not in combo_main]
                                if fallback_bonus:
                                    combo_bonus = sorted(random.sample(fallback_bonus, min(self.bonus_pick, len(fallback_bonus))))
                    
                    combinations.append({'main': combo_main, 'bonus': combo_bonus})
                    if combo_bonus:
                        used_bonus_numbers.update(combo_bonus)
            
            # Get actual drawn numbers
            actual_main = set(test_draw.numbers)
            actual_bonus = set(test_draw.bonus_numbers) if test_draw.bonus_numbers else set()
            
            # Evaluate all combinations
            all_combos_details = []
            best_combo_main = combinations[0]['main']
            best_combo_bonus = combinations[0]['bonus']
            best_main_hits = 0
            best_bonus_hits = 0
            best_total_score = 0
            
            for combo in combinations:
                main_set = set(combo['main'])
                bonus_set = set(combo['bonus'])
                
                main_hits = actual_main.intersection(main_set)
                bonus_hits = actual_bonus.intersection(bonus_set)
                
                n_main_hits = len(main_hits)
                n_bonus_hits = len(bonus_hits)
                
                main_hit_rate = n_main_hits / len(actual_main) if len(actual_main) > 0 else 0
                bonus_hit_rate = n_bonus_hits / len(actual_bonus) if len(actual_bonus) > 0 else 0
                
                total_score = n_main_hits + (n_bonus_hits * 0.5)
                division = self._get_prize_division(n_main_hits, n_bonus_hits)
                
                all_combos_details.append({
                    'main_combination': combo['main'],
                    'bonus_combination': combo['bonus'],
                    'main_hits': list(main_hits),
                    'bonus_hits': list(bonus_hits),
                    'n_main_hits': n_main_hits,
                    'n_bonus_hits': n_bonus_hits,
                    'main_hit_rate': main_hit_rate,
                    'bonus_hit_rate': bonus_hit_rate,
                    'total_score': total_score,
                    'division': division
                })
                
                if total_score > best_total_score:
                    best_total_score = total_score
                    best_combo_main = combo['main']
                    best_combo_bonus = combo['bonus']
                    best_main_hits = n_main_hits
                    best_bonus_hits = n_bonus_hits
            
            total_main_hits += best_main_hits
            total_main_possible += len(actual_main)
            total_bonus_hits += best_bonus_hits
            total_bonus_possible += len(actual_bonus)
            
            all_combos_details.sort(key=lambda x: x['total_score'], reverse=True)
            
            division_summary = {}
            for combo_detail in all_combos_details:
                div = combo_detail.get('division')
                if div is not None:
                    division_summary[f'div_{div}'] = division_summary.get(f'div_{div}', 0) + 1
            
            pred_dict = {
                'draw_date': test_draw.draw_date.isoformat(),
                'draw_id': str(test_draw.id),
                'predicted_main': best_combo_main,
                'predicted_bonus': best_combo_bonus,
                'actual_main': list(actual_main),
                'actual_bonus': list(actual_bonus),
                'main_hits': list(actual_main.intersection(set(best_combo_main))),
                'bonus_hits': list(actual_bonus.intersection(set(best_combo_bonus))),
                'n_main_hits': best_main_hits,
                'n_main_drawn': len(actual_main),
                'n_bonus_hits': best_bonus_hits,
                'n_bonus_drawn': len(actual_bonus),
                'main_hit_rate': best_main_hits / len(actual_main) if len(actual_main) > 0 else 0,
                'bonus_hit_rate': best_bonus_hits / len(actual_bonus) if len(actual_bonus) > 0 else 0,
                'n_combinations_tested': len(combinations),
                'n_combinations_requested': self.n_combinations,
                'n_non_predicted_main': len(non_predicted_main),
                'n_non_predicted_bonus': len(non_predicted_bonus) if self.bonus_enabled else 0,
                'diversity_max_common_main': self.max_common_main,
                'diversity_max_common_bonus': self.max_common_bonus,
                'all_combinations': all_combos_details,
                'division_summary': division_summary
            }
            print(f"[ANTI2 DEBUG] Created prediction with {len(combinations)} diverse combinations (requested {self.n_combinations})")
            predictions.append(pred_dict)
        
        # Compute aggregate statistics
        main_hit_rates = [p['main_hit_rate'] for p in predictions]
        bonus_hit_rates = [p['bonus_hit_rate'] for p in predictions if p['n_bonus_drawn'] > 0]
        
        avg_main_hit_rate = sum(main_hit_rates) / len(main_hit_rates) if main_hit_rates else 0
        avg_bonus_hit_rate = sum(bonus_hit_rates) / len(bonus_hit_rates) if bonus_hit_rates else 0
        
        main_pool_size = self.main_max - self.main_min + 1
        bonus_pool_size = self.bonus_max - self.bonus_min + 1 if self.bonus_enabled else 0
        expected_main_hit_rate = (self.main_pick / main_pool_size) if main_pool_size > 0 else 0
        expected_bonus_hit_rate = (self.bonus_pick / bonus_pool_size) if bonus_pool_size > 0 and self.bonus_pick > 0 else 0
        
        total_hits = total_main_hits + total_bonus_hits
        total_possible = total_main_possible + total_bonus_possible
        avg_hit_rate = total_hits / total_possible if total_possible > 0 else 0
        
        return {
            'model_name': 'ANTI2',
            'predictions': predictions,
            'statistics': {
                'total_tests': len(predictions),
                'total_main_hits': total_main_hits,
                'total_main_possible': total_main_possible,
                'total_bonus_hits': total_bonus_hits,
                'total_bonus_possible': total_bonus_possible,
                'total_hits': total_hits,
                'total_possible': total_possible,
                'avg_main_hit_rate': avg_main_hit_rate,
                'avg_bonus_hit_rate': avg_bonus_hit_rate,
                'avg_hit_rate': avg_hit_rate,
                'max_hit_rate': max(main_hit_rates) if main_hit_rates else 0,
                'expected_main_hit_rate': expected_main_hit_rate,
                'expected_bonus_hit_rate': expected_bonus_hit_rate,
                'lift_vs_random': avg_main_hit_rate / expected_main_hit_rate if expected_main_hit_rate > 0 else 0,
                'best_main_hit_rate': max(main_hit_rates) if main_hit_rates else 0,
                'worst_main_hit_rate': min(main_hit_rates) if main_hit_rates else 0,
                'best_bonus_hit_rate': max(bonus_hit_rates) if bonus_hit_rates else 0,
                'worst_bonus_hit_rate': min(bonus_hit_rates) if bonus_hit_rates else 0,
                'n_combinations_per_draw': self.n_combinations,
                'diversity_max_common_main': self.max_common_main,
                'diversity_max_common_bonus': self.max_common_bonus
            }
        }
    
    def _prepare_dataframe(self, draws: List[Draw]) -> pd.DataFrame:
        """Prepare DataFrame from draws."""
        data = []
        for draw in draws:
            data.append({
                'draw_date': draw.draw_date,
                'numbers': draw.numbers,
                'bonus_numbers': draw.bonus_numbers if draw.bonus_numbers else []
            })
        return pd.DataFrame(data)
    
    def _compute_comparison(self, models_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute comparative statistics across models.
        
        Args:
            models_results: Results from all models
        
        Returns:
            Comparative statistics
        """
        comparison = {
            'ranking': [],
            'best_model': None,
            'worst_model': None
        }
        
        # Rank models by average hit rate
        rankings = []
        for model_name, results in models_results.items():
            if 'statistics' in results:
                rankings.append({
                    'model': model_name,
                    'avg_hit_rate': results['statistics']['avg_hit_rate'],
                    'lift_vs_random': results['statistics']['lift_vs_random'],
                    'total_hits': results['statistics']['total_hits']
                })
        
        rankings.sort(key=lambda x: x['avg_hit_rate'], reverse=True)
        comparison['ranking'] = rankings
        
        if rankings:
            comparison['best_model'] = rankings[0]['model']
            comparison['worst_model'] = rankings[-1]['model']
        
        return comparison
    
    def generate_next_prediction(self, models: List[str] = None) -> Dict[str, Any]:
        """
        Generate Anti-Consensus prediction for the next draw using ALL historical data.
        
        Args:
            models: List of model names to use for consensus (default: M1-M15 except M0, M11)
        
        Returns:
            Dictionary with predicted numbers for the next draw
        """
        import random
        
        if models is None:
            models = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M12', 'M13', 'M14', 'M15']
        
        # Use ALL draws as training data
        all_draws = self.draws
        
        if len(all_draws) < 10:
            return {
                'error': 'Insufficient data for prediction',
                'main_numbers': [],
                'bonus_numbers': [],
                'combinations': []
            }
        
        # Get predictions from each model using all historical data
        other_predictions = {}
        for model_name in models:
            try:
                pred = self._get_top_n_predictions(model_name, all_draws)
                other_predictions[model_name] = pred
            except Exception as e:
                print(f"[PREDICT] Error with model {model_name}: {e}")
                continue
        
        # Collect all numbers predicted by other models
        predicted_main_by_others = set()
        predicted_bonus_by_others = set()
        for model_name, pred in other_predictions.items():
            predicted_main_by_others.update(pred.get('main_numbers', []))
            predicted_bonus_by_others.update(pred.get('bonus_numbers', []))
        
        # Get numbers NOT predicted by any model
        all_main_numbers = set(range(self.main_min, self.main_max + 1))
        all_bonus_numbers = set(range(self.bonus_min, self.bonus_max + 1)) if self.bonus_enabled else set()
        
        non_predicted_main = list(all_main_numbers - predicted_main_by_others)
        
        if self.bonus_separate_pool:
            non_predicted_bonus = list(all_bonus_numbers - predicted_bonus_by_others)
        else:
            non_predicted_bonus = non_predicted_main.copy()
        
        # Generate multiple combinations with ANTI2 diversity constraints
        combinations = []
        used_bonus = set()  # Track used bonus numbers for diversity
        max_attempts = 100
        
        for combo_idx in range(self.n_combinations):
            attempts = 0
            while attempts < max_attempts:
                attempts += 1
                
                # Generate main numbers
                if len(non_predicted_main) >= self.main_pick:
                    combo_main = sorted(random.sample(non_predicted_main, self.main_pick))
                else:
                    combo_main = sorted(random.sample(list(all_main_numbers), self.main_pick))
                
                # Check diversity constraint with existing combinations
                is_diverse = True
                for existing in combinations:
                    common_main = len(set(combo_main) & set(existing['main_numbers']))
                    if common_main > self.max_common_main:
                        is_diverse = False
                        break
                
                if not is_diverse:
                    continue
                
                # Generate bonus numbers
                combo_bonus = []
                if self.bonus_enabled and self.bonus_pick > 0:
                    if self.bonus_separate_pool:
                        # For separate bonus pool, try to get unique bonus numbers
                        available_bonus = [b for b in non_predicted_bonus if b not in used_bonus]
                        if len(available_bonus) >= self.bonus_pick:
                            combo_bonus = sorted(random.sample(available_bonus, self.bonus_pick))
                        elif len(non_predicted_bonus) >= self.bonus_pick:
                            combo_bonus = sorted(random.sample(non_predicted_bonus, self.bonus_pick))
                        else:
                            all_available = [b for b in all_bonus_numbers if b not in used_bonus]
                            if len(all_available) >= self.bonus_pick:
                                combo_bonus = sorted(random.sample(all_available, self.bonus_pick))
                            else:
                                combo_bonus = sorted(random.sample(list(all_bonus_numbers), self.bonus_pick))
                    else:
                        available_for_bonus = [n for n in non_predicted_bonus if n not in combo_main and n not in used_bonus]
                        if len(available_for_bonus) >= self.bonus_pick:
                            combo_bonus = sorted(random.sample(available_for_bonus, self.bonus_pick))
                        else:
                            remaining = [n for n in all_main_numbers if n not in combo_main]
                            combo_bonus = sorted(random.sample(remaining, min(self.bonus_pick, len(remaining))))
                    
                    # Check bonus diversity if max_common_bonus is 0
                    if self.max_common_bonus == 0 and combo_bonus:
                        if any(b in used_bonus for b in combo_bonus):
                            continue  # Try again for unique bonus
                
                # Valid combination found
                combinations.append({
                    'main_numbers': combo_main,
                    'bonus_numbers': combo_bonus
                })
                used_bonus.update(combo_bonus)
                break
            
            # If max attempts reached, add anyway
            if attempts >= max_attempts and len(combinations) < combo_idx + 1:
                if len(non_predicted_main) >= self.main_pick:
                    combo_main = sorted(random.sample(non_predicted_main, self.main_pick))
                else:
                    combo_main = sorted(random.sample(list(all_main_numbers), self.main_pick))
                combo_bonus = []
                if self.bonus_enabled and self.bonus_pick > 0:
                    if self.bonus_separate_pool:
                        combo_bonus = sorted(random.sample(list(all_bonus_numbers), self.bonus_pick))
                    else:
                        remaining = [n for n in all_main_numbers if n not in combo_main]
                        combo_bonus = sorted(random.sample(remaining, min(self.bonus_pick, len(remaining))))
                combinations.append({
                    'main_numbers': combo_main,
                    'bonus_numbers': combo_bonus
                })
        
        # Return the first combination as the main prediction, plus all combinations
        result = {
            'main_numbers': combinations[0]['main_numbers'] if combinations else [],
            'bonus_numbers': combinations[0]['bonus_numbers'] if combinations else [],
            'combinations': combinations,
            'excluded_main': sorted(list(predicted_main_by_others)),
            'excluded_bonus': sorted(list(predicted_bonus_by_others)),
            'models_used': list(other_predictions.keys()),
            'total_draws_analyzed': len(all_draws)
        }
        print(f"[PREDICT] Generated {len(combinations)} combinations")
        if combinations:
            print(f"[PREDICT] First combo: main={combinations[0]['main_numbers']}, bonus={combinations[0]['bonus_numbers']}")
        return result
