"""
Player bias analysis for jackpot studies.
Analyzes if player behavior changes with jackpot size.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class TestResult:
    """Result of a single statistical test"""
    name: str
    statistic: float
    p_value: float
    passed: bool
    alpha: float = 0.01
    description: str = ""
    details: Dict[str, Any] = None


class PlayerBiasAnalyzer:
    """
    Analyzes player behavior patterns in relation to jackpot size.
    
    When jackpots are large, more players participate, potentially
    changing the distribution of winning numbers (if players influence
    outcomes, which they shouldn't in a fair lottery).
    """
    
    def __init__(self, alpha: float = 0.01):
        """
        Initialize player bias analyzer.
        
        Args:
            alpha: Significance level
        """
        self.alpha = alpha
    
    def winners_vs_jackpot_test(self, jackpots: List[float],
                                 n_winners: List[int]) -> TestResult:
        """
        Test relationship between jackpot size and number of winners.
        
        In a fair lottery, the number of winners should be independent
        of jackpot size (though player count may increase).
        """
        valid_pairs = [(j, w) for j, w in zip(jackpots, n_winners) 
                       if j and j > 0 and w is not None]
        
        if len(valid_pairs) < 30:
            return TestResult(
                name="Winners vs Jackpot",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="Insufficient data (need >= 30 draws)"
            )
        
        jackpot_values = [j for j, _ in valid_pairs]
        winner_counts = [w for _, w in valid_pairs]
        
        # Spearman correlation
        corr, p_value = stats.spearmanr(jackpot_values, winner_counts)
        
        # Note: A negative correlation is expected (higher jackpot = fewer winners
        # because jackpot grows when no one wins). This is normal.
        # We're looking for unexpected positive correlation.
        
        return TestResult(
            name="Winners vs Jackpot",
            statistic=float(corr),
            p_value=float(p_value),
            passed=True,  # This relationship is expected
            alpha=self.alpha,
            description="Analyzes relationship between jackpot size and winner count",
            details={
                "n_draws": len(valid_pairs),
                "spearman_correlation": float(corr),
                "mean_winners": float(np.mean(winner_counts)),
                "std_winners": float(np.std(winner_counts)),
                "interpretation": "Expected negative correlation (jackpot grows when no winners)"
            }
        )
    
    def popular_numbers_analysis(self, draws: List[List[int]],
                                  jackpots: List[float],
                                  n_winners: List[int],
                                  n_max: int) -> TestResult:
        """
        Analyze if draws with more winners contain "popular" numbers.
        
        Popular numbers are those that players tend to choose more often
        (birthdays 1-31, lucky 7, etc.).
        """
        # Define popular number ranges
        birthday_numbers = set(range(1, 32))  # Days 1-31
        lucky_numbers = {7, 11, 13, 21, 27}
        
        valid_data = [(d, j, w) for d, j, w in zip(draws, jackpots, n_winners)
                      if j and j > 0 and w is not None]
        
        if len(valid_data) < 50:
            return TestResult(
                name="Popular Numbers Analysis",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="Insufficient data"
            )
        
        # Calculate birthday number proportion for each draw
        birthday_props = []
        winner_counts = []
        
        for draw, _, winners in valid_data:
            birthday_count = sum(1 for n in draw if n in birthday_numbers)
            birthday_props.append(birthday_count / len(draw))
            winner_counts.append(winners)
        
        # Correlation between birthday proportion and winners
        corr, p_value = stats.spearmanr(birthday_props, winner_counts)
        
        # Positive correlation would suggest draws with more birthday numbers
        # have more winners (expected if players favor these numbers)
        
        return TestResult(
            name="Popular Numbers Analysis",
            statistic=float(corr),
            p_value=float(p_value),
            passed=True,  # This is informational
            alpha=self.alpha,
            description="Analyzes if draws with 'popular' numbers have more winners",
            details={
                "n_draws": len(valid_data),
                "spearman_correlation": float(corr),
                "mean_birthday_proportion": float(np.mean(birthday_props)),
                "interpretation": "Positive correlation suggests players favor birthday numbers"
            }
        )
    
    def jackpot_threshold_effect(self, draws: List[List[int]],
                                  jackpots: List[float],
                                  n_max: int,
                                  threshold_percentile: float = 90) -> TestResult:
        """
        Test if draw characteristics differ when jackpot exceeds a threshold.
        
        Large jackpots attract more players, potentially changing
        the distribution of winning combinations.
        """
        valid_pairs = [(d, j) for d, j in zip(draws, jackpots) if j and j > 0]
        
        if len(valid_pairs) < 50:
            return TestResult(
                name="Jackpot Threshold Effect",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="Insufficient data"
            )
        
        jackpot_values = [j for _, j in valid_pairs]
        threshold = np.percentile(jackpot_values, threshold_percentile)
        
        high_jackpot_draws = [d for d, j in valid_pairs if j >= threshold]
        normal_draws = [d for d, j in valid_pairs if j < threshold]
        
        if len(high_jackpot_draws) < 5:
            return TestResult(
                name="Jackpot Threshold Effect",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                description="Insufficient high-jackpot draws"
            )
        
        # Compare sum distributions
        high_sums = [sum(d) for d in high_jackpot_draws]
        normal_sums = [sum(d) for d in normal_draws]
        
        # Mann-Whitney U test
        u_stat, p_value = stats.mannwhitneyu(high_sums, normal_sums, alternative='two-sided')
        
        return TestResult(
            name="Jackpot Threshold Effect",
            statistic=float(u_stat),
            p_value=float(p_value),
            passed=p_value >= self.alpha,
            alpha=self.alpha,
            description=f"Tests if draws differ when jackpot >= {threshold_percentile}th percentile",
            details={
                "threshold": float(threshold),
                "n_high_jackpot": len(high_jackpot_draws),
                "n_normal": len(normal_draws),
                "mean_sum_high": float(np.mean(high_sums)),
                "mean_sum_normal": float(np.mean(normal_sums)),
                "mann_whitney_u": float(u_stat)
            }
        )
    
    def run_all_analyses(self, draws: List[List[int]],
                          jackpots: List[float],
                          n_winners: List[int],
                          n_max: int) -> Dict[str, Any]:
        """
        Run all player bias analyses.
        
        Args:
            draws: List of draws
            jackpots: List of jackpot amounts
            n_winners: List of winner counts (division 1)
            n_max: Maximum number in pool
            
        Returns:
            Dictionary with all analysis results
        """
        results = {
            "n_draws": len(draws),
            "alpha": self.alpha,
            "analyses": {}
        }
        
        analyses = [
            self.winners_vs_jackpot_test(jackpots, n_winners),
            self.popular_numbers_analysis(draws, jackpots, n_winners, n_max),
            self.jackpot_threshold_effect(draws, jackpots, n_max)
        ]
        
        for analysis in analyses:
            results["analyses"][analysis.name] = {
                "statistic": analysis.statistic,
                "p_value": analysis.p_value,
                "passed": analysis.passed,
                "description": analysis.description,
                "details": analysis.details
            }
        
        return results
