"""
Must-Be-Won analysis for lottery jackpots.
Analyzes draws where the jackpot must be won (cap reached).
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class MBWResult:
    """Result of Must-Be-Won analysis"""
    name: str
    statistic: float
    p_value: float
    significant: bool
    alpha: float = 0.05
    description: str = ""
    details: Dict[str, Any] = None


class MustBeWonAnalyzer:
    """
    Analyzes "Must Be Won" (MBW) draws.
    
    In some lotteries, when the jackpot reaches a cap, it "must be won"
    in the next draw. This analyzer tests if MBW draws differ from
    regular draws.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize MBW analyzer.
        
        Args:
            alpha: Significance level
        """
        self.alpha = alpha
    
    def frequency_comparison_test(self, draws: List[List[int]],
                                   must_be_won: List[bool],
                                   n_max: int) -> MBWResult:
        """
        Compare number frequencies between MBW and regular draws.
        
        Uses chi-squared test for homogeneity.
        """
        mbw_draws = [d for d, m in zip(draws, must_be_won) if m]
        regular_draws = [d for d, m in zip(draws, must_be_won) if not m]
        
        if len(mbw_draws) < 5:
            return MBWResult(
                name="MBW Frequency Comparison",
                statistic=0.0,
                p_value=1.0,
                significant=False,
                alpha=self.alpha,
                description="Insufficient MBW draws (need >= 5)"
            )
        
        if len(regular_draws) < 20:
            return MBWResult(
                name="MBW Frequency Comparison",
                statistic=0.0,
                p_value=1.0,
                significant=False,
                alpha=self.alpha,
                description="Insufficient regular draws"
            )
        
        # Count frequencies
        mbw_freq = np.zeros(n_max)
        regular_freq = np.zeros(n_max)
        
        for draw in mbw_draws:
            for num in draw:
                if 1 <= num <= n_max:
                    mbw_freq[num - 1] += 1
        
        for draw in regular_draws:
            for num in draw:
                if 1 <= num <= n_max:
                    regular_freq[num - 1] += 1
        
        # Chi-squared test
        contingency = np.array([mbw_freq, regular_freq])
        
        # Remove zero columns
        nonzero = contingency.sum(axis=0) > 0
        contingency = contingency[:, nonzero]
        
        if contingency.shape[1] < 2:
            return MBWResult(
                name="MBW Frequency Comparison",
                statistic=0.0,
                p_value=1.0,
                significant=False,
                alpha=self.alpha,
                description="Insufficient variation"
            )
        
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        
        return MBWResult(
            name="MBW Frequency Comparison",
            statistic=float(chi2),
            p_value=float(p_value),
            significant=p_value < self.alpha,
            alpha=self.alpha,
            description="Tests if number frequencies differ in MBW draws",
            details={
                "n_mbw": len(mbw_draws),
                "n_regular": len(regular_draws),
                "chi2_statistic": float(chi2),
                "degrees_of_freedom": int(dof)
            }
        )
    
    def sum_comparison_test(self, draws: List[List[int]],
                             must_be_won: List[bool]) -> MBWResult:
        """
        Compare sum distributions between MBW and regular draws.
        
        Uses Mann-Whitney U test.
        """
        mbw_sums = [sum(d) for d, m in zip(draws, must_be_won) if m]
        regular_sums = [sum(d) for d, m in zip(draws, must_be_won) if not m]
        
        if len(mbw_sums) < 5 or len(regular_sums) < 20:
            return MBWResult(
                name="MBW Sum Comparison",
                statistic=0.0,
                p_value=1.0,
                significant=False,
                alpha=self.alpha,
                description="Insufficient data"
            )
        
        u_stat, p_value = stats.mannwhitneyu(mbw_sums, regular_sums, alternative='two-sided')
        
        return MBWResult(
            name="MBW Sum Comparison",
            statistic=float(u_stat),
            p_value=float(p_value),
            significant=p_value < self.alpha,
            alpha=self.alpha,
            description="Tests if draw sums differ in MBW draws",
            details={
                "n_mbw": len(mbw_sums),
                "n_regular": len(regular_sums),
                "mean_mbw": float(np.mean(mbw_sums)),
                "mean_regular": float(np.mean(regular_sums)),
                "std_mbw": float(np.std(mbw_sums)),
                "std_regular": float(np.std(regular_sums)),
                "mann_whitney_u": float(u_stat)
            }
        )
    
    def variance_comparison_test(self, draws: List[List[int]],
                                  must_be_won: List[bool]) -> MBWResult:
        """
        Compare variance of draws between MBW and regular draws.
        
        Uses Levene's test for equality of variances.
        """
        mbw_vars = [np.var(d) for d, m in zip(draws, must_be_won) if m]
        regular_vars = [np.var(d) for d, m in zip(draws, must_be_won) if not m]
        
        if len(mbw_vars) < 5 or len(regular_vars) < 20:
            return MBWResult(
                name="MBW Variance Comparison",
                statistic=0.0,
                p_value=1.0,
                significant=False,
                alpha=self.alpha,
                description="Insufficient data"
            )
        
        stat, p_value = stats.levene(mbw_vars, regular_vars)
        
        return MBWResult(
            name="MBW Variance Comparison",
            statistic=float(stat),
            p_value=float(p_value),
            significant=p_value < self.alpha,
            alpha=self.alpha,
            description="Tests if draw variances differ in MBW draws",
            details={
                "n_mbw": len(mbw_vars),
                "n_regular": len(regular_vars),
                "mean_var_mbw": float(np.mean(mbw_vars)),
                "mean_var_regular": float(np.mean(regular_vars)),
                "levene_statistic": float(stat)
            }
        )
    
    def winner_count_analysis(self, must_be_won: List[bool],
                               n_winners: List[int]) -> MBWResult:
        """
        Analyze winner counts in MBW vs regular draws.
        
        MBW draws should have at least one winner by definition.
        """
        mbw_winners = [w for m, w in zip(must_be_won, n_winners) if m and w is not None]
        regular_winners = [w for m, w in zip(must_be_won, n_winners) if not m and w is not None]
        
        if len(mbw_winners) < 3 or len(regular_winners) < 10:
            return MBWResult(
                name="MBW Winner Count Analysis",
                statistic=0.0,
                p_value=1.0,
                significant=False,
                alpha=self.alpha,
                description="Insufficient winner data"
            )
        
        # MBW draws should always have winners
        mbw_no_winner = sum(1 for w in mbw_winners if w == 0)
        
        # Compare distributions
        u_stat, p_value = stats.mannwhitneyu(mbw_winners, regular_winners, alternative='two-sided')
        
        return MBWResult(
            name="MBW Winner Count Analysis",
            statistic=float(u_stat),
            p_value=float(p_value),
            significant=p_value < self.alpha,
            alpha=self.alpha,
            description="Analyzes winner counts in MBW vs regular draws",
            details={
                "n_mbw": len(mbw_winners),
                "n_regular": len(regular_winners),
                "mean_winners_mbw": float(np.mean(mbw_winners)),
                "mean_winners_regular": float(np.mean(regular_winners)),
                "mbw_with_no_winner": int(mbw_no_winner),
                "note": "MBW draws should always have winners"
            }
        )
    
    def consecutive_mbw_analysis(self, draws: List[List[int]],
                                  must_be_won: List[bool],
                                  jackpots: List[float]) -> MBWResult:
        """
        Analyze patterns in consecutive MBW events.
        
        Tests if MBW events cluster in time.
        """
        mbw_indices = [i for i, m in enumerate(must_be_won) if m]
        
        if len(mbw_indices) < 5:
            return MBWResult(
                name="Consecutive MBW Analysis",
                statistic=0.0,
                p_value=1.0,
                significant=False,
                alpha=self.alpha,
                description="Insufficient MBW events"
            )
        
        # Calculate gaps between MBW events
        gaps = [mbw_indices[i+1] - mbw_indices[i] for i in range(len(mbw_indices)-1)]
        
        if len(gaps) < 3:
            return MBWResult(
                name="Consecutive MBW Analysis",
                statistic=0.0,
                p_value=1.0,
                significant=False,
                alpha=self.alpha,
                description="Insufficient gaps to analyze"
            )
        
        # Test if gaps follow expected distribution (geometric-like)
        mean_gap = np.mean(gaps)
        var_gap = np.var(gaps, ddof=1)
        
        # For geometric distribution, variance ≈ mean²
        # Coefficient of variation
        cv = np.std(gaps) / mean_gap if mean_gap > 0 else 0
        
        return MBWResult(
            name="Consecutive MBW Analysis",
            statistic=float(cv),
            p_value=1.0,  # Informational
            significant=False,
            alpha=self.alpha,
            description="Analyzes gaps between MBW events",
            details={
                "n_mbw_events": len(mbw_indices),
                "mean_gap": float(mean_gap),
                "std_gap": float(np.std(gaps)),
                "min_gap": int(min(gaps)),
                "max_gap": int(max(gaps)),
                "coefficient_of_variation": float(cv)
            }
        )
    
    def run_full_analysis(self, draws: List[List[int]],
                          must_be_won: List[bool],
                          n_max: int,
                          jackpots: List[float] = None,
                          n_winners: List[int] = None) -> Dict[str, Any]:
        """
        Run full MBW analysis.
        
        Args:
            draws: List of draws
            must_be_won: List of MBW flags
            n_max: Maximum number in pool
            jackpots: List of jackpot amounts (optional)
            n_winners: List of winner counts (optional)
            
        Returns:
            Complete MBW analysis results
        """
        n_mbw = sum(1 for m in must_be_won if m)
        
        results = {
            "n_draws": len(draws),
            "n_mbw_draws": n_mbw,
            "mbw_rate": n_mbw / len(draws) if draws else 0,
            "alpha": self.alpha,
            "tests": {}
        }
        
        if n_mbw < 3:
            results["error"] = "Insufficient MBW draws for analysis"
            return results
        
        # Run tests
        tests = [
            self.frequency_comparison_test(draws, must_be_won, n_max),
            self.sum_comparison_test(draws, must_be_won),
            self.variance_comparison_test(draws, must_be_won)
        ]
        
        if n_winners:
            tests.append(self.winner_count_analysis(must_be_won, n_winners))
        
        if jackpots:
            tests.append(self.consecutive_mbw_analysis(draws, must_be_won, jackpots))
        
        n_significant = 0
        for test in tests:
            results["tests"][test.name] = {
                "statistic": test.statistic,
                "p_value": test.p_value,
                "significant": test.significant,
                "description": test.description,
                "details": test.details
            }
            if test.significant:
                n_significant += 1
        
        results["summary"] = {
            "n_tests": len(tests),
            "n_significant": n_significant,
            "any_significant": n_significant > 0,
            "conclusion": "DIFFERENT" if n_significant > 0 else "NO_DIFFERENCE"
        }
        
        return results
