import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.analysis.randomness import RandomnessTests


def test_chi2_uniformity():
    rules = {
        "numbers": {"count": 5, "min": 1, "max": 49, "unique": True, "sorted": True},
        "bonus": {"enabled": True, "min": 1, "max": 10},
    }
    
    data = []
    np.random.seed(42)
    
    for i in range(200):
        numbers = sorted(np.random.choice(range(1, 50), 5, replace=False).tolist())
        data.append({
            "draw_date": datetime(2024, 1, 1) + timedelta(days=i),
            "numbers": numbers,
            "bonus": np.random.randint(1, 11),
        })
    
    df = pd.DataFrame(data)
    tester = RandomnessTests(df, rules)
    
    results = tester.run_all_tests()
    
    assert "uniformity" in results
    assert "chi2_numbers" in results["uniformity"]
    assert "statistic" in results["uniformity"]["chi2_numbers"]
    assert "p_value" in results["uniformity"]["chi2_numbers"]
    assert results["uniformity"]["chi2_numbers"]["p_value"] > 0.01


def test_fdr_correction():
    rules = {
        "numbers": {"count": 5, "min": 1, "max": 49, "unique": True, "sorted": True},
        "bonus": {"enabled": True, "min": 1, "max": 10},
    }
    
    data = []
    np.random.seed(42)
    
    for i in range(100):
        numbers = sorted(np.random.choice(range(1, 50), 5, replace=False).tolist())
        data.append({
            "draw_date": datetime(2024, 1, 1) + timedelta(days=i),
            "numbers": numbers,
            "bonus": np.random.randint(1, 11),
        })
    
    df = pd.DataFrame(data)
    tester = RandomnessTests(df, rules)
    
    results = tester.run_all_tests()
    
    assert "fdr_correction" in results
    assert "method" in results["fdr_correction"]
    assert results["fdr_correction"]["method"] == "Benjamini-Hochberg"
    assert "corrected_pvalues" in results["fdr_correction"]
    assert "rejected" in results["fdr_correction"]


def test_runs_test():
    rules = {
        "numbers": {"count": 5, "min": 1, "max": 49, "unique": True, "sorted": True},
        "bonus": {"enabled": False},
    }
    
    data = []
    for i in range(100):
        if i % 2 == 0:
            numbers = [2, 4, 6, 8, 10]
        else:
            numbers = [1, 3, 5, 7, 9]
        
        data.append({
            "draw_date": datetime(2024, 1, 1) + timedelta(days=i),
            "numbers": numbers,
            "bonus": None,
        })
    
    df = pd.DataFrame(data)
    tester = RandomnessTests(df, rules)
    
    results = tester.run_all_tests()
    
    assert "independence" in results
    assert "runs_even" in results["independence"]
    assert "runs" in results["independence"]["runs_even"]
