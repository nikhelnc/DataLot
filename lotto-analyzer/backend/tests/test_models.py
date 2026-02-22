import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.analysis.prob_models.m0_baseline import M0Baseline
from app.analysis.prob_models.m1_dirichlet import M1Dirichlet
from app.analysis.prob_models.m2_windowed import M2Windowed
from app.analysis.evaluation.walk_forward import WalkForwardEvaluator


def test_m0_baseline():
    rules = {
        "numbers": {"count": 5, "min": 1, "max": 49, "unique": True, "sorted": True},
    }
    
    model = M0Baseline(rules)
    probs = model.predict()
    
    assert len(probs) == 49
    assert all(abs(p - 1/49) < 1e-6 for p in probs.values())
    assert abs(sum(probs.values()) - 1.0) < 1e-6


def test_m1_dirichlet():
    rules = {
        "numbers": {"count": 5, "min": 1, "max": 49, "unique": True, "sorted": True},
    }
    
    data = []
    np.random.seed(42)
    
    for i in range(100):
        numbers = sorted(np.random.choice(range(1, 50), 5, replace=False).tolist())
        data.append({
            "draw_date": datetime(2024, 1, 1) + timedelta(days=i),
            "numbers": numbers,
            "bonus": None,
        })
    
    df = pd.DataFrame(data)
    
    model = M1Dirichlet(rules, alpha=1.0)
    model.fit(df)
    probs = model.predict()
    
    assert len(probs) == 49
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert all(p > 0 for p in probs.values())


def test_m2_windowed():
    rules = {
        "numbers": {"count": 5, "min": 1, "max": 49, "unique": True, "sorted": True},
    }
    
    data = []
    np.random.seed(42)
    
    for i in range(100):
        numbers = sorted(np.random.choice(range(1, 50), 5, replace=False).tolist())
        data.append({
            "draw_date": datetime(2024, 1, 1) + timedelta(days=i),
            "numbers": numbers,
            "bonus": None,
        })
    
    df = pd.DataFrame(data)
    
    model = M2Windowed(rules, window_size=50, lambda_shrink=0.3)
    model.fit(df)
    probs = model.predict()
    
    assert len(probs) == 49
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert all(p > 0 for p in probs.values())


def test_walk_forward_scoring():
    rules = {
        "numbers": {"count": 5, "min": 1, "max": 49, "unique": True, "sorted": True},
    }
    
    data = []
    np.random.seed(42)
    
    for i in range(60):
        numbers = sorted(np.random.choice(range(1, 50), 5, replace=False).tolist())
        data.append({
            "draw_date": datetime(2024, 1, 1) + timedelta(days=i),
            "numbers": numbers,
            "bonus": None,
        })
    
    df = pd.DataFrame(data)
    
    model = M0Baseline(rules)
    evaluator = WalkForwardEvaluator(df, rules, model)
    
    results = evaluator.evaluate()
    
    assert "brier_score" in results
    assert "baseline_brier" in results
    assert "ece" in results
    assert "n_predictions" in results
    
    assert results["brier_score"] >= 0
    assert results["brier_score"] <= 1
    assert results["ece"] >= 0
    assert results["ece"] <= 1
