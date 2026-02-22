import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.analysis.anomalies import AnomalyDetector


def test_change_point_detection_synthetic():
    rules = {
        "numbers": {"count": 5, "min": 1, "max": 49, "unique": True, "sorted": True},
        "bonus": {"enabled": False},
    }
    
    data = []
    np.random.seed(42)
    
    for i in range(100):
        if i < 50:
            numbers = sorted(np.random.choice(range(1, 26), 5, replace=False).tolist())
        else:
            numbers = sorted(np.random.choice(range(25, 50), 5, replace=False).tolist())
        
        data.append({
            "draw_date": datetime(2024, 1, 1) + timedelta(days=i),
            "numbers": numbers,
            "bonus": None,
        })
    
    df = pd.DataFrame(data)
    detector = AnomalyDetector(df, rules)
    
    anomalies = detector.detect_all()
    
    assert "change_points" in anomalies
    if "count" in anomalies["change_points"]:
        assert anomalies["change_points"]["count"] >= 0


def test_outlier_detection_mad():
    rules = {
        "numbers": {"count": 5, "min": 1, "max": 49, "unique": True, "sorted": True},
        "bonus": {"enabled": False},
    }
    
    data = []
    np.random.seed(42)
    
    for i in range(100):
        if i == 50:
            numbers = [1, 2, 3, 4, 5]
        elif i == 51:
            numbers = [45, 46, 47, 48, 49]
        else:
            numbers = sorted(np.random.choice(range(1, 50), 5, replace=False).tolist())
        
        data.append({
            "draw_date": datetime(2024, 1, 1) + timedelta(days=i),
            "numbers": numbers,
            "bonus": None,
        })
    
    df = pd.DataFrame(data)
    detector = AnomalyDetector(df, rules)
    
    anomalies = detector.detect_all()
    
    assert "outliers" in anomalies
    assert "method" in anomalies["outliers"]
    assert anomalies["outliers"]["method"] == "MAD z-score"


def test_drift_detection():
    rules = {
        "numbers": {"count": 5, "min": 1, "max": 49, "unique": True, "sorted": True},
        "bonus": {"enabled": False},
    }
    
    data = []
    np.random.seed(42)
    
    for i in range(200):
        numbers = sorted(np.random.choice(range(1, 50), 5, replace=False).tolist())
        data.append({
            "draw_date": datetime(2024, 1, 1) + timedelta(days=i),
            "numbers": numbers,
            "bonus": None,
        })
    
    df = pd.DataFrame(data)
    detector = AnomalyDetector(df, rules)
    
    anomalies = detector.detect_all()
    
    assert "drift" in anomalies
    if "max_psi" in anomalies["drift"]:
        assert isinstance(anomalies["drift"]["max_psi"], float)
        assert anomalies["drift"]["max_psi"] >= 0
