"""
Fraud detection module.
Implements dispersion tests, Benford's law, clustering tests, and fraud scoring.
"""

from .dispersion_tests import DispersionTests
from .benford_tests import BenfordTests
from .clustering_tests import ClusteringTests
from .jackpot_fraud_tests import JackpotFraudTests
from .fraud_score import FraudScoreCalculator
from .alert_manager import AlertManager

__all__ = [
    'DispersionTests',
    'BenfordTests',
    'ClusteringTests',
    'JackpotFraudTests',
    'FraudScoreCalculator',
    'AlertManager'
]
