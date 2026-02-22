"""
Probability models for lottery prediction.

Models M0-M20 implementing various statistical and machine learning approaches.
"""

from .m0_baseline import M0Baseline
from .m1_dirichlet import M1Dirichlet
from .m2_windowed import M2Windowed
from .m3_exponential_decay import M3ExponentialDecay
from .m4_hmm import M4HMM
from .m7_entropy import M7Entropy
from .m8_changepoint import M8Changepoint
from .m9_bayesian_network import M9BayesianNetwork
from .m11_lstm_hybrid import M11LSTMHybrid
from .m12_mixture_dirichlet import M12MixtureDirichlet
from .m13_spectral import M13Spectral
from .m14_copula import M14Copula
from .m15_thompson_sampling import M15ThompsonSampling
from .m16_gradient_boosting import M16GradientBoosting
from .m17_autoencoder import M17Autoencoder
from .m18_graph_neural import M18GraphNeural
from .m19_temporal_fusion import M19TemporalFusion
from .m20_meta_learner import M20MetaLearner

__all__ = [
    "M0Baseline",
    "M1Dirichlet",
    "M2Windowed",
    "M3ExponentialDecay",
    "M4HMM",
    "M7Entropy",
    "M8Changepoint",
    "M9BayesianNetwork",
    "M11LSTMHybrid",
    "M12MixtureDirichlet",
    "M13Spectral",
    "M14Copula",
    "M15ThompsonSampling",
    "M16GradientBoosting",
    "M17Autoencoder",
    "M18GraphNeural",
    "M19TemporalFusion",
    "M20MetaLearner",
]

# Model registry for easy access
MODELS = {
    "M0": M0Baseline,
    "M1": M1Dirichlet,
    "M2": M2Windowed,
    "M3": M3ExponentialDecay,
    "M4": M4HMM,
    "M7": M7Entropy,
    "M8": M8Changepoint,
    "M9": M9BayesianNetwork,
    "M11": M11LSTMHybrid,
    "M12": M12MixtureDirichlet,
    "M13": M13Spectral,
    "M14": M14Copula,
    "M15": M15ThompsonSampling,
    "M16": M16GradientBoosting,
    "M17": M17Autoencoder,
    "M18": M18GraphNeural,
    "M19": M19TemporalFusion,
    "M20": M20MetaLearner,
}
