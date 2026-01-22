"""
Topological Data Analysis Package
A package for analyzing financial data using graph theory and topological methods.
"""

from .weighted_graph import (
    compute_rolling_correlation,
    compute_graph_laplacian,
    simulate_heat_kernel,
    simulate_random_walk
)
from .residual_analysis import compute_residuals
from .tda_layer import (
    build_vietoris_rips_filtration,
    compute_persistent_homology,
    extract_h0_features,
    extract_h1_features,
    compute_persistence_landscape,
    compute_persistence_images,
    vectorize_persistence_diagrams,
    identify_market_regimes
)
from .feature_fusion import (
    FeatureFusion,
    create_local_features,
    create_global_features_vector
)
from .regime_detection import (
    MarketRegimeDetector,
    HMMRegimeDetector,
    ClusteringRegimeDetector,
    classify_regime_by_topology
)
from .risk_management import (
    RiskManager,
    KellyCalculator
)
from .market_mutual_model import MarketMutualModel
from .ensemble import (
    MetaLearner,
    SimpleEnsemble,
    combine_tda_with_neural_net
)
from .backtesting import (
    Backtester,
    RollingBacktester,
    compare_strategies
)

__version__ = '0.2.0'
__all__ = [
    # Graph and diffusion
    'compute_rolling_correlation',
    'compute_graph_laplacian',
    'simulate_heat_kernel',
    'simulate_random_walk',
    'compute_residuals',
    # TDA layer
    'build_vietoris_rips_filtration',
    'compute_persistent_homology',
    'extract_h0_features',
    'extract_h1_features',
    'compute_persistence_landscape',
    'compute_persistence_images',
    'vectorize_persistence_diagrams',
    'identify_market_regimes',
    # Feature fusion
    'FeatureFusion',
    'create_local_features',
    'create_global_features_vector',
    # Regime detection
    'MarketRegimeDetector',
    'HMMRegimeDetector',
    'ClusteringRegimeDetector',
    'classify_regime_by_topology',
    # Risk management
    'RiskManager',
    'KellyCalculator',
    # Market Mutual Model
    'MarketMutualModel',
    # Ensemble
    'MetaLearner',
    'SimpleEnsemble',
    'combine_tda_with_neural_net',
    # Backtesting
    'Backtester',
    'RollingBacktester',
    'compare_strategies'
]
