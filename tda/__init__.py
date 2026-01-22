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

__version__ = '0.1.0'
__all__ = [
    'compute_rolling_correlation',
    'compute_graph_laplacian',
    'simulate_heat_kernel',
    'simulate_random_walk',
    'compute_residuals',
    'build_vietoris_rips_filtration',
    'compute_persistent_homology',
    'extract_h0_features',
    'extract_h1_features',
    'compute_persistence_landscape',
    'compute_persistence_images',
    'vectorize_persistence_diagrams',
    'identify_market_regimes'
]
