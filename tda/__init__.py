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

__version__ = '0.1.0'
__all__ = [
    'compute_rolling_correlation',
    'compute_graph_laplacian',
    'simulate_heat_kernel',
    'simulate_random_walk',
    'compute_residuals'
]
