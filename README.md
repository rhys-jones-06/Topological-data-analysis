# Topological Data Analysis for Financial Data

A Python package for analyzing financial data using graph theory and topological methods. This package provides tools to model financial asset relationships as weighted graphs, extract global market shape through persistent homology, and analyze information diffusion and price shock propagation.

## Features

### Weighted Graph Generation
- **Rolling Correlation Matrices**: Calculate rolling correlation matrices between N assets to serve as edge weights in a graph representation
- **Graph Laplacian**: Compute the Graph Laplacian L = D - A, where D is the degree matrix and A is the adjacency matrix
- **Heat Kernel Diffusion**: Simulate heat kernel diffusion across the graph to smooth out noise and identify how price shocks propagate
- **Random Walk Simulation**: Model information or price shock propagation through random walks on the asset correlation graph

### TDA Layer - Algebraic Topology
- **Vietoris-Rips Filtration**: Build sequences of simplicial complexes from correlation distances between assets
- **Persistent Homology**: Compute persistence diagrams to identify topological features that persist across different scales
- **H₀ Features (Clusters)**: Detect and quantify market fragmentation into distinct clusters or sectors
- **H₁ Features (Loops/Cycles)**: Identify feedback loops and circular dependencies representing market regimes
- **Feature Vectorization**: Convert topological shapes into numerical features using:
  - Persistence Landscapes - Functional summaries for statistical analysis
  - Persistence Images - 2D representations suitable for CNNs
  - Statistical Features - Aggregate metrics for traditional ML models
- **Market Regime Identification**: Automatically classify market states based on topological features

### Residual Analysis
- **Mispricing Detection**: Subtract expected returns from actual returns to isolate local mispricings and anomalies

## Installation

1. Clone the repository:
```bash
git clone https://github.com/rhys-jones-06/Topological-data-analysis.git
cd Topological-data-analysis
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Graph-Based Analysis Example

```python
import numpy as np
import pandas as pd
from tda import (
    compute_rolling_correlation,
    compute_graph_laplacian,
    simulate_heat_kernel,
    simulate_random_walk,
    compute_residuals
)

# 1. Load or generate asset return data
asset_returns = pd.DataFrame({
    'Asset1': [...],
    'Asset2': [...],
    'Asset3': [...]
})

# 2. Compute rolling correlation matrices
correlation_matrices = compute_rolling_correlation(asset_returns, window=20)

# 3. Get the latest correlation matrix as adjacency matrix
adjacency_matrix = correlation_matrices[-1].drop(columns=['timestamp']).values
adjacency_matrix = np.abs(adjacency_matrix)
np.fill_diagonal(adjacency_matrix, 0)

# 4. Compute Graph Laplacian
laplacian = compute_graph_laplacian(adjacency_matrix)

# 5. Simulate heat kernel to see how a price shock propagates
initial_shock = np.array([1, 0, 0])  # Shock at first asset
diffused = simulate_heat_kernel(laplacian, time=1.0, initial_state=initial_shock)

# 6. Simulate random walk
final_state = simulate_random_walk(adjacency_matrix, n_steps=100)

# 7. Compute residuals for mispricing analysis
expected_returns = asset_returns.rolling(window=10).mean()
residuals = compute_residuals(asset_returns, expected_returns)
```

### TDA Layer Example - Extracting Global Market Shape

```python
import numpy as np
import pandas as pd
from tda import (
    compute_persistent_homology,
    extract_h0_features,
    extract_h1_features,
    vectorize_persistence_diagrams,
    identify_market_regimes
)

# 1. Compute correlation matrix from asset returns
correlation_matrix = asset_returns.corr().values

# 2. Compute persistent homology (Vietoris-Rips filtration)
persistence_result = compute_persistent_homology(
    correlation_matrix,
    max_dimension=2,
    use_correlation=False
)

# 3. Extract H_0 features (market clusters)
h0_features = extract_h0_features(persistence_result['dgms'][0])
print(f"Number of market clusters: {h0_features['num_components']}")
print(f"Cluster persistence: {h0_features['max_persistence']:.4f}")

# 4. Extract H_1 features (feedback loops/cycles)
h1_features = extract_h1_features(persistence_result['dgms'][1])
print(f"Number of feedback loops: {h1_features['num_loops']}")
print(f"Loop persistence: {h1_features['max_persistence']:.4f}")

# 5. Identify market regime
regime = identify_market_regimes(h0_features, h1_features)
print(f"Market regime: {regime['regime']}")

# 6. Vectorize for machine learning
# Option A: Statistical features
stat_features = vectorize_persistence_diagrams(persistence_result, method='statistics')

# Option B: Persistence landscapes
landscape_features = vectorize_persistence_diagrams(
    persistence_result,
    method='landscape',
    k=5,
    num_samples=100
)

# Option C: Persistence images
image_features = vectorize_persistence_diagrams(
    persistence_result,
    method='image',
    resolution=(20, 20)
)
```

### Complete Examples

See example scripts for complete working demonstrations:

```bash
# Graph-based analysis
python examples/example_usage.py

# TDA layer for market shape analysis
python examples/tda_layer_example.py
```

## API Reference

### weighted_graph module

#### compute_rolling_correlation(data, window, min_periods=None)
Calculate rolling correlation matrices between N assets.

**Parameters:**
- `data` (pd.DataFrame): Time series data where each column represents an asset
- `window` (int): Size of the rolling window for correlation calculation
- `min_periods` (int, optional): Minimum number of observations required

**Returns:**
- List of correlation matrices over time

#### compute_graph_laplacian(adjacency_matrix, normalized=False)
Compute the Graph Laplacian L = D - A.

**Parameters:**
- `adjacency_matrix` (np.ndarray): Edge weights between nodes
- `normalized` (bool): If True, compute normalized Laplacian

**Returns:**
- `np.ndarray`: The graph Laplacian matrix

#### simulate_heat_kernel(laplacian, time, initial_state=None)
Simulate heat kernel diffusion across the graph.

**Parameters:**
- `laplacian` (np.ndarray): The graph Laplacian matrix
- `time` (float): Diffusion time parameter
- `initial_state` (np.ndarray, optional): Initial heat distribution

**Returns:**
- `np.ndarray`: Heat distribution after time t

#### simulate_random_walk(adjacency_matrix, n_steps, initial_state=None, return_trajectory=False)
Simulate a random walk across the graph.

**Parameters:**
- `adjacency_matrix` (np.ndarray): Edge weights between nodes
- `n_steps` (int): Number of steps in the random walk
- `initial_state` (np.ndarray, optional): Initial probability distribution
- `return_trajectory` (bool): If True, return full trajectory

**Returns:**
- Final probability distribution, or (final_state, trajectory) if return_trajectory=True

### residual_analysis module

#### compute_residuals(actual_returns, expected_returns)
Compute residuals by subtracting expected returns from actual returns.

**Parameters:**
- `actual_returns` (np.ndarray, pd.DataFrame, or pd.Series): Observed returns
- `expected_returns` (np.ndarray, pd.DataFrame, or pd.Series): Expected returns

**Returns:**
- Residuals with same type as input

### tda_layer module

#### build_vietoris_rips_filtration(correlation_matrix, max_dimension=2, max_edge_length=None)
Build a Vietoris-Rips filtration from correlation distances between assets.

**Parameters:**
- `correlation_matrix` (np.ndarray): Correlation matrix between assets
- `max_dimension` (int, optional): Maximum homology dimension to compute (default: 2)
- `max_edge_length` (float, optional): Maximum edge length for filtration

**Returns:**
- Dictionary containing persistence diagrams and filtration information

#### compute_persistent_homology(data, max_dimension=2, max_edge_length=None, use_correlation=True)
Compute persistent homology from financial data.

**Parameters:**
- `data` (np.ndarray or pd.DataFrame): Time series data or correlation matrix
- `max_dimension` (int, optional): Maximum homology dimension (default: 2)
- `max_edge_length` (float, optional): Maximum edge length
- `use_correlation` (bool, optional): If True, compute correlation first (default: True)

**Returns:**
- Dictionary with persistence diagrams

#### extract_h0_features(persistence_diagram)
Extract features from H₀ persistence diagram (connected components/clusters).

**Parameters:**
- `persistence_diagram` (np.ndarray): H₀ persistence diagram

**Returns:**
- Dictionary with cluster features: num_components, max_persistence, mean_persistence, std_persistence, persistence_entropy

#### extract_h1_features(persistence_diagram)
Extract features from H₁ persistence diagram (loops/cycles).

**Parameters:**
- `persistence_diagram` (np.ndarray): H₁ persistence diagram

**Returns:**
- Dictionary with loop features: num_loops, max_persistence, mean_persistence, std_persistence, persistence_entropy, total_persistence

#### compute_persistence_landscape(persistence_diagram, k=5, num_samples=100, x_range=None)
Compute persistence landscape from a persistence diagram.

**Parameters:**
- `persistence_diagram` (np.ndarray): Persistence diagram
- `k` (int, optional): Number of landscape functions (default: 5)
- `num_samples` (int, optional): Number of sample points (default: 100)
- `x_range` (tuple, optional): Range of x values

**Returns:**
- np.ndarray of shape (k, num_samples) containing landscape functions

#### compute_persistence_images(persistence_diagram, resolution=(20, 20), sigma=0.1, x_range=None, y_range=None)
Compute persistence image from a persistence diagram.

**Parameters:**
- `persistence_diagram` (np.ndarray): Persistence diagram
- `resolution` (tuple, optional): Image resolution (default: (20, 20))
- `sigma` (float, optional): Gaussian kernel std dev (default: 0.1)
- `x_range` (tuple, optional): Range for birth times
- `y_range` (tuple, optional): Range for persistence values

**Returns:**
- np.ndarray of specified resolution containing the persistence image

#### vectorize_persistence_diagrams(persistence_result, method='landscape', **kwargs)
Convert persistence diagrams to feature vectors for machine learning.

**Parameters:**
- `persistence_result` (dict): Result from compute_persistent_homology
- `method` (str, optional): Vectorization method - 'landscape', 'image', or 'statistics' (default: 'landscape')
- `**kwargs`: Additional arguments for the vectorization method

**Returns:**
- Dictionary with feature vectors for each homology dimension

#### identify_market_regimes(h0_features, h1_features, threshold_clusters=3, threshold_loops=0.1)
Identify market regimes based on topological features.

**Parameters:**
- `h0_features` (dict): H₀ features from extract_h0_features
- `h1_features` (dict): H₁ features from extract_h1_features
- `threshold_clusters` (int, optional): Threshold for fragmentation (default: 3)
- `threshold_loops` (float, optional): Threshold for significant loops (default: 0.1)

**Returns:**
- Dictionary describing the market regime

## Testing

Run the test suite:

```bash
python -m unittest discover tests -v
```

All tests should pass, covering:
- Rolling correlation computation
- Graph Laplacian calculation
- Heat kernel diffusion
- Random walk simulation
- Residual analysis
- Vietoris-Rips filtration
- Persistent homology computation
- H₀ and H₁ feature extraction
- Persistence landscapes and images
- Market regime identification

## Dependencies

- numpy >= 1.21.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- ripser >= 0.6.0
- scikit-learn >= 1.0.0

## License

This project is open source and available under the MIT License.

## Applications

This package is particularly useful for:
- **Portfolio Analysis**: Understanding how assets are interconnected and identifying natural market clusters
- **Risk Management**: Analyzing how shocks propagate through a portfolio and detecting fragmentation
- **Anomaly Detection**: Identifying mispricings, unusual patterns, and emerging feedback loops
- **Network Analysis**: Studying the topological structure of financial markets using persistent homology
- **Systemic Risk**: Modeling contagion effects and feedback loops in financial systems
- **Market Regime Detection**: Using H₀ and H₁ features to classify market states (unified, fragmented, crisis)
- **Machine Learning**: Converting topological features into vectors for predictive models
