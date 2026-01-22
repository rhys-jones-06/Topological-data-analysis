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

### Market Mutual Model - Advanced Inference System
- **Feature Fusion**: Combine local residuals (graph diffusion) with global topological features (persistent homology)
- **Regime Detection**: Classify market states as "stable", "stressed", or "transitioning" using:
  - Hidden Markov Models (HMM)
  - Clustering algorithms (K-means, GMM)
  - Rule-based topological classification
- **Risk Management**: Scale position sizes based on topological persistence and market regime
  - Dynamic leverage adjustment based on market stability
  - Kelly criterion with topological confidence
  - Portfolio heat monitoring
- **Ensemble Integration**: Combine TDA signals with neural networks and other models
  - Meta-learning with Gradient Boosting
  - Weighted averaging strategies
  - Stacking approaches
- **Backtesting Engine**: Vectorized backtester with transaction costs and slippage
  - Walk-forward analysis
  - Strategy comparison
  - Performance metrics (Sharpe, Sortino, Calmar ratios)

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

### Market Mutual Model Example - Complete Inference Pipeline

```python
import pandas as pd
from tda import MarketMutualModel, Backtester

# 1. Initialize the Market Mutual Model
model = MarketMutualModel(
    regime_detector_type='hmm',  # or 'clustering', 'rule_based'
    n_regimes=3,                 # stable, transitioning, stressed
    random_state=42
)

# 2. Train on historical data
model.fit(
    asset_returns,  # DataFrame of asset returns
    window=20       # Rolling window for correlation
)

# 3. Generate predictions with detailed information
predictions = model.predict(test_returns, return_details=True)

print(f"Current regime: {predictions['regime']}")
print(f"Regime confidence: {predictions['regime_confidence']:.3f}")
print(f"Persistence score: {predictions['persistence_score']:.3f}")
print(f"Trading signals: {predictions['signals']}")

# 4. Get risk-adjusted positions for ensemble
# Combine with signals from neural network or other models
adjusted_positions = model.get_risk_adjusted_positions(
    asset_returns=test_returns,
    base_signals=neural_net_signals  # From your NN model
)

# 5. Backtest the strategy
backtester = Backtester(
    transaction_cost=0.001,  # 0.1%
    slippage=0.0005,        # 0.05%
    initial_capital=100000.0
)

results = backtester.run(
    returns=test_returns,
    signals=pd.DataFrame(predictions['signals'])
)

print(f"\nBacktest Results:")
print(f"Total Return: {results['metrics']['total_return']:.2%}")
print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.3f}")
print(f"Max Drawdown: {results['metrics']['max_drawdown']:.2%}")
```

### Ensemble Learning Example

```python
from tda import MetaLearner, combine_tda_with_neural_net

# Option 1: Simple weighted combination
combined_signal = combine_tda_with_neural_net(
    tda_signal=tda_predictions,
    nn_signal=neural_net_predictions,
    tda_confidence=0.7,
    nn_confidence=0.8,
    method='weighted'
)

# Option 2: Meta-learning with Gradient Boosting
meta_learner = MetaLearner(
    method='boosting',
    task='regression',
    n_estimators=100
)

# Train meta-learner
meta_learner.fit(
    tda_signals=tda_train_signals,
    other_signals=[nn_train_signals, momentum_signals],
    targets=actual_returns
)

# Generate ensemble predictions
ensemble_predictions = meta_learner.predict(
    tda_signals=tda_test_signals,
    other_signals=[nn_test_signals, momentum_signals]
)
```

### Complete Examples

See example scripts for complete working demonstrations:

```bash
# Graph-based analysis
python examples/example_usage.py

# TDA layer for market shape analysis
python examples/tda_layer_example.py

# Complete Market Mutual Model workflow
python examples/market_mutual_example.py
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

### market_mutual_model module

#### MarketMutualModel(regime_detector_type='hmm', n_regimes=3, ...)
Main inference model integrating feature fusion, regime detection, and risk management.

**Parameters:**
- `regime_detector_type` (str): Type of detector - 'hmm', 'clustering', or 'rule_based' (default: 'hmm')
- `n_regimes` (int): Number of regimes to detect (default: 3)
- `feature_fusion_params` (dict, optional): Parameters for FeatureFusion
- `risk_manager_params` (dict, optional): Parameters for RiskManager
- `random_state` (int, optional): Random seed

**Key Methods:**
- `fit(asset_returns, window=20)`: Train model on historical data
- `predict(asset_returns, return_details=False)`: Generate trading signals
- `predict_single_signal(asset_returns)`: Generate aggregate market signal
- `get_risk_adjusted_positions(asset_returns, base_signals)`: Adjust external signals with TDA risk management

### ensemble module

#### MetaLearner(method='weighted_average', task='regression', ...)
Ensemble combination of TDA and other model signals.

**Parameters:**
- `method` (str): 'weighted_average', 'stacking', or 'boosting' (default: 'weighted_average')
- `task` (str): 'regression' or 'classification' (default: 'regression')
- `random_state` (int, optional): Random seed

**Key Methods:**
- `fit(tda_signals, other_signals, targets)`: Train meta-learner
- `predict(tda_signals, other_signals)`: Generate ensemble predictions
- `get_feature_importance()`: Get importance scores if available

#### combine_tda_with_neural_net(tda_signal, nn_signal, tda_confidence, nn_confidence, method='weighted')
Convenience function to combine TDA and neural network signals.

### backtesting module

#### Backtester(transaction_cost=0.001, slippage=0.0005, ...)
Vectorized backtester with transaction costs and slippage.

**Parameters:**
- `transaction_cost` (float): Transaction cost as fraction (default: 0.001)
- `slippage` (float): Slippage as fraction (default: 0.0005)
- `max_position` (float): Maximum position size (default: 1.0)
- `initial_capital` (float): Initial capital (default: 100,000)

**Key Methods:**
- `run(returns, signals, rebalance_frequency=1)`: Run backtest

**Returns dictionary with:**
- `equity_curve`: Portfolio value over time
- `positions`: Position history
- `metrics`: Performance metrics (total_return, sharpe_ratio, max_drawdown, etc.)

#### RollingBacktester(train_window, test_window, backtester=None)
Rolling window backtester for walk-forward analysis.

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
- Feature fusion
- Regime detection (HMM, clustering, rule-based)
- Risk management and position sizing
- Ensemble learning
- Backtesting with transaction costs

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
- **Algorithmic Trading**: Combining TDA signals with neural networks for trading strategies
- **Ensemble Modeling**: Integrating topological insights with traditional quantitative models
- **Stress Testing**: Evaluating strategy performance during market regime transitions

## Architecture

The Market Mutual Model follows a modular architecture:

1. **Data Layer**: Rolling correlation matrices and graph construction
2. **Feature Extraction**:
   - Local: Residuals and heat kernel diffusion
   - Global: Persistent homology (H₀, H₁ features)
3. **Feature Fusion**: Combine local and global features with normalization
4. **Regime Detection**: Classify market state using HMM/clustering
5. **Risk Management**: Position sizing based on topological persistence
6. **Ensemble Integration**: Combine with neural networks and other models
7. **Backtesting**: Validate strategy with transaction costs

This modular design allows each component to be used independently or as part of the complete pipeline.
