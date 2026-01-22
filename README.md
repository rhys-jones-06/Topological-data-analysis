# Topological Data Analysis for Financial Data

A Python package for analyzing financial data using graph theory and topological methods. This package provides tools to model financial asset relationships as weighted graphs and analyze information diffusion and price shock propagation.

## Features

### Weighted Graph Generation
- **Rolling Correlation Matrices**: Calculate rolling correlation matrices between N assets to serve as edge weights in a graph representation
- **Graph Laplacian**: Compute the Graph Laplacian L = D - A, where D is the degree matrix and A is the adjacency matrix
- **Heat Kernel Diffusion**: Simulate heat kernel diffusion across the graph to smooth out noise and identify how price shocks propagate
- **Random Walk Simulation**: Model information or price shock propagation through random walks on the asset correlation graph

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

### Basic Example

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

### Complete Example

See `examples/example_usage.py` for a complete working example that demonstrates all features:

```bash
python examples/example_usage.py
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

## Dependencies

- numpy >= 1.21.0
- pandas >= 1.3.0
- scipy >= 1.7.0

## License

This project is open source and available under the MIT License.

## Applications

This package is particularly useful for:
- **Portfolio Analysis**: Understanding how assets are interconnected
- **Risk Management**: Analyzing how shocks propagate through a portfolio
- **Anomaly Detection**: Identifying mispricings and unusual patterns
- **Network Analysis**: Studying the topological structure of financial markets
- **Systemic Risk**: Modeling contagion effects in financial systems
