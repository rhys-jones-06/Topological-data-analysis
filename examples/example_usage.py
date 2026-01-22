"""
Example usage of the Topological Data Analysis package for financial data.

This script demonstrates:
1. Computing rolling correlation matrices for asset returns
2. Building a graph from correlation matrices  
3. Computing the graph Laplacian
4. Simulating heat kernel diffusion to analyze price shock propagation
5. Simulating random walks on the graph
6. Computing residuals for mispricing analysis
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from tda import (
    compute_rolling_correlation,
    compute_graph_laplacian,
    simulate_heat_kernel,
    simulate_random_walk,
    compute_residuals
)


def generate_sample_asset_data(n_assets=5, n_periods=100):
    """Generate sample asset return data for demonstration."""
    np.random.seed(42)
    
    # Generate correlated asset returns
    # Create a correlation structure
    correlation_matrix = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            correlation_matrix[i, j] = correlation_matrix[j, i] = np.random.uniform(0.3, 0.7)
    
    # Generate returns with correlation structure
    mean_returns = np.random.uniform(-0.01, 0.02, n_assets)
    std_returns = np.random.uniform(0.01, 0.03, n_assets)
    
    # Generate correlated random returns
    returns = np.random.multivariate_normal(
        mean_returns,
        np.diag(std_returns) @ correlation_matrix @ np.diag(std_returns),
        n_periods
    )
    
    # Create DataFrame
    asset_names = [f'Asset_{i+1}' for i in range(n_assets)]
    dates = pd.date_range('2023-01-01', periods=n_periods, freq='D')
    
    return pd.DataFrame(returns, index=dates, columns=asset_names)


def main():
    print("=" * 80)
    print("Topological Data Analysis for Financial Data - Example Usage")
    print("=" * 80)
    print()
    
    # 1. Generate sample data
    print("1. Generating sample asset return data...")
    asset_returns = generate_sample_asset_data(n_assets=5, n_periods=100)
    print(f"   Generated {len(asset_returns)} periods of returns for {len(asset_returns.columns)} assets")
    print(f"   Sample data:")
    print(asset_returns.head())
    print()
    
    # 2. Compute rolling correlation matrices
    print("2. Computing rolling correlation matrices...")
    window_size = 20
    correlation_matrices = compute_rolling_correlation(asset_returns, window=window_size)
    print(f"   Computed {len(correlation_matrices)} correlation matrices with window size {window_size}")
    print(f"   Latest correlation matrix:")
    latest_corr = correlation_matrices[-1].drop(columns=['timestamp'])
    print(latest_corr)
    print()
    
    # 3. Use the latest correlation as adjacency matrix and compute Graph Laplacian
    print("3. Computing Graph Laplacian...")
    adjacency_matrix = latest_corr.values
    # Convert correlation to positive weights (absolute value)
    adjacency_matrix = np.abs(adjacency_matrix)
    np.fill_diagonal(adjacency_matrix, 0)  # Remove self-loops
    
    laplacian = compute_graph_laplacian(adjacency_matrix)
    print(f"   Graph Laplacian:")
    print(laplacian)
    print()
    
    # Also compute normalized Laplacian
    laplacian_normalized = compute_graph_laplacian(adjacency_matrix, normalized=True)
    print(f"   Normalized Graph Laplacian:")
    print(laplacian_normalized)
    print()
    
    # 4. Simulate heat kernel diffusion
    print("4. Simulating heat kernel diffusion...")
    # Simulate a price shock at Asset_1
    initial_shock = np.zeros(5)
    initial_shock[0] = 1.0  # Shock at first asset
    
    print(f"   Initial shock: {initial_shock}")
    
    # Simulate diffusion at different time steps
    for time in [0.1, 0.5, 1.0, 2.0]:
        diffused = simulate_heat_kernel(laplacian, time=time, initial_state=initial_shock)
        print(f"   After time={time}: {diffused}")
    print()
    
    # 5. Simulate random walk
    print("5. Simulating random walk on the graph...")
    n_steps = 100
    final_state, trajectory = simulate_random_walk(
        adjacency_matrix, 
        n_steps=n_steps, 
        initial_state=initial_shock,
        return_trajectory=True
    )
    
    print(f"   Initial state: {trajectory[0]}")
    print(f"   Final state after {n_steps} steps: {final_state}")
    print(f"   Trajectory contains {len(trajectory)} states")
    print()
    
    # 6. Residual Analysis
    print("6. Computing residuals for mispricing analysis...")
    
    # Generate expected returns (e.g., from a model like CAPM)
    # For demonstration, use a simple moving average as expected returns
    expected_returns = asset_returns.rolling(window=10, min_periods=1).mean()
    
    # Compute residuals
    residuals = compute_residuals(asset_returns, expected_returns)
    
    print(f"   Residual statistics:")
    print(residuals.describe())
    print()
    
    print(f"   Assets with largest absolute residuals in last period:")
    last_residuals = residuals.iloc[-1].abs().sort_values(ascending=False)
    print(last_residuals)
    print()
    
    # Identify potential mispricings (large residuals)
    threshold = residuals.std().mean() * 2
    print(f"   Potential mispricings (|residual| > {threshold:.4f}):")
    mispricings = residuals[residuals.abs() > threshold]
    print(f"   Found {mispricings.count().sum()} instances of potential mispricing")
    print()
    
    print("=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
