"""
Weighted Graph Generation Module

This module provides functionality to:
1. Calculate rolling correlation matrices between N assets to serve as edge weights
2. Compute the Graph Laplacian L = D - A
3. Simulate heat kernel or random walk diffusion across the graph
"""

import numpy as np
import pandas as pd
from scipy.linalg import expm
from typing import Union, Tuple


def compute_rolling_correlation(
    data: pd.DataFrame,
    window: int,
    min_periods: int = None
) -> pd.DataFrame:
    """
    Calculate rolling correlation matrices between N assets.
    
    This function computes pairwise correlations between all assets in the dataset
    using a rolling window approach. The correlation matrix serves as edge weights
    in the graph representation.
    
    Parameters
    ----------
    data : pd.DataFrame
        Time series data where each column represents an asset and each row
        represents a time period (e.g., daily returns).
    window : int
        The size of the rolling window for correlation calculation.
    min_periods : int, optional
        Minimum number of observations required to have a value.
        If None, defaults to window size.
    
    Returns
    -------
    pd.DataFrame
        A 3D DataFrame structure containing correlation matrices over time.
        Returns a DataFrame with MultiIndex representing (time, asset1, asset2).
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = pd.DataFrame(np.random.randn(100, 5), columns=['A', 'B', 'C', 'D', 'E'])
    >>> corr_matrices = compute_rolling_correlation(data, window=20)
    """
    if min_periods is None:
        min_periods = window
    
    # Store correlation matrices for each time step
    correlation_matrices = []
    
    for i in range(window - 1, len(data)):
        # Extract the rolling window
        window_data = data.iloc[i - window + 1:i + 1]
        
        # Compute correlation matrix
        corr_matrix = window_data.corr()
        
        # Store with timestamp
        corr_matrix['timestamp'] = data.index[i] if hasattr(data, 'index') else i
        correlation_matrices.append(corr_matrix)
    
    return correlation_matrices


def compute_graph_laplacian(
    adjacency_matrix: np.ndarray,
    normalized: bool = False
) -> np.ndarray:
    """
    Compute the Graph Laplacian L = D - A.
    
    The graph Laplacian is a matrix representation of a graph that encodes
    its structure. It is defined as L = D - A, where:
    - D is the degree matrix (diagonal matrix of node degrees)
    - A is the adjacency matrix (edge weights)
    
    Parameters
    ----------
    adjacency_matrix : np.ndarray
        The adjacency matrix representing edge weights between nodes.
        Shape should be (n_nodes, n_nodes).
    normalized : bool, optional
        If True, compute the normalized Laplacian L_norm = D^(-1/2) L D^(-1/2).
        Default is False.
    
    Returns
    -------
    np.ndarray
        The graph Laplacian matrix of shape (n_nodes, n_nodes).
    
    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    >>> L = compute_graph_laplacian(A)
    >>> print(L)
    [[ 2 -1 -1]
     [-1  2 -1]
     [-1 -1  2]]
    """
    # Ensure the adjacency matrix is a numpy array
    A = np.asarray(adjacency_matrix)
    
    # Compute degree matrix (sum of edge weights for each node)
    degrees = np.sum(A, axis=1)
    D = np.diag(degrees)
    
    # Compute Laplacian
    L = D - A
    
    if normalized:
        # Compute normalized Laplacian: L_norm = D^(-1/2) L D^(-1/2)
        # Handle zero degrees to avoid division by zero
        degrees_sqrt_inv = np.zeros_like(degrees)
        non_zero_mask = degrees > 0
        degrees_sqrt_inv[non_zero_mask] = 1.0 / np.sqrt(degrees[non_zero_mask])
        
        D_sqrt_inv = np.diag(degrees_sqrt_inv)
        L = D_sqrt_inv @ L @ D_sqrt_inv
    
    return L


def simulate_heat_kernel(
    laplacian: np.ndarray,
    time: float,
    initial_state: np.ndarray = None
) -> np.ndarray:
    """
    Simulate heat kernel diffusion across the graph.
    
    The heat kernel describes how heat (or information) diffuses across a graph
    over time. It is computed as: H(t) = exp(-t * L), where L is the graph Laplacian.
    This can be used to smooth out noise and identify how price shocks propagate.
    
    Parameters
    ----------
    laplacian : np.ndarray
        The graph Laplacian matrix of shape (n_nodes, n_nodes).
    time : float
        The diffusion time parameter. Larger values result in more diffusion.
    initial_state : np.ndarray, optional
        Initial heat distribution on the graph. If None, uses uniform distribution.
        Shape should be (n_nodes,).
    
    Returns
    -------
    np.ndarray
        The heat distribution after time t. If initial_state is provided,
        returns a 1D array of shape (n_nodes,). Otherwise, returns the full
        heat kernel matrix of shape (n_nodes, n_nodes).
    
    Examples
    --------
    >>> import numpy as np
    >>> L = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]])
    >>> heat_kernel = simulate_heat_kernel(L, time=1.0)
    >>> initial = np.array([1, 0, 0])  # Shock at first node
    >>> diffused = simulate_heat_kernel(L, time=1.0, initial_state=initial)
    """
    # Compute heat kernel: H(t) = exp(-t * L)
    heat_kernel = expm(-time * laplacian)
    
    if initial_state is not None:
        # Apply heat kernel to initial state
        return heat_kernel @ initial_state
    else:
        # Return full heat kernel matrix
        return heat_kernel


def simulate_random_walk(
    adjacency_matrix: np.ndarray,
    n_steps: int,
    initial_state: np.ndarray = None,
    return_trajectory: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, list]]:
    """
    Simulate a random walk (diffusion) across the graph.
    
    A random walk simulates how information or price shocks propagate through
    a network. At each step, the walker moves to a neighboring node with
    probability proportional to the edge weight.
    
    Parameters
    ----------
    adjacency_matrix : np.ndarray
        The adjacency matrix representing edge weights between nodes.
        Shape should be (n_nodes, n_nodes).
    n_steps : int
        Number of steps in the random walk.
    initial_state : np.ndarray, optional
        Initial probability distribution over nodes. If None, uses uniform
        distribution. Shape should be (n_nodes,).
    return_trajectory : bool, optional
        If True, return the full trajectory of the random walk at each step.
        Default is False.
    
    Returns
    -------
    np.ndarray or tuple
        If return_trajectory is False, returns the final probability distribution
        of shape (n_nodes,). If return_trajectory is True, returns a tuple of
        (final_state, trajectory) where trajectory is a list of states at each step.
    
    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    >>> final_state = simulate_random_walk(A, n_steps=10)
    >>> final_state, trajectory = simulate_random_walk(A, n_steps=10, return_trajectory=True)
    """
    A = np.asarray(adjacency_matrix)
    n_nodes = A.shape[0]
    
    # Initialize state
    if initial_state is None:
        state = np.ones(n_nodes) / n_nodes
    else:
        state = np.asarray(initial_state)
        state = state / np.sum(state)  # Normalize
    
    # Compute transition matrix (row-stochastic)
    # P[i,j] = probability of moving from i to j
    row_sums = np.sum(A, axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    transition_matrix = A / row_sums
    
    trajectory = [state.copy()] if return_trajectory else None
    
    # Simulate random walk
    for _ in range(n_steps):
        state = transition_matrix.T @ state
        if return_trajectory:
            trajectory.append(state.copy())
    
    if return_trajectory:
        return state, trajectory
    else:
        return state
