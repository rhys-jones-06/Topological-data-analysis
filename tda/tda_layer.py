"""
TDA Layer Module - Algebraic Topology for Financial Data Analysis

This module implements topological data analysis techniques for extracting
global market shape from financial data. It includes:

1. Vietoris-Rips Filtration: Build simplicial complexes from correlation distances
2. Persistent Homology: Compute persistence diagrams using Ripser
3. Topological Features: Extract H_0 (clusters) and H_1 (loops/cycles)
4. Feature Vectorization: Convert topological shapes to numerical features
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union, Optional
from ripser import ripser


def _compute_persistence_entropy(persistences: np.ndarray) -> float:
    """
    Compute entropy of a persistence distribution.
    
    Helper function to calculate the Shannon entropy of persistence values,
    used as a measure of the complexity of the topological features.
    
    Parameters
    ----------
    persistences : np.ndarray
        Array of persistence values (death - birth).
    
    Returns
    -------
    float
        Entropy of the persistence distribution.
    """
    if len(persistences) == 0 or np.sum(persistences) == 0:
        return 0.0
    
    # Normalize to get probability distribution
    probs = persistences / np.sum(persistences)
    # Remove zeros to avoid log(0)
    probs = probs[probs > 0]
    # Calculate Shannon entropy
    return float(-np.sum(probs * np.log(probs)))


def build_vietoris_rips_filtration(
    correlation_matrix: np.ndarray,
    max_dimension: int = 2,
    max_edge_length: Optional[float] = None
) -> Dict:
    """
    Build a Vietoris-Rips filtration from correlation distances.
    
    The Vietoris-Rips complex is built from a distance matrix derived from
    correlation values. We use (1 - |correlation|) as the distance metric,
    where stronger correlations result in smaller distances.
    
    Parameters
    ----------
    correlation_matrix : np.ndarray
        Correlation matrix between assets. Shape (n_assets, n_assets).
    max_dimension : int, optional
        Maximum homology dimension to compute. Default is 2.
        H_0 = connected components (clusters)
        H_1 = loops/cycles
        H_2 = voids
    max_edge_length : float, optional
        Maximum edge length for the filtration. If None, uses diameter.
    
    Returns
    -------
    dict
        Dictionary containing the Vietoris-Rips filtration result with keys:
        - 'distance_matrix': The distance matrix used
        - 'dgms': Persistence diagrams for each homology dimension (from Ripser)
    
    Examples
    --------
    >>> import numpy as np
    >>> corr = np.array([[1.0, 0.8, 0.6], [0.8, 1.0, 0.7], [0.6, 0.7, 1.0]])
    >>> result = build_vietoris_rips_filtration(corr, max_dimension=1)
    >>> print(result['dgms'])
    """
    # Convert correlation to distance: d = 1 - |corr|
    # Strong positive/negative correlations -> small distances
    distance_matrix = 1.0 - np.abs(correlation_matrix)
    
    # Ensure diagonal is zero (distance from asset to itself)
    np.fill_diagonal(distance_matrix, 0)
    
    # Compute persistent homology using Ripser
    result = ripser(
        distance_matrix,
        maxdim=max_dimension,
        thresh=max_edge_length if max_edge_length is not None else np.inf,
        distance_matrix=True
    )
    
    # Add the distance matrix to the result for reference
    result['distance_matrix'] = distance_matrix
    
    return result


def compute_persistent_homology(
    data: Union[np.ndarray, pd.DataFrame],
    max_dimension: int = 2,
    max_edge_length: Optional[float] = None,
    use_correlation: bool = True
) -> Dict:
    """
    Compute persistent homology from financial data.
    
    This is a convenience function that handles both correlation matrices
    and raw data. If the input is raw data, it computes correlations first.
    
    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        Either a correlation matrix (n_assets x n_assets) or
        time series data (n_periods x n_assets).
    max_dimension : int, optional
        Maximum homology dimension to compute. Default is 2.
    max_edge_length : float, optional
        Maximum edge length for filtration. Default is None (no limit).
    use_correlation : bool, optional
        If True, treats data as time series and computes correlation first.
        If False, treats data as a distance/dissimilarity matrix.
        Default is True.
    
    Returns
    -------
    dict
        Persistent homology result from Ripser containing persistence diagrams.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> # From time series data
    >>> returns = pd.DataFrame(np.random.randn(100, 5))
    >>> result = compute_persistent_homology(returns, max_dimension=1)
    >>> 
    >>> # From correlation matrix
    >>> corr = returns.corr()
    >>> result = compute_persistent_homology(corr.values, use_correlation=False)
    """
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    if use_correlation:
        # If data is time series, compute correlation first
        if data.shape[0] > data.shape[1]:
            # Assume rows are time periods, columns are assets
            correlation_matrix = np.corrcoef(data.T)
        else:
            # Assume already a correlation matrix
            correlation_matrix = data
    else:
        # Treat as correlation/similarity matrix
        correlation_matrix = data
    
    return build_vietoris_rips_filtration(
        correlation_matrix,
        max_dimension=max_dimension,
        max_edge_length=max_edge_length
    )


def extract_h0_features(persistence_diagram: np.ndarray) -> Dict[str, float]:
    """
    Extract features from H_0 persistence diagram (connected components/clusters).
    
    H_0 represents connected components. The persistence of H_0 features
    indicates how long clusters persist as the filtration parameter increases.
    Long-lived components suggest stable market clusters.
    
    Parameters
    ----------
    persistence_diagram : np.ndarray
        H_0 persistence diagram from Ripser. Shape (n_features, 2).
        Each row is [birth, death] time of a topological feature.
    
    Returns
    -------
    dict
        Dictionary containing H_0 features:
        - 'num_components': Number of connected components (excluding infinite)
        - 'max_persistence': Maximum persistence value
        - 'mean_persistence': Mean persistence value
        - 'std_persistence': Standard deviation of persistence
        - 'persistence_entropy': Entropy of the persistence distribution
    
    Examples
    --------
    >>> dgm = np.array([[0.0, 0.1], [0.0, 0.3], [0.0, 0.5]])
    >>> features = extract_h0_features(dgm)
    >>> print(features['max_persistence'])
    0.5
    """
    # Remove infinite death times (the last component that never dies)
    finite_dgm = persistence_diagram[persistence_diagram[:, 1] < np.inf]
    
    if len(finite_dgm) == 0:
        return {
            'num_components': 0,
            'max_persistence': 0.0,
            'mean_persistence': 0.0,
            'std_persistence': 0.0,
            'persistence_entropy': 0.0
        }
    
    # Compute persistence (death - birth)
    persistences = finite_dgm[:, 1] - finite_dgm[:, 0]
    
    return {
        'num_components': len(finite_dgm),
        'max_persistence': float(np.max(persistences)) if len(persistences) > 0 else 0.0,
        'mean_persistence': float(np.mean(persistences)) if len(persistences) > 0 else 0.0,
        'std_persistence': float(np.std(persistences)) if len(persistences) > 0 else 0.0,
        'persistence_entropy': _compute_persistence_entropy(persistences)
    }


def extract_h1_features(persistence_diagram: np.ndarray) -> Dict[str, float]:
    """
    Extract features from H_1 persistence diagram (loops/cycles).
    
    H_1 represents 1-dimensional holes (loops/cycles) in the data.
    In financial markets, these can represent circular dependencies or
    feedback loops between assets. Long-lived H_1 features indicate
    stable cyclic market regimes.
    
    Parameters
    ----------
    persistence_diagram : np.ndarray
        H_1 persistence diagram from Ripser. Shape (n_features, 2).
        Each row is [birth, death] time of a topological feature.
    
    Returns
    -------
    dict
        Dictionary containing H_1 features:
        - 'num_loops': Number of loops/cycles detected
        - 'max_persistence': Maximum persistence value
        - 'mean_persistence': Mean persistence value
        - 'std_persistence': Standard deviation of persistence
        - 'persistence_entropy': Entropy of the persistence distribution
        - 'total_persistence': Sum of all persistence values
    
    Examples
    --------
    >>> dgm = np.array([[0.2, 0.5], [0.3, 0.6]])
    >>> features = extract_h1_features(dgm)
    >>> print(features['num_loops'])
    2
    """
    # H_1 features should not have infinite death times
    finite_dgm = persistence_diagram[persistence_diagram[:, 1] < np.inf]
    
    if len(finite_dgm) == 0:
        return {
            'num_loops': 0,
            'max_persistence': 0.0,
            'mean_persistence': 0.0,
            'std_persistence': 0.0,
            'persistence_entropy': 0.0,
            'total_persistence': 0.0
        }
    
    # Compute persistence (death - birth)
    persistences = finite_dgm[:, 1] - finite_dgm[:, 0]
    
    return {
        'num_loops': len(finite_dgm),
        'max_persistence': float(np.max(persistences)) if len(persistences) > 0 else 0.0,
        'mean_persistence': float(np.mean(persistences)) if len(persistences) > 0 else 0.0,
        'std_persistence': float(np.std(persistences)) if len(persistences) > 0 else 0.0,
        'persistence_entropy': _compute_persistence_entropy(persistences),
        'total_persistence': float(np.sum(persistences)) if len(persistences) > 0 else 0.0
    }


def compute_persistence_landscape(
    persistence_diagram: np.ndarray,
    k: int = 5,
    num_samples: int = 100,
    x_range: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Compute persistence landscape from a persistence diagram.
    
    Persistence landscapes are functional summaries of persistence diagrams
    that can be used as features for machine learning. The k-th landscape
    function represents the k-th largest persistence value at each point.
    
    Parameters
    ----------
    persistence_diagram : np.ndarray
        Persistence diagram. Shape (n_features, 2) with [birth, death] pairs.
    k : int, optional
        Number of landscape functions to compute. Default is 5.
    num_samples : int, optional
        Number of points to sample along the x-axis. Default is 100.
    x_range : tuple of (float, float), optional
        Range of x values to sample. If None, uses the full range of the diagram.
    
    Returns
    -------
    np.ndarray
        Persistence landscape. Shape (k, num_samples).
        Each row is one landscape function.
    
    Examples
    --------
    >>> dgm = np.array([[0.0, 0.5], [0.2, 0.7], [0.3, 0.6]])
    >>> landscape = compute_persistence_landscape(dgm, k=3, num_samples=50)
    >>> print(landscape.shape)
    (3, 50)
    """
    # Remove infinite values
    finite_dgm = persistence_diagram[persistence_diagram[:, 1] < np.inf]
    
    if len(finite_dgm) == 0:
        return np.zeros((k, num_samples))
    
    # Determine x range
    if x_range is None:
        x_min = np.min(finite_dgm[:, 0])
        x_max = np.max(finite_dgm[:, 1])
        x_range = (x_min, x_max)
    
    # Create x values
    x_values = np.linspace(x_range[0], x_range[1], num_samples)
    
    # Initialize landscape
    landscape = np.zeros((k, num_samples))
    
    # For each x value, compute the k largest lambda values
    for i, x in enumerate(x_values):
        lambda_values = []
        
        # For each persistence pair (b, d)
        for birth, death in finite_dgm:
            # Compute lambda(x) = min(x - birth, death - x) if birth <= x <= death
            if birth <= x <= death:
                lambda_val = min(x - birth, death - x)
                lambda_values.append(lambda_val)
        
        # Sort in descending order and take top k
        lambda_values = sorted(lambda_values, reverse=True)
        
        # Fill in the k landscape values (pad with zeros if needed)
        for j in range(k):
            if j < len(lambda_values):
                landscape[j, i] = lambda_values[j]
    
    return landscape


def compute_persistence_images(
    persistence_diagram: np.ndarray,
    resolution: Tuple[int, int] = (20, 20),
    sigma: float = 0.1,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Compute persistence image from a persistence diagram.
    
    Persistence images convert persistence diagrams into 2D images by
    placing Gaussian kernels at each point and summing them. The resulting
    image can be used as a feature vector for machine learning.
    
    Parameters
    ----------
    persistence_diagram : np.ndarray
        Persistence diagram. Shape (n_features, 2) with [birth, death] pairs.
    resolution : tuple of (int, int), optional
        Resolution of the output image (height, width). Default is (20, 20).
    sigma : float, optional
        Standard deviation of the Gaussian kernel. Default is 0.1.
    x_range : tuple of (float, float), optional
        Range for birth times. If None, uses the data range.
    y_range : tuple of (float, float), optional
        Range for persistence values. If None, uses the data range.
    
    Returns
    -------
    np.ndarray
        Persistence image. Shape resolution.
    
    Examples
    --------
    >>> dgm = np.array([[0.0, 0.5], [0.2, 0.7], [0.3, 0.6]])
    >>> image = compute_persistence_images(dgm, resolution=(10, 10))
    >>> print(image.shape)
    (10, 10)
    """
    # Remove infinite values
    finite_dgm = persistence_diagram[persistence_diagram[:, 1] < np.inf]
    
    if len(finite_dgm) == 0:
        return np.zeros(resolution)
    
    # Convert to birth-persistence coordinates
    births = finite_dgm[:, 0]
    persistences = finite_dgm[:, 1] - finite_dgm[:, 0]
    
    # Determine ranges
    if x_range is None:
        x_min, x_max = np.min(births), np.max(births)
        # Add padding
        x_padding = (x_max - x_min) * 0.1
        x_range = (x_min - x_padding, x_max + x_padding)
    
    if y_range is None:
        y_min, y_max = np.min(persistences), np.max(persistences)
        # Add padding
        y_padding = (y_max - y_min) * 0.1 if y_max > y_min else 0.1
        y_range = (y_min, y_max + y_padding)
    
    # Create grid
    x_grid = np.linspace(x_range[0], x_range[1], resolution[1])
    y_grid = np.linspace(y_range[0], y_range[1], resolution[0])
    
    # Initialize image
    image = np.zeros(resolution)
    
    # For each point in the persistence diagram
    for birth, persistence in zip(births, persistences):
        # Weight by persistence
        weight = persistence
        
        # Add Gaussian kernel centered at (birth, persistence)
        for i, y in enumerate(y_grid):
            for j, x in enumerate(x_grid):
                # Gaussian kernel
                distance_sq = ((x - birth) ** 2 + (y - persistence) ** 2)
                image[i, j] += weight * np.exp(-distance_sq / (2 * sigma ** 2))
    
    return image


def vectorize_persistence_diagrams(
    persistence_result: Dict,
    method: str = 'landscape',
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Convert persistence diagrams to feature vectors for machine learning.
    
    This function provides a unified interface for converting topological
    features into numerical vectors that can be used by ML models.
    
    Parameters
    ----------
    persistence_result : dict
        Result from compute_persistent_homology or build_vietoris_rips_filtration.
    method : str, optional
        Vectorization method. Options:
        - 'landscape': Persistence landscapes
        - 'image': Persistence images
        - 'statistics': Statistical summaries (default features)
        Default is 'landscape'.
    **kwargs
        Additional arguments passed to the vectorization method.
    
    Returns
    -------
    dict
        Dictionary with keys 'H_0', 'H_1', etc., each containing feature vectors.
    
    Examples
    --------
    >>> import numpy as np
    >>> corr = np.eye(5)
    >>> result = build_vietoris_rips_filtration(corr)
    >>> features = vectorize_persistence_diagrams(result, method='statistics')
    >>> print(features.keys())
    dict_keys(['H_0', 'H_1'])
    """
    dgms = persistence_result['dgms']
    features = {}
    
    for i, dgm in enumerate(dgms):
        dim_name = f'H_{i}'
        
        if len(dgm) == 0:
            # No features for this dimension
            if method == 'landscape':
                k = kwargs.get('k', 5)
                num_samples = kwargs.get('num_samples', 100)
                features[dim_name] = np.zeros((k, num_samples))
            elif method == 'image':
                resolution = kwargs.get('resolution', (20, 20))
                features[dim_name] = np.zeros(resolution)
            else:  # statistics
                features[dim_name] = {
                    'num_features': 0,
                    'max_persistence': 0.0,
                    'mean_persistence': 0.0,
                    'std_persistence': 0.0,
                    'persistence_entropy': 0.0
                }
        else:
            if method == 'landscape':
                features[dim_name] = compute_persistence_landscape(dgm, **kwargs)
            elif method == 'image':
                features[dim_name] = compute_persistence_images(dgm, **kwargs)
            else:  # statistics
                if i == 0:
                    features[dim_name] = extract_h0_features(dgm)
                elif i == 1:
                    features[dim_name] = extract_h1_features(dgm)
                else:
                    # Generic features for higher dimensions
                    finite_dgm = dgm[dgm[:, 1] < np.inf]
                    if len(finite_dgm) > 0:
                        persistences = finite_dgm[:, 1] - finite_dgm[:, 0]
                        features[dim_name] = {
                            'num_features': len(finite_dgm),
                            'max_persistence': float(np.max(persistences)),
                            'mean_persistence': float(np.mean(persistences)),
                            'std_persistence': float(np.std(persistences)),
                            'total_persistence': float(np.sum(persistences))
                        }
                    else:
                        features[dim_name] = {
                            'num_features': 0,
                            'max_persistence': 0.0,
                            'mean_persistence': 0.0,
                            'std_persistence': 0.0,
                            'total_persistence': 0.0
                        }
    
    return features


def identify_market_regimes(
    h0_features: Dict[str, float],
    h1_features: Dict[str, float],
    threshold_clusters: int = 3,
    threshold_loops: float = 0.1
) -> Dict[str, Union[str, bool]]:
    """
    Identify market regimes based on topological features.
    
    Uses H_0 (clusters) and H_1 (loops) features to characterize the
    current market regime.
    
    Parameters
    ----------
    h0_features : dict
        H_0 features from extract_h0_features.
    h1_features : dict
        H_1 features from extract_h1_features.
    threshold_clusters : int, optional
        Threshold for number of market clusters. Default is 3.
    threshold_loops : float, optional
        Threshold for significant loop persistence. Default is 0.1.
    
    Returns
    -------
    dict
        Dictionary describing the market regime:
        - 'regime': String description of the regime
        - 'is_fragmented': Boolean, True if market is fragmented into clusters
        - 'has_cycles': Boolean, True if significant feedback loops exist
        - 'num_clusters': Number of clusters
        - 'num_loops': Number of loops
    
    Examples
    --------
    >>> h0 = {'num_components': 5, 'max_persistence': 0.3}
    >>> h1 = {'num_loops': 2, 'max_persistence': 0.15}
    >>> regime = identify_market_regimes(h0, h1)
    >>> print(regime['regime'])
    'Fragmented with feedback loops'
    """
    num_clusters = h0_features['num_components']
    num_loops = h1_features['num_loops']
    max_loop_persistence = h1_features['max_persistence']
    
    is_fragmented = num_clusters >= threshold_clusters
    has_cycles = num_loops > 0 and max_loop_persistence >= threshold_loops
    
    # Determine regime description
    if is_fragmented and has_cycles:
        regime = "Fragmented with feedback loops"
    elif is_fragmented:
        regime = "Fragmented (multiple clusters)"
    elif has_cycles:
        regime = "Connected with feedback loops"
    else:
        regime = "Unified (single connected market)"
    
    return {
        'regime': regime,
        'is_fragmented': is_fragmented,
        'has_cycles': has_cycles,
        'num_clusters': num_clusters,
        'num_loops': num_loops,
        'max_loop_persistence': max_loop_persistence
    }
