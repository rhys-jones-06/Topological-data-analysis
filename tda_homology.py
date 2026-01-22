"""
TDA Homology Module using Giotto-TDA

Implements persistent homology analysis to identify market fragmentation
and structural stability using topological data analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
import ripser


class TDAHomology:
    """
    Topological Data Analysis using Persistent Homology.
    
    Uses Giotto-TDA to identify:
    - H0: Market clusters (connected components)
    - H1: Feedback loops and cycles
    
    Provides topological stability metrics for regime detection.
    """
    
    def __init__(
        self,
        max_edge_length: float = 2.0,
        max_dimension: int = 2,
        n_jobs: int = 1,
        baseline_window: int = 30
    ):
        """
        Initialize the TDA Homology analyzer.
        
        Parameters
        ----------
        max_edge_length : float, optional
            Maximum edge length for Vietoris-Rips. Default is 2.0.
        max_dimension : int, optional
            Maximum homology dimension to compute. Default is 2.
        n_jobs : int, optional
            Number of parallel jobs (not used with ripser). Default is 1.
        baseline_window : int, optional
            Number of days for rolling baseline. Default is 30.
        """
        self.max_edge_length = max_edge_length
        self.max_dimension = max_dimension
        self.n_jobs = n_jobs
        self.baseline_window = baseline_window
        
        self.last_diagrams = None
        self.persistence_history = []  # Rolling history of persistence scores
    
    def correlation_to_distance(self, correlation_matrix: np.ndarray) -> np.ndarray:
        """
        Convert correlation matrix to distance matrix.
        
        Parameters
        ----------
        correlation_matrix : np.ndarray
            Correlation matrix
        
        Returns
        -------
        np.ndarray
            Distance matrix
        """
        # Distance = sqrt(2 * (1 - correlation))
        # Clip to avoid negative values from numerical errors
        distance_matrix = np.sqrt(2 * np.clip(1 - correlation_matrix, 0, 2))
        return distance_matrix
    
    def compute_persistence_diagrams(
        self,
        correlation_matrix: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute persistence diagrams from correlation matrix.
        
        Parameters
        ----------
        correlation_matrix : np.ndarray
            Correlation matrix between assets
        
        Returns
        -------
        dict
            Dictionary with 'dgms' containing list of persistence diagrams
        """
        # Convert to distance matrix
        distance_matrix = self.correlation_to_distance(correlation_matrix)
        
        # Compute persistence using ripser
        result = ripser.ripser(
            distance_matrix,
            maxdim=self.max_dimension,
            thresh=self.max_edge_length,
            distance_matrix=True
        )
        
        self.last_diagrams = result['dgms']
        return result
    
    def extract_h0_features(self, diagrams: Optional[List[np.ndarray]] = None) -> Dict[str, float]:
        """
        Extract features from H0 persistence (connected components/clusters).
        
        Parameters
        ----------
        diagrams : list of np.ndarray, optional
            List of persistence diagrams. If None, uses last computed.
        
        Returns
        -------
        dict
            H0 features: num_components, max_persistence, mean_persistence, etc.
        """
        if diagrams is None:
            diagrams = self.last_diagrams
        
        if diagrams is None:
            raise ValueError("No diagrams available. Compute first.")
        
        # Extract H0 diagram (dimension 0)
        h0_diagram = diagrams[0]
        
        if len(h0_diagram) == 0:
            return {
                'num_components': 0,
                'max_persistence': 0.0,
                'mean_persistence': 0.0,
                'std_persistence': 0.0,
                'total_persistence': 0.0
            }
        
        # Persistence = death - birth
        persistence = h0_diagram[:, 1] - h0_diagram[:, 0]
        persistence = persistence[persistence < np.inf]  # Remove infinite bars
        
        features = {
            'num_components': len(persistence),
            'max_persistence': float(np.max(persistence)) if len(persistence) > 0 else 0.0,
            'mean_persistence': float(np.mean(persistence)) if len(persistence) > 0 else 0.0,
            'std_persistence': float(np.std(persistence)) if len(persistence) > 0 else 0.0,
            'total_persistence': float(np.sum(persistence)) if len(persistence) > 0 else 0.0
        }
        
        return features
    
    def extract_h1_features(self, diagrams: Optional[List[np.ndarray]] = None) -> Dict[str, float]:
        """
        Extract features from H1 persistence (loops/cycles).
        
        Parameters
        ----------
        diagrams : list of np.ndarray, optional
            List of persistence diagrams. If None, uses last computed.
        
        Returns
        -------
        dict
            H1 features: num_loops, max_persistence, mean_persistence, etc.
        """
        if diagrams is None:
            diagrams = self.last_diagrams
        
        if diagrams is None:
            raise ValueError("No diagrams available. Compute first.")
        
        # Extract H1 diagram (dimension 1)
        if len(diagrams) < 2:
            return {
                'num_loops': 0,
                'max_persistence': 0.0,
                'mean_persistence': 0.0,
                'std_persistence': 0.0,
                'total_persistence': 0.0
            }
        
        h1_diagram = diagrams[1]
        
        if len(h1_diagram) == 0:
            return {
                'num_loops': 0,
                'max_persistence': 0.0,
                'mean_persistence': 0.0,
                'std_persistence': 0.0,
                'total_persistence': 0.0
            }
        
        # Persistence = death - birth
        persistence = h1_diagram[:, 1] - h1_diagram[:, 0]
        persistence = persistence[persistence < np.inf]
        
        features = {
            'num_loops': len(persistence),
            'max_persistence': float(np.max(persistence)) if len(persistence) > 0 else 0.0,
            'mean_persistence': float(np.mean(persistence)) if len(persistence) > 0 else 0.0,
            'std_persistence': float(np.std(persistence)) if len(persistence) > 0 else 0.0,
            'total_persistence': float(np.sum(persistence)) if len(persistence) > 0 else 0.0
        }
        
        return features
    
    def compute_persistence_score(
        self,
        correlation_matrix: np.ndarray,
        normalization: str = 'max'
    ) -> float:
        """
        Compute overall persistence score for regime classification.
        
        Higher scores indicate more topological instability/fragmentation.
        
        Parameters
        ----------
        correlation_matrix : np.ndarray
            Correlation matrix
        normalization : str, optional
            How to normalize: 'max', 'mean', or 'total'. Default is 'max'.
        
        Returns
        -------
        float
            Persistence score (normalized to roughly [0, 1])
        """
        result = self.compute_persistence_diagrams(correlation_matrix)
        diagrams = result['dgms']
        
        h0_features = self.extract_h0_features(diagrams)
        h1_features = self.extract_h1_features(diagrams)
        
        # Combine H0 and H1 persistence
        if normalization == 'max':
            score = max(h0_features['max_persistence'], h1_features['max_persistence'])
        elif normalization == 'mean':
            score = (h0_features['mean_persistence'] + h1_features['mean_persistence']) / 2
        elif normalization == 'total':
            score = (h0_features['total_persistence'] + h1_features['total_persistence']) / 2
        else:
            raise ValueError(f"Unknown normalization: {normalization}")
        
        # Normalize to [0, 1] (using max_edge_length as reference)
        # Typical persistence values are in [0, max_edge_length]
        score = min(score / self.max_edge_length, 1.0)
        
        return score
    
    def update_baseline(self, persistence_score: float):
        """
        Update the rolling baseline with a new persistence score.
        
        Parameters
        ----------
        persistence_score : float
            New persistence score to add to history
        """
        self.persistence_history.append(persistence_score)
        
        # Keep only the last baseline_window entries
        if len(self.persistence_history) > self.baseline_window:
            self.persistence_history = self.persistence_history[-self.baseline_window:]
    
    def get_baseline_stats(self) -> Dict[str, float]:
        """
        Get statistics of the rolling baseline.
        
        Returns
        -------
        dict
            Baseline statistics including mean, std, min, max
        """
        if len(self.persistence_history) == 0:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'count': 0
            }
        
        history_array = np.array(self.persistence_history)
        return {
            'mean': float(np.mean(history_array)),
            'std': float(np.std(history_array)),
            'min': float(np.min(history_array)),
            'max': float(np.max(history_array)),
            'count': len(self.persistence_history)
        }
    
    def compute_adaptive_threshold(
        self,
        current_score: float,
        n_std: float = 2.0
    ) -> Dict[str, any]:
        """
        Compute adaptive threshold based on rolling baseline.
        
        A "hole" in the market structure might be normal during high-interest-rate
        environments but a sign of a crash during low-rate environments. This method
        compares current persistence to a rolling baseline to make gating adaptive.
        
        Parameters
        ----------
        current_score : float
            Current persistence score
        n_std : float, optional
            Number of standard deviations for threshold. Default is 2.0.
        
        Returns
        -------
        dict
            Adaptive threshold information including:
            - 'threshold': Adaptive threshold value
            - 'is_anomaly': Whether current score exceeds threshold
            - 'z_score': Z-score relative to baseline
            - 'baseline_mean': Mean of baseline
            - 'baseline_std': Std of baseline
        """
        baseline_stats = self.get_baseline_stats()
        
        # If we don't have enough history, use a conservative default
        if baseline_stats['count'] < 5:
            return {
                'threshold': 0.5,  # Default threshold
                'is_anomaly': current_score > 0.5,
                'z_score': 0.0,
                'baseline_mean': 0.0,
                'baseline_std': 0.0,
                'adaptive': False
            }
        
        # Compute adaptive threshold as mean + n_std * std
        threshold = baseline_stats['mean'] + n_std * baseline_stats['std']
        
        # Compute z-score
        if baseline_stats['std'] > 0:
            z_score = (current_score - baseline_stats['mean']) / baseline_stats['std']
        else:
            z_score = 0.0
        
        return {
            'threshold': threshold,
            'is_anomaly': current_score > threshold,
            'z_score': z_score,
            'baseline_mean': baseline_stats['mean'],
            'baseline_std': baseline_stats['std'],
            'adaptive': True
        }
    
    def classify_regime(
        self,
        correlation_matrix: np.ndarray,
        h0_threshold: float = 0.3,
        h1_threshold: float = 0.2,
        use_adaptive_threshold: bool = True
    ) -> Dict[str, any]:
        """
        Classify market regime based on topological features.
        
        Parameters
        ----------
        correlation_matrix : np.ndarray
            Correlation matrix
        h0_threshold : float, optional
            Threshold for H0 fragmentation. Default is 0.3.
        h1_threshold : float, optional
            Threshold for H1 loops. Default is 0.2.
        use_adaptive_threshold : bool, optional
            If True, use adaptive threshold based on rolling baseline. Default is True.
        
        Returns
        -------
        dict
            Regime classification with details
        """
        result = self.compute_persistence_diagrams(correlation_matrix)
        diagrams = result['dgms']
        
        h0_features = self.extract_h0_features(diagrams)
        h1_features = self.extract_h1_features(diagrams)
        
        # Compute current persistence score
        current_persistence = self.compute_persistence_score(correlation_matrix)
        
        # Update baseline
        self.update_baseline(current_persistence)
        
        # Compute adaptive threshold if requested
        if use_adaptive_threshold:
            adaptive_info = self.compute_adaptive_threshold(current_persistence)
        else:
            adaptive_info = None
        
        # Classification logic
        regime = "Stable"
        confidence = 1.0
        
        # Check for fragmentation (multiple persistent clusters)
        if h0_features['num_components'] > 3 or h0_features['max_persistence'] > h0_threshold:
            regime = "Fragmented"
            confidence = min(h0_features['max_persistence'] / h0_threshold, 1.0)
        
        # Check for loops (feedback cycles)
        if h1_features['num_loops'] > 0 and h1_features['max_persistence'] > h1_threshold:
            if regime == "Fragmented":
                regime = "Stressed"  # Both fragmented and has cycles
            else:
                regime = "Trending"  # Has coherent cycles but not fragmented
            confidence = min(h1_features['max_persistence'] / h1_threshold, 1.0)
        
        result_dict = {
            'regime': regime,
            'confidence': confidence,
            'h0_features': h0_features,
            'h1_features': h1_features,
            'persistence_score': current_persistence
        }
        
        # Add adaptive threshold info if available
        if adaptive_info is not None:
            result_dict['adaptive_threshold'] = adaptive_info
            result_dict['baseline_stats'] = self.get_baseline_stats()
        
        return result_dict
    
    def compute_topological_barcode(
        self,
        correlation_matrix: np.ndarray,
        dimension: int = 1
    ) -> List[Tuple[float, float]]:
        """
        Compute topological barcode for visualization.
        
        Parameters
        ----------
        correlation_matrix : np.ndarray
            Correlation matrix
        dimension : int, optional
            Homology dimension (0 or 1). Default is 1.
        
        Returns
        -------
        list of tuples
            List of (birth, death) intervals for the specified dimension
        """
        result = self.compute_persistence_diagrams(correlation_matrix)
        diagrams = result['dgms']
        
        # Extract diagram for the specified dimension
        if dimension >= len(diagrams):
            return []
        
        dim_diagram = diagrams[dimension]
        
        # Extract birth and death times
        barcode = [(float(p[0]), float(p[1])) for p in dim_diagram if p[1] < np.inf]
        
        return barcode


def create_sample_tda_homology(
    max_edge_length: float = 2.0,
    max_dimension: int = 2
) -> TDAHomology:
    """
    Factory function to create a sample TDA Homology analyzer.
    
    Parameters
    ----------
    max_edge_length : float, optional
        Maximum edge length. Default is 2.0.
    max_dimension : int, optional
        Maximum homology dimension. Default is 2.
    
    Returns
    -------
    TDAHomology
        Configured TDA homology analyzer
    """
    return TDAHomology(
        max_edge_length=max_edge_length,
        max_dimension=max_dimension,
        n_jobs=1
    )
