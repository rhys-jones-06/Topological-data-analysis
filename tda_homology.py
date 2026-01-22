"""
TDA Homology Module using Giotto-TDA

Implements persistent homology analysis to identify market fragmentation
and structural stability using topological data analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy, Amplitude


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
        homology_dimensions: Tuple[int, ...] = (0, 1, 2),
        n_jobs: int = 1
    ):
        """
        Initialize the TDA Homology analyzer.
        
        Parameters
        ----------
        max_edge_length : float, optional
            Maximum edge length for Vietoris-Rips. Default is 2.0.
        homology_dimensions : tuple, optional
            Homology dimensions to compute. Default is (0, 1, 2).
        n_jobs : int, optional
            Number of parallel jobs. Default is 1.
        """
        self.max_edge_length = max_edge_length
        self.homology_dimensions = homology_dimensions
        self.n_jobs = n_jobs
        
        # Initialize Giotto-TDA pipeline
        self.vr_persistence = VietorisRipsPersistence(
            homology_dimensions=homology_dimensions,
            max_edge_length=max_edge_length,
            n_jobs=n_jobs
        )
        
        self.persistence_entropy = PersistenceEntropy(n_jobs=n_jobs)
        self.persistence_amplitude = Amplitude(metric='persistence_image', n_jobs=n_jobs)
        
        self.last_diagrams = None
    
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
    ) -> np.ndarray:
        """
        Compute persistence diagrams from correlation matrix.
        
        Parameters
        ----------
        correlation_matrix : np.ndarray
            Correlation matrix between assets
        
        Returns
        -------
        np.ndarray
            Persistence diagrams (shape: (1, n_points, 3))
            Each point: (birth, death, homology_dimension)
        """
        # Convert to distance matrix
        distance_matrix = self.correlation_to_distance(correlation_matrix)
        
        # Giotto-TDA expects point clouds, but we can use distance matrix
        # Reshape to (n_samples=1, n_points, n_points) for distance matrix
        distance_matrix_3d = distance_matrix[np.newaxis, :, :]
        
        # Compute persistence diagrams
        diagrams = self.vr_persistence.fit_transform_plot(
            distance_matrix_3d,
            sample=0
        )
        
        self.last_diagrams = diagrams
        return diagrams
    
    def extract_h0_features(self, diagrams: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Extract features from H0 persistence (connected components/clusters).
        
        Parameters
        ----------
        diagrams : np.ndarray, optional
            Persistence diagrams. If None, uses last computed.
        
        Returns
        -------
        dict
            H0 features: num_components, max_persistence, mean_persistence, etc.
        """
        if diagrams is None:
            diagrams = self.last_diagrams
        
        if diagrams is None:
            raise ValueError("No diagrams available. Compute first.")
        
        # Extract H0 points (dimension 0)
        h0_points = diagrams[diagrams[:, :, 2] == 0]
        
        if len(h0_points) == 0:
            return {
                'num_components': 0,
                'max_persistence': 0.0,
                'mean_persistence': 0.0,
                'std_persistence': 0.0,
                'total_persistence': 0.0
            }
        
        # Persistence = death - birth
        persistence = h0_points[:, :, 1] - h0_points[:, :, 0]
        persistence = persistence[persistence < np.inf]  # Remove infinite bars
        
        features = {
            'num_components': len(persistence),
            'max_persistence': float(np.max(persistence)) if len(persistence) > 0 else 0.0,
            'mean_persistence': float(np.mean(persistence)) if len(persistence) > 0 else 0.0,
            'std_persistence': float(np.std(persistence)) if len(persistence) > 0 else 0.0,
            'total_persistence': float(np.sum(persistence)) if len(persistence) > 0 else 0.0
        }
        
        return features
    
    def extract_h1_features(self, diagrams: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Extract features from H1 persistence (loops/cycles).
        
        Parameters
        ----------
        diagrams : np.ndarray, optional
            Persistence diagrams. If None, uses last computed.
        
        Returns
        -------
        dict
            H1 features: num_loops, max_persistence, mean_persistence, etc.
        """
        if diagrams is None:
            diagrams = self.last_diagrams
        
        if diagrams is None:
            raise ValueError("No diagrams available. Compute first.")
        
        # Extract H1 points (dimension 1)
        h1_points = diagrams[diagrams[:, :, 2] == 1]
        
        if len(h1_points) == 0:
            return {
                'num_loops': 0,
                'max_persistence': 0.0,
                'mean_persistence': 0.0,
                'std_persistence': 0.0,
                'total_persistence': 0.0
            }
        
        # Persistence = death - birth
        persistence = h1_points[:, :, 1] - h1_points[:, :, 0]
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
        diagrams = self.compute_persistence_diagrams(correlation_matrix)
        
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
    
    def classify_regime(
        self,
        correlation_matrix: np.ndarray,
        h0_threshold: float = 0.3,
        h1_threshold: float = 0.2
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
        
        Returns
        -------
        dict
            Regime classification with details
        """
        diagrams = self.compute_persistence_diagrams(correlation_matrix)
        
        h0_features = self.extract_h0_features(diagrams)
        h1_features = self.extract_h1_features(diagrams)
        
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
        
        return {
            'regime': regime,
            'confidence': confidence,
            'h0_features': h0_features,
            'h1_features': h1_features,
            'persistence_score': self.compute_persistence_score(correlation_matrix)
        }
    
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
        diagrams = self.compute_persistence_diagrams(correlation_matrix)
        
        # Extract points for the specified dimension
        dim_points = diagrams[diagrams[:, :, 2] == dimension]
        
        # Extract birth and death times
        barcode = [(float(p[0]), float(p[1])) for p in dim_points if p[1] < np.inf]
        
        return barcode


def create_sample_tda_homology(
    max_edge_length: float = 2.0,
    homology_dimensions: Tuple[int, ...] = (0, 1, 2)
) -> TDAHomology:
    """
    Factory function to create a sample TDA Homology analyzer.
    
    Parameters
    ----------
    max_edge_length : float, optional
        Maximum edge length. Default is 2.0.
    homology_dimensions : tuple, optional
        Homology dimensions to compute. Default is (0, 1, 2).
    
    Returns
    -------
    TDAHomology
        Configured TDA homology analyzer
    """
    return TDAHomology(
        max_edge_length=max_edge_length,
        homology_dimensions=homology_dimensions,
        n_jobs=1
    )
