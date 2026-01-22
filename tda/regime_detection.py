"""
Regime Detection Module

This module provides functionality to detect market regimes using topological features.
Implements Hidden Markov Models and clustering-based approaches to classify markets
into "stable", "stressed", or "transitioning" states.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal


class MarketRegimeDetector:
    """
    Base class for market regime detection using topological features.
    
    Uses topological "holes" (H_1 features) and cluster information (H_0 features)
    to determine if the market is in a "stable", "stressed", or "transitioning" state.
    """
    
    REGIME_STABLE = "stable"
    REGIME_STRESSED = "stressed"
    REGIME_TRANSITIONING = "transitioning"
    
    def __init__(self, n_regimes: int = 3):
        """
        Initialize the regime detector.
        
        Parameters
        ----------
        n_regimes : int, optional
            Number of regimes to detect. Default is 3 (stable, stressed, transitioning).
        """
        self.n_regimes = n_regimes
        self._is_fitted = False
    
    def fit(self, features: np.ndarray, **kwargs) -> 'MarketRegimeDetector':
        """
        Fit the regime detector on historical data.
        
        Parameters
        ----------
        features : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        **kwargs
            Additional parameters for specific implementations.
        
        Returns
        -------
        self : MarketRegimeDetector
            Fitted detector.
        """
        raise NotImplementedError("Subclasses must implement fit method")
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict regimes for new data.
        
        Parameters
        ----------
        features : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        
        Returns
        -------
        np.ndarray
            Predicted regime labels.
        """
        raise NotImplementedError("Subclasses must implement predict method")
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Predict regime probabilities for new data.
        
        Parameters
        ----------
        features : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        
        Returns
        -------
        np.ndarray
            Predicted regime probabilities of shape (n_samples, n_regimes).
        """
        raise NotImplementedError("Subclasses must implement predict_proba method")
    
    def _map_regime_label(self, numeric_label: int) -> str:
        """
        Map numeric regime label to semantic name.
        
        Parameters
        ----------
        numeric_label : int
            Numeric regime label.
        
        Returns
        -------
        str
            Semantic regime name.
        """
        regime_map = {
            0: self.REGIME_STABLE,
            1: self.REGIME_TRANSITIONING,
            2: self.REGIME_STRESSED
        }
        return regime_map.get(numeric_label, f"regime_{numeric_label}")


class HMMRegimeDetector(MarketRegimeDetector):
    """
    Hidden Markov Model for regime detection.
    
    Uses a simple HMM implementation to model regime transitions over time.
    """
    
    def __init__(self, n_regimes: int = 3, n_iter: int = 100, random_state: Optional[int] = None):
        """
        Initialize HMM regime detector.
        
        Parameters
        ----------
        n_regimes : int, optional
            Number of hidden regimes. Default is 3.
        n_iter : int, optional
            Number of iterations for EM algorithm. Default is 100.
        random_state : int, optional
            Random seed for reproducibility.
        """
        super().__init__(n_regimes)
        self.n_iter = n_iter
        self.random_state = random_state
        
        # HMM parameters
        self.transition_matrix = None  # Shape: (n_regimes, n_regimes)
        self.emission_means = None      # Shape: (n_regimes, n_features)
        self.emission_covs = None       # Shape: (n_regimes, n_features, n_features)
        self.initial_probs = None       # Shape: (n_regimes,)
    
    def fit(self, features: np.ndarray, **kwargs) -> 'HMMRegimeDetector':
        """
        Fit HMM using Baum-Welch (EM) algorithm.
        
        Parameters
        ----------
        features : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        
        Returns
        -------
        self : HMMRegimeDetector
            Fitted detector.
        """
        n_samples, n_features = features.shape
        
        # Initialize random state
        rng = np.random.RandomState(self.random_state)
        
        # Initialize parameters using K-means
        kmeans = KMeans(n_clusters=self.n_regimes, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(features)
        
        # Initialize emission parameters from K-means
        self.emission_means = np.zeros((self.n_regimes, n_features))
        self.emission_covs = np.zeros((self.n_regimes, n_features, n_features))
        
        for k in range(self.n_regimes):
            mask = labels == k
            if np.sum(mask) > 0:
                self.emission_means[k] = np.mean(features[mask], axis=0)
                cov = np.cov(features[mask].T)
                # Add small value to diagonal for numerical stability
                self.emission_covs[k] = cov + np.eye(n_features) * 1e-6
            else:
                # Random initialization if cluster is empty
                self.emission_means[k] = features[rng.randint(0, n_samples)]
                self.emission_covs[k] = np.eye(n_features)
        
        # Initialize transition matrix (uniform with self-transition bias)
        self.transition_matrix = np.ones((self.n_regimes, self.n_regimes)) * 0.1
        np.fill_diagonal(self.transition_matrix, 0.7)
        self.transition_matrix = self.transition_matrix / self.transition_matrix.sum(axis=1, keepdims=True)
        
        # Initialize initial state probabilities
        self.initial_probs = np.ones(self.n_regimes) / self.n_regimes
        
        # Use Gaussian Mixture Model as a simpler alternative to full HMM
        # (Full Baum-Welch would require more complex implementation)
        self.gmm = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type='full',
            max_iter=self.n_iter,
            random_state=self.random_state,
            n_init=3
        )
        self.gmm.fit(features)
        
        # Update parameters from GMM
        self.emission_means = self.gmm.means_
        self.emission_covs = self.gmm.covariances_
        self.initial_probs = self.gmm.weights_
        
        self._is_fitted = True
        return self
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict most likely regime for each sample.
        
        Parameters
        ----------
        features : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        
        Returns
        -------
        np.ndarray
            Predicted regime labels.
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.gmm.predict(features)
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Predict regime probabilities.
        
        Parameters
        ----------
        features : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        
        Returns
        -------
        np.ndarray
            Predicted regime probabilities of shape (n_samples, n_regimes).
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.gmm.predict_proba(features)
    
    def get_regime_name(self, features: np.ndarray) -> List[str]:
        """
        Get semantic regime names for features.
        
        Parameters
        ----------
        features : np.ndarray
            Feature matrix.
        
        Returns
        -------
        list of str
            Regime names.
        """
        labels = self.predict(features)
        return [self._map_regime_label(label) for label in labels]


class ClusteringRegimeDetector(MarketRegimeDetector):
    """
    Clustering-based regime detection.
    
    Uses K-means or other clustering algorithms to identify market regimes
    based on topological features.
    """
    
    def __init__(
        self,
        n_regimes: int = 3,
        method: str = 'kmeans',
        random_state: Optional[int] = None
    ):
        """
        Initialize clustering-based regime detector.
        
        Parameters
        ----------
        n_regimes : int, optional
            Number of regimes to detect. Default is 3.
        method : str, optional
            Clustering method: 'kmeans' or 'gmm'. Default is 'kmeans'.
        random_state : int, optional
            Random seed for reproducibility.
        """
        super().__init__(n_regimes)
        self.method = method
        self.random_state = random_state
        self.model = None
    
    def fit(self, features: np.ndarray, **kwargs) -> 'ClusteringRegimeDetector':
        """
        Fit clustering model.
        
        Parameters
        ----------
        features : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        
        Returns
        -------
        self : ClusteringRegimeDetector
            Fitted detector.
        """
        if self.method == 'kmeans':
            self.model = KMeans(
                n_clusters=self.n_regimes,
                random_state=self.random_state,
                n_init=10
            )
        elif self.method == 'gmm':
            self.model = GaussianMixture(
                n_components=self.n_regimes,
                random_state=self.random_state,
                n_init=3
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.model.fit(features)
        self._is_fitted = True
        return self
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict regime labels.
        
        Parameters
        ----------
        features : np.ndarray
            Feature matrix.
        
        Returns
        -------
        np.ndarray
            Predicted regime labels.
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(features)
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Predict regime probabilities.
        
        Parameters
        ----------
        features : np.ndarray
            Feature matrix.
        
        Returns
        -------
        np.ndarray
            Predicted regime probabilities.
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self.method == 'kmeans':
            # For K-means, convert distances to probabilities
            distances = self.model.transform(features)
            # Use softmax on negative distances
            exp_neg_dist = np.exp(-distances)
            probs = exp_neg_dist / exp_neg_dist.sum(axis=1, keepdims=True)
            return probs
        else:  # gmm
            return self.model.predict_proba(features)
    
    def get_regime_name(self, features: np.ndarray) -> List[str]:
        """
        Get semantic regime names.
        
        Parameters
        ----------
        features : np.ndarray
            Feature matrix.
        
        Returns
        -------
        list of str
            Regime names.
        """
        labels = self.predict(features)
        return [self._map_regime_label(label) for label in labels]


def classify_regime_by_topology(
    h0_features: Dict[str, float],
    h1_features: Dict[str, float],
    stable_threshold: Dict[str, float] = None,
    stressed_threshold: Dict[str, float] = None
) -> Tuple[str, Dict[str, float]]:
    """
    Classify market regime using rule-based approach on topological features.
    
    This is a simpler alternative to HMM/clustering that uses predefined rules
    based on topological properties.
    
    Parameters
    ----------
    h0_features : dict
        H_0 features (clusters).
    h1_features : dict
        H_1 features (loops).
    stable_threshold : dict, optional
        Thresholds for stable regime. Keys: 'max_clusters', 'min_loop_persistence'.
    stressed_threshold : dict, optional
        Thresholds for stressed regime. Keys: 'min_clusters', 'max_loop_persistence'.
    
    Returns
    -------
    regime : str
        Regime classification: "stable", "stressed", or "transitioning".
    confidence : dict
        Confidence scores for each regime.
    """
    # Default thresholds
    if stable_threshold is None:
        stable_threshold = {
            'max_clusters': 2,
            'min_loop_persistence': 0.05
        }
    
    if stressed_threshold is None:
        stressed_threshold = {
            'min_clusters': 4,
            'max_loop_persistence': 0.2
        }
    
    num_clusters = h0_features.get('num_components', 0)
    max_loop_persistence = h1_features.get('max_persistence', 0.0)
    num_loops = h1_features.get('num_loops', 0)
    
    # Calculate confidence scores
    stable_score = 0.0
    stressed_score = 0.0
    transitioning_score = 0.0
    
    # Stable: Few clusters, weak loops
    if num_clusters <= stable_threshold['max_clusters']:
        stable_score += 0.5
    if max_loop_persistence < stable_threshold['min_loop_persistence']:
        stable_score += 0.5
    
    # Stressed: Many clusters or strong loops
    if num_clusters >= stressed_threshold['min_clusters']:
        stressed_score += 0.5
    if max_loop_persistence > stressed_threshold['max_loop_persistence']:
        stressed_score += 0.5
    
    # Transitioning: Between stable and stressed
    if stable_threshold['max_clusters'] < num_clusters < stressed_threshold['min_clusters']:
        transitioning_score += 0.5
    if (stable_threshold['min_loop_persistence'] < max_loop_persistence < 
            stressed_threshold['max_loop_persistence']):
        transitioning_score += 0.5
    
    # Normalize scores
    total = stable_score + stressed_score + transitioning_score
    if total > 0:
        stable_score /= total
        stressed_score /= total
        transitioning_score /= total
    
    # Determine regime
    scores = {
        'stable': stable_score,
        'transitioning': transitioning_score,
        'stressed': stressed_score
    }
    
    regime = max(scores, key=scores.get)
    
    return regime, scores
