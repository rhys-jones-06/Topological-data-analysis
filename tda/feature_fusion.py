"""
Feature Fusion Module

This module provides functionality to combine local residuals (from graph diffusion)
with global features (from persistent homology) into a unified feature representation
for the Market Mutual Model.
"""

import numpy as np
from typing import Dict, Union, Optional, List
from sklearn.preprocessing import StandardScaler


class FeatureFusion:
    """
    Combines local residuals with global topological features.
    
    This class fuses:
    1. Local features: Residuals from graph diffusion (heat kernel results)
    2. Global features: Persistent homology features (H_0, H_1, etc.)
    
    The fusion process includes normalization and optional weighting of feature groups.
    """
    
    def __init__(
        self,
        normalize: bool = True,
        local_weight: float = 0.5,
        global_weight: float = 0.5
    ):
        """
        Initialize the FeatureFusion module.
        
        Parameters
        ----------
        normalize : bool, optional
            If True, normalize features using StandardScaler. Default is True.
        local_weight : float, optional
            Weight for local features (0-1). Default is 0.5.
        global_weight : float, optional
            Weight for global features (0-1). Default is 0.5.
        """
        self.normalize = normalize
        self.local_weight = local_weight
        self.global_weight = global_weight
        self.local_scaler = StandardScaler() if normalize else None
        self.global_scaler = StandardScaler() if normalize else None
        self._is_fitted = False
    
    def fit(
        self,
        local_features: np.ndarray,
        global_features: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> 'FeatureFusion':
        """
        Fit the feature fusion scalers on training data.
        
        Parameters
        ----------
        local_features : np.ndarray
            Local features (e.g., residuals, diffusion patterns).
            Shape: (n_samples, n_local_features).
        global_features : np.ndarray or dict
            Global features from topological analysis.
            If dict, should contain feature vectors for each dimension.
            Shape: (n_samples, n_global_features).
        
        Returns
        -------
        self : FeatureFusion
            Fitted instance.
        """
        # Handle 1D local features
        if local_features.ndim == 1:
            local_features = local_features.reshape(-1, 1)
        
        n_samples = local_features.shape[0]
        
        # Convert global features dict to array if needed
        global_array = self._flatten_global_features(global_features, n_samples)
        
        if self.normalize:
            # Fit scalers
            if global_array.ndim == 1:
                global_array = global_array.reshape(-1, 1)
                
            self.local_scaler.fit(local_features)
            self.global_scaler.fit(global_array)
        
        self._is_fitted = True
        return self
    
    def transform(
        self,
        local_features: np.ndarray,
        global_features: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> np.ndarray:
        """
        Transform and fuse local and global features.
        
        Parameters
        ----------
        local_features : np.ndarray
            Local features to transform.
            Shape: (n_samples, n_local_features).
        global_features : np.ndarray or dict
            Global features to transform.
        
        Returns
        -------
        np.ndarray
            Fused feature vector.
            Shape: (n_samples, n_local_features + n_global_features).
        """
        # Handle 1D local arrays
        if local_features.ndim == 1:
            local_features = local_features.reshape(-1, 1)
        
        n_samples = local_features.shape[0]
        
        # Convert global features dict to array if needed
        global_array = self._flatten_global_features(global_features, n_samples)
        
        # Handle 1D global arrays
        if global_array.ndim == 1:
            global_array = global_array.reshape(-1, 1)
        
        # Normalize if fitted
        if self.normalize and self._is_fitted:
            local_normalized = self.local_scaler.transform(local_features)
            global_normalized = self.global_scaler.transform(global_array)
        else:
            local_normalized = local_features
            global_normalized = global_array
        
        # Apply weights
        local_weighted = local_normalized * self.local_weight
        global_weighted = global_normalized * self.global_weight
        
        # Concatenate features
        fused = np.concatenate([local_weighted, global_weighted], axis=1)
        
        return fused
    
    def fit_transform(
        self,
        local_features: np.ndarray,
        global_features: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> np.ndarray:
        """
        Fit scalers and transform features in one step.
        
        Parameters
        ----------
        local_features : np.ndarray
            Local features.
        global_features : np.ndarray or dict
            Global features.
        
        Returns
        -------
        np.ndarray
            Fused feature vector.
        """
        self.fit(local_features, global_features)
        return self.transform(local_features, global_features)
    
    def _flatten_global_features(
        self,
        global_features: Union[np.ndarray, Dict[str, np.ndarray]],
        n_samples: int = 1
    ) -> np.ndarray:
        """
        Convert global features from dict format to array.
        
        Parameters
        ----------
        global_features : np.ndarray or dict
            Global features, either as array or dict of arrays.
        n_samples : int
            Number of samples to repeat features for.
        
        Returns
        -------
        np.ndarray
            Flattened global features.
        """
        if isinstance(global_features, dict):
            # Extract features from dictionary
            feature_list = []
            for key in sorted(global_features.keys()):
                value = global_features[key]
                if isinstance(value, dict):
                    # If value is a dict (e.g., statistical features), extract values
                    feature_list.extend(list(value.values()))
                elif isinstance(value, np.ndarray):
                    # If value is an array, flatten it
                    feature_list.extend(value.flatten())
                else:
                    # If value is a scalar, add it directly
                    feature_list.append(float(value))
            
            # Create array and repeat for n_samples
            feature_array = np.array(feature_list).reshape(1, -1)
            if n_samples > 1:
                feature_array = np.repeat(feature_array, n_samples, axis=0)
            return feature_array
        else:
            # Already an array
            if global_features.ndim == 1:
                feature_array = global_features.reshape(1, -1)
                if n_samples > 1:
                    feature_array = np.repeat(feature_array, n_samples, axis=0)
                return feature_array
            return global_features
    
    def get_feature_names(
        self,
        local_feature_names: Optional[List[str]] = None,
        global_feature_names: Optional[List[str]] = None
    ) -> List[str]:
        """
        Get names of fused features.
        
        Parameters
        ----------
        local_feature_names : list of str, optional
            Names of local features.
        global_feature_names : list of str, optional
            Names of global features.
        
        Returns
        -------
        list of str
            Names of all fused features.
        """
        names = []
        
        if local_feature_names:
            names.extend([f"local_{name}" for name in local_feature_names])
        else:
            n_local = self.local_scaler.n_features_in_ if self._is_fitted else 0
            names.extend([f"local_{i}" for i in range(n_local)])
        
        if global_feature_names:
            names.extend([f"global_{name}" for name in global_feature_names])
        else:
            n_global = self.global_scaler.n_features_in_ if self._is_fitted else 0
            names.extend([f"global_{i}" for i in range(n_global)])
        
        return names


def create_local_features(
    residuals: np.ndarray,
    diffusion_state: np.ndarray,
    laplacian_spectrum: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Create local feature vector from residuals and diffusion patterns.
    
    Parameters
    ----------
    residuals : np.ndarray
        Residuals from price model (actual - expected returns).
        Shape: (n_assets,) or (n_samples, n_assets).
    diffusion_state : np.ndarray
        State after heat kernel diffusion or random walk.
        Shape: (n_assets,) or (n_samples, n_assets).
    laplacian_spectrum : np.ndarray, optional
        Eigenvalues of the graph Laplacian (spectral features).
        Shape: (n_assets,).
    
    Returns
    -------
    np.ndarray
        Local feature vector combining residuals, diffusion, and spectral info.
    """
    features = []
    
    # Handle 1D or 2D inputs
    if residuals.ndim == 1:
        residuals = residuals.reshape(1, -1)
    if diffusion_state.ndim == 1:
        diffusion_state = diffusion_state.reshape(1, -1)
    
    # Add residuals
    features.append(residuals)
    
    # Add diffusion state
    features.append(diffusion_state)
    
    # Add spectral features if provided
    if laplacian_spectrum is not None:
        if laplacian_spectrum.ndim == 1:
            laplacian_spectrum = laplacian_spectrum.reshape(1, -1)
        # Repeat for each sample if needed
        if laplacian_spectrum.shape[0] == 1 and residuals.shape[0] > 1:
            laplacian_spectrum = np.repeat(laplacian_spectrum, residuals.shape[0], axis=0)
        features.append(laplacian_spectrum)
    
    # Concatenate all features
    return np.concatenate(features, axis=1)


def create_global_features_vector(
    h0_features: Dict[str, float],
    h1_features: Dict[str, float],
    additional_features: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """
    Create global feature vector from topological features.
    
    Parameters
    ----------
    h0_features : dict
        H_0 topological features (clusters).
    h1_features : dict
        H_1 topological features (loops/cycles).
    additional_features : dict, optional
        Additional global features to include.
    
    Returns
    -------
    np.ndarray
        Global feature vector.
    """
    features = []
    
    # Add H_0 features
    features.extend([
        h0_features.get('num_components', 0),
        h0_features.get('max_persistence', 0.0),
        h0_features.get('mean_persistence', 0.0),
        h0_features.get('std_persistence', 0.0),
        h0_features.get('persistence_entropy', 0.0)
    ])
    
    # Add H_1 features
    features.extend([
        h1_features.get('num_loops', 0),
        h1_features.get('max_persistence', 0.0),
        h1_features.get('mean_persistence', 0.0),
        h1_features.get('std_persistence', 0.0),
        h1_features.get('persistence_entropy', 0.0),
        h1_features.get('total_persistence', 0.0)
    ])
    
    # Add additional features if provided
    if additional_features:
        features.extend(list(additional_features.values()))
    
    return np.array(features).reshape(1, -1)
