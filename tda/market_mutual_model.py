"""
Market Mutual Model

Main inference model that integrates feature fusion, regime detection,
and risk management for topological market analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, Tuple, List

from .feature_fusion import FeatureFusion, create_local_features, create_global_features_vector
from .regime_detection import (
    MarketRegimeDetector, HMMRegimeDetector, ClusteringRegimeDetector,
    classify_regime_by_topology
)
from .risk_management import RiskManager
from .weighted_graph import (
    compute_rolling_correlation, compute_graph_laplacian,
    simulate_heat_kernel, simulate_random_walk
)
from .tda_layer import (
    compute_persistent_homology, extract_h0_features, extract_h1_features
)
from .residual_analysis import compute_residuals


class MarketMutualModel:
    """
    Market Mutual Model for topological market analysis and inference.
    
    This model combines:
    1. Feature Fusion: Local residuals + global topological features
    2. Regime Detection: HMM or clustering to detect market states
    3. Risk Management: Position sizing based on persistence
    
    Provides a .predict() interface for integration with ensemble models.
    """
    
    def __init__(
        self,
        regime_detector_type: str = 'hmm',
        n_regimes: int = 3,
        feature_fusion_params: Optional[Dict] = None,
        risk_manager_params: Optional[Dict] = None,
        random_state: Optional[int] = None
    ):
        """
        Initialize the Market Mutual Model.
        
        Parameters
        ----------
        regime_detector_type : str, optional
            Type of regime detector: 'hmm', 'clustering', or 'rule_based'.
            Default is 'hmm'.
        n_regimes : int, optional
            Number of regimes to detect. Default is 3.
        feature_fusion_params : dict, optional
            Parameters for FeatureFusion.
        risk_manager_params : dict, optional
            Parameters for RiskManager.
        random_state : int, optional
            Random seed for reproducibility.
        """
        self.regime_detector_type = regime_detector_type
        self.n_regimes = n_regimes
        self.random_state = random_state
        
        # Initialize feature fusion
        fusion_params = feature_fusion_params or {}
        self.feature_fusion = FeatureFusion(**fusion_params)
        
        # Initialize regime detector
        if regime_detector_type == 'hmm':
            self.regime_detector = HMMRegimeDetector(
                n_regimes=n_regimes,
                random_state=random_state
            )
        elif regime_detector_type == 'clustering':
            self.regime_detector = ClusteringRegimeDetector(
                n_regimes=n_regimes,
                random_state=random_state
            )
        elif regime_detector_type == 'rule_based':
            self.regime_detector = None  # Use rule-based classification
        else:
            raise ValueError(f"Unknown regime detector type: {regime_detector_type}")
        
        # Initialize risk manager
        risk_params = risk_manager_params or {}
        self.risk_manager = RiskManager(**risk_params)
        
        # State variables
        self._is_fitted = False
        self.feature_names = None
    
    def fit(
        self,
        asset_returns: pd.DataFrame,
        window: int = 20,
        expected_returns: Optional[pd.DataFrame] = None
    ) -> 'MarketMutualModel':
        """
        Fit the model on historical data.
        
        Parameters
        ----------
        asset_returns : pd.DataFrame
            Historical asset returns. Shape: (n_periods, n_assets).
        window : int, optional
            Rolling window for correlation. Default is 20.
        expected_returns : pd.DataFrame, optional
            Expected returns for residual calculation.
            If None, uses rolling mean.
        
        Returns
        -------
        self : MarketMutualModel
            Fitted model.
        """
        n_periods = len(asset_returns)
        
        # Compute expected returns if not provided
        if expected_returns is None:
            expected_returns = asset_returns.rolling(window=10, min_periods=1).mean()
        
        # Prepare training data
        all_local_features = []
        all_global_features = []
        
        # Compute rolling correlation
        correlation_matrices = compute_rolling_correlation(asset_returns, window=window)
        
        for i in range(len(correlation_matrices)):
            # Get current correlation matrix
            corr_df = correlation_matrices[i]
            timestamp = corr_df['timestamp']
            corr_matrix = corr_df.drop(columns=['timestamp']).values
            
            # Get index for this timestamp
            idx = i + window - 1
            if idx >= n_periods:
                break
            
            # Compute local features
            residuals = compute_residuals(
                asset_returns.iloc[idx],
                expected_returns.iloc[idx]
            ).values
            
            # Compute graph and diffusion
            adjacency = np.abs(corr_matrix)
            np.fill_diagonal(adjacency, 0)
            laplacian = compute_graph_laplacian(adjacency)
            
            # Simulate diffusion
            diffusion = simulate_heat_kernel(laplacian, time=1.0, initial_state=residuals)
            
            # Create local feature vector
            local_feat = create_local_features(residuals, diffusion)
            
            # Compute global features
            persistence = compute_persistent_homology(
                corr_matrix,
                max_dimension=1,
                use_correlation=False
            )
            
            h0_feat = extract_h0_features(persistence['dgms'][0])
            h1_feat = extract_h1_features(persistence['dgms'][1])
            
            global_feat = create_global_features_vector(h0_feat, h1_feat)
            
            all_local_features.append(local_feat)
            all_global_features.append(global_feat)
        
        # Convert to arrays
        local_features = np.vstack(all_local_features)
        global_features = np.vstack(all_global_features)
        
        # Fit feature fusion
        fused_features = self.feature_fusion.fit_transform(local_features, global_features)
        
        # Fit regime detector
        if self.regime_detector is not None:
            self.regime_detector.fit(fused_features)
        
        self._is_fitted = True
        return self
    
    def predict(
        self,
        asset_returns: pd.DataFrame,
        expected_returns: Optional[pd.DataFrame] = None,
        return_details: bool = False
    ) -> Union[np.ndarray, Dict]:
        """
        Generate trading signals/confidence scores.
        
        Parameters
        ----------
        asset_returns : pd.DataFrame
            Recent asset returns for prediction.
            Shape: (n_periods, n_assets).
        expected_returns : pd.DataFrame, optional
            Expected returns. If None, uses rolling mean.
        return_details : bool, optional
            If True, return detailed information. Default is False.
        
        Returns
        -------
        signals : np.ndarray or dict
            If return_details=False: Array of confidence scores (-1 to 1).
            If return_details=True: Dictionary with signals and metadata.
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get the most recent data
        if len(asset_returns) < 20:
            raise ValueError("Need at least 20 periods of data for prediction")
        
        # Compute correlation matrix
        corr_matrix = asset_returns.iloc[-20:].corr().values
        
        # Compute expected returns if not provided
        if expected_returns is None:
            expected_returns = asset_returns.rolling(window=10, min_periods=1).mean()
        
        # Get latest residuals
        residuals = compute_residuals(
            asset_returns.iloc[-1],
            expected_returns.iloc[-1]
        ).values
        
        # Compute graph and diffusion
        adjacency = np.abs(corr_matrix)
        np.fill_diagonal(adjacency, 0)
        laplacian = compute_graph_laplacian(adjacency)
        diffusion = simulate_heat_kernel(laplacian, time=1.0, initial_state=residuals)
        
        # Create local features
        local_features = create_local_features(residuals, diffusion)
        
        # Compute global features
        persistence = compute_persistent_homology(
            corr_matrix,
            max_dimension=1,
            use_correlation=False  # corr_matrix is already a correlation, not time series
        )
        
        h0_features = extract_h0_features(persistence['dgms'][0])
        h1_features = extract_h1_features(persistence['dgms'][1])
        global_features = create_global_features_vector(h0_features, h1_features)
        
        # Fuse features
        fused_features = self.feature_fusion.transform(local_features, global_features)
        
        # Detect regime
        if self.regime_detector is not None:
            regime_probs = self.regime_detector.predict_proba(fused_features)
            regime_label = self.regime_detector.predict(fused_features)[0]
            regime = self.regime_detector._map_regime_label(regime_label)
            regime_confidence = np.max(regime_probs[0])
        else:
            # Use rule-based classification
            regime, regime_scores = classify_regime_by_topology(h0_features, h1_features)
            regime_confidence = regime_scores[regime]
        
        # Calculate persistence score
        persistence_score = self.risk_manager.calculate_persistence_score(
            h0_features, h1_features
        )
        
        # Generate signals based on residuals and diffusion
        # Positive residual + positive diffusion = buy signal
        # Negative residual + negative diffusion = sell signal
        raw_signals = residuals * diffusion
        
        # Normalize signals
        signal_strength = np.tanh(raw_signals / (np.std(raw_signals) + 1e-8))
        
        # Scale signals by persistence and regime
        scaled_signals = np.array([
            self.risk_manager.calculate_position_size(
                signal,
                persistence_score,
                regime,
                regime_confidence
            )
            for signal in signal_strength
        ])
        
        if return_details:
            return {
                'signals': scaled_signals,
                'raw_signals': signal_strength,
                'regime': regime,
                'regime_confidence': regime_confidence,
                'persistence_score': persistence_score,
                'h0_features': h0_features,
                'h1_features': h1_features,
                'residuals': residuals,
                'diffusion': diffusion
            }
        else:
            return scaled_signals
    
    def predict_single_signal(
        self,
        asset_returns: pd.DataFrame,
        expected_returns: Optional[pd.DataFrame] = None
    ) -> float:
        """
        Generate a single aggregate signal for the entire market.
        
        Parameters
        ----------
        asset_returns : pd.DataFrame
            Recent asset returns.
        expected_returns : pd.DataFrame, optional
            Expected returns.
        
        Returns
        -------
        float
            Aggregate market signal (-1 to 1).
        """
        signals = self.predict(asset_returns, expected_returns, return_details=False)
        
        # Return mean signal across all assets
        return np.mean(signals)
    
    def get_risk_adjusted_positions(
        self,
        asset_returns: pd.DataFrame,
        base_signals: np.ndarray,
        expected_returns: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """
        Adjust base signals with topological risk management.
        
        This method is designed for ensemble integration, where base_signals
        come from another model (e.g., neural network).
        
        Parameters
        ----------
        asset_returns : pd.DataFrame
            Recent asset returns.
        base_signals : np.ndarray
            Base trading signals from another model.
        expected_returns : pd.DataFrame, optional
            Expected returns.
        
        Returns
        -------
        np.ndarray
            Risk-adjusted position sizes.
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get regime and persistence info
        details = self.predict(asset_returns, expected_returns, return_details=True)
        
        regime = details['regime']
        regime_confidence = details['regime_confidence']
        persistence_score = details['persistence_score']
        
        # Adjust each signal
        adjusted_signals = np.array([
            self.risk_manager.calculate_position_size(
                signal,
                persistence_score,
                regime,
                regime_confidence
            )
            for signal in base_signals
        ])
        
        return adjusted_signals
