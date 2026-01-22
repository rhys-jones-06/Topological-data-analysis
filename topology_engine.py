"""
The TopologyEngine - Multi-Scale Structural Alpha Engine

Integrates Neural Network, Graph Diffusion, and TDA Homology into a unified
ensemble system using feature-level fusion with topological gating.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, List
import json
from datetime import datetime

from nn_strategy import NeuralNetStrategy
from graph_diffusion import GraphDiffusion
from tda_homology import TDAHomology


class DataOrchestrator:
    """
    Central data orchestrator that feeds synchronized OHLCV data
    to all three modules (NN, Graph, TDA) simultaneously.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize the Data Orchestrator.
        
        Parameters
        ----------
        window_size : int, optional
            Size of rolling window for correlation computation. Default is 100.
        """
        self.window_size = window_size
        self.data_cache = {}
    
    def prepare_data(
        self,
        ohlcv_data: pd.DataFrame,
        lookback: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Prepare and synchronize data for all modules.
        
        Parameters
        ----------
        ohlcv_data : pd.DataFrame
            OHLCV data for multiple assets.
            Expected columns: asset names or multi-index with (asset, field)
        lookback : int, optional
            Number of periods to use. If None, uses all available.
        
        Returns
        -------
        dict
            Synchronized data package containing:
            - 'ohlcv': Raw OHLCV data
            - 'returns': Returns matrix
            - 'correlation': Correlation matrix
            - 'timestamps': Timestamp array
        """
        if lookback is not None:
            ohlcv_data = ohlcv_data.iloc[-lookback:]
        
        # Extract close prices for each asset
        if isinstance(ohlcv_data.columns, pd.MultiIndex):
            # Multi-index format: (asset, field)
            close_prices = ohlcv_data.xs('Close', level=1, axis=1)
        else:
            # Assume all columns are close prices
            close_prices = ohlcv_data
        
        # Compute returns
        returns = close_prices.pct_change().dropna()
        
        # Compute correlation matrix
        if len(returns) >= self.window_size:
            correlation_matrix = returns.iloc[-self.window_size:].corr().values
        else:
            correlation_matrix = returns.corr().values
        
        data_package = {
            'ohlcv': ohlcv_data,
            'close_prices': close_prices,
            'returns': returns,
            'correlation': correlation_matrix,
            'timestamps': ohlcv_data.index.values if hasattr(ohlcv_data.index, 'values') else None,
            'asset_names': list(close_prices.columns)
        }
        
        return data_package


class GatingNetwork:
    """
    Gating Network / Ensemble class that combines NN predictions
    with TDA persistence scores using topological gating logic.
    
    If TDA detects high instability (H1 persistence > threshold),
    the engine forces a Neutral/Cash signal regardless of NN prediction.
    """
    
    def __init__(
        self,
        instability_threshold: float = 0.5,
        confidence_decay: float = 0.5,
        regime_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the Gating Network.
        
        Parameters
        ----------
        instability_threshold : float, optional
            Persistence score threshold for forcing neutral. Default is 0.5.
        confidence_decay : float, optional
            Decay factor for confidence when in unstable regime. Default is 0.5.
        regime_weights : dict, optional
            Weights for different regimes. If None, uses defaults.
        """
        self.instability_threshold = instability_threshold
        self.confidence_decay = confidence_decay
        
        if regime_weights is None:
            regime_weights = {
                'Stable': 1.0,
                'Trending': 0.8,
                'Fragmented': 0.3,
                'Stressed': 0.1
            }
        self.regime_weights = regime_weights
    
    def combine_signals(
        self,
        nn_proba: float,
        persistence_score: float,
        regime: str,
        graph_leakage: Optional[float] = None
    ) -> Dict[str, any]:
        """
        Combine NN probability with TDA persistence using gating logic.
        
        Parameters
        ----------
        nn_proba : float
            Neural network probability of upward movement (0 to 1)
        persistence_score : float
            TDA persistence score (0 to 1, higher = more unstable)
        regime : str
            Market regime classification
        graph_leakage : float, optional
            Graph leakage score (0 to 1)
        
        Returns
        -------
        dict
            Combined signal with final_signal, confidence_score, etc.
        """
        # Check gating condition
        if persistence_score > self.instability_threshold:
            # Force neutral/cash due to high topological instability
            final_signal = "NEUTRAL"
            confidence_score = 0.2  # Low confidence
            reason = "TopologicalInstability"
        else:
            # Normal operation: use NN prediction
            # Apply regime-based weighting
            regime_weight = self.regime_weights.get(regime, 0.5)
            
            # Adjust NN probability based on regime
            adjusted_proba = 0.5 + (nn_proba - 0.5) * regime_weight
            
            # Determine signal
            if adjusted_proba > 0.55:
                final_signal = "LONG"
            elif adjusted_proba < 0.45:
                final_signal = "SHORT"
            else:
                final_signal = "NEUTRAL"
            
            # Confidence score combines NN confidence and topological stability
            nn_confidence = abs(nn_proba - 0.5) * 2  # 0 to 1
            tda_confidence = 1.0 - persistence_score  # Inverse of instability
            
            # Weighted combination
            confidence_score = (nn_confidence * 0.6 + tda_confidence * 0.4) * regime_weight
            reason = "Normal"
        
        # Include graph leakage in confidence if available
        if graph_leakage is not None:
            leakage_penalty = graph_leakage * 0.2
            confidence_score = max(confidence_score * (1 - leakage_penalty), 0.0)
        
        return {
            'final_signal': final_signal,
            'confidence_score': min(confidence_score, 1.0),
            'nn_proba': nn_proba,
            'persistence_score': persistence_score,
            'regime': regime,
            'reason': reason,
            'graph_leakage': graph_leakage
        }
    
    def compute_confidence_interval(
        self,
        confidence_score: float,
        n_samples: int = 100
    ) -> List[float]:
        """
        Compute confidence interval around the signal.
        
        Parameters
        ----------
        confidence_score : float
            Base confidence score
        n_samples : int, optional
            Sample size for estimation. Default is 100.
        
        Returns
        -------
        list
            [lower_bound, upper_bound]
        """
        # Simple confidence interval based on score
        # Higher confidence = narrower interval
        width = (1.0 - confidence_score) * 0.5
        
        return [max(0.0, confidence_score - width), min(1.0, confidence_score + width)]


class TopologyEngine:
    """
    The main TopologyEngine class integrating all components.
    
    Orchestrates data flow, runs NN/Graph/TDA modules, applies gating,
    and produces risk-adjusted alpha map as output.
    """
    
    def __init__(
        self,
        nn_strategy: Optional[NeuralNetStrategy] = None,
        graph_diffusion: Optional[GraphDiffusion] = None,
        tda_homology: Optional[TDAHomology] = None,
        data_orchestrator: Optional[DataOrchestrator] = None,
        gating_network: Optional[GatingNetwork] = None,
        instability_threshold: float = 0.5
    ):
        """
        Initialize The TopologyEngine.
        
        Parameters
        ----------
        nn_strategy : NeuralNetStrategy, optional
            Neural network strategy instance
        graph_diffusion : GraphDiffusion, optional
            Graph diffusion instance
        tda_homology : TDAHomology, optional
            TDA homology instance
        data_orchestrator : DataOrchestrator, optional
            Data orchestrator instance
        gating_network : GatingNetwork, optional
            Gating network instance
        instability_threshold : float, optional
            Threshold for topological instability. Default is 0.5.
        """
        # Initialize components with defaults if not provided
        self.nn_strategy = nn_strategy or NeuralNetStrategy()
        self.graph_diffusion = graph_diffusion or GraphDiffusion()
        self.tda_homology = tda_homology or TDAHomology()
        self.data_orchestrator = data_orchestrator or DataOrchestrator()
        self.gating_network = gating_network or GatingNetwork(
            instability_threshold=instability_threshold
        )
        
        self.is_fitted = False
    
    def fit(
        self,
        train_data: pd.DataFrame,
        horizon: int = 1,
        single_asset_mode: bool = False
    ):
        """
        Train the ensemble on historical data.
        
        Parameters
        ----------
        train_data : pd.DataFrame
            Training OHLCV data
        horizon : int, optional
            Forecast horizon for NN. Default is 1.
        single_asset_mode : bool, optional
            If True, treats data as single asset. Default is False.
        """
        if single_asset_mode:
            # Train NN on single asset
            self.nn_strategy.fit(train_data, horizon=horizon)
        else:
            # For multi-asset, would need to select primary asset or aggregate
            # For now, use first asset
            if isinstance(train_data.columns, pd.MultiIndex):
                # Get first asset's OHLCV
                first_asset = train_data.columns.get_level_values(0)[0]
                asset_data = train_data.xs(first_asset, level=0, axis=1)
            else:
                # Assume single asset or use as-is
                asset_data = train_data
            
            self.nn_strategy.fit(asset_data, horizon=horizon)
        
        self.is_fitted = True
    
    def predict(
        self,
        data: pd.DataFrame,
        return_details: bool = True
    ) -> Dict[str, any]:
        """
        Generate ensemble prediction with topological gating.
        
        Parameters
        ----------
        data : pd.DataFrame
            Recent OHLCV data for prediction
        return_details : bool, optional
            If True, return detailed breakdown. Default is True.
        
        Returns
        -------
        dict
            Prediction results including final_signal, confidence, regime, etc.
        """
        if not self.is_fitted:
            raise ValueError("Engine must be fitted before prediction")
        
        # Prepare data through orchestrator
        data_package = self.data_orchestrator.prepare_data(data)
        
        # Run NN prediction
        # For single asset, extract OHLCV
        if isinstance(data.columns, pd.MultiIndex):
            first_asset = data.columns.get_level_values(0)[0]
            asset_data = data.xs(first_asset, level=0, axis=1)
        else:
            asset_data = data
        
        nn_proba = self.nn_strategy.predict_proba(asset_data)
        
        # Run Graph Diffusion
        self.graph_diffusion.build_correlation_graph(
            data_package['correlation'],
            asset_names=data_package.get('asset_names')
        )
        self.graph_diffusion.compute_laplacian(normalized=True)
        graph_leakage = self.graph_diffusion.compute_leakage_score()
        
        # Run TDA Homology
        regime_info = self.tda_homology.classify_regime(data_package['correlation'])
        persistence_score = regime_info['persistence_score']
        regime = regime_info['regime']
        
        # Apply gating network
        result = self.gating_network.combine_signals(
            nn_proba=nn_proba,
            persistence_score=persistence_score,
            regime=regime,
            graph_leakage=graph_leakage
        )
        
        # Compute confidence interval
        confidence_interval = self.gating_network.compute_confidence_interval(
            result['confidence_score']
        )
        
        # Compute dynamic hedge ratio (simple version)
        hedge_ratio = self._compute_hedge_ratio(
            persistence_score=persistence_score,
            graph_leakage=graph_leakage,
            regime=regime
        )
        
        # Build output
        output = {
            'timestamp': str(datetime.now()),
            'final_signal': result['final_signal'],
            'confidence_score': result['confidence_score'],
            'confidence_interval': confidence_interval,
            'regime_classification': regime,
            'nn_predict_proba': nn_proba,
            'persistence_score': persistence_score,
            'graph_leakage': graph_leakage,
            'suggested_hedge': hedge_ratio
        }
        
        if return_details:
            output['details'] = {
                'regime_info': regime_info,
                'reason': result['reason'],
                'graph_metrics': self.graph_diffusion.compute_graph_metrics()
            }
        
        return output
    
    def _compute_hedge_ratio(
        self,
        persistence_score: float,
        graph_leakage: float,
        regime: str
    ) -> Dict[str, any]:
        """
        Compute dynamic hedge ratio based on topological metrics.
        
        Parameters
        ----------
        persistence_score : float
            TDA persistence score
        graph_leakage : float
            Graph leakage score
        regime : str
            Market regime
        
        Returns
        -------
        dict
            Hedge suggestion with instrument and ratio
        """
        # Simple heuristic: higher instability -> higher hedge ratio
        base_hedge = (persistence_score + graph_leakage) / 2
        
        # Regime-based adjustment
        regime_multipliers = {
            'Stable': 0.5,
            'Trending': 0.7,
            'Fragmented': 1.2,
            'Stressed': 1.5
        }
        
        multiplier = regime_multipliers.get(regime, 1.0)
        hedge_ratio = min(base_hedge * multiplier, 1.0)
        
        # Determine hedge instrument based on regime
        if regime in ['Fragmented', 'Stressed']:
            instrument = 'VX_FUT'  # Volatility futures
        elif regime == 'Trending':
            instrument = 'SECTOR_ETF'  # Sector rotation
        else:
            instrument = 'CASH'
        
        return {
            'instrument': instrument,
            'ratio': round(hedge_ratio, 3)
        }
    
    def to_json(self, prediction: Dict[str, any]) -> str:
        """
        Convert prediction to JSON string.
        
        Parameters
        ----------
        prediction : dict
            Prediction output from predict()
        
        Returns
        -------
        str
            JSON string
        """
        # Create a simplified version for JSON export
        json_output = {
            'timestamp': prediction['timestamp'],
            'final_signal': prediction['final_signal'],
            'confidence_score': prediction['confidence_score'],
            'confidence_interval': prediction['confidence_interval'],
            'regime_classification': prediction['regime_classification'],
            'nn_predict_proba': prediction['nn_predict_proba'],
            'persistence_score': prediction['persistence_score'],
            'graph_leakage': prediction['graph_leakage'],
            'suggested_hedge': prediction['suggested_hedge']
        }
        
        return json.dumps(json_output, indent=2)


def create_topology_engine(
    instability_threshold: float = 0.5,
    nn_hidden_layers: tuple = (100, 50),
    correlation_threshold: float = 0.3,
    max_edge_length: float = 2.0
) -> TopologyEngine:
    """
    Factory function to create a configured TopologyEngine.
    
    Parameters
    ----------
    instability_threshold : float, optional
        TDA instability threshold. Default is 0.5.
    nn_hidden_layers : tuple, optional
        NN architecture. Default is (100, 50).
    correlation_threshold : float, optional
        Graph correlation threshold. Default is 0.3.
    max_edge_length : float, optional
        TDA max edge length. Default is 2.0.
    
    Returns
    -------
    TopologyEngine
        Configured engine ready for training
    """
    nn_strategy = NeuralNetStrategy(hidden_layers=nn_hidden_layers)
    graph_diffusion = GraphDiffusion(correlation_threshold=correlation_threshold)
    tda_homology = TDAHomology(max_edge_length=max_edge_length)
    
    engine = TopologyEngine(
        nn_strategy=nn_strategy,
        graph_diffusion=graph_diffusion,
        tda_homology=tda_homology,
        instability_threshold=instability_threshold
    )
    
    return engine
