"""
Neural Network Strategy Module

A simple neural network strategy for short-term (24-48 hour) price prediction.
This module provides the "micro" bottom-up signal for the TopologyEngine.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


class NeuralNetStrategy:
    """
    Neural Network strategy for short-horizon trading signals.
    
    This provides the "micro lens" that looks at price, volume, and momentum
    to predict the next 24-48 hours of movement.
    """
    
    def __init__(
        self,
        hidden_layers: tuple = (100, 50),
        lookback_period: int = 20,
        random_state: Optional[int] = 42
    ):
        """
        Initialize the Neural Network strategy.
        
        Parameters
        ----------
        hidden_layers : tuple, optional
            Size of hidden layers. Default is (100, 50).
        lookback_period : int, optional
            Number of periods to look back for features. Default is 20.
        random_state : int, optional
            Random seed for reproducibility.
        """
        self.hidden_layers = hidden_layers
        self.lookback_period = lookback_period
        self.random_state = random_state
        
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def _create_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Create features from OHLCV data.
        
        Parameters
        ----------
        data : pd.DataFrame
            OHLCV data with columns: Open, High, Low, Close, Volume
        
        Returns
        -------
        np.ndarray
            Feature matrix
        """
        features = []
        
        # Price momentum features
        for period in [5, 10, 20]:
            if len(data) >= period:
                features.append(data['Close'].pct_change(period).fillna(0))
        
        # Volume features
        features.append(data['Volume'].pct_change().fillna(0))
        
        # Volatility (rolling std of returns)
        for period in [5, 10]:
            if len(data) >= period:
                returns = data['Close'].pct_change()
                features.append(returns.rolling(period).std().fillna(0))
        
        # Price relative to moving averages
        for period in [10, 20]:
            if len(data) >= period:
                ma = data['Close'].rolling(period).mean()
                features.append((data['Close'] / ma - 1).fillna(0))
        
        # High-Low spread
        features.append((data['High'] - data['Low']) / data['Close'])
        
        return np.column_stack(features)
    
    def _create_targets(self, data: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        """
        Create binary targets (1 = up, 0 = down) based on future returns.
        
        Parameters
        ----------
        data : pd.DataFrame
            OHLCV data
        horizon : int, optional
            Forecast horizon in periods. Default is 1.
        
        Returns
        -------
        np.ndarray
            Binary targets
        """
        future_returns = data['Close'].shift(-horizon) / data['Close'] - 1
        targets = (future_returns > 0).astype(int)
        return targets.values[:-horizon]
    
    def fit(self, data: pd.DataFrame, horizon: int = 1):
        """
        Train the neural network on historical data.
        
        Parameters
        ----------
        data : pd.DataFrame
            OHLCV data with columns: Open, High, Low, Close, Volume
        horizon : int, optional
            Forecast horizon in periods. Default is 1 (next period).
        """
        X = self._create_features(data)
        y = self._create_targets(data, horizon)
        
        # Align X and y (remove last 'horizon' rows from X)
        X = X[:-horizon]
        
        # Remove any rows with NaN
        valid_idx = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
    
    def predict_proba(self, data: pd.DataFrame) -> float:
        """
        Predict probability of upward movement.
        
        Parameters
        ----------
        data : pd.DataFrame
            Recent OHLCV data for prediction
        
        Returns
        -------
        float
            Probability of upward movement (0 to 1)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._create_features(data)
        
        # Use only the last row for prediction
        X_last = X[-1:].reshape(1, -1)
        
        # Handle NaN values
        if np.isnan(X_last).any():
            # Return neutral probability if features are invalid
            return 0.5
        
        X_scaled = self.scaler.transform(X_last)
        
        # Predict probability of class 1 (upward movement)
        proba = self.model.predict_proba(X_scaled)[0, 1]
        
        return proba
    
    def predict_signal(self, data: pd.DataFrame, threshold: float = 0.5) -> int:
        """
        Generate trading signal based on prediction.
        
        Parameters
        ----------
        data : pd.DataFrame
            Recent OHLCV data
        threshold : float, optional
            Probability threshold for signal. Default is 0.5.
        
        Returns
        -------
        int
            1 for long, -1 for short, 0 for neutral
        """
        proba = self.predict_proba(data)
        
        if proba > threshold:
            return 1
        elif proba < (1 - threshold):
            return -1
        else:
            return 0


def create_sample_nn_strategy(random_state: int = 42) -> NeuralNetStrategy:
    """
    Factory function to create a sample NN strategy with default parameters.
    
    Parameters
    ----------
    random_state : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    NeuralNetStrategy
        Configured neural network strategy
    """
    return NeuralNetStrategy(
        hidden_layers=(100, 50),
        lookback_period=20,
        random_state=random_state
    )
