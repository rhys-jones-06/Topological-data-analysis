"""
Ensemble Module

Provides meta-learning functionality to combine TDA signals with other models
(e.g., neural networks) using various ensemble strategies.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Callable
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import Ridge, LogisticRegression


class MetaLearner:
    """
    Meta-learner for ensemble combination of TDA and other model signals.
    
    Combines signals from:
    1. TDA/Market Mutual Model
    2. Neural Networks
    3. Other quantitative models
    
    Using various ensemble strategies.
    """
    
    def __init__(
        self,
        method: str = 'weighted_average',
        task: str = 'regression',
        random_state: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the meta-learner.
        
        Parameters
        ----------
        method : str, optional
            Ensemble method:
            - 'weighted_average': Simple weighted average
            - 'stacking': Use a meta-model (Ridge/Logistic)
            - 'boosting': Gradient Boosting Tree
            Default is 'weighted_average'.
        task : str, optional
            Task type: 'regression' or 'classification'. Default is 'regression'.
        random_state : int, optional
            Random seed for reproducibility.
        **kwargs
            Additional parameters for the ensemble method.
        """
        self.method = method
        self.task = task
        self.random_state = random_state
        self.kwargs = kwargs
        
        # Model weights or meta-model
        self.weights = None
        self.meta_model = None
        self._is_fitted = False
    
    def fit(
        self,
        tda_signals: np.ndarray,
        other_signals: Union[np.ndarray, List[np.ndarray]],
        targets: np.ndarray
    ) -> 'MetaLearner':
        """
        Fit the meta-learner on training data.
        
        Parameters
        ----------
        tda_signals : np.ndarray
            Signals from TDA model. Shape: (n_samples, n_assets) or (n_samples,).
        other_signals : np.ndarray or list of np.ndarray
            Signals from other models. Each array has shape (n_samples, n_assets).
        targets : np.ndarray
            Target values (returns or labels). Shape: (n_samples, n_assets).
        
        Returns
        -------
        self : MetaLearner
            Fitted meta-learner.
        """
        # Prepare input signals
        if isinstance(other_signals, list):
            all_signals = [tda_signals] + other_signals
        else:
            all_signals = [tda_signals, other_signals]
        
        # Validate shapes match
        expected_shape = all_signals[0].shape
        for i, sig in enumerate(all_signals[1:], 1):
            if sig.shape != expected_shape:
                raise ValueError(
                    f"Signal {i} shape {sig.shape} does not match expected shape {expected_shape}"
                )
        
        # Stack signals as features
        X = np.column_stack([s.flatten() for s in all_signals])
        y = targets.flatten()
        
        if self.method == 'weighted_average':
            # Learn optimal weights using ridge regression
            ridge = Ridge(alpha=1.0, random_state=self.random_state)
            ridge.fit(X, y)
            
            # Extract and normalize weights
            raw_weights = ridge.coef_
            self.weights = raw_weights / np.sum(np.abs(raw_weights))
            
        elif self.method == 'stacking':
            # Use a meta-model
            if self.task == 'regression':
                self.meta_model = Ridge(
                    alpha=self.kwargs.get('alpha', 1.0),
                    random_state=self.random_state
                )
            else:  # classification
                self.meta_model = LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000
                )
            
            self.meta_model.fit(X, y)
            
        elif self.method == 'boosting':
            # Use gradient boosting
            if self.task == 'regression':
                self.meta_model = GradientBoostingRegressor(
                    n_estimators=self.kwargs.get('n_estimators', 100),
                    max_depth=self.kwargs.get('max_depth', 3),
                    learning_rate=self.kwargs.get('learning_rate', 0.1),
                    random_state=self.random_state
                )
            else:  # classification
                self.meta_model = GradientBoostingClassifier(
                    n_estimators=self.kwargs.get('n_estimators', 100),
                    max_depth=self.kwargs.get('max_depth', 3),
                    learning_rate=self.kwargs.get('learning_rate', 0.1),
                    random_state=self.random_state
                )
            
            self.meta_model.fit(X, y)
        
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")
        
        self._is_fitted = True
        return self
    
    def predict(
        self,
        tda_signals: np.ndarray,
        other_signals: Union[np.ndarray, List[np.ndarray]]
    ) -> np.ndarray:
        """
        Generate ensemble predictions.
        
        Parameters
        ----------
        tda_signals : np.ndarray
            Signals from TDA model.
        other_signals : np.ndarray or list of np.ndarray
            Signals from other models.
        
        Returns
        -------
        np.ndarray
            Ensemble predictions.
        """
        if not self._is_fitted:
            raise ValueError("Meta-learner must be fitted before prediction")
        
        # Prepare input signals
        if isinstance(other_signals, list):
            all_signals = [tda_signals] + other_signals
        else:
            all_signals = [tda_signals, other_signals]
        
        original_shape = all_signals[0].shape
        
        # Stack signals as features
        X = np.column_stack([s.flatten() for s in all_signals])
        
        if self.method == 'weighted_average':
            # Weighted combination
            predictions = np.dot(X, self.weights)
        else:
            # Use meta-model
            predictions = self.meta_model.predict(X)
        
        # Reshape to original shape
        return predictions.reshape(original_shape)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance if available.
        
        Returns
        -------
        np.ndarray or None
            Feature importance scores, or None if not available.
        """
        if not self._is_fitted:
            return None
        
        if self.method == 'weighted_average':
            return np.abs(self.weights)
        elif self.method == 'boosting' and hasattr(self.meta_model, 'feature_importances_'):
            return self.meta_model.feature_importances_
        elif hasattr(self.meta_model, 'coef_'):
            return np.abs(self.meta_model.coef_)
        else:
            return None


class SimpleEnsemble:
    """
    Simple ensemble strategies without meta-learning.
    """
    
    @staticmethod
    def weighted_average(
        signals: List[np.ndarray],
        weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Compute weighted average of signals.
        
        Parameters
        ----------
        signals : list of np.ndarray
            List of signal arrays to combine.
        weights : list of float, optional
            Weights for each signal. If None, uses equal weights.
        
        Returns
        -------
        np.ndarray
            Combined signals.
        """
        if weights is None:
            weights = [1.0 / len(signals)] * len(signals)
        else:
            # Normalize weights
            total = sum(weights)
            weights = [w / total for w in weights]
        
        result = np.zeros_like(signals[0])
        for signal, weight in zip(signals, weights):
            result += signal * weight
        
        return result
    
    @staticmethod
    def majority_vote(
        signals: List[np.ndarray],
        threshold: float = 0.0
    ) -> np.ndarray:
        """
        Majority vote ensemble for binary signals.
        
        Parameters
        ----------
        signals : list of np.ndarray
            List of signal arrays (positive = buy, negative = sell).
        threshold : float, optional
            Threshold for considering a signal. Default is 0.0.
        
        Returns
        -------
        np.ndarray
            Combined signals based on majority vote.
        """
        # Convert to binary decisions
        votes = [np.sign(s) for s in signals]
        
        # Stack and sum votes
        vote_sum = np.sum(np.stack(votes, axis=0), axis=0)
        
        # Return normalized average signal strength
        return vote_sum / len(signals)
    
    @staticmethod
    def max_confidence(
        signals: List[np.ndarray],
        confidences: List[np.ndarray]
    ) -> np.ndarray:
        """
        Select signal with highest confidence at each point.
        
        Parameters
        ----------
        signals : list of np.ndarray
            List of signal arrays.
        confidences : list of np.ndarray
            Confidence scores for each signal.
        
        Returns
        -------
        np.ndarray
            Combined signals using max confidence strategy.
        """
        # Stack signals and confidences
        signal_stack = np.stack(signals, axis=0)
        confidence_stack = np.stack(confidences, axis=0)
        
        # Find indices of maximum confidence
        max_idx = np.argmax(confidence_stack, axis=0)
        
        # Select signals with max confidence
        result = np.zeros_like(signals[0])
        for i in range(signal_stack.shape[1]):
            result[i] = signal_stack[max_idx[i], i]
        
        return result


def combine_tda_with_neural_net(
    tda_signal: np.ndarray,
    nn_signal: np.ndarray,
    tda_confidence: float,
    nn_confidence: float,
    method: str = 'weighted'
) -> np.ndarray:
    """
    Convenience function to combine TDA and Neural Net signals.
    
    Parameters
    ----------
    tda_signal : np.ndarray
        Signal from TDA model.
    nn_signal : np.ndarray
        Signal from Neural Network.
    tda_confidence : float
        Confidence in TDA signal (0 to 1).
    nn_confidence : float
        Confidence in NN signal (0 to 1).
    method : str, optional
        Combination method: 'weighted', 'max', or 'average'.
        Default is 'weighted'.
    
    Returns
    -------
    np.ndarray
        Combined signal.
    """
    if method == 'weighted':
        # Weight by confidence
        total_confidence = tda_confidence + nn_confidence
        if total_confidence > 0:
            tda_weight = tda_confidence / total_confidence
            nn_weight = nn_confidence / total_confidence
        else:
            tda_weight = nn_weight = 0.5
        
        return tda_signal * tda_weight + nn_signal * nn_weight
    
    elif method == 'max':
        # Use signal with higher confidence
        if tda_confidence > nn_confidence:
            return tda_signal
        else:
            return nn_signal
    
    elif method == 'average':
        # Simple average
        return (tda_signal + nn_signal) / 2
    
    else:
        raise ValueError(f"Unknown method: {method}")
