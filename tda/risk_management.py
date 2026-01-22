"""
Risk Management Layer

This module provides functionality to scale positions based on the persistence
of market shape. Higher stability in topological structure allows for higher leverage.
"""

import numpy as np
from typing import Dict, Optional, Tuple


class RiskManager:
    """
    Risk management based on topological persistence and market regime.
    
    Scales position sizes based on:
    1. Persistence of topological features (stability)
    2. Current market regime (stable, transitioning, stressed)
    3. Confidence in predictions
    """
    
    def __init__(
        self,
        base_leverage: float = 1.0,
        max_leverage: float = 3.0,
        min_leverage: float = 0.1,
        regime_multipliers: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the risk manager.
        
        Parameters
        ----------
        base_leverage : float, optional
            Base leverage multiplier. Default is 1.0.
        max_leverage : float, optional
            Maximum allowed leverage. Default is 3.0.
        min_leverage : float, optional
            Minimum leverage (defensive mode). Default is 0.1.
        regime_multipliers : dict, optional
            Leverage multipliers for each regime.
            Default: {'stable': 1.5, 'transitioning': 1.0, 'stressed': 0.3}
        """
        self.base_leverage = base_leverage
        self.max_leverage = max_leverage
        self.min_leverage = min_leverage
        
        if regime_multipliers is None:
            self.regime_multipliers = {
                'stable': 1.5,
                'transitioning': 1.0,
                'stressed': 0.3
            }
        else:
            self.regime_multipliers = regime_multipliers
    
    def calculate_position_size(
        self,
        base_signal: float,
        persistence_score: float,
        regime: str,
        regime_confidence: float = 1.0
    ) -> float:
        """
        Calculate position size based on signal and risk factors.
        
        Parameters
        ----------
        base_signal : float
            Base trading signal strength (-1 to 1).
        persistence_score : float
            Topological persistence score (0 to 1, higher = more stable).
        regime : str
            Current market regime: 'stable', 'transitioning', or 'stressed'.
        regime_confidence : float, optional
            Confidence in regime classification (0 to 1). Default is 1.0.
        
        Returns
        -------
        float
            Position size as a fraction of capital (-max_leverage to max_leverage).
        """
        # Get regime multiplier
        regime_multiplier = self.regime_multipliers.get(regime, 1.0)
        
        # Calculate leverage based on persistence
        # Higher persistence = higher confidence = higher leverage
        persistence_leverage = self.base_leverage + (
            (self.max_leverage - self.base_leverage) * persistence_score
        )
        
        # Apply regime multiplier
        adjusted_leverage = persistence_leverage * regime_multiplier
        
        # Apply confidence scaling
        adjusted_leverage *= regime_confidence
        
        # Clip to allowed range
        adjusted_leverage = np.clip(adjusted_leverage, self.min_leverage, self.max_leverage)
        
        # Calculate final position size
        position_size = base_signal * adjusted_leverage
        
        return position_size
    
    def calculate_persistence_score(
        self,
        h0_features: Dict[str, float],
        h1_features: Dict[str, float],
        weight_h0: float = 0.4,
        weight_h1: float = 0.6
    ) -> float:
        """
        Calculate persistence score from topological features.
        
        Parameters
        ----------
        h0_features : dict
            H_0 features (clusters).
        h1_features : dict
            H_1 features (loops).
        weight_h0 : float, optional
            Weight for H_0 features. Default is 0.4.
        weight_h1 : float, optional
            Weight for H_1 features. Default is 0.6.
        
        Returns
        -------
        float
            Persistence score between 0 and 1.
        """
        # H_0 persistence score: fewer clusters and higher persistence = more stable
        max_h0_persistence = h0_features.get('max_persistence', 0.0)
        num_components = h0_features.get('num_components', 0)
        
        # Normalize H_0 score (inverse of clusters, direct for persistence)
        h0_score = 0.0
        if num_components > 0:
            # Fewer components is better (more unified market)
            h0_score += (1.0 / (1.0 + num_components)) * 0.5
        # Higher persistence is better
        h0_score += min(max_h0_persistence, 1.0) * 0.5
        
        # H_1 persistence score: stable loops indicate regime stability
        max_h1_persistence = h1_features.get('max_persistence', 0.0)
        mean_h1_persistence = h1_features.get('mean_persistence', 0.0)
        
        # Moderate H_1 persistence is good, too high or too low is unstable
        # Optimal range: 0.1 to 0.3
        optimal_h1 = 0.2
        h1_deviation = abs(max_h1_persistence - optimal_h1)
        h1_score = max(0.0, 1.0 - h1_deviation / optimal_h1)
        
        # Combine scores
        persistence_score = weight_h0 * h0_score + weight_h1 * h1_score
        
        # Clip to [0, 1]
        return np.clip(persistence_score, 0.0, 1.0)
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        position_size: float,
        volatility: float,
        persistence_score: float
    ) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit levels.
        
        Parameters
        ----------
        entry_price : float
            Entry price for the position.
        position_size : float
            Position size (positive for long, negative for short).
        volatility : float
            Current volatility estimate (e.g., standard deviation of returns).
        persistence_score : float
            Topological persistence score (0 to 1).
        
        Returns
        -------
        stop_loss : float
            Stop loss price level.
        take_profit : float
            Take profit price level.
        """
        # Base stop loss at 2 * volatility
        # Tighter stops in low persistence (unstable) markets
        stop_multiplier = 2.0 - persistence_score  # Range: 1.0 to 2.0
        stop_distance = volatility * stop_multiplier
        
        # Take profit at 3 * volatility
        # Wider targets in high persistence (stable) markets
        profit_multiplier = 2.0 + persistence_score  # Range: 2.0 to 3.0
        profit_distance = volatility * profit_multiplier
        
        if position_size > 0:  # Long position
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + profit_distance
        else:  # Short position
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - profit_distance
        
        return stop_loss, take_profit
    
    def calculate_portfolio_heat(
        self,
        positions: np.ndarray,
        values: np.ndarray,
        volatilities: np.ndarray
    ) -> float:
        """
        Calculate portfolio heat (total risk exposure).
        
        Parameters
        ----------
        positions : np.ndarray
            Current position sizes for each asset.
        values : np.ndarray
            Current values/prices for each asset.
        volatilities : np.ndarray
            Volatility estimates for each asset.
        
        Returns
        -------
        float
            Portfolio heat as a fraction of total capital.
        """
        # Calculate risk for each position
        position_risks = np.abs(positions * values * volatilities)
        
        # Sum total risk
        total_risk = np.sum(position_risks)
        
        # Calculate portfolio value
        portfolio_value = np.sum(np.abs(positions * values))
        
        # Return heat as fraction
        if portfolio_value > 0:
            return total_risk / portfolio_value
        else:
            return 0.0
    
    def adjust_for_portfolio_heat(
        self,
        proposed_position: float,
        current_heat: float,
        max_heat: float = 0.2
    ) -> float:
        """
        Adjust proposed position to respect portfolio heat limit.
        
        Parameters
        ----------
        proposed_position : float
            Proposed position size.
        current_heat : float
            Current portfolio heat (0 to 1).
        max_heat : float, optional
            Maximum allowed portfolio heat. Default is 0.2 (20%).
        
        Returns
        -------
        float
            Adjusted position size.
        """
        if current_heat >= max_heat:
            # Already at max heat, reduce proposed position
            return proposed_position * 0.5
        else:
            # Scale position to respect heat limit
            heat_headroom = max_heat - current_heat
            heat_factor = min(1.0, heat_headroom / max_heat)
            return proposed_position * heat_factor


class KellyCalculator:
    """
    Calculate Kelly criterion for position sizing.
    
    Uses topological persistence as a measure of confidence.
    """
    
    def __init__(self, fraction: float = 0.25):
        """
        Initialize Kelly calculator.
        
        Parameters
        ----------
        fraction : float, optional
            Fraction of Kelly to use (for safety). Default is 0.25 (quarter Kelly).
        """
        self.fraction = fraction
    
    def calculate(
        self,
        win_prob: float,
        win_loss_ratio: float,
        persistence_score: float
    ) -> float:
        """
        Calculate Kelly criterion position size.
        
        Parameters
        ----------
        win_prob : float
            Probability of winning (0 to 1).
        win_loss_ratio : float
            Ratio of average win to average loss.
        persistence_score : float
            Topological persistence score (0 to 1).
        
        Returns
        -------
        float
            Kelly fraction (0 to 1).
        """
        # Basic Kelly formula: f* = (p*b - q) / b
        # where p = win probability, q = loss probability, b = win/loss ratio
        q = 1.0 - win_prob
        
        if win_loss_ratio <= 0:
            return 0.0
        
        kelly = (win_prob * win_loss_ratio - q) / win_loss_ratio
        
        # Clip to valid range
        kelly = np.clip(kelly, 0.0, 1.0)
        
        # Apply fractional Kelly
        kelly *= self.fraction
        
        # Scale by persistence (higher persistence = more confidence)
        kelly *= persistence_score
        
        return kelly
