"""
Unit tests for risk management module.
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tda.risk_management import RiskManager, KellyCalculator


class TestRiskManager(unittest.TestCase):
    """Test cases for RiskManager class."""
    
    def setUp(self):
        """Set up test data."""
        self.risk_manager = RiskManager(
            base_leverage=1.0,
            max_leverage=3.0,
            min_leverage=0.1
        )
    
    def test_initialization(self):
        """Test RiskManager initialization."""
        self.assertEqual(self.risk_manager.base_leverage, 1.0)
        self.assertEqual(self.risk_manager.max_leverage, 3.0)
        self.assertEqual(self.risk_manager.min_leverage, 0.1)
    
    def test_calculate_position_size_stable(self):
        """Test position size calculation in stable regime."""
        position = self.risk_manager.calculate_position_size(
            base_signal=0.5,
            persistence_score=0.8,
            regime='stable',
            regime_confidence=1.0
        )
        
        # In stable regime with high persistence, should have high leverage
        self.assertGreater(position, 0.5)
        self.assertLessEqual(position, 3.0)  # Max leverage
    
    def test_calculate_position_size_stressed(self):
        """Test position size calculation in stressed regime."""
        position = self.risk_manager.calculate_position_size(
            base_signal=0.5,
            persistence_score=0.3,
            regime='stressed',
            regime_confidence=1.0
        )
        
        # In stressed regime, should have lower leverage
        self.assertLess(position, 0.5)
        self.assertGreaterEqual(position, 0.1)  # Min leverage
    
    def test_calculate_position_size_negative_signal(self):
        """Test position size with negative signal (short)."""
        position = self.risk_manager.calculate_position_size(
            base_signal=-0.5,
            persistence_score=0.6,
            regime='stable',
            regime_confidence=0.8
        )
        
        # Should be negative (short position)
        self.assertLess(position, 0)
        self.assertGreaterEqual(position, -3.0)
    
    def test_calculate_position_size_low_confidence(self):
        """Test position size with low regime confidence."""
        position = self.risk_manager.calculate_position_size(
            base_signal=0.5,
            persistence_score=0.8,
            regime='stable',
            regime_confidence=0.3
        )
        
        # Low confidence should reduce position size
        self.assertGreater(position, 0)
        self.assertLess(position, 1.0)
    
    def test_calculate_persistence_score(self):
        """Test persistence score calculation."""
        h0_features = {
            'num_components': 2,
            'max_persistence': 0.7
        }
        h1_features = {
            'num_loops': 1,
            'max_persistence': 0.2,
            'mean_persistence': 0.15
        }
        
        score = self.risk_manager.calculate_persistence_score(h0_features, h1_features)
        
        # Score should be between 0 and 1
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_calculate_persistence_score_fragmented(self):
        """Test persistence score with fragmented market."""
        h0_features = {
            'num_components': 5,
            'max_persistence': 0.3
        }
        h1_features = {
            'num_loops': 3,
            'max_persistence': 0.4,
            'mean_persistence': 0.3
        }
        
        score = self.risk_manager.calculate_persistence_score(h0_features, h1_features)
        
        # Fragmented market should have lower score
        self.assertLess(score, 0.5)
    
    def test_calculate_stop_loss_long(self):
        """Test stop loss calculation for long position."""
        entry_price = 100.0
        stop_loss, take_profit = self.risk_manager.calculate_stop_loss(
            entry_price=entry_price,
            position_size=1.0,
            volatility=2.0,
            persistence_score=0.7
        )
        
        # Stop loss should be below entry
        self.assertLess(stop_loss, entry_price)
        # Take profit should be above entry
        self.assertGreater(take_profit, entry_price)
        
        # Distance should be related to volatility
        stop_distance = entry_price - stop_loss
        self.assertGreater(stop_distance, 0)
        self.assertLess(stop_distance, 5.0)  # Should be reasonable
    
    def test_calculate_stop_loss_short(self):
        """Test stop loss calculation for short position."""
        entry_price = 100.0
        stop_loss, take_profit = self.risk_manager.calculate_stop_loss(
            entry_price=entry_price,
            position_size=-1.0,
            volatility=2.0,
            persistence_score=0.7
        )
        
        # For short: stop loss above entry
        self.assertGreater(stop_loss, entry_price)
        # Take profit below entry
        self.assertLess(take_profit, entry_price)
    
    def test_calculate_portfolio_heat(self):
        """Test portfolio heat calculation."""
        positions = np.array([1.0, -0.5, 0.8])
        values = np.array([100.0, 50.0, 75.0])
        volatilities = np.array([0.02, 0.03, 0.025])
        
        heat = self.risk_manager.calculate_portfolio_heat(
            positions, values, volatilities
        )
        
        # Heat should be positive and reasonable
        self.assertGreater(heat, 0)
        self.assertLess(heat, 1.0)
    
    def test_adjust_for_portfolio_heat_under_limit(self):
        """Test position adjustment when under heat limit."""
        proposed = 1.0
        adjusted = self.risk_manager.adjust_for_portfolio_heat(
            proposed_position=proposed,
            current_heat=0.1,
            max_heat=0.2
        )
        
        # Should allow position (may scale it down slightly)
        self.assertGreater(adjusted, 0)
        self.assertLessEqual(adjusted, proposed)
    
    def test_adjust_for_portfolio_heat_over_limit(self):
        """Test position adjustment when over heat limit."""
        proposed = 1.0
        adjusted = self.risk_manager.adjust_for_portfolio_heat(
            proposed_position=proposed,
            current_heat=0.25,
            max_heat=0.2
        )
        
        # Should reduce position
        self.assertLess(adjusted, proposed)
        self.assertGreater(adjusted, 0)


class TestKellyCalculator(unittest.TestCase):
    """Test cases for KellyCalculator class."""
    
    def setUp(self):
        """Set up test data."""
        self.kelly = KellyCalculator(fraction=0.25)
    
    def test_initialization(self):
        """Test Kelly calculator initialization."""
        self.assertEqual(self.kelly.fraction, 0.25)
    
    def test_calculate_positive_edge(self):
        """Test Kelly calculation with positive edge."""
        kelly_fraction = self.kelly.calculate(
            win_prob=0.6,
            win_loss_ratio=2.0,
            persistence_score=0.8
        )
        
        # Should be positive with winning edge
        self.assertGreater(kelly_fraction, 0)
        self.assertLessEqual(kelly_fraction, 1.0)
    
    def test_calculate_negative_edge(self):
        """Test Kelly calculation with negative edge."""
        kelly_fraction = self.kelly.calculate(
            win_prob=0.4,
            win_loss_ratio=1.0,
            persistence_score=0.8
        )
        
        # Should be zero or very small with losing edge
        self.assertGreaterEqual(kelly_fraction, 0)
        self.assertLess(kelly_fraction, 0.1)
    
    def test_calculate_low_persistence(self):
        """Test Kelly with low persistence score."""
        kelly_fraction = self.kelly.calculate(
            win_prob=0.6,
            win_loss_ratio=2.0,
            persistence_score=0.2
        )
        
        # Low persistence should reduce Kelly fraction
        self.assertGreater(kelly_fraction, 0)
        self.assertLess(kelly_fraction, 0.2)
    
    def test_calculate_invalid_ratio(self):
        """Test Kelly with invalid win/loss ratio."""
        kelly_fraction = self.kelly.calculate(
            win_prob=0.6,
            win_loss_ratio=0.0,
            persistence_score=0.8
        )
        
        # Should return 0 for invalid ratio
        self.assertEqual(kelly_fraction, 0.0)
    
    def test_full_kelly(self):
        """Test with full Kelly (fraction=1.0)."""
        full_kelly = KellyCalculator(fraction=1.0)
        
        kelly_fraction = full_kelly.calculate(
            win_prob=0.6,
            win_loss_ratio=2.0,
            persistence_score=1.0
        )
        
        # Full Kelly with perfect persistence
        self.assertGreater(kelly_fraction, 0)
        # Should be close to theoretical Kelly
        theoretical = (0.6 * 2.0 - 0.4) / 2.0
        self.assertAlmostEqual(kelly_fraction, theoretical, places=2)


if __name__ == '__main__':
    unittest.main()
