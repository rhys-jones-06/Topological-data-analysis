"""
Test cases for TDA enhancements:
1. Adaptive TDA Persistence Score with Rolling Baseline
2. Synthetic Data with Correlation Preservation (GBM and Copula)
3. JSON Output with Topological Attribution
"""

import unittest
import numpy as np
import pandas as pd
import json

from tda_homology import TDAHomology
from monte_carlo_stress import MonteCarloStressTest
from topology_engine import create_topology_engine


class TestAdaptivePersistence(unittest.TestCase):
    """Test adaptive persistence scoring with rolling baseline."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.tda = TDAHomology(baseline_window=30)
    
    def test_baseline_initialization(self):
        """Test that baseline starts empty."""
        stats = self.tda.get_baseline_stats()
        self.assertEqual(stats['count'], 0)
        self.assertEqual(stats['mean'], 0.0)
    
    def test_baseline_update(self):
        """Test updating the baseline."""
        # Add some scores
        for i in range(10):
            self.tda.update_baseline(0.3 + i * 0.01)
        
        stats = self.tda.get_baseline_stats()
        self.assertEqual(stats['count'], 10)
        self.assertGreater(stats['mean'], 0)
        self.assertGreater(stats['std'], 0)
    
    def test_baseline_rolling_window(self):
        """Test that baseline respects window size."""
        # Add more scores than window size
        for i in range(50):
            self.tda.update_baseline(0.3 + i * 0.01)
        
        stats = self.tda.get_baseline_stats()
        # Should only keep last 30
        self.assertEqual(stats['count'], 30)
    
    def test_adaptive_threshold_insufficient_data(self):
        """Test adaptive threshold with insufficient baseline data."""
        # Add only a few data points
        for i in range(3):
            self.tda.update_baseline(0.3)
        
        adaptive_info = self.tda.compute_adaptive_threshold(0.5)
        
        # Should use default threshold
        self.assertEqual(adaptive_info['threshold'], 0.5)
        self.assertFalse(adaptive_info['adaptive'])
    
    def test_adaptive_threshold_with_data(self):
        """Test adaptive threshold with sufficient baseline data."""
        # Add baseline data
        baseline_scores = [0.2, 0.25, 0.22, 0.23, 0.24, 0.21, 0.26, 0.23, 0.22, 0.25]
        for score in baseline_scores:
            self.tda.update_baseline(score)
        
        # Current score is within normal range
        adaptive_info = self.tda.compute_adaptive_threshold(0.23)
        
        self.assertTrue(adaptive_info['adaptive'])
        self.assertFalse(adaptive_info['is_anomaly'])
        self.assertLess(abs(adaptive_info['z_score']), 1.0)
        
        # Current score is anomalous
        adaptive_info = self.tda.compute_adaptive_threshold(0.7)
        
        self.assertTrue(adaptive_info['adaptive'])
        self.assertTrue(adaptive_info['is_anomaly'])
        self.assertGreater(adaptive_info['z_score'], 2.0)
    
    def test_classify_regime_with_adaptive_threshold(self):
        """Test regime classification uses adaptive threshold."""
        # Create correlation matrix
        n_assets = 5
        correlation_matrix = np.eye(n_assets) * 0.8 + 0.2
        np.fill_diagonal(correlation_matrix, 1.0)
        
        # Build baseline
        for _ in range(10):
            self.tda.classify_regime(correlation_matrix)
        
        # Classify again
        result = self.tda.classify_regime(correlation_matrix)
        
        # Should have adaptive threshold info
        self.assertIn('adaptive_threshold', result)
        self.assertIn('baseline_stats', result)
        
        adaptive_info = result['adaptive_threshold']
        self.assertTrue(adaptive_info['adaptive'])
        self.assertIn('threshold', adaptive_info)
        self.assertIn('z_score', adaptive_info)


class TestSyntheticDataGeneration(unittest.TestCase):
    """Test synthetic data generation with correlation preservation."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create correlated returns
        n_periods = 100
        n_assets = 3
        
        # Generate correlated data
        mean = [0.001, 0.0015, 0.0012]
        cov = [[0.0004, 0.0002, 0.0001],
               [0.0002, 0.0005, 0.00015],
               [0.0001, 0.00015, 0.0003]]
        
        returns_array = np.random.multivariate_normal(mean, cov, n_periods)
        self.returns = pd.DataFrame(
            returns_array,
            columns=['Asset1', 'Asset2', 'Asset3']
        )
        
        self.mc_tester = MonteCarloStressTest(
            n_simulations=10,
            random_state=42
        )
    
    def test_gbm_generation(self):
        """Test Geometric Brownian Motion generation."""
        synthetic = self.mc_tester.generate_synthetic_returns(
            self.returns,
            method='gbm'
        )
        
        # Check shape
        self.assertEqual(synthetic.shape, self.returns.shape)
        
        # Check columns
        self.assertEqual(list(synthetic.columns), list(self.returns.columns))
    
    def test_copula_generation(self):
        """Test Copula-based generation."""
        synthetic = self.mc_tester.generate_synthetic_returns(
            self.returns,
            method='copula'
        )
        
        # Check shape
        self.assertEqual(synthetic.shape, self.returns.shape)
        
        # Check columns
        self.assertEqual(list(synthetic.columns), list(self.returns.columns))
    
    def test_correlation_preservation_gbm(self):
        """Test that GBM preserves correlations."""
        synthetic = self.mc_tester.generate_synthetic_returns(
            self.returns,
            method='gbm'
        )
        
        validation = self.mc_tester.validate_correlation_preservation(
            self.returns,
            synthetic,
            tolerance=0.2  # Allow some variance
        )
        
        # Should preserve correlations reasonably well
        self.assertLess(validation['mean_correlation_diff'], 0.20)
        self.assertIn('max_correlation_diff', validation)
        self.assertIn('is_valid', validation)
    
    def test_correlation_preservation_copula(self):
        """Test that Copula preserves correlations."""
        synthetic = self.mc_tester.generate_synthetic_returns(
            self.returns,
            method='copula'
        )
        
        validation = self.mc_tester.validate_correlation_preservation(
            self.returns,
            synthetic,
            tolerance=0.2
        )
        
        # Should preserve correlations
        self.assertLess(validation['mean_correlation_diff'], 0.20)
    
    def test_validation_detects_poor_correlation(self):
        """Test that validation detects when correlations are not preserved."""
        # Generate uncorrelated data
        synthetic = pd.DataFrame(
            np.random.randn(*self.returns.shape),
            columns=self.returns.columns
        )
        
        validation = self.mc_tester.validate_correlation_preservation(
            self.returns,
            synthetic,
            tolerance=0.1  # Strict tolerance
        )
        
        # Should detect correlation loss
        self.assertFalse(validation['is_valid'])
        self.assertGreater(validation['max_correlation_diff'], 0.1)
    
    def test_stress_test_with_validation(self):
        """Test stress test includes correlation validation."""
        def simple_evaluator(data):
            return {'sharpe': np.random.randn()}
        
        results = self.mc_tester.run_stress_test(
            self.returns,
            simple_evaluator,
            generation_method='gbm',
            validate_correlations=True
        )
        
        # Should include validation results
        self.assertIn('correlation_validation', results)
        
        validation = results['correlation_validation']
        self.assertIn('n_validated', validation)
        self.assertIn('validation_rate', validation)
        self.assertIn('mean_max_diff', validation)


class TestTopologicalAttribution(unittest.TestCase):
    """Test JSON output with topological attribution."""
    
    def setUp(self):
        """Set up test engine."""
        np.random.seed(42)
        
        # Create engine
        self.engine = create_topology_engine(
            instability_threshold=0.5
        )
        
        # Create training data
        n_periods = 100
        dates = pd.date_range('2020-01-01', periods=n_periods, freq='D')
        
        self.train_data = pd.DataFrame({
            'Open': 100 + np.random.randn(n_periods).cumsum(),
            'High': 102 + np.random.randn(n_periods).cumsum(),
            'Low': 98 + np.random.randn(n_periods).cumsum(),
            'Close': 100 + np.random.randn(n_periods).cumsum(),
            'Volume': 1000000 + np.random.randint(-100000, 100000, n_periods)
        }, index=dates)
        
        # Train the engine
        self.engine.fit(self.train_data, single_asset_mode=True)
    
    def test_prediction_includes_reason(self):
        """Test that prediction includes reason field."""
        # Make prediction
        prediction = self.engine.predict(self.train_data[-20:])
        
        # Should include reason
        self.assertIn('reason', prediction)
        self.assertIsInstance(prediction['reason'], str)
    
    def test_prediction_includes_reason_details(self):
        """Test that prediction includes detailed reason."""
        prediction = self.engine.predict(self.train_data[-20:])
        
        # Should include reason_details
        self.assertIn('reason_details', prediction)
        self.assertIsInstance(prediction['reason_details'], list)
    
    def test_json_output_includes_attribution(self):
        """Test that JSON output includes topological attribution."""
        prediction = self.engine.predict(self.train_data[-20:])
        json_output = self.engine.to_json(prediction)
        
        # Parse JSON
        parsed = json.loads(json_output)
        
        # Should include attribution fields
        self.assertIn('reason', parsed)
        self.assertIn('reason_details', parsed)
        
        # Check reason is not "Unknown"
        self.assertNotEqual(parsed['reason'], 'Unknown')
    
    def test_reason_types(self):
        """Test different reason types based on conditions."""
        # Get multiple predictions
        for i in range(5):
            start_idx = -30 - i * 10
            end_idx = -10 - i * 10
            prediction = self.engine.predict(self.train_data[start_idx:end_idx])
            
            reason = prediction['reason']
            
            # Should be one of the expected reason types
            self.assertIn(reason, [
                'H1_instability_exceeded_threshold',
                'NN_prediction_bullish',
                'NN_prediction_bearish',
                'NN_prediction_neutral'
            ])
    
    def test_reason_details_format(self):
        """Test that reason_details contains expected information."""
        prediction = self.engine.predict(self.train_data[-20:])
        
        reason_details = prediction['reason_details']
        
        # Should be a list of strings
        self.assertIsInstance(reason_details, list)
        
        # Should contain some information
        if len(reason_details) > 0:
            for detail in reason_details:
                self.assertIsInstance(detail, str)
    
    def test_attribution_interpretability(self):
        """Test that attribution makes model interpretable."""
        prediction = self.engine.predict(self.train_data[-20:])
        
        # Extract key information
        signal = prediction['final_signal']
        reason = prediction['reason']
        
        # NEUTRAL signal should have clear reason
        if signal == 'NEUTRAL':
            self.assertTrue(
                reason in ['H1_instability_exceeded_threshold', 'NN_prediction_neutral'],
                f"NEUTRAL signal should have clear reason, got: {reason}"
            )
        
        # LONG/SHORT should have NN-based reason (if not gated)
        if signal in ['LONG', 'SHORT']:
            if reason != 'H1_instability_exceeded_threshold':
                self.assertTrue(
                    reason in ['NN_prediction_bullish', 'NN_prediction_bearish'],
                    f"LONG/SHORT signal should have NN-based reason, got: {reason}"
                )


if __name__ == '__main__':
    unittest.main()
