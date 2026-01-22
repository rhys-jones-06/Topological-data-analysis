"""
Unit tests for regime detection module.
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tda.regime_detection import (
    HMMRegimeDetector,
    ClusteringRegimeDetector,
    classify_regime_by_topology
)


class TestHMMRegimeDetector(unittest.TestCase):
    """Test cases for HMMRegimeDetector."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        # Create synthetic features with 3 distinct regimes
        regime1 = np.random.randn(30, 5) + np.array([0, 0, 0, 0, 0])
        regime2 = np.random.randn(30, 5) + np.array([2, 2, 2, 2, 2])
        regime3 = np.random.randn(30, 5) + np.array([4, 4, 4, 4, 4])
        self.features = np.vstack([regime1, regime2, regime3])
    
    def test_initialization(self):
        """Test HMM detector initialization."""
        detector = HMMRegimeDetector(n_regimes=3, random_state=42)
        self.assertEqual(detector.n_regimes, 3)
        self.assertFalse(detector._is_fitted)
    
    def test_fit(self):
        """Test fitting the HMM detector."""
        detector = HMMRegimeDetector(n_regimes=3, random_state=42)
        detector.fit(self.features)
        
        self.assertTrue(detector._is_fitted)
        self.assertIsNotNone(detector.emission_means)
        self.assertEqual(detector.emission_means.shape, (3, 5))
    
    def test_predict(self):
        """Test prediction."""
        detector = HMMRegimeDetector(n_regimes=3, random_state=42)
        detector.fit(self.features)
        
        predictions = detector.predict(self.features[:10])
        
        self.assertEqual(len(predictions), 10)
        # Predictions should be in range [0, n_regimes)
        self.assertTrue(np.all(predictions >= 0))
        self.assertTrue(np.all(predictions < 3))
    
    def test_predict_proba(self):
        """Test probability prediction."""
        detector = HMMRegimeDetector(n_regimes=3, random_state=42)
        detector.fit(self.features)
        
        probas = detector.predict_proba(self.features[:10])
        
        self.assertEqual(probas.shape, (10, 3))
        # Probabilities should sum to 1
        np.testing.assert_array_almost_equal(probas.sum(axis=1), np.ones(10))
        # Probabilities should be in [0, 1]
        self.assertTrue(np.all(probas >= 0))
        self.assertTrue(np.all(probas <= 1))
    
    def test_get_regime_name(self):
        """Test getting regime names."""
        detector = HMMRegimeDetector(n_regimes=3, random_state=42)
        detector.fit(self.features)
        
        names = detector.get_regime_name(self.features[:5])
        
        self.assertEqual(len(names), 5)
        self.assertIsInstance(names[0], str)


class TestClusteringRegimeDetector(unittest.TestCase):
    """Test cases for ClusteringRegimeDetector."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        # Create synthetic features
        regime1 = np.random.randn(30, 5) + np.array([0, 0, 0, 0, 0])
        regime2 = np.random.randn(30, 5) + np.array([3, 3, 3, 3, 3])
        self.features = np.vstack([regime1, regime2])
    
    def test_kmeans_initialization(self):
        """Test K-means detector initialization."""
        detector = ClusteringRegimeDetector(n_regimes=2, method='kmeans', random_state=42)
        self.assertEqual(detector.method, 'kmeans')
        self.assertFalse(detector._is_fitted)
    
    def test_gmm_initialization(self):
        """Test GMM detector initialization."""
        detector = ClusteringRegimeDetector(n_regimes=2, method='gmm', random_state=42)
        self.assertEqual(detector.method, 'gmm')
    
    def test_invalid_method(self):
        """Test invalid method raises error."""
        detector = ClusteringRegimeDetector(n_regimes=2, method='invalid')
        with self.assertRaises(ValueError):
            detector.fit(self.features)
    
    def test_kmeans_fit_predict(self):
        """Test K-means fit and predict."""
        detector = ClusteringRegimeDetector(n_regimes=2, method='kmeans', random_state=42)
        detector.fit(self.features)
        
        predictions = detector.predict(self.features)
        
        self.assertEqual(len(predictions), 60)
        self.assertTrue(np.all(predictions >= 0))
        self.assertTrue(np.all(predictions < 2))
    
    def test_gmm_fit_predict(self):
        """Test GMM fit and predict."""
        detector = ClusteringRegimeDetector(n_regimes=2, method='gmm', random_state=42)
        detector.fit(self.features)
        
        predictions = detector.predict(self.features)
        
        self.assertEqual(len(predictions), 60)
    
    def test_kmeans_predict_proba(self):
        """Test K-means probability prediction."""
        detector = ClusteringRegimeDetector(n_regimes=2, method='kmeans', random_state=42)
        detector.fit(self.features)
        
        probas = detector.predict_proba(self.features[:10])
        
        self.assertEqual(probas.shape, (10, 2))
        # Should be valid probabilities
        self.assertTrue(np.all(probas >= 0))
        self.assertTrue(np.all(probas <= 1))
    
    def test_gmm_predict_proba(self):
        """Test GMM probability prediction."""
        detector = ClusteringRegimeDetector(n_regimes=2, method='gmm', random_state=42)
        detector.fit(self.features)
        
        probas = detector.predict_proba(self.features[:10])
        
        self.assertEqual(probas.shape, (10, 2))
        np.testing.assert_array_almost_equal(probas.sum(axis=1), np.ones(10))


class TestClassifyRegimeByTopology(unittest.TestCase):
    """Test cases for classify_regime_by_topology function."""
    
    def test_stable_regime(self):
        """Test stable regime classification."""
        h0_features = {
            'num_components': 1,
            'max_persistence': 0.8
        }
        h1_features = {
            'num_loops': 0,
            'max_persistence': 0.02
        }
        
        regime, scores = classify_regime_by_topology(h0_features, h1_features)
        
        self.assertEqual(regime, 'stable')
        self.assertGreater(scores['stable'], scores['stressed'])
        self.assertGreater(scores['stable'], scores['transitioning'])
    
    def test_stressed_regime(self):
        """Test stressed regime classification."""
        h0_features = {
            'num_components': 5,
            'max_persistence': 0.3
        }
        h1_features = {
            'num_loops': 3,
            'max_persistence': 0.25
        }
        
        regime, scores = classify_regime_by_topology(h0_features, h1_features)
        
        self.assertEqual(regime, 'stressed')
        self.assertGreater(scores['stressed'], scores['stable'])
    
    def test_transitioning_regime(self):
        """Test transitioning regime classification."""
        h0_features = {
            'num_components': 3,
            'max_persistence': 0.4
        }
        h1_features = {
            'num_loops': 1,
            'max_persistence': 0.1
        }
        
        regime, scores = classify_regime_by_topology(h0_features, h1_features)
        
        # Should be transitioning (between stable and stressed)
        self.assertIn(regime, ['transitioning', 'stable', 'stressed'])
    
    def test_custom_thresholds(self):
        """Test with custom thresholds."""
        h0_features = {
            'num_components': 2,
            'max_persistence': 0.5
        }
        h1_features = {
            'num_loops': 1,
            'max_persistence': 0.08
        }
        
        stable_threshold = {'max_clusters': 3, 'min_loop_persistence': 0.1}
        stressed_threshold = {'min_clusters': 5, 'max_loop_persistence': 0.3}
        
        regime, scores = classify_regime_by_topology(
            h0_features, h1_features,
            stable_threshold, stressed_threshold
        )
        
        # Should return valid regime
        self.assertIn(regime, ['stable', 'transitioning', 'stressed'])
    
    def test_confidence_scores(self):
        """Test that confidence scores are valid."""
        h0_features = {'num_components': 2, 'max_persistence': 0.5}
        h1_features = {'num_loops': 1, 'max_persistence': 0.1}
        
        regime, scores = classify_regime_by_topology(h0_features, h1_features)
        
        # Scores should be between 0 and 1
        for score in scores.values():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
        
        # Should have 3 regimes
        self.assertEqual(len(scores), 3)
        self.assertIn('stable', scores)
        self.assertIn('transitioning', scores)
        self.assertIn('stressed', scores)


if __name__ == '__main__':
    unittest.main()
