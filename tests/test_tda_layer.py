"""
Unit tests for the TDA layer module.
"""

import unittest
import numpy as np
import pandas as pd
from tda.tda_layer import (
    build_vietoris_rips_filtration,
    compute_persistent_homology,
    extract_h0_features,
    extract_h1_features,
    compute_persistence_landscape,
    compute_persistence_images,
    vectorize_persistence_diagrams,
    identify_market_regimes
)


class TestVietorisRipsFiltration(unittest.TestCase):
    """Test cases for Vietoris-Rips filtration."""
    
    def setUp(self):
        """Set up test data."""
        # Simple correlation matrix
        self.corr_matrix = np.array([
            [1.0, 0.8, 0.6],
            [0.8, 1.0, 0.7],
            [0.6, 0.7, 1.0]
        ])
    
    def test_vr_filtration_basic(self):
        """Test basic Vietoris-Rips filtration computation."""
        result = build_vietoris_rips_filtration(self.corr_matrix, max_dimension=1)
        
        # Check that result has expected keys
        self.assertIn('dgms', result)
        self.assertIn('distance_matrix', result)
        
        # Check that persistence diagrams exist for H_0 and H_1
        self.assertEqual(len(result['dgms']), 2)  # H_0 and H_1
    
    def test_distance_conversion(self):
        """Test that correlation is properly converted to distance."""
        result = build_vietoris_rips_filtration(self.corr_matrix)
        distance_matrix = result['distance_matrix']
        
        # Check diagonal is zero
        np.testing.assert_array_almost_equal(np.diag(distance_matrix), np.zeros(3))
        
        # Check distance formula: d = 1 - |corr|
        expected_distance = 1.0 - np.abs(self.corr_matrix)
        np.fill_diagonal(expected_distance, 0)
        np.testing.assert_array_almost_equal(distance_matrix, expected_distance)
    
    def test_max_dimension(self):
        """Test that max_dimension parameter works correctly."""
        result = build_vietoris_rips_filtration(self.corr_matrix, max_dimension=2)
        
        # Should have H_0, H_1, and H_2
        self.assertEqual(len(result['dgms']), 3)


class TestPersistentHomology(unittest.TestCase):
    """Test cases for persistent homology computation."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        # Create time series data
        self.returns = pd.DataFrame(
            np.random.randn(100, 5),
            columns=['Asset1', 'Asset2', 'Asset3', 'Asset4', 'Asset5']
        )
        self.corr_matrix = self.returns.corr().values
    
    def test_from_time_series(self):
        """Test computing persistence from time series data."""
        result = compute_persistent_homology(self.returns, max_dimension=1)
        
        self.assertIn('dgms', result)
        self.assertEqual(len(result['dgms']), 2)
    
    def test_from_correlation_matrix(self):
        """Test computing persistence from correlation matrix."""
        result = compute_persistent_homology(
            self.corr_matrix,
            max_dimension=1,
            use_correlation=False
        )
        
        self.assertIn('dgms', result)
        self.assertEqual(len(result['dgms']), 2)
    
    def test_with_dataframe(self):
        """Test that function works with pandas DataFrame."""
        result = compute_persistent_homology(self.returns, max_dimension=1)
        self.assertIsNotNone(result)


class TestH0Features(unittest.TestCase):
    """Test cases for H_0 feature extraction."""
    
    def test_h0_features_basic(self):
        """Test basic H_0 feature extraction."""
        # Create a simple persistence diagram
        dgm = np.array([
            [0.0, 0.1],
            [0.0, 0.3],
            [0.0, 0.5],
            [0.0, np.inf]  # One component that never dies
        ])
        
        features = extract_h0_features(dgm)
        
        # Check all expected keys are present
        self.assertIn('num_components', features)
        self.assertIn('max_persistence', features)
        self.assertIn('mean_persistence', features)
        self.assertIn('std_persistence', features)
        self.assertIn('persistence_entropy', features)
        
        # Check values
        self.assertEqual(features['num_components'], 3)  # Excluding infinite
        self.assertAlmostEqual(features['max_persistence'], 0.5)
        self.assertGreater(features['persistence_entropy'], 0)
    
    def test_h0_features_empty(self):
        """Test H_0 features with empty diagram."""
        dgm = np.array([]).reshape(0, 2)
        features = extract_h0_features(dgm)
        
        self.assertEqual(features['num_components'], 0)
        self.assertEqual(features['max_persistence'], 0.0)
    
    def test_h0_features_only_infinite(self):
        """Test H_0 features with only infinite component."""
        dgm = np.array([[0.0, np.inf]])
        features = extract_h0_features(dgm)
        
        self.assertEqual(features['num_components'], 0)


class TestH1Features(unittest.TestCase):
    """Test cases for H_1 feature extraction."""
    
    def test_h1_features_basic(self):
        """Test basic H_1 feature extraction."""
        # Create a simple persistence diagram
        dgm = np.array([
            [0.2, 0.5],
            [0.3, 0.6],
            [0.4, 0.7]
        ])
        
        features = extract_h1_features(dgm)
        
        # Check all expected keys are present
        self.assertIn('num_loops', features)
        self.assertIn('max_persistence', features)
        self.assertIn('mean_persistence', features)
        self.assertIn('std_persistence', features)
        self.assertIn('persistence_entropy', features)
        self.assertIn('total_persistence', features)
        
        # Check values
        self.assertEqual(features['num_loops'], 3)
        self.assertAlmostEqual(features['max_persistence'], 0.3)
        self.assertAlmostEqual(features['total_persistence'], 0.9)
    
    def test_h1_features_empty(self):
        """Test H_1 features with empty diagram."""
        dgm = np.array([]).reshape(0, 2)
        features = extract_h1_features(dgm)
        
        self.assertEqual(features['num_loops'], 0)
        self.assertEqual(features['total_persistence'], 0.0)


class TestPersistenceLandscape(unittest.TestCase):
    """Test cases for persistence landscape computation."""
    
    def test_landscape_shape(self):
        """Test that landscape has correct shape."""
        dgm = np.array([
            [0.0, 0.5],
            [0.2, 0.7],
            [0.3, 0.6]
        ])
        
        k = 3
        num_samples = 50
        landscape = compute_persistence_landscape(dgm, k=k, num_samples=num_samples)
        
        self.assertEqual(landscape.shape, (k, num_samples))
    
    def test_landscape_empty_diagram(self):
        """Test landscape computation with empty diagram."""
        dgm = np.array([]).reshape(0, 2)
        landscape = compute_persistence_landscape(dgm, k=3, num_samples=50)
        
        # Should return zeros
        np.testing.assert_array_equal(landscape, np.zeros((3, 50)))
    
    def test_landscape_values(self):
        """Test that landscape values are non-negative."""
        dgm = np.array([
            [0.0, 0.5],
            [0.2, 0.7]
        ])
        
        landscape = compute_persistence_landscape(dgm, k=2, num_samples=50)
        
        # All values should be non-negative
        self.assertTrue(np.all(landscape >= 0))


class TestPersistenceImages(unittest.TestCase):
    """Test cases for persistence image computation."""
    
    def test_image_shape(self):
        """Test that image has correct shape."""
        dgm = np.array([
            [0.0, 0.5],
            [0.2, 0.7],
            [0.3, 0.6]
        ])
        
        resolution = (10, 10)
        image = compute_persistence_images(dgm, resolution=resolution)
        
        self.assertEqual(image.shape, resolution)
    
    def test_image_empty_diagram(self):
        """Test image computation with empty diagram."""
        dgm = np.array([]).reshape(0, 2)
        resolution = (10, 10)
        image = compute_persistence_images(dgm, resolution=resolution)
        
        # Should return zeros
        np.testing.assert_array_equal(image, np.zeros(resolution))
    
    def test_image_values(self):
        """Test that image values are non-negative."""
        dgm = np.array([
            [0.0, 0.5],
            [0.2, 0.7]
        ])
        
        image = compute_persistence_images(dgm, resolution=(10, 10))
        
        # All values should be non-negative
        self.assertTrue(np.all(image >= 0))


class TestVectorizePersistenceDiagrams(unittest.TestCase):
    """Test cases for persistence diagram vectorization."""
    
    def setUp(self):
        """Set up test data."""
        self.corr_matrix = np.array([
            [1.0, 0.8, 0.6],
            [0.8, 1.0, 0.7],
            [0.6, 0.7, 1.0]
        ])
        self.persistence_result = build_vietoris_rips_filtration(
            self.corr_matrix,
            max_dimension=1
        )
    
    def test_vectorize_statistics(self):
        """Test vectorization using statistics method."""
        features = vectorize_persistence_diagrams(
            self.persistence_result,
            method='statistics'
        )
        
        # Should have H_0 and H_1
        self.assertIn('H_0', features)
        self.assertIn('H_1', features)
        
        # Check that features are dictionaries with expected keys
        self.assertIsInstance(features['H_0'], dict)
        self.assertIsInstance(features['H_1'], dict)
    
    def test_vectorize_landscape(self):
        """Test vectorization using landscape method."""
        features = vectorize_persistence_diagrams(
            self.persistence_result,
            method='landscape',
            k=3,
            num_samples=50
        )
        
        # Should have H_0 and H_1
        self.assertIn('H_0', features)
        self.assertIn('H_1', features)
        
        # Check shapes
        self.assertEqual(features['H_0'].shape, (3, 50))
        self.assertEqual(features['H_1'].shape, (3, 50))
    
    def test_vectorize_image(self):
        """Test vectorization using image method."""
        features = vectorize_persistence_diagrams(
            self.persistence_result,
            method='image',
            resolution=(10, 10)
        )
        
        # Should have H_0 and H_1
        self.assertIn('H_0', features)
        self.assertIn('H_1', features)
        
        # Check shapes
        self.assertEqual(features['H_0'].shape, (10, 10))
        self.assertEqual(features['H_1'].shape, (10, 10))


class TestIdentifyMarketRegimes(unittest.TestCase):
    """Test cases for market regime identification."""
    
    def test_fragmented_with_cycles(self):
        """Test identification of fragmented market with cycles."""
        h0_features = {
            'num_components': 5,
            'max_persistence': 0.3,
            'mean_persistence': 0.2,
            'std_persistence': 0.1,
            'persistence_entropy': 1.5
        }
        h1_features = {
            'num_loops': 2,
            'max_persistence': 0.15,
            'mean_persistence': 0.12,
            'std_persistence': 0.03,
            'persistence_entropy': 0.8,
            'total_persistence': 0.24
        }
        
        regime = identify_market_regimes(h0_features, h1_features)
        
        self.assertTrue(regime['is_fragmented'])
        self.assertTrue(regime['has_cycles'])
        self.assertEqual(regime['regime'], "Fragmented with feedback loops")
    
    def test_unified_market(self):
        """Test identification of unified market."""
        h0_features = {
            'num_components': 1,
            'max_persistence': 0.1,
            'mean_persistence': 0.1,
            'std_persistence': 0.0,
            'persistence_entropy': 0.0
        }
        h1_features = {
            'num_loops': 0,
            'max_persistence': 0.0,
            'mean_persistence': 0.0,
            'std_persistence': 0.0,
            'persistence_entropy': 0.0,
            'total_persistence': 0.0
        }
        
        regime = identify_market_regimes(h0_features, h1_features)
        
        self.assertFalse(regime['is_fragmented'])
        self.assertFalse(regime['has_cycles'])
        self.assertEqual(regime['regime'], "Unified (single connected market)")
    
    def test_fragmented_only(self):
        """Test identification of fragmented market without cycles."""
        h0_features = {
            'num_components': 4,
            'max_persistence': 0.2,
            'mean_persistence': 0.15,
            'std_persistence': 0.05,
            'persistence_entropy': 1.2
        }
        h1_features = {
            'num_loops': 1,
            'max_persistence': 0.05,  # Below threshold
            'mean_persistence': 0.05,
            'std_persistence': 0.0,
            'persistence_entropy': 0.0,
            'total_persistence': 0.05
        }
        
        regime = identify_market_regimes(h0_features, h1_features)
        
        self.assertTrue(regime['is_fragmented'])
        self.assertFalse(regime['has_cycles'])  # Below persistence threshold
        self.assertEqual(regime['regime'], "Fragmented (multiple clusters)")


if __name__ == '__main__':
    unittest.main()
