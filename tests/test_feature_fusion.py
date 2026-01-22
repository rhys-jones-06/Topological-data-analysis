"""
Unit tests for feature fusion module.
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tda.feature_fusion import (
    FeatureFusion,
    create_local_features,
    create_global_features_vector
)


class TestFeatureFusion(unittest.TestCase):
    """Test cases for FeatureFusion class."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.local_features = np.random.randn(10, 5)
        self.global_features = np.random.randn(10, 3)
    
    def test_initialization(self):
        """Test FeatureFusion initialization."""
        fusion = FeatureFusion(normalize=True, local_weight=0.6, global_weight=0.4)
        self.assertEqual(fusion.local_weight, 0.6)
        self.assertEqual(fusion.global_weight, 0.4)
        self.assertTrue(fusion.normalize)
    
    def test_fit_transform(self):
        """Test fit_transform method."""
        fusion = FeatureFusion()
        fused = fusion.fit_transform(self.local_features, self.global_features)
        
        # Check output shape
        expected_features = self.local_features.shape[1] + self.global_features.shape[1]
        self.assertEqual(fused.shape, (10, expected_features))
        
        # Check that model is fitted
        self.assertTrue(fusion._is_fitted)
    
    def test_transform_after_fit(self):
        """Test transform on new data after fitting."""
        fusion = FeatureFusion()
        fusion.fit(self.local_features, self.global_features)
        
        # Transform new data
        new_local = np.random.randn(5, 5)
        new_global = np.random.randn(5, 3)
        fused = fusion.transform(new_local, new_global)
        
        self.assertEqual(fused.shape, (5, 8))
    
    def test_without_normalization(self):
        """Test feature fusion without normalization."""
        fusion = FeatureFusion(normalize=False)
        fused = fusion.fit_transform(self.local_features, self.global_features)
        
        # Without normalization, should just concatenate weighted features
        self.assertEqual(fused.shape, (10, 8))
    
    def test_dict_global_features(self):
        """Test with dictionary global features."""
        fusion = FeatureFusion()
        
        # Create dict of global features
        global_dict = {
            'H_0': {'num_components': 3, 'max_persistence': 0.5},
            'H_1': {'num_loops': 2, 'max_persistence': 0.3}
        }
        
        fused = fusion.fit_transform(self.local_features, global_dict)
        
        # Should have local features + 4 global features
        self.assertEqual(fused.shape[1], 5 + 4)
    
    def test_1d_input(self):
        """Test with 1D input arrays."""
        fusion = FeatureFusion()
        
        local_1d = np.array([1, 2, 3, 4, 5])
        global_1d = np.array([6, 7, 8])
        
        fused = fusion.fit_transform(local_1d, global_1d)
        
        # Should reshape to 2D
        self.assertEqual(fused.ndim, 2)
        # Local has 5 elements, global has 3 elements
        # After reshaping: local (5, 1), global (1, 3) repeated to (5, 3)
        # Final shape: (5, 1+3) = (5, 4)
        self.assertEqual(fused.shape, (5, 4))


class TestCreateLocalFeatures(unittest.TestCase):
    """Test cases for create_local_features function."""
    
    def test_basic_creation(self):
        """Test basic local feature creation."""
        residuals = np.array([0.1, -0.2, 0.3])
        diffusion = np.array([0.05, -0.1, 0.15])
        
        features = create_local_features(residuals, diffusion)
        
        # Should concatenate residuals and diffusion
        self.assertEqual(features.shape, (1, 6))
        np.testing.assert_array_equal(features[0, :3], residuals)
        np.testing.assert_array_equal(features[0, 3:], diffusion)
    
    def test_with_laplacian_spectrum(self):
        """Test with Laplacian spectrum."""
        residuals = np.array([0.1, -0.2, 0.3])
        diffusion = np.array([0.05, -0.1, 0.15])
        spectrum = np.array([0.0, 1.5, 2.0])
        
        features = create_local_features(residuals, diffusion, spectrum)
        
        # Should include spectrum
        self.assertEqual(features.shape, (1, 9))
    
    def test_2d_input(self):
        """Test with 2D input."""
        residuals = np.random.randn(5, 3)
        diffusion = np.random.randn(5, 3)
        
        features = create_local_features(residuals, diffusion)
        
        self.assertEqual(features.shape, (5, 6))


class TestCreateGlobalFeaturesVector(unittest.TestCase):
    """Test cases for create_global_features_vector function."""
    
    def test_basic_creation(self):
        """Test basic global feature vector creation."""
        h0_features = {
            'num_components': 3,
            'max_persistence': 0.5,
            'mean_persistence': 0.3,
            'std_persistence': 0.1,
            'persistence_entropy': 0.8
        }
        
        h1_features = {
            'num_loops': 2,
            'max_persistence': 0.4,
            'mean_persistence': 0.25,
            'std_persistence': 0.08,
            'persistence_entropy': 0.6,
            'total_persistence': 0.5
        }
        
        features = create_global_features_vector(h0_features, h1_features)
        
        # Should have 11 features (5 from H0 + 6 from H1)
        self.assertEqual(features.shape, (1, 11))
        self.assertEqual(features[0, 0], 3)  # num_components
        self.assertEqual(features[0, 5], 2)  # num_loops
    
    def test_with_additional_features(self):
        """Test with additional features."""
        h0_features = {'num_components': 3, 'max_persistence': 0.5,
                      'mean_persistence': 0.3, 'std_persistence': 0.1,
                      'persistence_entropy': 0.8}
        h1_features = {'num_loops': 2, 'max_persistence': 0.4,
                      'mean_persistence': 0.25, 'std_persistence': 0.08,
                      'persistence_entropy': 0.6, 'total_persistence': 0.5}
        additional = {'custom_metric': 0.9}
        
        features = create_global_features_vector(h0_features, h1_features, additional)
        
        # Should have 12 features
        self.assertEqual(features.shape, (1, 12))
    
    def test_missing_keys(self):
        """Test with missing keys (should default to 0)."""
        h0_features = {'num_components': 3}
        h1_features = {'num_loops': 2}
        
        features = create_global_features_vector(h0_features, h1_features)
        
        # Should still create vector with defaults
        self.assertEqual(features.shape, (1, 11))
        self.assertEqual(features[0, 0], 3)
        self.assertEqual(features[0, 1], 0.0)  # default max_persistence


if __name__ == '__main__':
    unittest.main()
