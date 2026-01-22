"""
Unit tests for the residual analysis module.
"""

import unittest
import numpy as np
import pandas as pd
from tda.residual_analysis import compute_residuals


class TestComputeResiduals(unittest.TestCase):
    """Test cases for compute_residuals function."""
    
    def test_residuals_numpy_arrays(self):
        """Test residual computation with numpy arrays."""
        actual = np.array([0.05, 0.03, -0.02, 0.04])
        expected = np.array([0.02, 0.02, 0.02, 0.02])
        residuals = compute_residuals(actual, expected)
        
        expected_residuals = np.array([0.03, 0.01, -0.04, 0.02])
        np.testing.assert_array_almost_equal(residuals, expected_residuals)
    
    def test_residuals_pandas_series(self):
        """Test residual computation with pandas Series."""
        actual = pd.Series([0.05, 0.03, -0.02, 0.04])
        expected = pd.Series([0.02, 0.02, 0.02, 0.02])
        residuals = compute_residuals(actual, expected)
        
        self.assertIsInstance(residuals, pd.Series)
        expected_residuals = pd.Series([0.03, 0.01, -0.04, 0.02])
        pd.testing.assert_series_equal(residuals, expected_residuals)
    
    def test_residuals_pandas_dataframe(self):
        """Test residual computation with pandas DataFrame."""
        actual = pd.DataFrame({
            'Asset1': [0.05, 0.03, -0.02],
            'Asset2': [0.04, -0.01, 0.06]
        })
        expected = pd.DataFrame({
            'Asset1': [0.02, 0.02, 0.02],
            'Asset2': [0.02, 0.02, 0.02]
        })
        residuals = compute_residuals(actual, expected)
        
        self.assertIsInstance(residuals, pd.DataFrame)
        expected_residuals = pd.DataFrame({
            'Asset1': [0.03, 0.01, -0.04],
            'Asset2': [0.02, -0.03, 0.04]
        })
        pd.testing.assert_frame_equal(residuals, expected_residuals)
    
    def test_residuals_2d_numpy_array(self):
        """Test residual computation with 2D numpy arrays."""
        actual = np.array([[0.05, 0.04], [0.03, -0.01], [-0.02, 0.06]])
        expected = np.array([[0.02, 0.02], [0.02, 0.02], [0.02, 0.02]])
        residuals = compute_residuals(actual, expected)
        
        expected_residuals = np.array([[0.03, 0.02], [0.01, -0.03], [-0.04, 0.04]])
        np.testing.assert_array_almost_equal(residuals, expected_residuals)
    
    def test_residuals_shape_mismatch(self):
        """Test that ValueError is raised for shape mismatch."""
        actual = np.array([0.05, 0.03, -0.02])
        expected = np.array([0.02, 0.02])
        
        with self.assertRaises(ValueError) as context:
            compute_residuals(actual, expected)
        
        self.assertIn("Shape mismatch", str(context.exception))
    
    def test_residuals_zero_expected(self):
        """Test residuals when expected returns are zero."""
        actual = np.array([0.05, 0.03, -0.02, 0.04])
        expected = np.zeros(4)
        residuals = compute_residuals(actual, expected)
        
        # Residuals should equal actual returns when expected is zero
        np.testing.assert_array_equal(residuals, actual)
    
    def test_residuals_equal_values(self):
        """Test residuals when actual equals expected."""
        actual = np.array([0.02, 0.02, 0.02, 0.02])
        expected = np.array([0.02, 0.02, 0.02, 0.02])
        residuals = compute_residuals(actual, expected)
        
        # Residuals should be zero when actual equals expected
        np.testing.assert_array_almost_equal(residuals, np.zeros(4))
    
    def test_residuals_preserves_type(self):
        """Test that output type matches input type."""
        # Test with numpy array
        actual_np = np.array([0.05, 0.03])
        expected_np = np.array([0.02, 0.02])
        residuals_np = compute_residuals(actual_np, expected_np)
        self.assertIsInstance(residuals_np, np.ndarray)
        
        # Test with pandas Series
        actual_series = pd.Series([0.05, 0.03])
        expected_series = pd.Series([0.02, 0.02])
        residuals_series = compute_residuals(actual_series, expected_series)
        self.assertIsInstance(residuals_series, pd.Series)
        
        # Test with pandas DataFrame
        actual_df = pd.DataFrame({'A': [0.05, 0.03]})
        expected_df = pd.DataFrame({'A': [0.02, 0.02]})
        residuals_df = compute_residuals(actual_df, expected_df)
        self.assertIsInstance(residuals_df, pd.DataFrame)


if __name__ == '__main__':
    unittest.main()
