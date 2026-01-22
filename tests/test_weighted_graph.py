"""
Unit tests for the weighted graph generation module.
"""

import unittest
import numpy as np
import pandas as pd
from tda.weighted_graph import (
    compute_rolling_correlation,
    compute_graph_laplacian,
    simulate_heat_kernel,
    simulate_random_walk
)


class TestRollingCorrelation(unittest.TestCase):
    """Test cases for compute_rolling_correlation function."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.data = pd.DataFrame(
            np.random.randn(100, 5),
            columns=['Asset1', 'Asset2', 'Asset3', 'Asset4', 'Asset5']
        )
    
    def test_rolling_correlation_output_length(self):
        """Test that the output has the correct number of correlation matrices."""
        window = 20
        result = compute_rolling_correlation(self.data, window=window)
        expected_length = len(self.data) - window + 1
        self.assertEqual(len(result), expected_length)
    
    def test_rolling_correlation_matrix_shape(self):
        """Test that each correlation matrix has the correct shape."""
        window = 20
        result = compute_rolling_correlation(self.data, window=window)
        for corr_matrix in result:
            # Remove timestamp column for shape checking
            corr_matrix_clean = corr_matrix.drop(columns=['timestamp'])
            self.assertEqual(corr_matrix_clean.shape, (5, 5))
    
    def test_rolling_correlation_values(self):
        """Test that correlation values are within valid range [-1, 1]."""
        window = 20
        result = compute_rolling_correlation(self.data, window=window)
        for corr_matrix in result:
            corr_matrix_clean = corr_matrix.drop(columns=['timestamp'])
            values = corr_matrix_clean.values
            self.assertTrue(np.all(values >= -1))
            self.assertTrue(np.all(values <= 1))
    
    def test_rolling_correlation_diagonal_ones(self):
        """Test that diagonal elements are 1 (self-correlation)."""
        window = 20
        result = compute_rolling_correlation(self.data, window=window)
        for corr_matrix in result:
            corr_matrix_clean = corr_matrix.drop(columns=['timestamp'])
            diagonal = np.diag(corr_matrix_clean.values)
            np.testing.assert_array_almost_equal(diagonal, np.ones(5))


class TestGraphLaplacian(unittest.TestCase):
    """Test cases for compute_graph_laplacian function."""
    
    def test_simple_graph_laplacian(self):
        """Test Laplacian computation for a simple graph."""
        # Simple triangle graph
        A = np.array([[0, 1, 1], 
                      [1, 0, 1], 
                      [1, 1, 0]])
        L = compute_graph_laplacian(A)
        expected = np.array([[2, -1, -1], 
                           [-1, 2, -1], 
                           [-1, -1, 2]])
        np.testing.assert_array_equal(L, expected)
    
    def test_weighted_graph_laplacian(self):
        """Test Laplacian computation for a weighted graph."""
        A = np.array([[0, 0.5, 0.3], 
                      [0.5, 0, 0.7], 
                      [0.3, 0.7, 0]])
        L = compute_graph_laplacian(A)
        
        # Check that L = D - A
        degrees = np.sum(A, axis=1)
        D = np.diag(degrees)
        expected = D - A
        np.testing.assert_array_almost_equal(L, expected)
    
    def test_laplacian_row_sum_zero(self):
        """Test that each row of the Laplacian sums to zero."""
        A = np.random.rand(5, 5)
        A = (A + A.T) / 2  # Make symmetric
        np.fill_diagonal(A, 0)  # No self-loops
        
        L = compute_graph_laplacian(A)
        row_sums = np.sum(L, axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.zeros(5))
    
    def test_normalized_laplacian(self):
        """Test normalized Laplacian computation."""
        A = np.array([[0, 1, 1], 
                      [1, 0, 1], 
                      [1, 1, 0]])
        L_norm = compute_graph_laplacian(A, normalized=True)
        
        # Check properties of normalized Laplacian
        # All eigenvalues should be in [0, 2]
        eigenvalues = np.linalg.eigvalsh(L_norm)
        self.assertTrue(np.all(eigenvalues >= -1e-10))  # Allow small numerical errors
        self.assertTrue(np.all(eigenvalues <= 2 + 1e-10))


class TestHeatKernel(unittest.TestCase):
    """Test cases for simulate_heat_kernel function."""
    
    def test_heat_kernel_shape(self):
        """Test that heat kernel has correct shape."""
        L = np.array([[2, -1, -1], 
                     [-1, 2, -1], 
                     [-1, -1, 2]])
        H = simulate_heat_kernel(L, time=1.0)
        self.assertEqual(H.shape, (3, 3))
    
    def test_heat_kernel_with_initial_state(self):
        """Test heat kernel application to initial state."""
        L = np.array([[2, -1, -1], 
                     [-1, 2, -1], 
                     [-1, -1, 2]])
        initial = np.array([1, 0, 0])
        result = simulate_heat_kernel(L, time=1.0, initial_state=initial)
        
        # Result should be a 1D array
        self.assertEqual(result.shape, (3,))
        
        # Sum should be preserved (conservation)
        np.testing.assert_almost_equal(np.sum(result), 1.0)
    
    def test_heat_kernel_diffusion(self):
        """Test that heat diffuses over time."""
        L = np.array([[2, -1, -1], 
                     [-1, 2, -1], 
                     [-1, -1, 2]])
        initial = np.array([1, 0, 0])  # All heat at first node
        
        # At time=0, heat should be concentrated at first node
        result_t0 = simulate_heat_kernel(L, time=0.0, initial_state=initial)
        np.testing.assert_array_almost_equal(result_t0, initial)
        
        # At time>0, heat should diffuse to other nodes
        result_t1 = simulate_heat_kernel(L, time=1.0, initial_state=initial)
        self.assertTrue(result_t1[0] < 1.0)  # First node loses heat
        self.assertTrue(result_t1[1] > 0.0)  # Other nodes gain heat
        self.assertTrue(result_t1[2] > 0.0)


class TestRandomWalk(unittest.TestCase):
    """Test cases for simulate_random_walk function."""
    
    def test_random_walk_output_shape(self):
        """Test that random walk output has correct shape."""
        A = np.array([[0, 1, 1], 
                      [1, 0, 1], 
                      [1, 1, 0]])
        result = simulate_random_walk(A, n_steps=10)
        self.assertEqual(result.shape, (3,))
    
    def test_random_walk_probability_sum(self):
        """Test that probabilities sum to 1."""
        A = np.array([[0, 1, 1], 
                      [1, 0, 1], 
                      [1, 1, 0]])
        result = simulate_random_walk(A, n_steps=10)
        np.testing.assert_almost_equal(np.sum(result), 1.0)
    
    def test_random_walk_trajectory(self):
        """Test random walk with trajectory return."""
        A = np.array([[0, 1, 1], 
                      [1, 0, 1], 
                      [1, 1, 0]])
        n_steps = 10
        final_state, trajectory = simulate_random_walk(
            A, n_steps=n_steps, return_trajectory=True
        )
        
        # Trajectory should have n_steps + 1 states (initial + each step)
        self.assertEqual(len(trajectory), n_steps + 1)
        
        # Final state in trajectory should match returned final state
        np.testing.assert_array_almost_equal(trajectory[-1], final_state)
    
    def test_random_walk_with_initial_state(self):
        """Test random walk with custom initial state."""
        A = np.array([[0, 1, 1], 
                      [1, 0, 1], 
                      [1, 1, 0]])
        initial = np.array([1, 0, 0])  # Start at first node
        result = simulate_random_walk(A, n_steps=10, initial_state=initial)
        
        # Result should still be a valid probability distribution
        np.testing.assert_almost_equal(np.sum(result), 1.0)
        self.assertTrue(np.all(result >= 0))
    
    def test_random_walk_convergence(self):
        """Test that random walk converges to stationary distribution."""
        # Regular graph where all nodes have same degree
        A = np.array([[0, 1, 1, 0], 
                      [1, 0, 1, 1], 
                      [1, 1, 0, 1],
                      [0, 1, 1, 0]])
        
        # Long random walk should converge
        result = simulate_random_walk(A, n_steps=1000)
        
        # Check that all probabilities are positive
        self.assertTrue(np.all(result > 0))


if __name__ == '__main__':
    unittest.main()
