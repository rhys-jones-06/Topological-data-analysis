"""
Unit tests for The TopologyEngine
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn_strategy import NeuralNetStrategy, create_sample_nn_strategy
from graph_diffusion import GraphDiffusion, create_sample_graph_diffusion
from tda_homology import TDAHomology, create_sample_tda_homology
from topology_engine import (
    DataOrchestrator,
    GatingNetwork,
    TopologyEngine,
    create_topology_engine
)


class TestNeuralNetStrategy(unittest.TestCase):
    """Test cases for Neural Network Strategy."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create synthetic OHLCV data
        n_periods = 200
        dates = pd.date_range('2020-01-01', periods=n_periods, freq='D')
        
        prices = 100 + np.cumsum(np.random.randn(n_periods) * 2)
        
        self.ohlcv_data = pd.DataFrame({
            'Open': prices + np.random.randn(n_periods) * 0.5,
            'High': prices + np.abs(np.random.randn(n_periods)) * 1.5,
            'Low': prices - np.abs(np.random.randn(n_periods)) * 1.5,
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, n_periods)
        }, index=dates)
    
    def test_nn_initialization(self):
        """Test NN strategy initialization."""
        nn = NeuralNetStrategy()
        self.assertIsNotNone(nn)
        self.assertFalse(nn.is_fitted)
    
    def test_nn_training(self):
        """Test NN strategy training."""
        nn = NeuralNetStrategy(random_state=42)
        nn.fit(self.ohlcv_data, horizon=1)
        self.assertTrue(nn.is_fitted)
    
    def test_nn_prediction(self):
        """Test NN strategy prediction."""
        nn = NeuralNetStrategy(random_state=42)
        nn.fit(self.ohlcv_data[:150], horizon=1)
        
        proba = nn.predict_proba(self.ohlcv_data[150:])
        
        self.assertIsInstance(proba, float)
        self.assertGreaterEqual(proba, 0.0)
        self.assertLessEqual(proba, 1.0)
    
    def test_factory_function(self):
        """Test factory function."""
        nn = create_sample_nn_strategy()
        self.assertIsInstance(nn, NeuralNetStrategy)


class TestGraphDiffusion(unittest.TestCase):
    """Test cases for Graph Diffusion."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create correlation matrix
        n_assets = 5
        self.correlation_matrix = np.random.rand(n_assets, n_assets)
        self.correlation_matrix = (self.correlation_matrix + self.correlation_matrix.T) / 2
        np.fill_diagonal(self.correlation_matrix, 1.0)
        
        self.asset_names = [f'Asset_{i}' for i in range(n_assets)]
    
    def test_graph_initialization(self):
        """Test graph diffusion initialization."""
        gd = GraphDiffusion()
        self.assertIsNotNone(gd)
    
    def test_build_graph(self):
        """Test graph building."""
        gd = GraphDiffusion(correlation_threshold=0.3)
        graph = gd.build_correlation_graph(self.correlation_matrix, self.asset_names)
        
        self.assertEqual(graph.number_of_nodes(), len(self.asset_names))
        self.assertGreater(graph.number_of_edges(), 0)
    
    def test_laplacian_computation(self):
        """Test Laplacian computation."""
        gd = GraphDiffusion()
        gd.build_correlation_graph(self.correlation_matrix, self.asset_names)
        L = gd.compute_laplacian(normalized=True)
        
        self.assertEqual(L.shape, (len(self.asset_names), len(self.asset_names)))
    
    def test_heat_diffusion(self):
        """Test heat kernel diffusion."""
        gd = GraphDiffusion()
        gd.build_correlation_graph(self.correlation_matrix, self.asset_names)
        gd.compute_laplacian()
        
        diffused = gd.heat_kernel_diffusion()
        
        self.assertEqual(len(diffused), len(self.asset_names))
        # With normalized Laplacian, sum may not be exactly 1.0
        self.assertGreater(np.sum(diffused), 0.9)
        self.assertLess(np.sum(diffused), 1.1)
    
    def test_leakage_score(self):
        """Test leakage score computation."""
        gd = GraphDiffusion()
        gd.build_correlation_graph(self.correlation_matrix, self.asset_names)
        gd.compute_laplacian()
        
        leakage = gd.compute_leakage_score()
        
        self.assertIsInstance(leakage, float)
        self.assertGreaterEqual(leakage, 0.0)
        self.assertLessEqual(leakage, 1.0)


class TestTDAHomology(unittest.TestCase):
    """Test cases for TDA Homology."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create correlation matrix
        n_assets = 5
        self.correlation_matrix = np.random.rand(n_assets, n_assets)
        self.correlation_matrix = (self.correlation_matrix + self.correlation_matrix.T) / 2
        np.fill_diagonal(self.correlation_matrix, 1.0)
    
    def test_tda_initialization(self):
        """Test TDA homology initialization."""
        tda = TDAHomology()
        self.assertIsNotNone(tda)
    
    def test_correlation_to_distance(self):
        """Test correlation to distance conversion."""
        tda = TDAHomology()
        distance = tda.correlation_to_distance(self.correlation_matrix)
        
        self.assertEqual(distance.shape, self.correlation_matrix.shape)
        self.assertAlmostEqual(distance[0, 0], 0.0, places=5)
    
    def test_persistence_diagrams(self):
        """Test persistence diagram computation."""
        tda = TDAHomology()
        diagrams = tda.compute_persistence_diagrams(self.correlation_matrix)
        
        self.assertIsNotNone(diagrams)
        self.assertGreater(len(diagrams), 0)
    
    def test_h0_features(self):
        """Test H0 feature extraction."""
        tda = TDAHomology()
        tda.compute_persistence_diagrams(self.correlation_matrix)
        features = tda.extract_h0_features()
        
        self.assertIn('num_components', features)
        self.assertIn('max_persistence', features)
    
    def test_persistence_score(self):
        """Test persistence score computation."""
        tda = TDAHomology()
        score = tda.compute_persistence_score(self.correlation_matrix)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_regime_classification(self):
        """Test regime classification."""
        tda = TDAHomology()
        regime_info = tda.classify_regime(self.correlation_matrix)
        
        self.assertIn('regime', regime_info)
        self.assertIn('confidence', regime_info)
        self.assertIn(regime_info['regime'], ['Stable', 'Fragmented', 'Trending', 'Stressed'])


class TestDataOrchestrator(unittest.TestCase):
    """Test cases for Data Orchestrator."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        n_periods = 100
        n_assets = 3
        dates = pd.date_range('2020-01-01', periods=n_periods, freq='D')
        
        # Create multi-asset data
        data = {}
        for i in range(n_assets):
            prices = 100 + np.cumsum(np.random.randn(n_periods))
            data[f'Asset_{i}'] = prices
        
        self.price_data = pd.DataFrame(data, index=dates)
    
    def test_orchestrator_initialization(self):
        """Test data orchestrator initialization."""
        orch = DataOrchestrator()
        self.assertIsNotNone(orch)
    
    def test_prepare_data(self):
        """Test data preparation."""
        orch = DataOrchestrator(window_size=50)
        data_package = orch.prepare_data(self.price_data)
        
        self.assertIn('correlation', data_package)
        self.assertIn('returns', data_package)
        self.assertEqual(data_package['correlation'].shape[0], len(self.price_data.columns))


class TestGatingNetwork(unittest.TestCase):
    """Test cases for Gating Network."""
    
    def test_gating_initialization(self):
        """Test gating network initialization."""
        gate = GatingNetwork()
        self.assertIsNotNone(gate)
    
    def test_signal_combination_normal(self):
        """Test signal combination in normal regime."""
        gate = GatingNetwork(instability_threshold=0.5)
        
        result = gate.combine_signals(
            nn_proba=0.7,
            persistence_score=0.3,
            regime='Stable',
            graph_leakage=0.2
        )
        
        self.assertIn('final_signal', result)
        self.assertIn('confidence_score', result)
        self.assertIn(result['final_signal'], ['LONG', 'SHORT', 'NEUTRAL'])
    
    def test_signal_combination_gated(self):
        """Test signal combination with gating."""
        gate = GatingNetwork(instability_threshold=0.5)
        
        result = gate.combine_signals(
            nn_proba=0.9,  # Strong NN signal
            persistence_score=0.8,  # High instability
            regime='Fragmented'
        )
        
        # Should be forced to NEUTRAL due to high persistence
        self.assertEqual(result['final_signal'], 'NEUTRAL')
        self.assertEqual(result['reason'], 'H1_instability_exceeded_threshold')  # Updated reason string


class TestTopologyEngine(unittest.TestCase):
    """Test cases for The TopologyEngine."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create synthetic OHLCV data
        n_periods = 200
        dates = pd.date_range('2020-01-01', periods=n_periods, freq='D')
        
        prices = 100 + np.cumsum(np.random.randn(n_periods) * 2)
        
        self.ohlcv_data = pd.DataFrame({
            'Open': prices + np.random.randn(n_periods) * 0.5,
            'High': prices + np.abs(np.random.randn(n_periods)) * 1.5,
            'Low': prices - np.abs(np.random.randn(n_periods)) * 1.5,
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, n_periods)
        }, index=dates)
    
    def test_engine_initialization(self):
        """Test topology engine initialization."""
        engine = TopologyEngine()
        self.assertIsNotNone(engine)
        self.assertFalse(engine.is_fitted)
    
    def test_engine_fit(self):
        """Test engine training."""
        engine = TopologyEngine()
        engine.fit(self.ohlcv_data[:150], single_asset_mode=True)
        self.assertTrue(engine.is_fitted)
    
    def test_engine_predict(self):
        """Test engine prediction."""
        engine = TopologyEngine()
        engine.fit(self.ohlcv_data[:150], single_asset_mode=True)
        
        prediction = engine.predict(self.ohlcv_data[100:], return_details=True)
        
        self.assertIn('final_signal', prediction)
        self.assertIn('confidence_score', prediction)
        self.assertIn('regime_classification', prediction)
        self.assertIn(prediction['final_signal'], ['LONG', 'SHORT', 'NEUTRAL'])
    
    def test_engine_json_output(self):
        """Test JSON output generation."""
        engine = TopologyEngine()
        engine.fit(self.ohlcv_data[:150], single_asset_mode=True)
        
        prediction = engine.predict(self.ohlcv_data[100:])
        json_output = engine.to_json(prediction)
        
        self.assertIsInstance(json_output, str)
        self.assertIn('final_signal', json_output)
    
    def test_factory_function(self):
        """Test factory function."""
        engine = create_topology_engine()
        self.assertIsInstance(engine, TopologyEngine)


if __name__ == '__main__':
    unittest.main()
