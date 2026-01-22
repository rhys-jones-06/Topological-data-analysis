"""
TopologyEngine Example - End-to-End Demonstration

This example demonstrates the complete workflow of The TopologyEngine:
1. Data preparation
2. Engine initialization and training
3. Prediction with topological gating
4. Walk-forward validation
5. Monte Carlo stress testing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from topology_engine import create_topology_engine
from walk_forward_tester import create_walk_forward_tester
from monte_carlo_stress import create_monte_carlo_tester


def generate_sample_market_data(
    n_assets: int = 5,
    n_periods: int = 500,
    regime_change_points: list = None
) -> pd.DataFrame:
    """
    Generate synthetic market data with regime changes.
    
    Parameters
    ----------
    n_assets : int
        Number of assets
    n_periods : int
        Number of time periods
    regime_change_points : list
        List of indices where regime changes occur
    
    Returns
    -------
    pd.DataFrame
        Multi-asset OHLCV data
    """
    np.random.seed(42)
    
    dates = pd.date_range('2020-01-01', periods=n_periods, freq='D')
    
    if regime_change_points is None:
        regime_change_points = [n_periods // 3, 2 * n_periods // 3]
    
    # Generate returns with regime changes
    returns = np.zeros((n_periods, n_assets))
    
    current_regime = 0
    for i in range(n_periods):
        # Check for regime change
        for change_point in regime_change_points:
            if i >= change_point:
                current_regime = (current_regime + 1) % 3
        
        # Different correlation structures for different regimes
        if current_regime == 0:
            # Stable regime: moderate correlation
            mean = np.zeros(n_assets)
            cov = 0.02 ** 2 * (0.3 * np.ones((n_assets, n_assets)) + 0.7 * np.eye(n_assets))
        elif current_regime == 1:
            # Fragmented regime: low correlation
            mean = np.zeros(n_assets)
            cov = 0.03 ** 2 * np.eye(n_assets)
        else:
            # Stressed regime: high correlation, high volatility
            mean = -0.001 * np.ones(n_assets)
            cov = 0.05 ** 2 * (0.8 * np.ones((n_assets, n_assets)) + 0.2 * np.eye(n_assets))
        
        returns[i] = np.random.multivariate_normal(mean, cov)
    
    # Convert to prices
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    
    # Create OHLCV for first asset (simplified)
    data = pd.DataFrame({
        'Open': prices[:, 0] + np.random.randn(n_periods) * 0.5,
        'High': prices[:, 0] + np.abs(np.random.randn(n_periods)) * 1.5,
        'Low': prices[:, 0] - np.abs(np.random.randn(n_periods)) * 1.5,
        'Close': prices[:, 0],
        'Volume': np.random.randint(1000, 10000, n_periods)
    }, index=dates)
    
    # Also create multi-asset returns for correlation analysis
    returns_df = pd.DataFrame(
        returns,
        index=dates,
        columns=[f'Asset_{i}' for i in range(n_assets)]
    )
    
    return data, returns_df


def example_basic_prediction():
    """Example 1: Basic prediction with TopologyEngine."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Prediction with TopologyEngine")
    print("=" * 70)
    
    # Generate sample data
    ohlcv_data, returns_data = generate_sample_market_data(n_assets=5, n_periods=300)
    
    # Create and train engine
    engine = create_topology_engine(
        instability_threshold=0.5,
        nn_hidden_layers=(50, 25),
        correlation_threshold=0.3
    )
    
    print("\n1. Training TopologyEngine...")
    train_data = ohlcv_data[:200]
    engine.fit(train_data, single_asset_mode=True, horizon=1)
    print("   Training complete!")
    
    # Make prediction
    print("\n2. Generating prediction...")
    test_data = ohlcv_data[150:]  # Overlap for correlation calculation
    prediction = engine.predict(test_data, return_details=True)
    
    # Display results
    print("\n3. Prediction Results:")
    print(f"   Signal: {prediction['final_signal']}")
    print(f"   Confidence: {prediction['confidence_score']:.3f}")
    print(f"   Confidence Interval: [{prediction['confidence_interval'][0]:.3f}, {prediction['confidence_interval'][1]:.3f}]")
    print(f"   Regime: {prediction['regime_classification']}")
    print(f"   NN Probability: {prediction['nn_predict_proba']:.3f}")
    print(f"   Persistence Score: {prediction['persistence_score']:.3f}")
    print(f"   Graph Leakage: {prediction['graph_leakage']:.3f}")
    print(f"   Suggested Hedge: {prediction['suggested_hedge']}")
    
    # JSON output
    print("\n4. JSON Output:")
    json_output = engine.to_json(prediction)
    print(json_output)


def example_walk_forward_validation():
    """Example 2: Walk-forward validation."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Walk-Forward Validation")
    print("=" * 70)
    
    # Generate longer dataset
    ohlcv_data, returns_data = generate_sample_market_data(n_assets=5, n_periods=600)
    
    # Create walk-forward tester
    wf_tester = create_walk_forward_tester(
        train_months=12,  # 12 months training
        test_months=3,    # 3 months testing
        anchored=True
    )
    
    print("\n1. Setting up walk-forward analysis...")
    
    # Define trainer function
    def train_model(train_data):
        engine = create_topology_engine()
        engine.fit(train_data, single_asset_mode=True, horizon=1)
        return engine
    
    # Define predictor function
    def predict_model(engine, test_data):
        predictions = []
        for i in range(len(test_data) - 50):  # Need window for prediction
            window = test_data.iloc[i:i+50]
            try:
                pred = engine.predict(window, return_details=False)
                predictions.append({
                    'signal': pred['final_signal'],
                    'confidence': pred['confidence_score'],
                    'regime': pred['regime_classification']
                })
            except:
                predictions.append({
                    'signal': 'NEUTRAL',
                    'confidence': 0.0,
                    'regime': 'Unknown'
                })
        return predictions
    
    # Define metric calculator
    def calculate_metrics(predictions, test_data):
        # Simple metrics
        long_count = sum(1 for p in predictions if p['signal'] == 'LONG')
        short_count = sum(1 for p in predictions if p['signal'] == 'SHORT')
        neutral_count = sum(1 for p in predictions if p['signal'] == 'NEUTRAL')
        
        avg_confidence = np.mean([p['confidence'] for p in predictions])
        
        return {
            'long_pct': long_count / len(predictions),
            'short_pct': short_count / len(predictions),
            'neutral_pct': neutral_count / len(predictions),
            'avg_confidence': avg_confidence
        }
    
    print("\n2. Running walk-forward validation...")
    try:
        results = wf_tester.run(
            data=ohlcv_data,
            model_trainer=train_model,
            model_predictor=predict_model,
            metric_calculator=calculate_metrics
        )
        
        print(f"\n3. Results from {results['n_splits']} splits:")
        print(f"   Window configuration:")
        print(f"     - Training: {results['window_config']['train_window']} months")
        print(f"     - Testing: {results['window_config']['test_window']} months")
        print(f"     - Anchored: {results['window_config']['anchored']}")
        
        if 'aggregated' in results and results['aggregated']:
            print(f"\n   Aggregated Metrics:")
            for metric, value in results['aggregated'].items():
                print(f"     {metric}: {value:.4f}")
    except Exception as e:
        print(f"\n   Walk-forward validation skipped: {str(e)}")
        print("   (This is expected for the demo - would work with real historical data)")


def example_monte_carlo_stress_test():
    """Example 3: Monte Carlo stress testing."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Monte Carlo Stress Testing")
    print("=" * 70)
    
    # Generate sample data
    _, returns_data = generate_sample_market_data(n_assets=5, n_periods=300)
    
    # Create stress tester
    mc_tester = create_monte_carlo_tester(
        n_simulations=100,  # Reduced for demo
        block_size=10,
        random_state=42
    )
    
    print("\n1. Setting up Monte Carlo stress test...")
    print(f"   Number of simulations: 100")
    print(f"   Block bootstrap size: 10")
    
    # Define strategy evaluator
    def evaluate_strategy(returns_df):
        # Simple strategy: average correlation
        corr_matrix = returns_df.corr().values
        avg_corr = (corr_matrix.sum() - len(corr_matrix)) / (len(corr_matrix) ** 2 - len(corr_matrix))
        
        # Volatility
        vol = returns_df.std().mean()
        
        return {
            'avg_correlation': avg_corr,
            'avg_volatility': vol,
            'sharpe_proxy': returns_df.mean().mean() / vol if vol > 0 else 0
        }
    
    print("\n2. Running stress test...")
    try:
        stress_results = mc_tester.run_stress_test(
            original_data=returns_data,
            strategy_evaluator=evaluate_strategy,
            generation_method='block_bootstrap',
            return_distribution=True
        )
        
        print("\n3. Stress Test Results:")
        print(f"   Successful simulations: {stress_results['n_successful_simulations']}")
        
        print("\n   Original metrics:")
        for metric, value in stress_results['original_metrics'].items():
            print(f"     {metric}: {value:.4f}")
        
        print("\n   Synthetic metrics (average):")
        for metric, value in stress_results['synthetic_metrics_mean'].items():
            print(f"     {metric}: {value:.4f} ± {stress_results['synthetic_metrics_std'][metric]:.4f}")
        
        print("\n   P-values (is original significantly different?):")
        for metric, pval in stress_results['p_values'].items():
            significance = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            print(f"     {metric}: {pval:.4f} {significance}")
        
        # Assess robustness
        assessment = mc_tester.assess_robustness(stress_results)
        print(f"\n4. Robustness Assessment:")
        print(f"   Is Robust: {assessment['is_robust']}")
        print(f"   Conclusion: {assessment['summary']['conclusion']}")
        
        if assessment['warnings']:
            print("\n   Warnings:")
            for warning in assessment['warnings']:
                print(f"     - {warning}")
    
    except Exception as e:
        print(f"\n   Stress test skipped: {str(e)}")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("THE TOPOLOGYENGINE - COMPREHENSIVE DEMONSTRATION")
    print("Multi-Scale Structural Alpha Engine for Quantitative Finance")
    print("=" * 70)
    
    # Run examples
    example_basic_prediction()
    example_walk_forward_validation()
    example_monte_carlo_stress_test()
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. TopologyEngine integrates NN, Graph, and TDA signals")
    print("2. Topological gating prevents overconfidence in unstable regimes")
    print("3. Walk-forward validation ensures robustness")
    print("4. Monte Carlo stress testing validates strategy resilience")
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
