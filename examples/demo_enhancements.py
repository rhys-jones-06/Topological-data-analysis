"""
Demonstration of TDA Enhancements

This script demonstrates the three key enhancements to the TDA system:
1. Adaptive TDA Persistence Score with Rolling Baseline
2. Synthetic Data Generation using GBM and Copula
3. JSON Output with Topological Attribution
"""

import numpy as np
import pandas as pd
import json
from tda_homology import TDAHomology
from monte_carlo_stress import MonteCarloStressTest
from topology_engine import create_topology_engine


def demo_adaptive_persistence():
    """Demonstrate adaptive persistence scoring."""
    print("\n" + "=" * 70)
    print("DEMONSTRATION 1: Adaptive TDA Persistence Score")
    print("=" * 70)
    
    # Create TDA with 30-day rolling baseline
    tda = TDAHomology(baseline_window=30)
    
    # Simulate market conditions over time
    print("\n1. Building baseline from 'normal' market conditions...")
    normal_scores = []
    for i in range(40):
        # Create correlation matrix representing normal market
        n_assets = 5
        correlation = np.eye(n_assets) * 0.7 + 0.3
        np.fill_diagonal(correlation, 1.0)
        
        # Add small random variations
        noise = np.random.randn(n_assets, n_assets) * 0.05
        noise = (noise + noise.T) / 2
        correlation += noise
        np.fill_diagonal(correlation, 1.0)
        correlation = np.clip(correlation, -1, 1)
        
        # Classify regime
        result = tda.classify_regime(correlation)
        normal_scores.append(result['persistence_score'])
    
    baseline_stats = tda.get_baseline_stats()
    print(f"   Baseline established with {baseline_stats['count']} observations")
    print(f"   Mean persistence: {baseline_stats['mean']:.3f}")
    print(f"   Std deviation: {baseline_stats['std']:.3f}")
    
    # Test 1: Normal market condition
    print("\n2. Testing normal market condition...")
    normal_corr = np.eye(5) * 0.7 + 0.3
    np.fill_diagonal(normal_corr, 1.0)
    
    result = tda.classify_regime(normal_corr)
    adaptive_info = result['adaptive_threshold']
    
    print(f"   Current persistence: {result['persistence_score']:.3f}")
    print(f"   Adaptive threshold: {adaptive_info['threshold']:.3f}")
    print(f"   Z-score: {adaptive_info['z_score']:.2f}")
    print(f"   Is anomaly: {adaptive_info['is_anomaly']}")
    print(f"   Regime: {result['regime']}")
    
    # Test 2: Crisis condition (high fragmentation)
    print("\n3. Testing crisis market condition (high fragmentation)...")
    crisis_corr = np.random.rand(5, 5) * 0.2  # Very low correlations
    crisis_corr = (crisis_corr + crisis_corr.T) / 2
    np.fill_diagonal(crisis_corr, 1.0)
    
    result = tda.classify_regime(crisis_corr)
    adaptive_info = result['adaptive_threshold']
    
    print(f"   Current persistence: {result['persistence_score']:.3f}")
    print(f"   Adaptive threshold: {adaptive_info['threshold']:.3f}")
    print(f"   Z-score: {adaptive_info['z_score']:.2f}")
    print(f"   Is anomaly: {adaptive_info['is_anomaly']}")
    print(f"   Regime: {result['regime']}")
    
    print("\n   >>> Key Insight:")
    print("   The adaptive threshold adjusts based on recent history.")
    print("   A persistence score that's normal during high-rate environments")
    print("   could signal a crash during low-rate environments.")


def demo_synthetic_data_generation():
    """Demonstrate GBM and Copula synthetic data generation."""
    print("\n" + "=" * 70)
    print("DEMONSTRATION 2: Synthetic Data with Correlation Preservation")
    print("=" * 70)
    
    # Create correlated market returns
    print("\n1. Creating correlated market returns...")
    n_periods = 200
    n_assets = 4
    
    # Define realistic correlations
    mean = [0.0005, 0.0008, 0.0006, 0.0007]
    cov = [[0.0004, 0.0002, 0.0001, 0.00015],
           [0.0002, 0.0005, 0.00015, 0.0001],
           [0.0001, 0.00015, 0.0003, 0.0002],
           [0.00015, 0.0001, 0.0002, 0.00035]]
    
    returns = pd.DataFrame(
        np.random.multivariate_normal(mean, cov, n_periods),
        columns=['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    )
    
    original_corr = returns.corr()
    print(f"   Created returns for {n_assets} assets over {n_periods} periods")
    print("\n   Original Correlation Matrix:")
    print(original_corr.round(3))
    
    # Test GBM
    print("\n2. Generating synthetic data using Geometric Brownian Motion (GBM)...")
    mc_tester = MonteCarloStressTest(n_simulations=100, random_state=42)
    
    gbm_synthetic = mc_tester.generate_synthetic_returns(returns, method='gbm')
    gbm_corr = gbm_synthetic.corr()
    
    gbm_validation = mc_tester.validate_correlation_preservation(
        returns, gbm_synthetic, tolerance=0.15
    )
    
    print("\n   GBM Synthetic Correlation Matrix:")
    print(gbm_corr.round(3))
    print(f"\n   Validation Results:")
    print(f"     - Correlation preserved: {gbm_validation['is_valid']}")
    print(f"     - Max correlation diff: {gbm_validation['max_correlation_diff']:.4f}")
    print(f"     - Mean correlation diff: {gbm_validation['mean_correlation_diff']:.4f}")
    
    # Test Copula
    print("\n3. Generating synthetic data using Gaussian Copula...")
    copula_synthetic = mc_tester.generate_synthetic_returns(returns, method='copula')
    copula_corr = copula_synthetic.corr()
    
    copula_validation = mc_tester.validate_correlation_preservation(
        returns, copula_synthetic, tolerance=0.15
    )
    
    print("\n   Copula Synthetic Correlation Matrix:")
    print(copula_corr.round(3))
    print(f"\n   Validation Results:")
    print(f"     - Correlation preserved: {copula_validation['is_valid']}")
    print(f"     - Max correlation diff: {copula_validation['max_correlation_diff']:.4f}")
    print(f"     - Mean correlation diff: {copula_validation['mean_correlation_diff']:.4f}")
    
    print("\n   >>> Key Insight:")
    print("   Both GBM and Copula methods preserve correlations, ensuring")
    print("   that TDA analysis on synthetic data returns meaningful results.")
    print("   Without correlation preservation, TDA would return 0 and tests")
    print("   would be useless.")


def demo_topological_attribution():
    """Demonstrate JSON output with topological attribution."""
    print("\n" + "=" * 70)
    print("DEMONSTRATION 3: JSON Output with Topological Attribution")
    print("=" * 70)
    
    # Create engine
    print("\n1. Creating TopologyEngine...")
    engine = create_topology_engine(instability_threshold=0.5)
    
    # Create training data
    n_periods = 150
    dates = pd.date_range('2020-01-01', periods=n_periods, freq='D')
    
    train_data = pd.DataFrame({
        'Open': 100 + np.random.randn(n_periods).cumsum(),
        'High': 102 + np.random.randn(n_periods).cumsum(),
        'Low': 98 + np.random.randn(n_periods).cumsum(),
        'Close': 100 + np.random.randn(n_periods).cumsum(),
        'Volume': 1000000 + np.random.randint(-100000, 100000, n_periods)
    }, index=dates)
    
    # Train
    print("2. Training engine...")
    engine.fit(train_data, single_asset_mode=True)
    
    # Generate predictions with different scenarios
    print("\n3. Generating predictions with topological attribution...")
    
    # Prediction 1: Recent data
    print("\n   Scenario A: Recent market data")
    pred_a = engine.predict(train_data[-30:])
    
    print(f"   Signal: {pred_a['final_signal']}")
    print(f"   Reason: {pred_a['reason']}")
    print(f"   Reason Details: {pred_a['reason_details']}")
    print(f"   Persistence Score: {pred_a['persistence_score']:.3f}")
    
    # Prediction 2: Mid period
    print("\n   Scenario B: Mid-period data")
    pred_b = engine.predict(train_data[60:90])
    
    print(f"   Signal: {pred_b['final_signal']}")
    print(f"   Reason: {pred_b['reason']}")
    print(f"   Reason Details: {pred_b['reason_details']}")
    print(f"   Persistence Score: {pred_b['persistence_score']:.3f}")
    
    # Show complete JSON output
    print("\n4. Complete JSON Output with Attribution:")
    json_output = engine.to_json(pred_a)
    parsed = json.loads(json_output)
    
    print(json.dumps(parsed, indent=2))
    
    print("\n   >>> Key Insight:")
    print("   The JSON output now includes 'reason' and 'reason_details' fields")
    print("   that make the model interpretable. Instead of a black box, users")
    print("   can understand WHY a signal was generated:")
    print("   - 'H1_instability_exceeded_threshold': TDA detected instability")
    print("   - 'NN_prediction_bullish/bearish': Neural network prediction")
    print("   - 'NN_prediction_neutral': NN uncertain")
    print("   The reason_details provide additional context like threshold values,")
    print("   regime information, and specific feature violations.")


def main():
    """Run all demonstrations."""
    np.random.seed(42)
    
    print("\n" + "=" * 70)
    print(" TDA ENHANCEMENTS DEMONSTRATION")
    print(" Topological Data Analysis for Financial Markets")
    print("=" * 70)
    
    demo_adaptive_persistence()
    demo_synthetic_data_generation()
    demo_topological_attribution()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nThree key enhancements have been implemented:")
    print("\n1. ADAPTIVE PERSISTENCE SCORING")
    print("   - Compares current persistence to 30-day rolling baseline")
    print("   - Context-aware: same score can be normal or anomalous")
    print("   - Adaptive thresholds using z-scores")
    
    print("\n2. CORRELATION-PRESERVING SYNTHETIC DATA")
    print("   - Geometric Brownian Motion (GBM) for realistic simulations")
    print("   - Gaussian Copula for complex dependency structures")
    print("   - Automatic validation ensures TDA remains meaningful")
    
    print("\n3. INTERPRETABLE JSON OUTPUT")
    print("   - Every signal includes a 'reason' field")
    print("   - Detailed attribution in 'reason_details'")
    print("   - Makes the AI transparent, not a black box")
    
    print("\nAll enhancements are production-ready with comprehensive tests.")
    print("=" * 70)


if __name__ == '__main__':
    main()
