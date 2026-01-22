"""
Market Mutual Model Example

This example demonstrates the complete Market Mutual Model workflow:
1. Feature Fusion: Combining local residuals with global topological features
2. Regime Detection: Detecting market states (stable, stressed, transitioning)
3. Risk Management: Scaling positions based on topological persistence
4. Ensemble: Combining TDA signals with neural network signals
5. Backtesting: Testing the strategy on historical data
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tda import (
    MarketMutualModel,
    MetaLearner,
    Backtester,
    compare_strategies
)


def generate_market_data(n_assets=10, n_periods=500, regime_changes=True):
    """
    Generate synthetic market data with different regimes.
    
    Parameters
    ----------
    n_assets : int
        Number of assets.
    n_periods : int
        Number of time periods.
    regime_changes : bool
        Whether to simulate regime changes.
    
    Returns
    -------
    pd.DataFrame
        Asset returns.
    """
    np.random.seed(42)
    
    returns = []
    dates = pd.date_range('2020-01-01', periods=n_periods, freq='D')
    
    if regime_changes:
        # Create three regimes
        regime_1_end = n_periods // 3
        regime_2_end = 2 * n_periods // 3
        
        # Regime 1: Stable (low volatility, high correlation)
        corr_1 = np.eye(n_assets) * 0.3 + 0.7
        vol_1 = 0.01
        mean_1 = 0.0005
        
        returns_1 = np.random.multivariate_normal(
            [mean_1] * n_assets,
            corr_1 * vol_1 ** 2,
            regime_1_end
        )
        
        # Regime 2: Transitioning (medium volatility, medium correlation)
        corr_2 = np.eye(n_assets) * 0.5 + 0.5
        vol_2 = 0.02
        mean_2 = 0.0002
        
        returns_2 = np.random.multivariate_normal(
            [mean_2] * n_assets,
            corr_2 * vol_2 ** 2,
            regime_2_end - regime_1_end
        )
        
        # Regime 3: Stressed (high volatility, fragmented)
        corr_3 = np.eye(n_assets) * 0.8 + 0.2
        # Create clusters
        for i in range(0, n_assets, 3):
            for j in range(i, min(i+3, n_assets)):
                for k in range(i, min(i+3, n_assets)):
                    corr_3[j, k] = 0.9
                    corr_3[k, j] = 0.9
        
        vol_3 = 0.03
        mean_3 = -0.0003
        
        returns_3 = np.random.multivariate_normal(
            [mean_3] * n_assets,
            corr_3 * vol_3 ** 2,
            n_periods - regime_2_end
        )
        
        returns = np.vstack([returns_1, returns_2, returns_3])
    else:
        # Single regime
        corr = np.eye(n_assets) * 0.5 + 0.5
        vol = 0.015
        mean = 0.0003
        
        returns = np.random.multivariate_normal(
            [mean] * n_assets,
            corr * vol ** 2,
            n_periods
        )
    
    asset_names = [f'Asset_{i+1}' for i in range(n_assets)]
    return pd.DataFrame(returns, index=dates, columns=asset_names)


def create_dummy_neural_net_signals(returns):
    """
    Create dummy neural network signals for ensemble demonstration.
    
    In practice, this would be replaced with actual neural network predictions.
    """
    # Simple momentum-based signals as proxy for NN
    momentum = returns.rolling(window=10).mean()
    signals = np.tanh(momentum / returns.rolling(window=20).std())
    return signals.fillna(0)


def main():
    print("=" * 80)
    print("Market Mutual Model - Complete Example")
    print("=" * 80)
    print()
    
    # 1. Generate market data
    print("1. Generating synthetic market data with regime changes...")
    returns = generate_market_data(n_assets=10, n_periods=500, regime_changes=True)
    print(f"   Generated {len(returns)} periods of returns for {len(returns.columns)} assets")
    print(f"   Date range: {returns.index[0]} to {returns.index[-1]}")
    print()
    
    # Split into train and test
    train_size = int(len(returns) * 0.7)
    train_returns = returns.iloc[:train_size]
    test_returns = returns.iloc[train_size:]
    
    print(f"   Training set: {len(train_returns)} periods")
    print(f"   Test set: {len(test_returns)} periods")
    print()
    
    # 2. Initialize and train Market Mutual Model
    print("2. Training Market Mutual Model...")
    print("   - Using HMM for regime detection")
    print("   - Detecting 3 regimes: stable, transitioning, stressed")
    
    model = MarketMutualModel(
        regime_detector_type='hmm',
        n_regimes=3,
        random_state=42
    )
    
    model.fit(train_returns, window=20)
    print("   ✓ Model trained successfully")
    print()
    
    # 3. Generate predictions on test set
    print("3. Generating predictions on test set...")
    
    # Get detailed predictions
    predictions = model.predict(test_returns, return_details=True)
    
    print(f"   Current regime: {predictions['regime']}")
    print(f"   Regime confidence: {predictions['regime_confidence']:.3f}")
    print(f"   Persistence score: {predictions['persistence_score']:.3f}")
    print()
    
    print("   Topological features:")
    print(f"   - H_0 (clusters): {predictions['h0_features']['num_components']}")
    print(f"   - H_0 max persistence: {predictions['h0_features']['max_persistence']:.4f}")
    print(f"   - H_1 (loops): {predictions['h1_features']['num_loops']}")
    print(f"   - H_1 max persistence: {predictions['h1_features']['max_persistence']:.4f}")
    print()
    
    # 4. Ensemble with Neural Network
    print("4. Creating ensemble with Neural Network signals...")
    
    # Generate dummy NN signals
    nn_signals_train = create_dummy_neural_net_signals(train_returns)
    nn_signals_test = create_dummy_neural_net_signals(test_returns)
    
    # Get TDA signals
    tda_signals_test = predictions['signals']
    
    # Create ensemble using simple weighted average
    print("   - Combining TDA and NN signals using weighted average...")
    
    # Ensure arrays have the same shape
    tda_signals_array = tda_signals_test.reshape(-1)
    nn_signals_array = nn_signals_test.iloc[-1].values.reshape(-1)
    
    # Verify shapes match
    assert tda_signals_array.shape == nn_signals_array.shape, \
        f"Shape mismatch: TDA {tda_signals_array.shape} vs NN {nn_signals_array.shape}"
    
    # Simple weighted combination based on regime confidence
    tda_weight = predictions['regime_confidence']
    nn_weight = 1.0 - tda_weight
    
    ensemble_signals = (tda_signals_array * tda_weight + 
                       nn_signals_array * nn_weight)
    
    print(f"   ✓ Ensemble signals generated for {len(ensemble_signals)} assets")
    print(f"   - TDA weight: {tda_weight:.3f}")
    print(f"   - NN weight: {nn_weight:.3f}")
    print()
    
    # 5. Backtesting
    print("5. Running backtest...")
    
    backtester = Backtester(
        transaction_cost=0.001,  # 0.1%
        slippage=0.0005,         # 0.05%
        initial_capital=100000.0
    )
    
    # Create signal DataFrames for backtesting
    # Use a simple momentum strategy for baseline comparison
    baseline_signals = test_returns.rolling(window=10).mean() / test_returns.rolling(window=20).std()
    baseline_signals = baseline_signals.fillna(0)
    baseline_signals = np.clip(baseline_signals, -1, 1)
    
    # For TDA strategy, we need to generate signals for the entire test period
    # Simplified: use predictions from the model
    tda_signal_series = []
    for i in range(20, len(test_returns)):
        window_data = test_returns.iloc[max(0, i-50):i+1]
        if len(window_data) >= 20:
            pred = model.predict(window_data, return_details=False)
            tda_signal_series.append(pred)
    
    if tda_signal_series:
        tda_signals_df = pd.DataFrame(
            tda_signal_series,
            index=test_returns.index[20:20+len(tda_signal_series)],
            columns=test_returns.columns
        )
    else:
        tda_signals_df = baseline_signals  # Fallback
    
    # Align signals with returns
    common_idx = test_returns.index.intersection(tda_signals_df.index)
    test_returns_aligned = test_returns.loc[common_idx]
    tda_signals_aligned = tda_signals_df.loc[common_idx]
    baseline_signals_aligned = baseline_signals.loc[common_idx]
    
    # Run backtests
    strategies = {
        'TDA_Strategy': tda_signals_aligned,
        'Baseline_Momentum': baseline_signals_aligned
    }
    
    comparison = compare_strategies(test_returns_aligned, strategies, backtester)
    
    print("\n   Backtest Results:")
    print("   " + "=" * 70)
    print(comparison.to_string())
    print("   " + "=" * 70)
    print()
    
    # 6. Performance Summary
    print("6. Performance Summary:")
    print()
    
    tda_metrics = comparison.loc['TDA_Strategy']
    baseline_metrics = comparison.loc['Baseline_Momentum']
    
    print(f"   TDA Strategy:")
    print(f"   - Total Return: {tda_metrics['total_return']:.2%}")
    print(f"   - Sharpe Ratio: {tda_metrics['sharpe_ratio']:.3f}")
    print(f"   - Max Drawdown: {tda_metrics['max_drawdown']:.2%}")
    print()
    
    print(f"   Baseline Momentum:")
    print(f"   - Total Return: {baseline_metrics['total_return']:.2%}")
    print(f"   - Sharpe Ratio: {baseline_metrics['sharpe_ratio']:.3f}")
    print(f"   - Max Drawdown: {baseline_metrics['max_drawdown']:.2%}")
    print()
    
    if tda_metrics['sharpe_ratio'] > baseline_metrics['sharpe_ratio']:
        print("   ✓ TDA Strategy outperforms baseline on risk-adjusted basis!")
    else:
        print("   ✗ Baseline performs better (may need parameter tuning)")
    print()
    
    print("=" * 80)
    print("Example complete!")
    print("=" * 80)
    print()
    print("Next Steps:")
    print("1. Fine-tune regime detection parameters")
    print("2. Integrate with actual neural network models")
    print("3. Test on real market data (especially 2022 bear market)")
    print("4. Optimize risk management parameters")
    print("5. Implement portfolio optimization")


if __name__ == '__main__':
    main()
