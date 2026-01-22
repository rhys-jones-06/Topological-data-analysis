"""
Backtesting Engine

Vectorized backtester that accounts for transaction costs and slippage,
specifically designed for testing topological trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Callable, Tuple, List


class Backtester:
    """
    Vectorized backtester for trading strategies.
    
    Features:
    - Transaction costs
    - Slippage modeling
    - Position limits
    - Performance metrics
    - Equity curve generation
    """
    
    def __init__(
        self,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        max_position: float = 1.0,
        initial_capital: float = 100000.0
    ):
        """
        Initialize the backtester.
        
        Parameters
        ----------
        transaction_cost : float, optional
            Transaction cost as a fraction (e.g., 0.001 = 0.1%). Default is 0.001.
        slippage : float, optional
            Slippage as a fraction. Default is 0.0005.
        max_position : float, optional
            Maximum position size as a fraction of capital. Default is 1.0.
        initial_capital : float, optional
            Initial capital. Default is 100,000.
        """
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.max_position = max_position
        self.initial_capital = initial_capital
    
    def run(
        self,
        returns: pd.DataFrame,
        signals: pd.DataFrame,
        rebalance_frequency: int = 1
    ) -> Dict:
        """
        Run backtest on historical data.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Asset returns. Shape: (n_periods, n_assets).
        signals : pd.DataFrame
            Trading signals. Shape: (n_periods, n_assets).
            Values between -1 (max short) and 1 (max long).
        rebalance_frequency : int, optional
            Rebalance every N periods. Default is 1 (daily).
        
        Returns
        -------
        dict
            Backtest results including equity curve and metrics.
        """
        n_periods, n_assets = returns.shape
        
        # Ensure signals and returns are aligned
        signals = signals.iloc[:n_periods]
        
        # Initialize tracking arrays
        positions = np.zeros((n_periods, n_assets))
        portfolio_values = np.zeros(n_periods)
        cash = np.zeros(n_periods)
        
        # Start with initial capital
        cash[0] = self.initial_capital
        portfolio_values[0] = self.initial_capital
        
        for t in range(n_periods):
            if t == 0:
                # Initial positions based on first signal
                target_positions = signals.iloc[t].values * self.max_position
                positions[t] = np.clip(target_positions, -self.max_position, self.max_position)
                
                # Calculate costs for initial positions
                position_values = np.abs(positions[t]) * portfolio_values[t]
                costs = position_values * (self.transaction_cost + self.slippage)
                cash[t] -= np.sum(costs)
                
            else:
                # Calculate portfolio value from previous positions and returns
                position_returns = positions[t-1] * returns.iloc[t].values
                portfolio_values[t] = portfolio_values[t-1] * (1 + np.sum(position_returns))
                cash[t] = cash[t-1]
                
                # Rebalance if needed
                if t % rebalance_frequency == 0:
                    # Target positions based on signal
                    target_positions = signals.iloc[t].values * self.max_position
                    target_positions = np.clip(target_positions, -self.max_position, self.max_position)
                    
                    # Calculate position changes
                    position_changes = target_positions - positions[t-1]
                    
                    # Calculate transaction costs on changes
                    change_values = np.abs(position_changes) * portfolio_values[t]
                    costs = change_values * (self.transaction_cost + self.slippage)
                    
                    # Apply costs
                    cash[t] -= np.sum(costs)
                    portfolio_values[t] -= np.sum(costs)
                    
                    # Update positions
                    positions[t] = target_positions
                else:
                    # Keep previous positions
                    positions[t] = positions[t-1]
        
        # Calculate metrics
        equity_curve = pd.Series(portfolio_values, index=returns.index)
        metrics = self._calculate_metrics(equity_curve, returns)
        
        return {
            'equity_curve': equity_curve,
            'positions': pd.DataFrame(positions, index=returns.index, columns=returns.columns),
            'cash': pd.Series(cash, index=returns.index),
            'metrics': metrics
        }
    
    def _calculate_metrics(
        self,
        equity_curve: pd.Series,
        benchmark_returns: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Parameters
        ----------
        equity_curve : pd.Series
            Portfolio equity over time.
        benchmark_returns : pd.DataFrame
            Benchmark returns for comparison.
        
        Returns
        -------
        dict
            Performance metrics.
        """
        # Calculate returns
        portfolio_returns = equity_curve.pct_change().fillna(0)
        
        # Total return
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        
        # Annualized return (assuming 252 trading days)
        n_years = len(equity_curve) / 252
        annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        # Volatility
        volatility = portfolio_returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (portfolio_returns > 0).sum() / len(portfolio_returns)
        
        # Sortino ratio (downside deviation)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino_ratio = annualized_return / downside_std if downside_std > 0 else 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'final_value': equity_curve.iloc[-1]
        }


class RollingBacktester:
    """
    Rolling window backtester for walk-forward analysis.
    """
    
    def __init__(
        self,
        train_window: int,
        test_window: int,
        backtester: Optional[Backtester] = None
    ):
        """
        Initialize rolling backtester.
        
        Parameters
        ----------
        train_window : int
            Size of training window in periods.
        test_window : int
            Size of test window in periods.
        backtester : Backtester, optional
            Base backtester to use. If None, creates default.
        """
        self.train_window = train_window
        self.test_window = test_window
        self.backtester = backtester or Backtester()
    
    def run(
        self,
        returns: pd.DataFrame,
        model: Callable,
        **model_kwargs
    ) -> Dict:
        """
        Run rolling window backtest.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Historical returns.
        model : callable
            Model that has fit() and predict() methods.
        **model_kwargs
            Additional arguments for model.
        
        Returns
        -------
        dict
            Combined backtest results across all windows.
        """
        n_periods = len(returns)
        all_results = []
        
        # Iterate through rolling windows
        start_idx = 0
        while start_idx + self.train_window + self.test_window <= n_periods:
            # Define train and test periods
            train_end = start_idx + self.train_window
            test_end = train_end + self.test_window
            
            train_returns = returns.iloc[start_idx:train_end]
            test_returns = returns.iloc[train_end:test_end]
            
            # Train model
            model.fit(train_returns, **model_kwargs)
            
            # Generate signals for test period
            test_signals = model.predict(test_returns)
            
            # Run backtest on test period
            test_results = self.backtester.run(test_returns, pd.DataFrame(test_signals))
            
            all_results.append({
                'start': start_idx,
                'train_end': train_end,
                'test_end': test_end,
                'metrics': test_results['metrics'],
                'equity_curve': test_results['equity_curve']
            })
            
            # Move to next window
            start_idx += self.test_window
        
        # Combine results
        combined_equity = pd.concat([r['equity_curve'] for r in all_results])
        combined_metrics = self._aggregate_metrics([r['metrics'] for r in all_results])
        
        return {
            'equity_curve': combined_equity,
            'metrics': combined_metrics,
            'window_results': all_results
        }
    
    def _aggregate_metrics(self, metrics_list: List[Dict]) -> Dict:
        """
        Aggregate metrics across windows.
        
        Parameters
        ----------
        metrics_list : list of dict
            Metrics from each window.
        
        Returns
        -------
        dict
            Aggregated metrics.
        """
        keys = metrics_list[0].keys()
        aggregated = {}
        
        for key in keys:
            values = [m[key] for m in metrics_list]
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
            aggregated[f'{key}_min'] = np.min(values)
            aggregated[f'{key}_max'] = np.max(values)
        
        return aggregated


def compare_strategies(
    returns: pd.DataFrame,
    strategies: Dict[str, pd.DataFrame],
    backtester: Optional[Backtester] = None
) -> pd.DataFrame:
    """
    Compare multiple strategies.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Historical returns.
    strategies : dict
        Dictionary mapping strategy names to signal DataFrames.
    backtester : Backtester, optional
        Backtester to use. If None, creates default.
    
    Returns
    -------
    pd.DataFrame
        Comparison table of strategy metrics.
    """
    if backtester is None:
        backtester = Backtester()
    
    results = {}
    
    for name, signals in strategies.items():
        backtest_result = backtester.run(returns, signals)
        results[name] = backtest_result['metrics']
    
    # Convert to DataFrame
    comparison_df = pd.DataFrame(results).T
    
    return comparison_df
