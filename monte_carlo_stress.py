"""
Monte Carlo Stress Testing Module

Generates synthetic datasets preserving topological structure but permuting
calendar ordering to test robustness of the TopologyEngine.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Callable
import warnings


class MonteCarloStressTest:
    """
    Monte Carlo stress testing for topology-based strategies.
    
    Generates synthetic market scenarios that preserve topological features
    but shuffle temporal ordering to test strategy robustness.
    """
    
    def __init__(
        self,
        n_simulations: int = 1000,
        preserve_correlation: bool = True,
        preserve_volatility: bool = True,
        block_size: Optional[int] = None,
        random_state: Optional[int] = None
    ):
        """
        Initialize Monte Carlo stress tester.
        
        Parameters
        ----------
        n_simulations : int, optional
            Number of synthetic scenarios. Default is 1000.
        preserve_correlation : bool, optional
            If True, preserve correlation structure. Default is True.
        preserve_volatility : bool, optional
            If True, preserve volatility structure. Default is True.
        block_size : int, optional
            Block size for block bootstrap. If None, uses single-day shuffle.
        random_state : int, optional
            Random seed for reproducibility.
        """
        self.n_simulations = n_simulations
        self.preserve_correlation = preserve_correlation
        self.preserve_volatility = preserve_volatility
        self.block_size = block_size
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def generate_synthetic_returns(
        self,
        original_returns: pd.DataFrame,
        method: str = 'block_bootstrap'
    ) -> pd.DataFrame:
        """
        Generate synthetic returns preserving topological structure.
        
        Parameters
        ----------
        original_returns : pd.DataFrame
            Original returns data
        method : str, optional
            Generation method: 'block_bootstrap', 'shuffle', or 'parametric'.
            Default is 'block_bootstrap'.
        
        Returns
        -------
        pd.DataFrame
            Synthetic returns with same shape as original
        """
        if method == 'block_bootstrap':
            return self._block_bootstrap(original_returns)
        elif method == 'shuffle':
            return self._shuffle_returns(original_returns)
        elif method == 'parametric':
            return self._parametric_generation(original_returns)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _block_bootstrap(
        self,
        returns: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Block bootstrap preserving short-term dependencies.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Original returns
        
        Returns
        -------
        pd.DataFrame
            Bootstrapped returns
        """
        n_periods = len(returns)
        
        if self.block_size is None:
            # Auto-determine block size (typically 5-20 days)
            self.block_size = min(20, max(5, n_periods // 20))
        
        n_blocks = int(np.ceil(n_periods / self.block_size))
        
        # Sample blocks with replacement
        synthetic_returns = []
        
        for _ in range(n_blocks):
            # Random starting point
            start_idx = np.random.randint(0, max(1, n_periods - self.block_size))
            end_idx = min(start_idx + self.block_size, n_periods)
            
            block = returns.iloc[start_idx:end_idx].copy()
            synthetic_returns.append(block)
        
        # Concatenate and trim to original length
        synthetic_df = pd.concat(synthetic_returns, ignore_index=True)
        synthetic_df = synthetic_df.iloc[:n_periods]
        
        # Restore index
        synthetic_df.index = returns.index
        
        return synthetic_df
    
    def _shuffle_returns(
        self,
        returns: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Shuffle returns while optionally preserving correlation.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Original returns
        
        Returns
        -------
        pd.DataFrame
            Shuffled returns
        """
        if self.preserve_correlation:
            # Shuffle rows (time) but keep cross-sectional correlation
            shuffled_idx = np.random.permutation(len(returns))
            synthetic = returns.iloc[shuffled_idx].copy()
            synthetic.index = returns.index
        else:
            # Shuffle each column independently
            synthetic = returns.copy()
            for col in synthetic.columns:
                synthetic[col] = np.random.permutation(synthetic[col].values)
        
        return synthetic
    
    def _parametric_generation(
        self,
        returns: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate returns from fitted parametric distribution.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Original returns
        
        Returns
        -------
        pd.DataFrame
            Parametrically generated returns
        """
        n_periods, n_assets = returns.shape
        
        # Estimate parameters
        means = returns.mean().values
        
        if self.preserve_correlation:
            cov_matrix = returns.cov().values
        else:
            # Diagonal covariance (independent assets)
            variances = returns.var().values
            cov_matrix = np.diag(variances)
        
        # Generate from multivariate normal
        synthetic_values = np.random.multivariate_normal(
            mean=means,
            cov=cov_matrix,
            size=n_periods
        )
        
        synthetic = pd.DataFrame(
            synthetic_values,
            index=returns.index,
            columns=returns.columns
        )
        
        return synthetic
    
    def run_stress_test(
        self,
        original_data: pd.DataFrame,
        strategy_evaluator: Callable,
        generation_method: str = 'block_bootstrap',
        return_distribution: bool = True
    ) -> Dict[str, any]:
        """
        Run Monte Carlo stress test on a strategy.
        
        Parameters
        ----------
        original_data : pd.DataFrame
            Original returns or price data
        strategy_evaluator : callable
            Function that evaluates strategy: evaluator(data) -> metrics_dict
        generation_method : str, optional
            Synthetic data generation method. Default is 'block_bootstrap'.
        return_distribution : bool, optional
            If True, return full distribution of metrics. Default is True.
        
        Returns
        -------
        dict
            Stress test results including:
            - 'original_metrics': Metrics on original data
            - 'synthetic_metrics_mean': Average metrics across simulations
            - 'synthetic_metrics_std': Std dev of metrics
            - 'synthetic_metrics_distribution': Full distribution if requested
            - 'p_values': Statistical significance of original vs synthetic
        """
        print(f"Running Monte Carlo stress test with {self.n_simulations} simulations...")
        
        # Evaluate on original data
        try:
            original_metrics = strategy_evaluator(original_data)
        except Exception as e:
            warnings.warn(f"Original evaluation failed: {str(e)}")
            original_metrics = {}
        
        # Run simulations
        synthetic_results = []
        
        for i in range(self.n_simulations):
            if (i + 1) % 100 == 0:
                print(f"  Simulation {i+1}/{self.n_simulations}...")
            
            try:
                # Generate synthetic data
                synthetic_data = self.generate_synthetic_returns(
                    original_data,
                    method=generation_method
                )
                
                # Evaluate strategy on synthetic data
                metrics = strategy_evaluator(synthetic_data)
                synthetic_results.append(metrics)
                
            except Exception as e:
                warnings.warn(f"Simulation {i+1} failed: {str(e)}")
                continue
        
        if len(synthetic_results) == 0:
            raise ValueError("All simulations failed")
        
        # Aggregate results
        aggregated = self._aggregate_stress_results(
            original_metrics,
            synthetic_results,
            return_distribution
        )
        
        return aggregated
    
    def _aggregate_stress_results(
        self,
        original_metrics: Dict[str, float],
        synthetic_results: List[Dict[str, float]],
        return_distribution: bool
    ) -> Dict[str, any]:
        """
        Aggregate stress test results across simulations.
        
        Parameters
        ----------
        original_metrics : dict
            Metrics from original data
        synthetic_results : list of dict
            Metrics from each simulation
        return_distribution : bool
            Whether to include full distribution
        
        Returns
        -------
        dict
            Aggregated results
        """
        # Convert to DataFrame for easier analysis
        if len(synthetic_results) > 0:
            synthetic_df = pd.DataFrame(synthetic_results)
        else:
            synthetic_df = pd.DataFrame()
        
        aggregated = {
            'original_metrics': original_metrics,
            'n_successful_simulations': len(synthetic_results)
        }
        
        if not synthetic_df.empty:
            # Statistics
            aggregated['synthetic_metrics_mean'] = synthetic_df.mean().to_dict()
            aggregated['synthetic_metrics_std'] = synthetic_df.std().to_dict()
            aggregated['synthetic_metrics_median'] = synthetic_df.median().to_dict()
            aggregated['synthetic_metrics_min'] = synthetic_df.min().to_dict()
            aggregated['synthetic_metrics_max'] = synthetic_df.max().to_dict()
            
            # Percentiles
            aggregated['synthetic_metrics_5pct'] = synthetic_df.quantile(0.05).to_dict()
            aggregated['synthetic_metrics_95pct'] = synthetic_df.quantile(0.95).to_dict()
            
            # P-values (how extreme is original compared to synthetic?)
            p_values = {}
            for metric in original_metrics:
                if metric in synthetic_df.columns:
                    original_value = original_metrics[metric]
                    synthetic_values = synthetic_df[metric].values
                    
                    # Two-tailed p-value
                    n_extreme = np.sum(np.abs(synthetic_values) >= np.abs(original_value))
                    p_values[metric] = n_extreme / len(synthetic_values)
            
            aggregated['p_values'] = p_values
            
            # Full distribution if requested
            if return_distribution:
                aggregated['synthetic_metrics_distribution'] = synthetic_df.to_dict('list')
        
        return aggregated
    
    def assess_robustness(
        self,
        stress_results: Dict[str, any],
        significance_level: float = 0.05
    ) -> Dict[str, any]:
        """
        Assess strategy robustness from stress test results.
        
        Parameters
        ----------
        stress_results : dict
            Results from run_stress_test()
        significance_level : float, optional
            Significance level for statistical tests. Default is 0.05.
        
        Returns
        -------
        dict
            Robustness assessment
        """
        assessment = {
            'is_robust': True,
            'warnings': [],
            'summary': {}
        }
        
        original = stress_results['original_metrics']
        synthetic_mean = stress_results.get('synthetic_metrics_mean', {})
        p_values = stress_results.get('p_values', {})
        
        for metric, original_value in original.items():
            if metric not in synthetic_mean:
                continue
            
            synthetic_avg = synthetic_mean[metric]
            p_value = p_values.get(metric, 1.0)
            
            # Check if original is significantly different from synthetic
            if p_value < significance_level:
                assessment['warnings'].append(
                    f"{metric}: Original ({original_value:.4f}) is significantly "
                    f"different from synthetic average ({synthetic_avg:.4f}), "
                    f"p-value = {p_value:.4f}"
                )
                assessment['is_robust'] = False
            
            # Check if performance degrades substantially in synthetic scenarios
            if original_value > 0 and synthetic_avg < original_value * 0.5:
                assessment['warnings'].append(
                    f"{metric}: Performance degrades >50% in synthetic scenarios"
                )
                assessment['is_robust'] = False
        
        if assessment['is_robust']:
            assessment['summary']['conclusion'] = "Strategy appears robust to temporal permutations"
        else:
            assessment['summary']['conclusion'] = "Strategy may be overfit to specific temporal ordering"
        
        return assessment


def create_monte_carlo_tester(
    n_simulations: int = 1000,
    block_size: int = 10,
    random_state: int = 42
) -> MonteCarloStressTest:
    """
    Factory function to create a configured Monte Carlo stress tester.
    
    Parameters
    ----------
    n_simulations : int, optional
        Number of simulations. Default is 1000.
    block_size : int, optional
        Block bootstrap size. Default is 10.
    random_state : int, optional
        Random seed. Default is 42.
    
    Returns
    -------
    MonteCarloStressTest
        Configured stress tester
    """
    return MonteCarloStressTest(
        n_simulations=n_simulations,
        block_size=block_size,
        random_state=random_state
    )
