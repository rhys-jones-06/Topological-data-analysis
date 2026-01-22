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
        method: str = 'gbm'
    ) -> pd.DataFrame:
        """
        Generate synthetic returns preserving topological structure.
        
        Parameters
        ----------
        original_returns : pd.DataFrame
            Original returns data
        method : str, optional
            Generation method: 'gbm', 'copula', 'block_bootstrap', 'shuffle', or 'parametric'.
            Default is 'gbm' (Geometric Brownian Motion).
        
        Returns
        -------
        pd.DataFrame
            Synthetic returns with same shape as original
        """
        if method == 'gbm':
            return self._gbm_generation(original_returns)
        elif method == 'copula':
            return self._copula_generation(original_returns)
        elif method == 'block_bootstrap':
            return self._block_bootstrap(original_returns)
        elif method == 'shuffle':
            return self._shuffle_returns(original_returns)
        elif method == 'parametric':
            return self._parametric_generation(original_returns)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'gbm', 'copula', 'block_bootstrap', 'shuffle', or 'parametric'.")
    
    def _gbm_generation(
        self,
        returns: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate returns using Geometric Brownian Motion (GBM).
        
        GBM preserves correlations between stocks by using the covariance matrix
        to generate correlated random walks, ensuring the synthetic data maintains
        the same correlation structure as the real market.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Original returns
        
        Returns
        -------
        pd.DataFrame
            Synthetic returns generated via GBM
        """
        n_periods, n_assets = returns.shape
        
        # Estimate drift (mu) and volatility (sigma) from historical returns
        mu = returns.mean().values
        
        # Get covariance matrix to preserve correlations
        cov_matrix = returns.cov().values
        
        # Generate correlated random increments using Cholesky decomposition
        # This ensures the correlation structure is maintained
        try:
            cholesky = np.linalg.cholesky(cov_matrix)
        except np.linalg.LinAlgError:
            # If covariance matrix is not positive definite, regularize it
            min_eig = np.min(np.real(np.linalg.eigvals(cov_matrix)))
            if min_eig < 0:
                cov_matrix -= 1.5 * min_eig * np.eye(n_assets)
            cholesky = np.linalg.cholesky(cov_matrix)
        
        # Generate independent standard normal random variables
        z = np.random.standard_normal((n_periods, n_assets))
        
        # Transform to correlated random variables
        correlated_z = z @ cholesky.T
        
        # Generate GBM returns: r_t = mu * dt + sigma * dW
        # Using dt = 1 (daily returns)
        dt = 1.0
        synthetic_returns = mu * dt + correlated_z
        
        # Create DataFrame with same structure as input
        synthetic_df = pd.DataFrame(
            synthetic_returns,
            index=returns.index,
            columns=returns.columns
        )
        
        return synthetic_df
    
    def _copula_generation(
        self,
        returns: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate returns using Gaussian Copula approach.
        
        Copula separates the marginal distributions from the dependency structure,
        allowing us to preserve both the individual return distributions and the
        correlation structure between assets.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Original returns
        
        Returns
        -------
        pd.DataFrame
            Synthetic returns generated via Gaussian copula
        """
        n_periods, n_assets = returns.shape
        
        # Step 1: Transform each marginal to uniform distribution via empirical CDF
        uniform_data = np.zeros_like(returns.values)
        for i in range(n_assets):
            # Rank transform to get empirical CDF values
            ranks = returns.iloc[:, i].rank(method='average')
            uniform_data[:, i] = ranks / (len(ranks) + 1)
        
        # Step 2: Transform to standard normal via inverse CDF
        from scipy.stats import norm
        normal_data = norm.ppf(uniform_data)
        
        # Step 3: Estimate correlation matrix from normal-transformed data
        # Handle any NaN/Inf values
        normal_data = np.nan_to_num(normal_data, nan=0.0, posinf=3.0, neginf=-3.0)
        correlation_matrix = np.corrcoef(normal_data.T)
        
        # Step 4: Generate new correlated normal data
        try:
            synthetic_normal = np.random.multivariate_normal(
                mean=np.zeros(n_assets),
                cov=correlation_matrix,
                size=n_periods
            )
        except np.linalg.LinAlgError:
            # If correlation matrix is not positive definite, use Cholesky with regularization
            min_eig = np.min(np.real(np.linalg.eigvals(correlation_matrix)))
            if min_eig < 0:
                correlation_matrix -= 1.5 * min_eig * np.eye(n_assets)
            synthetic_normal = np.random.multivariate_normal(
                mean=np.zeros(n_assets),
                cov=correlation_matrix,
                size=n_periods
            )
        
        # Step 5: Transform back to uniform
        synthetic_uniform = norm.cdf(synthetic_normal)
        
        # Step 6: Transform to original marginal distributions via inverse empirical CDF
        synthetic_returns = np.zeros_like(synthetic_uniform)
        for i in range(n_assets):
            # Sort original returns to get empirical quantiles
            sorted_returns = np.sort(returns.iloc[:, i].values)
            # Map uniform values to quantiles
            indices = (synthetic_uniform[:, i] * len(sorted_returns)).astype(int)
            indices = np.clip(indices, 0, len(sorted_returns) - 1)
            synthetic_returns[:, i] = sorted_returns[indices]
        
        # Create DataFrame with same structure as input
        synthetic_df = pd.DataFrame(
            synthetic_returns,
            index=returns.index,
            columns=returns.columns
        )
        
        return synthetic_df
    
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
    
    def validate_correlation_preservation(
        self,
        original_returns: pd.DataFrame,
        synthetic_returns: pd.DataFrame,
        tolerance: float = 0.15
    ) -> Dict[str, any]:
        """
        Validate that synthetic data preserves correlations from original data.
        
        As per requirement: "If the synthetic data lacks correlation, the TDA will 
        return 0, and the test will be useless." This method ensures correlations
        are maintained.
        
        Parameters
        ----------
        original_returns : pd.DataFrame
            Original returns data
        synthetic_returns : pd.DataFrame
            Synthetic returns data
        tolerance : float, optional
            Maximum allowed difference in correlations. Default is 0.15.
        
        Returns
        -------
        dict
            Validation results including:
            - 'is_valid': Whether correlations are preserved within tolerance
            - 'max_correlation_diff': Maximum absolute difference in correlations
            - 'mean_correlation_diff': Mean absolute difference
            - 'original_correlation_matrix': Original correlation matrix
            - 'synthetic_correlation_matrix': Synthetic correlation matrix
        """
        # Compute correlation matrices
        orig_corr = original_returns.corr().values
        synth_corr = synthetic_returns.corr().values
        
        # Compute differences (excluding diagonal)
        n = orig_corr.shape[0]
        mask = ~np.eye(n, dtype=bool)
        
        corr_diff = np.abs(orig_corr - synth_corr)
        max_diff = np.max(corr_diff[mask])
        mean_diff = np.mean(corr_diff[mask])
        
        is_valid = max_diff <= tolerance
        
        return {
            'is_valid': is_valid,
            'max_correlation_diff': float(max_diff),
            'mean_correlation_diff': float(mean_diff),
            'tolerance': tolerance,
            'original_correlation_matrix': orig_corr,
            'synthetic_correlation_matrix': synth_corr
        }
    
    def run_stress_test(
        self,
        original_data: pd.DataFrame,
        strategy_evaluator: Callable,
        generation_method: str = 'gbm',
        return_distribution: bool = True,
        validate_correlations: bool = True
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
            Synthetic data generation method. Default is 'gbm' (Geometric Brownian Motion).
        return_distribution : bool, optional
            If True, return full distribution of metrics. Default is True.
        validate_correlations : bool, optional
            If True, validate that synthetic data preserves correlations. Default is True.
        
        Returns
        -------
        dict
            Stress test results including:
            - 'original_metrics': Metrics on original data
            - 'synthetic_metrics_mean': Average metrics across simulations
            - 'synthetic_metrics_std': Std dev of metrics
            - 'synthetic_metrics_distribution': Full distribution if requested
            - 'p_values': Statistical significance of original vs synthetic
            - 'correlation_validation': Validation results if validate_correlations=True
        """
        print(f"Running Monte Carlo stress test with {self.n_simulations} simulations using {generation_method}...")
        
        # Evaluate on original data
        try:
            original_metrics = strategy_evaluator(original_data)
        except Exception as e:
            warnings.warn(f"Original evaluation failed: {str(e)}")
            original_metrics = {}
        
        # Run simulations
        synthetic_results = []
        correlation_validations = []
        
        for i in range(self.n_simulations):
            if (i + 1) % 100 == 0:
                print(f"  Simulation {i+1}/{self.n_simulations}...")
            
            try:
                # Generate synthetic data
                synthetic_data = self.generate_synthetic_returns(
                    original_data,
                    method=generation_method
                )
                
                # Validate correlations if requested (only for first few to avoid overhead)
                if validate_correlations and i < 10:
                    validation = self.validate_correlation_preservation(
                        original_data,
                        synthetic_data
                    )
                    correlation_validations.append(validation)
                    
                    if not validation['is_valid'] and i == 0:
                        warnings.warn(
                            f"Synthetic data may not preserve correlations adequately. "
                            f"Max diff: {validation['max_correlation_diff']:.3f}, "
                            f"Mean diff: {validation['mean_correlation_diff']:.3f}"
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
        
        # Add correlation validation summary if available
        if correlation_validations:
            valid_count = sum(1 for v in correlation_validations if v['is_valid'])
            aggregated['correlation_validation'] = {
                'n_validated': len(correlation_validations),
                'n_valid': valid_count,
                'validation_rate': valid_count / len(correlation_validations),
                'mean_max_diff': np.mean([v['max_correlation_diff'] for v in correlation_validations]),
                'mean_mean_diff': np.mean([v['mean_correlation_diff'] for v in correlation_validations])
            }
        
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
