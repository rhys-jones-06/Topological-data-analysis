"""
Walk-Forward Tester with Purged Cross-Validation

Implements industry-standard walk-forward analysis with anchored rolling windows,
purging functions to prevent data leakage, and Combinatorial Purged Cross-Validation (CPCV).
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple, Callable
from datetime import datetime, timedelta
import warnings


class PurgingUtils:
    """
    Utilities for purging data to prevent look-ahead bias in time series.
    """
    
    @staticmethod
    def compute_purge_window(
        test_start_idx: int,
        test_end_idx: int,
        purge_pct: float = 0.02
    ) -> Tuple[int, int]:
        """
        Compute purge window around test set.
        
        Parameters
        ----------
        test_start_idx : int
            Starting index of test set
        test_end_idx : int
            Ending index of test set
        purge_pct : float, optional
            Percentage of test period to purge on each side. Default is 0.02 (2%).
        
        Returns
        -------
        tuple
            (purge_start_idx, purge_end_idx)
        """
        test_length = test_end_idx - test_start_idx
        purge_length = max(1, int(test_length * purge_pct))
        
        purge_start = max(0, test_start_idx - purge_length)
        purge_end = test_end_idx + purge_length
        
        return purge_start, purge_end
    
    @staticmethod
    def purge_train_set(
        train_indices: np.ndarray,
        purge_start: int,
        purge_end: int
    ) -> np.ndarray:
        """
        Remove purge window from training indices.
        
        Parameters
        ----------
        train_indices : np.ndarray
            Training set indices
        purge_start : int
            Start of purge window
        purge_end : int
            End of purge window
        
        Returns
        -------
        np.ndarray
            Purged training indices
        """
        mask = (train_indices < purge_start) | (train_indices >= purge_end)
        return train_indices[mask]
    
    @staticmethod
    def create_embargo_period(
        n_samples: int,
        embargo_pct: float = 0.01
    ) -> int:
        """
        Create embargo period to account for non-independent samples.
        
        Parameters
        ----------
        n_samples : int
            Total number of samples
        embargo_pct : float, optional
            Percentage to embargo. Default is 0.01 (1%).
        
        Returns
        -------
        int
            Embargo period length
        """
        return max(1, int(n_samples * embargo_pct))


class WalkForwardTester:
    """
    Walk-Forward Analysis tester with anchored rolling windows.
    
    Implements:
    - Anchored expanding window (train on increasing history)
    - Fixed-size rolling test windows
    - Purging to prevent data leakage
    - Combinatorial Purged Cross-Validation (CPCV)
    """
    
    def __init__(
        self,
        train_window: int = 36,  # months
        test_window: int = 12,   # months
        step_size: Optional[int] = None,  # If None, equals test_window
        purge_pct: float = 0.02,
        embargo_pct: float = 0.01,
        min_train_samples: int = 100,
        anchored: bool = True
    ):
        """
        Initialize Walk-Forward Tester.
        
        Parameters
        ----------
        train_window : int, optional
            Initial training window in months. Default is 36.
        test_window : int, optional
            Test window in months. Default is 12.
        step_size : int, optional
            Step size for rolling forward. If None, equals test_window.
        purge_pct : float, optional
            Percentage to purge around test borders. Default is 0.02 (2%).
        embargo_pct : float, optional
            Embargo percentage for non-independent samples. Default is 0.01.
        min_train_samples : int, optional
            Minimum training samples required. Default is 100.
        anchored : bool, optional
            If True, use anchored (expanding) window. Default is True.
        """
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size if step_size is not None else test_window
        self.purge_pct = purge_pct
        self.embargo_pct = embargo_pct
        self.min_train_samples = min_train_samples
        self.anchored = anchored
        
        self.purging_utils = PurgingUtils()
        self.results = []
    
    def _convert_months_to_periods(
        self,
        data: pd.DataFrame,
        n_months: int
    ) -> int:
        """
        Convert months to number of periods based on data frequency.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data with DatetimeIndex
        n_months : int
            Number of months
        
        Returns
        -------
        int
            Number of periods
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            # Assume daily data if no datetime index
            return n_months * 21  # ~21 trading days per month
        
        # Estimate periods per month from data
        time_diff = (data.index[-1] - data.index[0]).days
        n_periods = len(data)
        periods_per_day = n_periods / max(time_diff, 1)
        
        return int(n_months * 30 * periods_per_day)
    
    def generate_splits(
        self,
        data: pd.DataFrame
    ) -> List[Dict[str, any]]:
        """
        Generate train/test splits for walk-forward analysis.
        
        Parameters
        ----------
        data : pd.DataFrame
            Full dataset with DatetimeIndex
        
        Returns
        -------
        list of dict
            Each dict contains:
            - 'train_idx': Training indices
            - 'test_idx': Test indices
            - 'train_start': Training start index
            - 'train_end': Training end index
            - 'test_start': Test start index
            - 'test_end': Test end index
        """
        n_samples = len(data)
        
        # Convert windows to periods
        train_periods = self._convert_months_to_periods(data, self.train_window)
        test_periods = self._convert_months_to_periods(data, self.test_window)
        step_periods = self._convert_months_to_periods(data, self.step_size)
        
        splits = []
        test_start = train_periods
        
        while test_start + test_periods <= n_samples:
            test_end = test_start + test_periods
            
            # Determine training window
            if self.anchored:
                # Anchored: always start from beginning
                train_start = 0
                train_end = test_start
            else:
                # Rolling: fixed-size window
                train_start = max(0, test_start - train_periods)
                train_end = test_start
            
            # Check minimum training samples
            if (train_end - train_start) < self.min_train_samples:
                test_start += step_periods
                continue
            
            # Compute purge window
            purge_start, purge_end = self.purging_utils.compute_purge_window(
                test_start, test_end, self.purge_pct
            )
            
            # Create indices
            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)
            
            # Apply purging to training set
            train_idx = self.purging_utils.purge_train_set(
                train_idx, purge_start, purge_end
            )
            
            split = {
                'train_idx': train_idx,
                'test_idx': test_idx,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'purge_start': purge_start,
                'purge_end': purge_end
            }
            
            splits.append(split)
            
            # Move to next test period
            test_start += step_periods
        
        return splits
    
    def run(
        self,
        data: pd.DataFrame,
        model_trainer: Callable,
        model_predictor: Callable,
        metric_calculator: Optional[Callable] = None
    ) -> Dict[str, any]:
        """
        Run walk-forward analysis.
        
        Parameters
        ----------
        data : pd.DataFrame
            Full dataset
        model_trainer : callable
            Function to train model: model_trainer(train_data) -> fitted_model
        model_predictor : callable
            Function to predict: model_predictor(model, test_data) -> predictions
        metric_calculator : callable, optional
            Function to calculate metrics: metric_calculator(predictions, actual) -> dict
        
        Returns
        -------
        dict
            Walk-forward results including metrics for each fold and aggregated stats
        """
        splits = self.generate_splits(data)
        
        if len(splits) == 0:
            raise ValueError("No valid splits generated. Check data size and window parameters.")
        
        self.results = []
        
        for i, split in enumerate(splits):
            print(f"Processing split {i+1}/{len(splits)}...")
            
            # Extract train and test data
            train_data = data.iloc[split['train_idx']]
            test_data = data.iloc[split['test_idx']]
            
            # Train model
            try:
                model = model_trainer(train_data)
            except Exception as e:
                warnings.warn(f"Training failed for split {i+1}: {str(e)}")
                continue
            
            # Make predictions
            try:
                predictions = model_predictor(model, test_data)
            except Exception as e:
                warnings.warn(f"Prediction failed for split {i+1}: {str(e)}")
                continue
            
            # Calculate metrics if provided
            metrics = {}
            if metric_calculator is not None:
                try:
                    # Extract actual values from test data
                    metrics = metric_calculator(predictions, test_data)
                except Exception as e:
                    warnings.warn(f"Metric calculation failed for split {i+1}: {str(e)}")
            
            result = {
                'split_index': i,
                'train_start': split['train_start'],
                'train_end': split['train_end'],
                'test_start': split['test_start'],
                'test_end': split['test_end'],
                'n_train': len(split['train_idx']),
                'n_test': len(split['test_idx']),
                'predictions': predictions,
                'metrics': metrics
            }
            
            self.results.append(result)
        
        # Aggregate results
        aggregated = self._aggregate_results()
        
        return {
            'splits': self.results,
            'aggregated': aggregated,
            'n_splits': len(self.results),
            'window_config': {
                'train_window': self.train_window,
                'test_window': self.test_window,
                'step_size': self.step_size,
                'anchored': self.anchored
            }
        }
    
    def _aggregate_results(self) -> Dict[str, float]:
        """
        Aggregate metrics across all splits.
        
        Returns
        -------
        dict
            Aggregated metrics
        """
        if not self.results:
            return {}
        
        # Collect all metrics
        all_metrics = {}
        
        for result in self.results:
            for metric_name, metric_value in result['metrics'].items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(metric_value)
        
        # Compute statistics
        aggregated = {}
        
        for metric_name, values in all_metrics.items():
            aggregated[f'{metric_name}_mean'] = np.mean(values)
            aggregated[f'{metric_name}_std'] = np.std(values)
            aggregated[f'{metric_name}_min'] = np.min(values)
            aggregated[f'{metric_name}_max'] = np.max(values)
        
        return aggregated


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation for financial time series.
    
    Implements the methodology from "Advances in Financial Machine Learning"
    by Marcos López de Prado.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        n_test_groups: int = 2,
        purge_pct: float = 0.02,
        embargo_pct: float = 0.01
    ):
        """
        Initialize CPCV.
        
        Parameters
        ----------
        n_splits : int, optional
            Number of splits/groups. Default is 5.
        n_test_groups : int, optional
            Number of groups to use for testing in each combination. Default is 2.
        purge_pct : float, optional
            Purge percentage. Default is 0.02.
        embargo_pct : float, optional
            Embargo percentage. Default is 0.01.
        """
        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.purge_pct = purge_pct
        self.embargo_pct = embargo_pct
        self.purging_utils = PurgingUtils()
    
    def generate_combinatorial_splits(
        self,
        data: pd.DataFrame
    ) -> List[Dict[str, np.ndarray]]:
        """
        Generate combinatorial purged splits.
        
        Parameters
        ----------
        data : pd.DataFrame
            Full dataset
        
        Returns
        -------
        list of dict
            Each dict contains 'train_idx' and 'test_idx'
        """
        from itertools import combinations
        
        n_samples = len(data)
        indices = np.arange(n_samples)
        
        # Divide into n_splits groups
        group_size = n_samples // self.n_splits
        groups = []
        
        for i in range(self.n_splits):
            start = i * group_size
            end = (i + 1) * group_size if i < self.n_splits - 1 else n_samples
            groups.append(np.arange(start, end))
        
        # Generate all combinations of test groups
        test_combinations = list(combinations(range(self.n_splits), self.n_test_groups))
        
        splits = []
        
        for test_group_ids in test_combinations:
            # Combine test groups
            test_idx = np.concatenate([groups[i] for i in test_group_ids])
            
            # All other groups for training
            train_group_ids = [i for i in range(self.n_splits) if i not in test_group_ids]
            train_idx = np.concatenate([groups[i] for i in train_group_ids])
            
            # Apply purging around test sets
            test_start = test_idx.min()
            test_end = test_idx.max() + 1
            
            purge_start, purge_end = self.purging_utils.compute_purge_window(
                test_start, test_end, self.purge_pct
            )
            
            train_idx = self.purging_utils.purge_train_set(
                train_idx, purge_start, purge_end
            )
            
            splits.append({
                'train_idx': train_idx,
                'test_idx': test_idx
            })
        
        return splits


def create_walk_forward_tester(
    train_months: int = 36,
    test_months: int = 12,
    anchored: bool = True
) -> WalkForwardTester:
    """
    Factory function to create a configured WalkForwardTester.
    
    Parameters
    ----------
    train_months : int, optional
        Training window in months. Default is 36.
    test_months : int, optional
        Test window in months. Default is 12.
    anchored : bool, optional
        Use anchored expanding window. Default is True.
    
    Returns
    -------
    WalkForwardTester
        Configured tester
    """
    return WalkForwardTester(
        train_window=train_months,
        test_window=test_months,
        anchored=anchored
    )
