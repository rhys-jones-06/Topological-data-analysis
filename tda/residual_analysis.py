"""
Residual Analysis Module

This module provides functionality to analyze residuals by subtracting
expected returns from actual returns to isolate local mispricings.
"""

import numpy as np
import pandas as pd
from typing import Union


def compute_residuals(
    actual_returns: Union[np.ndarray, pd.DataFrame, pd.Series],
    expected_returns: Union[np.ndarray, pd.DataFrame, pd.Series]
) -> Union[np.ndarray, pd.DataFrame, pd.Series]:
    """
    Compute residuals by subtracting expected returns from actual returns.
    
    This function isolates local mispricings by computing the difference between
    observed (actual) returns and predicted (expected) returns. Large residuals
    may indicate mispricing or anomalies in the market.
    
    Parameters
    ----------
    actual_returns : np.ndarray, pd.DataFrame, or pd.Series
        The observed returns for assets over time.
        Shape: (n_periods,) for single asset or (n_periods, n_assets) for multiple assets.
    expected_returns : np.ndarray, pd.DataFrame, or pd.Series
        The expected/predicted returns for assets over time.
        Should have the same shape as actual_returns.
    
    Returns
    -------
    np.ndarray, pd.DataFrame, or pd.Series
        The residuals (actual - expected) with the same shape and type as input.
        Positive residuals indicate returns higher than expected,
        negative residuals indicate returns lower than expected.
    
    Raises
    ------
    ValueError
        If actual_returns and expected_returns have different shapes.
    
    Examples
    --------
    >>> import numpy as np
    >>> actual = np.array([0.05, 0.03, -0.02, 0.04])
    >>> expected = np.array([0.02, 0.02, 0.02, 0.02])
    >>> residuals = compute_residuals(actual, expected)
    >>> print(residuals)
    [ 0.03  0.01 -0.04  0.02]
    
    >>> import pandas as pd
    >>> actual_df = pd.DataFrame({'Asset1': [0.05, 0.03], 'Asset2': [0.04, -0.01]})
    >>> expected_df = pd.DataFrame({'Asset1': [0.02, 0.02], 'Asset2': [0.02, 0.02]})
    >>> residuals_df = compute_residuals(actual_df, expected_df)
    """
    # Convert to numpy if needed for shape checking
    actual_array = np.asarray(actual_returns)
    expected_array = np.asarray(expected_returns)
    
    # Validate shapes match
    if actual_array.shape != expected_array.shape:
        raise ValueError(
            f"Shape mismatch: actual_returns has shape {actual_array.shape} "
            f"but expected_returns has shape {expected_array.shape}"
        )
    
    # Compute residuals
    if isinstance(actual_returns, pd.DataFrame):
        residuals = actual_returns - expected_returns
    elif isinstance(actual_returns, pd.Series):
        residuals = actual_returns - expected_returns
    else:
        residuals = actual_array - expected_array
    
    return residuals
