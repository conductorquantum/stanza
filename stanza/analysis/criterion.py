"""Fit quality assessment criteria for curve fitting.

This module provides functions to evaluate the quality of nonlinear curve fits
using statistically robust metrics like R² and NRMSE.
"""

import numpy as np


def fit_quality_criterion(
    x_data: np.ndarray,
    y_data: np.ndarray,
    y_pred: np.ndarray,
    r_squared_threshold: float = 0.7,
    nrmse_threshold: float = 0.2,
) -> bool:
    """Evaluate fit quality using R² and NRMSE metrics.

    This criterion evaluates both how well the model explains variance (R²) and how
    small the errors are relative to the data range (NRMSE).

    Args:
        x_data: Input x values (used for context, not in calculation)
        y_data: Observed y values
        y_pred: Predicted y values from the fitted model
        r_squared_threshold: Minimum R² value for acceptable fit. Default 0.7
                            means the model must explain at least 70% of variance.
        nrmse_threshold: Maximum NRMSE (normalized RMSE) for acceptable fit.
                        Default 0.2 means errors must be less than 20% of data range.

    Returns:
        True if fit quality is GOOD (passes both criteria), False if POOR

    Notes:
        - R² (coefficient of determination) measures the proportion of variance
          in the data explained by the model. Range: [0, 1], higher is better.
        - NRMSE (normalized root mean square error) measures prediction error
          relative to data range. Range: [0, ∞), lower is better.
        - These metrics are appropriate for nonlinear models, unlike reduced
          chi-squared which requires known degrees of freedom.
    """
    residuals = y_data - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)

    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    rmse = np.sqrt(ss_res / len(y_data))
    data_range = y_data.max() - y_data.min()
    nrmse = rmse / data_range if data_range > 0 else float("inf")

    return bool(r_squared > r_squared_threshold and nrmse < nrmse_threshold)
