import numpy as np


def pcov_fail_criterion(pcov: np.ndarray, ubound: float = 0.01) -> bool:
    """Fail criterion for the covariance matrix of the fitted parameters.

    Args:
        pcov: Covariance matrix from curve fitting
        ubound: Upper bound threshold for diagonal elements. Default 0.01 balances
               accepting good pinchoff fits while rejecting linear or poor quality data.

    Returns:
        True if fit quality is poor (high uncertainty or degenerate covariance)
    """
    return bool(np.any(np.diag(pcov) > ubound) or np.allclose(pcov, 0.0, atol=1e-14))
