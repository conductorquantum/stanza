import numpy as np


def pcov_fail_criterion(pcov: np.ndarray, ubound: float = 1e10) -> bool:
    """Fail criterion for the covariance matrix of a non-linear fit using the scipy module."""
    return bool(np.any(np.diag(pcov) > ubound) or np.allclose(pcov, 0.0))
