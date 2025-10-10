import numpy as np


def pcov_fail_criterion(pcov: np.ndarray, ubound: float = 1e-8) -> bool:
    """Fail criterion for the covariance matrix of the fitted parameters."""
    return bool(np.any(np.diag(pcov) > ubound) or np.allclose(pcov, 0.0, atol=1e-14))
