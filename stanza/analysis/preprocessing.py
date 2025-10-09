import numpy as np


def normalize(a: np.ndarray) -> np.ndarray:
    """Scale array to [0, 1]. Constant arrays map to 0."""
    amin, amax = a.min(), a.max()
    return np.zeros_like(a, dtype=float) if amin == amax else (a - amin) / (amax - amin)
