import numpy as np


def normalize(a: np.ndarray) -> np.ndarray:
    """Normalize array values to the range [0, 1] using min-max scaling.

    Applies linear transformation to scale all values in the input array
    to the [0, 1] range. The minimum value in the array maps to 0, and
    the maximum value maps to 1. This is useful for standardizing data
    ranges before curve fitting or other numerical operations.

    Args:
        a: Input array to normalize

    Returns:
        Normalized array with values in [0, 1]. If the input is a constant
        array (all values equal), returns an array of zeros with the same
        shape and float dtype.
    """
    amin, amax = a.min(), a.max()
    return np.zeros_like(a, dtype=float) if amin == amax else (a - amin) / (amax - amin)
