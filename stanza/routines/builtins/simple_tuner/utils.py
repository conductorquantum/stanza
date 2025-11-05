import numpy as np


def generate_linear_sweep(
    start_point: np.ndarray,
    direction: np.ndarray,
    total_sweep_dist: float,
    n_points: int,
) -> np.ndarray:
    """Generate evenly-spaced points along a line segment.

    Args:
        start_point: (d,) array with starting coordinates
        direction: (d,) array with normalized direction vector
        total_sweep_dist: The total sweep distance
        n_points: The number of points in the trace

    Returns:
        (n_points, d) array of coordinates along the line
    """
    t = np.linspace(0, total_sweep_dist, n_points)
    return start_point + np.outer(t, direction)
