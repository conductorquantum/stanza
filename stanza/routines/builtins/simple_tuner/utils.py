import numpy as np


def generate_linear_sweep(
    start_point: np.ndarray,
    direction: np.ndarray,
    total_sweep_dist: float,
    n_points: int,
) -> np.ndarray:
    """Generate evenly-spaced points along a line segment.

    Args:
        start_point: (2,) array with starting coordinates [X, Y]
        direction: (2,) array with normalized direction vector [uX, uY]
        total_sweep_dist: The total sweep distance
        n_points: The number of points in the trace

    Returns:
        (n_points, 2) array of [x, y] coordinates along the line
    """
    t = np.linspace(0, total_sweep_dist, n_points)
    return start_point + np.outer(t, direction)
