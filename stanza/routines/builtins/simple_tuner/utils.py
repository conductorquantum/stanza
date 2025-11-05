from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class SweepGeometry:
    """Random sweep configuration in 2D voltage space."""

    start: NDArray[np.float64]
    end: NDArray[np.float64]
    direction: NDArray[np.float64]
    angle: float
    total_distance: float


@dataclass
class GateIndices:
    """Indices for different gate types in the voltage array."""

    plunger: list[int]
    reservoir: list[int]
    barrier: list[int]


def generate_linear_sweep(
    start_point: np.ndarray,
    direction: np.ndarray,
    total_sweep_dist: float,
    n_points: int,
) -> NDArray[np.float64]:
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


def generate_random_sweep(
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    scale: float,
    num_points: int,
) -> SweepGeometry | None:
    """Generate random sweep that stays within bounds.

    Args:
        x_bounds: (min, max) voltage bounds for X axis
        y_bounds: (min, max) voltage bounds for Y axis
        scale: Voltage spacing per step
        num_points: Number of points in sweep

    Returns:
        SweepGeometry if sweep stays in bounds, None otherwise
    """
    angle = np.random.uniform(0, 2 * np.pi)
    direction = np.array([np.cos(angle), np.sin(angle)])
    start = np.array(
        [
            np.random.uniform(*x_bounds),
            np.random.uniform(*y_bounds),
        ]
    )

    total_distance = scale * (num_points - 1)
    end = start + direction * total_distance

    # Check bounds
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    if not (x_min <= end[0] <= x_max and y_min <= end[1] <= y_max):
        return None

    return SweepGeometry(start, end, direction, angle, total_distance)


def build_voltage_array(
    sweep: SweepGeometry,
    num_points: int,
    gates: list[str],
    gate_idx: GateIndices,
    reservoir_voltages: dict[str, float],
    barrier_voltages: dict[str, float],
) -> NDArray[np.float64]:
    """Construct full voltage array with sweep and fixed gate voltages.

    Args:
        sweep: Sweep geometry configuration
        num_points: Number of points in sweep
        gates: List of all gate names
        gate_idx: Indices for each gate type
        reservoir_voltages: Fixed voltages for reservoir gates
        barrier_voltages: Fixed voltages for barrier gates

    Returns:
        (num_points, num_gates) array of voltages
    """
    sweep_voltages = generate_linear_sweep(
        sweep.start, sweep.direction, sweep.total_distance, num_points
    )

    voltages = np.zeros((num_points, len(gates)))
    voltages[:, gate_idx.plunger] = sweep_voltages

    # Vectorized assignment for fixed voltages
    for idx in gate_idx.reservoir:
        voltages[:, idx] = reservoir_voltages[gates[idx]]
    for idx in gate_idx.barrier:
        voltages[:, idx] = barrier_voltages[gates[idx]]

    return voltages


def compute_peak_spacings(
    peak_indices: NDArray[np.int64],
    sweep_voltages: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    """Calculate inter-peak spacings in voltage space.

    Args:
        peak_indices: Indices of detected peaks
        sweep_voltages: (num_points, 2) array of voltage coordinates

    Returns:
        Array of inter-peak spacings, or None if fewer than 3 peaks
    """
    if len(peak_indices) < 3:
        return None

    peak_positions = sweep_voltages[peak_indices]
    start_position = sweep_voltages[0]

    # Compute Euclidean distances from start point
    distances = np.linalg.norm(peak_positions - start_position, axis=1)

    # Inter-peak spacings
    return np.diff(distances)
