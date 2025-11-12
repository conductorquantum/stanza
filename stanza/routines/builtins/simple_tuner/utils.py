from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from stanza.device import Device
from stanza.models import GateType
from stanza.registry import ResultsRegistry


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


def build_gate_indices(gates: list[str], device: Device) -> GateIndices:
    """Extract indices for each gate type from the gate list.

    Args:
        gates: List of gate electrode names
        device: Device instance with get_gate_by_type method

    Returns:
        GateIndices with plunger, reservoir, and barrier indices
    """
    plunger_gates = device.get_gates_by_type(GateType.PLUNGER)
    reservoir_gates = device.get_gates_by_type(GateType.RESERVOIR)
    barrier_gates = device.get_gates_by_type(GateType.BARRIER)

    return GateIndices(
        plunger=[i for i, g in enumerate(gates) if g in plunger_gates],
        reservoir=[i for i, g in enumerate(gates) if g in reservoir_gates],
        barrier=[i for i, g in enumerate(gates) if g in barrier_gates],
    )


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


def build_full_voltages(
    sweep_voltages: NDArray[np.float64],
    gates: list[str],
    gate_idx: GateIndices,
    transition_voltages: dict[str, float],
    cutoff_voltages: dict[str, float],
    saturation_voltages: dict[str, float],
    barrier_voltages: dict[str, float] | None = None,
) -> NDArray[np.float64]:
    """Construct full voltage array from plunger sweep voltages.

    Works with arbitrary-dimensional sweep arrays (1D, 2D, 3D, etc).

    Args:
        sweep_voltages: (..., 2) array of plunger voltages
        gates: List of all gate names
        gate_idx: Indices for each gate type
        saturation_voltages: Fixed saturation voltages for all gates
        barrier_voltages: Optional explicit barrier voltages. If None, uses default logic.

    Returns:
        (..., num_gates) array of voltages
    """
    shape = sweep_voltages.shape[:-1]  # All dims except last (plunger coords)
    voltages = np.zeros(shape + (len(gates),))

    # Plunger voltages
    voltages[..., gate_idx.plunger] = sweep_voltages

    # Fixed voltages
    for idx in gate_idx.reservoir:
        voltages[..., idx] = saturation_voltages[gates[idx]]

    # Barrier voltages - use override if provided, otherwise default logic
    if barrier_voltages is not None:
        for idx in gate_idx.barrier:
            voltages[..., idx] = barrier_voltages[gates[idx]]
    else:
        for i, idx in enumerate(gate_idx.barrier):
            if i == 1:
                voltages[..., idx] = transition_voltages[gates[idx]]
            else:
                voltages[..., idx] = saturation_voltages[gates[idx]]

    return voltages


def check_voltages_in_bounds(
    voltages: NDArray[np.float64], safe_bounds: tuple[float, float]
) -> bool:
    """Check if all voltages are within safe bounds.

    Args:
        voltages: Array of voltages of any shape (..., num_gates)
        safe_bounds: (min, max) safe voltage bounds

    Returns:
        True if all voltages are within bounds, False otherwise
    """
    min_voltage, max_voltage = safe_bounds
    return bool(np.all((voltages >= min_voltage) & (voltages <= max_voltage)))


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


def get_global_turn_on_voltage(results: ResultsRegistry) -> float:
    """Get global turn voltage from results.

    Args:
        results: ResultsRegistry instance

    Returns:
        Global turn voltage
    """
    if not (res := results["global_accumulation"]):
        raise ValueError("Global turn on voltage not found")
    return float(res["global_turn_on_voltage"])


def get_voltages(
    gates: list[str],
    key: str,
    results: ResultsRegistry,
) -> dict[str, float]:
    """Get voltages for all gates.

    Args:
        gates: List of gate names
        key: Key to extract from characterization results, one of ["saturation_voltage", "cutoff_voltage", "transition_voltage"]
        results: ResultsRegistry instance

    Returns:
        Dict mapping gate names to voltages
    """
    if not (
        res := results.get("reservoir_characterization")["reservoir_characterization"]
    ) or not (
        fg := results.get("finger_gate_characterization")[
            "finger_gate_characterization"
        ]
    ):
        raise ValueError("Reservoir and finger gate characterization results not found")
    return {g: {**res, **fg}[g][key] for g in gates}


def get_plunger_gate_bounds(
    plunger_gates: list[str], results: ResultsRegistry
) -> dict[str, tuple[float, float]]:
    """Get plunger bounds for all plunger gates.

    Args:
        plunger_gates: List of plunger gate names
        results: ResultsRegistry instance

    Returns:
        Dict mapping plunger gate names to bounds
    """
    if not (
        res := results.get("reservoir_characterization")["reservoir_characterization"]
    ) or not (
        fg := results.get("finger_gate_characterization")[
            "finger_gate_characterization"
        ]
    ):
        raise ValueError("Reservoir and finger gate characterization results not found")
    v = {**res, **fg}
    return {
        g: tuple(sorted([v[g]["saturation_voltage"], v[g]["cutoff_voltage"]]))
        for g in plunger_gates
    }


def get_gate_safe_bounds(
    _gates: list[str], results: ResultsRegistry
) -> tuple[float, float]:
    """Get safe bounds for all gates.

    Args:
        _gates: List of gate names (unused, kept for API consistency)
        results: ResultsRegistry instance

    Returns:
        Tuple of (min, max) safe voltage bounds
    """
    if not (res := results.get("leakage_test")):
        raise ValueError("Leakage test results not found")
    return (res["min_safe_voltage_bound"], res["max_safe_voltage_bound"])
