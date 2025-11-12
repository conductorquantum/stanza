from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from stanza.device import Device
from stanza.logger.session import LoggerSession
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
        gates: List of gate electrode names.
        device: Device instance with get_gate_by_type method.

    Returns:
        GateIndices with plunger, reservoir, and barrier indices.
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
        start_point: (d,) array with starting coordinates.
        direction: (d,) array with normalized direction vector.
        total_sweep_dist: Total sweep distance.
        n_points: Number of points in the trace.

    Returns:
        (n_points, d) array of coordinates along the line.
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
        x_bounds: (min, max) voltage bounds for X axis.
        y_bounds: (min, max) voltage bounds for Y axis.
        scale: Voltage spacing per step.
        num_points: Number of points in sweep.

    Returns:
        SweepGeometry if sweep stays in bounds, None otherwise.
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


def build_full_voltages(
    sweep_voltages: NDArray[np.float64],
    gates: list[str],
    gate_idx: GateIndices,
    transition_voltages: dict[str, float],
    saturation_voltages: dict[str, float],
    barrier_voltages: dict[str, float] | None = None,
) -> NDArray[np.float64]:
    """Construct full voltage array from plunger sweep voltages.

    Works with arbitrary-dimensional sweep arrays (1D, 2D, 3D, etc). Applies
    fixed voltages to reservoir and barrier gates based on characterization data.

    Args:
        sweep_voltages: (..., 2) array of plunger voltages.
        gates: List of all gate names.
        gate_idx: Indices for each gate type.
        transition_voltages: Transition voltages for all gates.
        saturation_voltages: Saturation voltages for all gates.
        barrier_voltages: Optional explicit barrier voltages (overrides defaults).

    Returns:
        (..., num_gates) array of voltages.
    """
    shape = sweep_voltages.shape[:-1]
    voltages = np.zeros(shape + (len(gates),))

    voltages[..., gate_idx.plunger] = sweep_voltages

    for idx in gate_idx.reservoir:
        voltages[..., idx] = saturation_voltages[gates[idx]]

    if barrier_voltages:
        for idx in gate_idx.barrier:
            voltages[..., idx] = barrier_voltages[gates[idx]]
    else:
        # Default: middle barrier at transition, outer barriers at saturation
        for i, idx in enumerate(gate_idx.barrier):
            voltage = transition_voltages if i == 1 else saturation_voltages
            voltages[..., idx] = voltage[gates[idx]]

    return voltages


def check_voltages_in_bounds(
    voltages: NDArray[np.float64], safe_bounds: tuple[float, float]
) -> bool:
    """Check if all voltages are within safe bounds.

    Args:
        voltages: Array of voltages of any shape (..., num_gates).
        safe_bounds: (min, max) safe voltage bounds.

    Returns:
        True if all voltages are within bounds, False otherwise.
    """
    min_voltage, max_voltage = safe_bounds
    return bool(np.all((voltages >= min_voltage) & (voltages <= max_voltage)))


def compute_peak_spacings(
    peak_indices: NDArray[np.int64],
    sweep_voltages: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    """Calculate inter-peak spacings in voltage space.

    Args:
        peak_indices: Indices of detected peaks.
        sweep_voltages: (num_points, 2) array of voltage coordinates.

    Returns:
        Array of inter-peak spacings, or None if fewer than 3 peaks.
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
    """Get global turn-on voltage from results.

    Args:
        results: ResultsRegistry instance.

    Returns:
        Global turn-on voltage.

    Raises:
        ValueError: If global accumulation results not found.
    """
    if not (res := results["global_accumulation"]):
        raise ValueError("Global turn on voltage not found")
    return float(res["global_turn_on_voltage"])


def _get_gate_characterization(results: ResultsRegistry) -> dict[str, dict[str, float]]:
    """Get combined gate characterization results.

    Args:
        results: ResultsRegistry instance.

    Returns:
        Combined dict of reservoir and finger gate characterization.

    Raises:
        ValueError: If characterization results not found.
    """
    reservoir = results.get("reservoir_characterization", {}).get(
        "reservoir_characterization"
    )
    finger = results.get("finger_gate_characterization", {}).get(
        "finger_gate_characterization"
    )

    if not reservoir or not finger:
        raise ValueError("Gate characterization results not found")

    return {**reservoir, **finger}


def get_voltages(
    gates: list[str],
    key: str,
    results: ResultsRegistry,
) -> dict[str, float]:
    """Get voltages for all gates.

    Args:
        gates: List of gate names.
        key: Key to extract ("saturation_voltage", "cutoff_voltage", or "transition_voltage").
        results: ResultsRegistry instance.

    Returns:
        Dict mapping gate names to voltages.
    """
    characterization = _get_gate_characterization(results)
    return {g: characterization[g][key] for g in gates}


def get_plunger_gate_bounds(
    plunger_gates: list[str], results: ResultsRegistry
) -> dict[str, tuple[float, float]]:
    """Get plunger bounds for all plunger gates.

    Args:
        plunger_gates: List of plunger gate names.
        results: ResultsRegistry instance.

    Returns:
        Dict mapping plunger gate names to (min, max) voltage bounds.
    """
    characterization = _get_gate_characterization(results)
    bounds: dict[str, tuple[float, float]] = {}
    for g in plunger_gates:
        voltages = sorted(
            [
                characterization[g]["saturation_voltage"],
                characterization[g]["cutoff_voltage"],
            ]
        )
        bounds[g] = (voltages[0], voltages[1])
    return bounds


def get_gate_safe_bounds(results: ResultsRegistry) -> tuple[float, float]:
    """Get safe voltage bounds.

    Args:
        results: ResultsRegistry instance.

    Returns:
        (min, max) safe voltage bounds.

    Raises:
        ValueError: If leakage test results not found.
    """
    leakage = results.get("leakage_test")
    if not leakage:
        raise ValueError("Leakage test results not found")
    return (leakage["min_safe_voltage_bound"], leakage["max_safe_voltage_bound"])


def measure_and_classify(
    device: Any,
    client: Any,
    gates: list[str],
    voltages: Any,
    measure_electrode: str,
    model: str,
    reshape: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, bool, float]:
    """Measure current and classify result.

    Performs device measurement sweep and classifies the resulting current data
    using a specified ML model.

    Args:
        device: Device instance.
        client: Models client.
        gates: Gate electrode names.
        voltages: Voltage array.
        measure_electrode: Current measurement electrode.
        model: Classification model name.
        reshape: Optional reshape dimensions for output.

    Returns:
        Tuple of (currents, classification, score).
    """
    _, currents = device.sweep_nd(gates, voltages.tolist(), measure_electrode)
    currents = np.array(currents)

    if reshape is not None:
        currents = currents.reshape(reshape)

    result = client.models.execute(model=model, data=currents).output
    classification = result["classification"]
    score = result.get("score", 0.0)

    return currents, classification, score


def log_measurement(
    session: LoggerSession | None,
    name: str,
    measurement_id: str,
    data: dict[str, Any],
    metadata: dict[str, Any],
    routine_name: str,
) -> None:
    """Log measurement data to session.

    Args:
        session: Logger session instance (optional).
        name: Measurement name.
        measurement_id: Unique measurement identifier.
        data: Measurement data dict.
        metadata: Measurement metadata dict.
        routine_name: Name of the routine performing the measurement.
    """
    if session:
        session.log_measurement(
            name, {**data, "id": measurement_id}, metadata, routine_name
        )


def log_classification(
    session: LoggerSession | None,
    measurement_id: str,
    measurement_type: str,
    classification: bool,
    score: float,
    metadata: dict[str, Any],
    routine_name: str,
) -> None:
    """Log classification result to session.

    Args:
        session: Logger session instance (optional).
        measurement_id: Unique measurement identifier.
        measurement_type: Type of measurement being classified.
        classification: Classification result (True/False).
        score: Classification confidence score.
        metadata: Classification metadata dict.
        routine_name: Name of the routine performing the classification.
    """
    if session:
        session.log_analysis(
            "dqd_search_classification",
            {
                "measurement_id": measurement_id,
                "measurement_type": measurement_type,
                "classification": bool(classification),
                "score": float(score),
            },
            metadata,
            routine_name,
        )
